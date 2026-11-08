"""Stage B / C analysis: supervised prediction, group-level ablation, and group-level CCA.

Five independent classes separate responsibilities rather than splitting training, scoring, and ablation by script name:

- BottleneckMLP: defines only the network architecture.
- FeaturePrep: stratification, feature transforms, column selection, concatenation, and column standardization.
- HeadTrainer: validation splits, training, batch inference, and checkpoints; `run` is the stage B entry point.
- MetricScorer: discrete / continuous scoring and table-level summaries. FeatureWriter writes to uns/runs/<run_name>/.
- FeatureGroupEval: group-level leave-one-out / leave-one-in and group-level CCA.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts_formal.utils.elements_morphology import FEATURE_GROUP_ORDER, build_feature_groups
from scripts_formal.utils.elements_result_io import FeatureReader, FeatureWriter
from scripts_formal.utils.elements_statistician import CellStatistician, resolve_torch_device


INPUT_EMB = "emb*"
INPUT_MORPH = "morph_ext"
INPUT_CONCAT = "emb*||morph_ext"
HEAD_PROJ64 = "proj64_mlp"
PCA_EV_THR = 1e-8
N_EMB_PCA = 64


class BottleneckMLP(nn.Module):
    """Supervised bottleneck network: in_dim -> bottleneck_dim -> mlp_hidden -> out_dim.

    Regression (continuous_head=True) adds sigmoid after the final linear output, yielding values in (0, 1).
    The classification head omits this layer.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        mlp_hidden: int,
        out_dim: int = 1,
        continuous_head: bool = False,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.continuous_head = bool(continuous_head)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, self.out_dim),
        )

    def forward(self, x):
        out = self.mlp(self.proj(x))
        if self.continuous_head:
            out = torch.sigmoid(out)
        return out

    def extract_proj(self, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
        """Collect projection-layer outputs in batches for stage E Proj64-PCA."""

        self.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[start:start + batch_size]).to(device, non_blocking=True)
                out.append(self.proj(xb).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


class FeaturePrep:
    """Prepare training / evaluation data from the cell table.

    Read target specifications, build evaluation slices by obs columns, scale continuous targets, apply per-image
    z-score / row L2 to raw features, standardize columns, concatenate three default inputs, and select morphology groups.
    """

    class FixedMaxScaler:
        """Divide continuous targets by a fixed upper bound; inverse_transform multiplies it back. Match the StandardScaler interface."""

        def __init__(self, scale_max: float):
            self.scale_max = float(scale_max)

        def transform(self, y: np.ndarray) -> np.ndarray:
            return np.asarray(y, dtype=np.float64) / self.scale_max

        def inverse_transform(self, y: np.ndarray) -> np.ndarray:
            return np.asarray(y, dtype=np.float64) * self.scale_max

    def __init__(
        self,
        cell_table: dict[str, Any],
        n_mean: int = 1024,
        column_zscore: bool = True,
        feature_transforms: list[bool] | None = None,
        stratify_eval_key: str = "",
        meta_key: str = "gt_cell_type",
        target_kind: str = "discrete",
        meta_map: dict | None = None,
        meta_map_override: dict | None = None,
        type_name_to_index: dict[str, int] | None = None,
        type_index_to_name: dict[int, str] | None = None,
        obs_equals: dict[str, Any] | None = None,
        exclude_obs_in: dict[str, Any] | None = None,
        target_scale_max: float | None = None,
        lda_label_key: str = "",
        label_unit: str = "",
    ):
        self.source_table = cell_table
        self.cell_table = cell_table
        self.n_mean = int(n_mean)
        self.column_zscore = bool(column_zscore)
        self.feature_transforms = (
            list(feature_transforms) if feature_transforms is not None
            else [False, False, False, False]
        )
        self.stratify_eval_key = str(stratify_eval_key or "")
        self.meta_key = str(meta_key)
        self.target_kind = str(target_kind)
        self.meta_map = dict(meta_map or {})
        self.meta_map_override = meta_map_override
        self.type_name_to_index = dict(type_name_to_index or {})
        self.type_index_to_name = dict(type_index_to_name or {})
        self.obs_equals = dict(obs_equals or {})
        self.exclude_obs_in = dict(exclude_obs_in or {})
        self.target_scale_max = target_scale_max
        self.lda_label_key = str(lda_label_key or "")
        self.label_unit = str(label_unit or "")
        self.spec: dict[str, Any] | None = None
        self.y_train: np.ndarray | None = None
        self.y_eval: np.ndarray | None = None
        self.train_mask: np.ndarray | None = None
        self.test_mask: np.ndarray | None = None
        self.scopes: list[tuple[str, np.ndarray]] | None = None
        self.y_scaler: Any = None
        self.X_emb: np.ndarray | None = None
        self.X_morph: np.ndarray | None = None

    def copy_cell_table(self, cell_table: dict[str, Any]) -> dict[str, Any]:
        """Copy the main table dictionary so multiple runs cannot overwrite the same extraction."""

        copied = {
            "obs": cell_table["obs"].copy(),
            "X_sam": np.asarray(cell_table["X_sam"]),
            "embeddings": np.asarray(cell_table.get("embeddings", cell_table["X_sam"])),
            "extended_morph": np.asarray(cell_table["extended_morph"]),
            "extended_morph_raw": np.asarray(
                cell_table.get("extended_morph_raw", cell_table["extended_morph"])
            ),
            "extended_morph_columns": list(cell_table["extended_morph_columns"]),
            "extended_morph_raw_columns": list(
                cell_table.get("extended_morph_raw_columns", cell_table["extended_morph_columns"])
            ),
            "embedding_columns": list(cell_table.get("embedding_columns", [])),
            "feature_groups": dict(cell_table.get("feature_groups") or {}),
            "config": dict(cell_table.get("config") or {}),
        }
        return copied

    def prepare(self) -> FeaturePrep:
        """Filter in memory, encode targets, build slices, scale targets, and transform raw features."""

        table = self.copy_cell_table(self.source_table)
        meta_keys = [
            self.stratify_eval_key,
            self.lda_label_key,
            self.meta_key,
            *self.obs_equals.keys(),
            *self.exclude_obs_in.keys(),
        ]
        obs = FeatureReader.fill_obs_from_meta_map(
            table["obs"].reset_index(drop=True),
            meta_keys,
            self.meta_map,
        )
        if "obs_name" in obs.columns:
            obs = obs.set_index("obs_name", drop=False)
        table["obs"] = obs
        if self.stratify_eval_key:
            self.stratify_eval_key = FeatureReader.resolve_obs_key(
                obs, self.stratify_eval_key, self.meta_map,
            )
        if self.lda_label_key:
            self.lda_label_key = FeatureReader.resolve_obs_key(
                obs, self.lda_label_key, self.meta_map,
            )
        # Filter before target encoding; translate gt_cell_type to names and compare other columns in their native types.
        table = CellStatistician.apply_obs_equals_filter(
            table, self.obs_equals, self.meta_map, self.type_index_to_name,
        )
        table = CellStatistician.apply_obs_in_exclude(
            table, self.exclude_obs_in, self.meta_map, self.type_index_to_name,
        )
        obs, encoding = FeatureReader.resolve_prediction_target(
            obs=table["obs"].reset_index(drop=True),
            meta_key=self.meta_key,
            target_kind=self.target_kind,
            meta_map=self.meta_map,
            meta_map_override=self.meta_map_override,
            type_name_to_index=self.type_name_to_index,
            type_index_to_name=self.type_index_to_name,
        )
        if "obs_name" in obs.columns:
            obs = obs.set_index("obs_name", drop=False)
        table["obs"] = obs
        if not table["feature_groups"]:
            table["feature_groups"] = build_feature_groups(
                pd.DataFrame(columns=table["extended_morph_columns"])
            )
        self.cell_table = table
        self.spec = self.build_target_spec(encoding)
        self.prepare_target_arrays()
        self.X_emb, self.X_morph = self.apply_feature_transforms()
        return self

    def build_target_spec(self, target_encoding: dict[str, Any]) -> dict[str, Any]:
        """Assemble the prediction specification from this run's in-memory encoding."""

        class_index_to_name = dict(target_encoding.get("class_index_to_name") or {})
        n_classes = int(target_encoding.get("n_classes") or 0)
        class_names = [
            str(class_index_to_name.get(str(i), class_index_to_name.get(i, str(i))))
            for i in range(n_classes)
        ]
        return {
            "target_kind": self.target_kind,
            "meta_key": self.meta_key,
            "n_classes": n_classes,
            "class_names": class_names,
            "target_scale_max": self.target_scale_max,
            "lda_label_key": self.lda_label_key,
            "label_unit": self.label_unit,
            "class_index_to_name": class_index_to_name,
            "class_index_to_raw": dict(target_encoding.get("class_index_to_raw") or {}),
            "continuous_value_map": dict(target_encoding.get("continuous_value_map") or {}),
            "type_index_to_name": dict(self.type_index_to_name),
        }

    def read_target_spec(self) -> dict[str, Any]:
        """Support legacy calls by running prepare first if it has not been called."""

        if self.spec is None:
            self.prepare()
        return self.spec

    def build_stratify_scopes(self) -> list[tuple[str, np.ndarray]] | None:
        """Build evaluation slices for each nonempty obs column value; training still uses all train rows.

        Sort numerically if all values are numeric-convertible, otherwise as strings. Slice names are str() of native values.
        Translate gt_cell_type to type names before slicing; do not extract numbers from other columns.
        """

        key = self.stratify_eval_key
        if not key:
            return None
        obs = self.cell_table["obs"].reset_index(drop=True)
        key = FeatureReader.resolve_obs_key(obs, key, self.meta_map)
        if key == "gt_cell_type":
            series = FeatureReader.decode_gt_cell_type(obs[key], self.type_index_to_name)
        else:
            series = obs[key]
        values = [v for v in pd.unique(series) if pd.notna(v)]
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        if numeric.notna().all():
            values = [values[i] for i in np.argsort(numeric.to_numpy())]
        else:
            values = sorted(values, key=str)
        scopes: list[tuple[str, np.ndarray]] = []
        for value in values:
            name = str(value)
            mask = (series == value).to_numpy()
            scopes.append((name, mask))
            print(f"[B] stratify {key}={value!r} -> scope {name} n={int(mask.sum())}", flush=True)
        return scopes

    def prepare_target_arrays(self) -> None:
        """Extract targets and split train/test; scale continuous targets by a fixed upper bound or training-set z-score.

        Append stratified slices after all-test and intersect them with test_mask without changing the training split.
        All train / all-test rows must cover all K discrete classes; stratified slices may omit classes.
        """

        spec = self.spec
        extra_scopes = self.build_stratify_scopes()
        obs = self.cell_table["obs"].reset_index(drop=True)
        self.train_mask = obs["split"].to_numpy() == "train"
        self.test_mask = obs["split"].to_numpy() == "test"
        scopes = [("all-test", self.test_mask)]
        if extra_scopes:
            for name, mask in extra_scopes:
                if str(name) == "all-test":
                    continue
                scopes.append((str(name), np.asarray(mask, dtype=bool) & self.test_mask))
        self.scopes = scopes
        if spec["target_kind"] == "discrete":
            y = pd.to_numeric(obs["target"], errors="raise").to_numpy(dtype=np.int64)
            if np.unique(y).size != spec["n_classes"]:
                raise RuntimeError(
                    f"target classes {np.unique(y).tolist()} are inconsistent with n_classes={spec['n_classes']}."
                )
            if np.unique(y[self.train_mask]).size != spec["n_classes"]:
                raise RuntimeError("The training set does not contain all discrete classes")
            if np.unique(y[self.test_mask]).size != spec["n_classes"]:
                raise RuntimeError("The test set does not contain all discrete classes")
            for name, mask in scopes:
                print(
                    f"[B] scope {name}: n={int(mask.sum())} "
                    f"counts={np.bincount(y[mask], minlength=spec['n_classes']).tolist()}",
                    flush=True,
                )
            self.y_train = y
            self.y_eval = y
            self.y_scaler = None
            return

        # Continuous targets: fit on scaled values but evaluate in original units; divide by target_scale_max if provided.
        y = pd.to_numeric(obs["target"], errors="raise").to_numpy(dtype=np.float64)
        scale_max = spec.get("target_scale_max")
        if scale_max is not None:
            self.y_scaler = FeaturePrep.FixedMaxScaler(float(scale_max))
            print(f"[B] target_scale_max={float(scale_max):.4f} (y_train = y / scale_max)", flush=True)
        else:
            self.y_scaler = StandardScaler().fit(y[self.train_mask].reshape(-1, 1))
            print("[B] target scaled by train StandardScaler", flush=True)
        self.y_train = self.y_scaler.transform(y.reshape(-1, 1)).ravel().astype(np.float64)
        self.y_eval = y
        for name, mask in scopes:
            print(
                f"[B] scope {name}: n={int(mask.sum())} "
                f"y_mean={float(y[mask].mean()):.4f} y_std={float(y[mask].std()):.4f}",
                flush=True,
            )

    def apply_feature_transforms(self) -> tuple[np.ndarray, np.ndarray]:
        """Apply per-image z-score / per-cell L2 to raw matrices before column standardization.

        The four feature_transforms booleans control, in order: embedding per-image z-score,
        embedding row L2, morphology per-image z-score, and morphology row L2.
        """

        emb_zscore, emb_l2, morph_zscore, morph_l2 = self.feature_transforms
        X_emb = np.nan_to_num(
            np.asarray(self.cell_table["X_sam"][:, :self.n_mean], dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        X_m = np.nan_to_num(
            np.asarray(self.cell_table["extended_morph"], dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if emb_zscore or morph_zscore:
            image_names = self.cell_table["obs"].reset_index(drop=True)["image_name"].to_numpy()
            if emb_zscore:
                X_emb = self.zscore_by_image(X_emb, image_names)
            if morph_zscore:
                X_m = self.zscore_by_image(X_m, image_names)
        if emb_l2:
            X_emb = self.l2_normalize_rows(X_emb)
        if morph_l2:
            X_m = self.l2_normalize_rows(X_m)
        return X_emb, X_m

    def zscore_by_image(self, X: np.ndarray, image_names: np.ndarray) -> np.ndarray:
        """Apply z-score per image and channel using that image's cells; set zero-standard-deviation channels to 0."""

        out = np.array(X, dtype=np.float32, copy=True)
        for name in pd.unique(image_names):
            idx = image_names == name
            sub = out[idx]
            mu = sub.mean(axis=0)
            sd = sub.std(axis=0)
            zero = sd == 0
            sd = np.where(zero, 1.0, sd)
            scaled = (sub - mu) / sd
            scaled[:, zero] = 0.0
            out[idx] = scaled
        return out

    def l2_normalize_rows(self, X: np.ndarray) -> np.ndarray:
        """Divide each cell vector by its L2 norm; leave zero vectors unchanged."""

        X = np.asarray(X, dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.where(nrm == 0, 1.0, nrm)
        return (X / nrm).astype(np.float32)

    def column_standardize(self, X: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        """Estimate column means / standard deviations on train and standardize all rows; if disabled, only cast to float32."""

        if self.column_zscore:
            return StandardScaler().fit(X[train_mask]).transform(X).astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def build_input_matrices(
        self,
        X_emb: np.ndarray | None = None,
        X_morph: np.ndarray | None = None,
        train_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Build the three default input groups: emb*, morph_ext, and emb*||morph_ext."""

        if X_emb is None:
            X_emb = self.X_emb
            X_morph = self.X_morph
            train_mask = self.train_mask
        X_emb_u = self.column_standardize(X_emb, train_mask)
        X_morph_u = self.column_standardize(X_morph, train_mask)
        return {
            INPUT_EMB: X_emb_u,
            INPUT_MORPH: X_morph_u,
            INPUT_CONCAT: np.hstack([X_emb_u, X_morph_u]).astype(np.float32),
        }

    def morph_groups(self) -> dict[str, list[str]]:
        """Select this run's nonempty morphology group columns in a fixed group order."""

        raw = self.cell_table["feature_groups"]
        kept_set = set(self.cell_table["extended_morph_columns"])
        kept = {}
        for name in FEATURE_GROUP_ORDER:
            cols = [c for c in raw.get(name, []) if c in kept_set]
            if cols:
                kept[name] = list(cols)
        return kept


class HeadTrainer:
    """Train BottleneckMLP and run batch inference.

    Discrete tasks: cross entropy with balanced class weights and early stopping on validation macro AP.
    Continuous tasks: MSE and early stopping on validation R-squared. Restore the epoch with the best validation primary metric.
    """

    def __init__(
        self,
        device: torch.device,
        bottleneck_dim: int,
        mlp_hidden: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        min_delta_ap: float,
        val_fraction: float,
    ):
        self.device = device
        self.bottleneck_dim = int(bottleneck_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_delta_ap = float(min_delta_ap)
        self.val_fraction = float(val_fraction)

    def run(
        self,
        prep: FeaturePrep,
        h5ad_path: Path | str,
        run_name: str,
        mlp_seeds: tuple[int, ...],
        extra_metrics: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Stage B: prepare three default inputs -> train / evaluate -> write metrics and seed0 to uns/runs."""

        X_dict = prep.build_input_matrices()
        cell_table = prep.cell_table
        # Train and score proj64_mlp for each seed.
        scorer = MetricScorer()
        metrics, per_class = self.evaluate_inputs(
            X_dict=X_dict,
            prep=prep,
            mlp_seeds=mlp_seeds,
            extra_metrics=extra_metrics,
            scorer=scorer,
            h5ad_path=h5ad_path,
            run_name=run_name,
            obs=cell_table["obs"],
        )
        FeatureWriter.write_uns(h5ad_path, "type_prediction/metrics", metrics, run_name=run_name)
        if len(per_class):
            FeatureWriter.write_uns(h5ad_path, "type_prediction/metrics_per_class", per_class, run_name=run_name)
        FeatureWriter.write_uns(
            h5ad_path,
            "type_prediction/delta_metrics",
            scorer.build_delta_tables(metrics, prep.spec["target_kind"]), run_name=run_name)
        FeatureWriter.write_uns(
            h5ad_path,
            "type_prediction/metrics_summary",
            scorer.summarize_metrics(metrics, prep.spec["target_kind"], extra_metrics), run_name=run_name)
        return metrics

    def evaluate_inputs(
        self,
        X_dict: dict[str, np.ndarray],
        prep: FeaturePrep,
        mlp_seeds: tuple[int, ...],
        extra_metrics: tuple[str, ...],
        scorer: MetricScorer,
        h5ad_path: Path | None = None,
        run_name: str | None = None,
        obs: pd.DataFrame | None = None,
        log_tag: str = "[B]",
        slot: str = "input",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Train, infer, and score each X_dict input group for each seed. Seed 0 can be saved to h5ad_path."""

        spec = prep.spec
        y_train = prep.y_train
        y_eval = prep.y_eval
        y_scaler = prep.y_scaler
        target_kind = spec["target_kind"]
        metric_rows = []
        per_class_rows = []
        pred_records: list[dict[str, Any]] = []
        idx_tr = np.flatnonzero(prep.train_mask)
        n_classes = spec["n_classes"]
        out_dim = n_classes if target_kind == "discrete" else 1

        # Split validation data per seed (stratified for discrete, random for continuous), then train all inputs.
        for seed in mlp_seeds:
            idx_fit, idx_val, class_w = self.split_fit_val(idx_tr, y_train, spec, seed)
            for input_name, X in X_dict.items():
                self.set_seed(seed)
                print(
                    f"{log_tag} seed={seed} {HEAD_PROJ64} {slot}={input_name} dim={X.shape[1]}",
                    flush=True,
                )
                model = BottleneckMLP(
                    X.shape[1], self.bottleneck_dim, self.mlp_hidden, out_dim=out_dim,
                    continuous_head=(target_kind == "continuous"),
                ).to(self.device)
                fit_info = self.train(
                    model, X, y_train, idx_fit, idx_val, seed,
                    target_kind=target_kind, n_classes=n_classes, class_weight=class_w,
                )
                pred_all = self.predict(model, X)
                if target_kind == "continuous":
                    pred_eval = y_scaler.inverse_transform(
                        np.asarray(pred_all, dtype=np.float64).reshape(-1, 1)
                    ).ravel()
                else:
                    pred_eval = pred_all
                print(f"  fit={fit_info}", flush=True)
                if seed == 0 and h5ad_path is not None:
                    self.save_seed0(h5ad_path, input_name, model, out_dim, run_name=run_name)
                metric_rows.extend(scorer.score_rows(
                    seed, HEAD_PROJ64, input_name, y_eval, prep.scopes, pred_eval, spec,
                    per_class_rows, extra_metrics=extra_metrics, obs=obs,
                ))
                pred_records.append({
                    "seed": int(seed), "head": HEAD_PROJ64, "input": input_name, "pred": pred_eval,
                })
                del model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        scorer.write_fov_time_means(obs, y_eval, pred_records, spec, h5ad_path, run_name=run_name)
        return pd.DataFrame(metric_rows), pd.DataFrame(per_class_rows)

    def split_fit_val(
        self,
        idx_tr: np.ndarray,
        y_train: np.ndarray,
        spec: dict[str, Any],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Split validation data from train; stratify discrete targets by class and compute balanced class weights."""

        if spec["target_kind"] == "discrete":
            idx_fit, idx_val = train_test_split(
                idx_tr, test_size=self.val_fraction, stratify=y_train[idx_tr], random_state=seed,
            )
            class_w = self.balanced_class_weight(y_train[idx_fit], spec["n_classes"])
        else:
            idx_fit, idx_val = train_test_split(
                idx_tr, test_size=self.val_fraction, random_state=seed,
            )
            class_w = None
        return idx_fit, idx_val, class_w

    def train(
        self,
        model,
        X_z,
        y,
        idx_fit,
        idx_val,
        seed,
        target_kind: str = "discrete",
        n_classes: int = 2,
        class_weight: np.ndarray | None = None,
    ) -> dict:
        """Use Adam with early stopping on the validation primary metric; restore the epoch with the best metric."""

        device = self.device
        batch_size = self.batch_size
        X_fit = torch.from_numpy(X_z[idx_fit])
        X_val = torch.from_numpy(X_z[idx_val]).to(device)
        y_val_np = y[idx_val]
        # Continuous: MSE between final sigmoid outputs and y/scale_max; validate with R-squared.
        # Discrete: balanced-class-weight cross entropy and validation macro AP. No sigmoid in the classification head.
        if target_kind == "continuous":
            y_fit = torch.from_numpy(y[idx_fit].astype(np.float32))
            loss_fn = nn.MSELoss()
            score_name = "r2"
        else:
            y_fit = torch.from_numpy(y[idx_fit].astype(np.int64))
            weight = None
            if class_weight is not None:
                weight = torch.as_tensor(class_weight, dtype=torch.float32, device=device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            score_name = "macro_ap"
        loader = DataLoader(
            TensorDataset(X_fit, y_fit),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            drop_last=False,
        )
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_score = -np.inf
        best_state = None
        best_epoch = 0
        stale = 0
        last_epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            last_epoch = epoch
            model.train()
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                if target_kind == "continuous":
                    loss = loss_fn(out.view(-1), yb.view(-1))
                else:
                    loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_out = []
                for start in range(0, len(X_val), batch_size):
                    val_out.append(model(X_val[start:start + batch_size]))
                val_out = torch.cat(val_out, dim=0).cpu().numpy()
            if target_kind == "continuous":
                score = float(r2_score(y_val_np, val_out.reshape(-1)))
            else:
                score = float(np.nanmean(
                    MetricScorer.per_class_ap(y_val_np, MetricScorer.softmax_np(val_out), n_classes)
                ))
            if score > best_score + self.min_delta_ap:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.to(device)
        return {
            "best_epoch": int(best_epoch),
            "n_epochs": int(last_epoch),
            "best_val_score": float(best_score),
            "best_val_score_name": score_name,
            "n_params": int(sum(p.numel() for p in model.parameters())),
        }

    def predict(self, model, X_z) -> np.ndarray:
        """Run batched forward passes. Return (N,) for out_dim=1, otherwise (N, K)."""

        model.eval()
        logits_all = []
        with torch.no_grad():
            for start in range(0, len(X_z), self.batch_size):
                xb = torch.from_numpy(X_z[start:start + self.batch_size]).to(
                    self.device, non_blocking=True,
                )
                out = model(xb)
                if out.ndim == 2 and out.shape[1] == 1:
                    out = out.view(-1)
                logits_all.append(out.cpu())
        return torch.cat(logits_all, dim=0).numpy()

    def save_seed0(
        self,
        h5ad_path: Path,
        input_name: str,
        model,
        out_dim: int,
        run_name: str | None = None,
    ) -> None:
        """Save the seed 0 BottleneckMLP for stage E extraction of the 64-dimensional bottleneck."""

        if input_name == INPUT_EMB:
            stem = "emb"
        elif input_name == INPUT_MORPH:
            stem = "morph"
        else:
            stem = str(input_name).replace("*", "").replace("||", "_concat_").replace("/", "_")
        FeatureWriter.write_seed0(
            h5ad_path,
            stem,
            {
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "in_dim": int(model.proj[0].in_features),
                "bottleneck_dim": int(self.bottleneck_dim),
                "mlp_hidden": int(self.mlp_hidden),
                "out_dim": int(out_dim),
                "input_name": str(input_name),
                "seed": 0,
            }, run_name=run_name)

    @staticmethod
    def balanced_class_weight(y: np.ndarray, n_classes: int) -> np.ndarray:
        """sklearn class_weight='balanced'：w_k = n / (K * n_k)。"""

        counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(np.float64)
        n = float(len(y))
        return n / (float(n_classes) * counts)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set random seeds for numpy / torch / cuda."""

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class MetricScorer:
    """Score predictions: discrete macro / weighted AP, continuous R-squared / RMSE, and slice / cross-seed summaries."""

    def score_rows(
        self,
        seed: int,
        head: str,
        input_name: str,
        y: np.ndarray,
        scopes: list[tuple[str, np.ndarray]],
        pred_all: np.ndarray,
        spec: dict[str, Any],
        per_class_rows: list[dict[str, Any]],
        extra_metrics: tuple[str, ...] = (),
        obs: pd.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        """Collect default metrics across all slices for one (seed, head, input) into rows."""

        extra = set(extra_metrics)
        rows = []
        target_kind = spec["target_kind"]
        for scope_name, mask in scopes:
            if not mask.any():
                continue
            row = {
                "seed": int(seed),
                "head": head,
                "input": input_name,
                "scope": scope_name,
                "meta_key": spec["meta_key"],
                "target_kind": target_kind,
                "n_classes": spec["n_classes"],
                "n_test": int(mask.sum()),
            }
            row.update(self.count_eval_units(obs, mask))
            if target_kind == "discrete":
                met = self.score_discrete(y[mask], pred_all[mask], spec["class_names"])
                row["macro_ap"] = met["macro_ap"]
                row["weighted_ap"] = met["weighted_ap"]
                row["n_classes_present"] = met["n_classes_present"]
                row["random_macro_ap"] = 1.0 / float(spec["n_classes"])
                if "macro_auc" in extra:
                    row["macro_auc"] = met["macro_auc"]
                for item in met["per_class"]:
                    per_class_rows.append({
                        "seed": int(seed),
                        "head": head,
                        "input": input_name,
                        "scope": scope_name,
                        **item,
                    })
                print(
                    f"  {head} {input_name} {scope_name}: macro_AP={met['macro_ap']:.4f} "
                    f"weighted_AP={met['weighted_ap']:.4f} n_present={met['n_classes_present']}",
                    flush=True,
                )
            else:
                met = self.score_continuous(y[mask], pred_all[mask])
                row["r2"] = met["r2"]
                row["rmse"] = met["rmse"]
                row["n_test"] = met["n_test"]
                if "mae" in extra:
                    row["mae"] = met["mae"]
                if "spearman" in extra:
                    row["spearman"] = met["spearman"]
                print(
                    f"  {head} {input_name} {scope_name}: R2={met['r2']:.4f} RMSE={met['rmse']:.4f}",
                    flush=True,
                )
            rows.append(row)
        return rows

    def score_discrete(self, y_true: np.ndarray, logits: np.ndarray, class_names: list[str]) -> dict:
        """Discrete tasks: macro AP, weighted AP, macro AUC, and per-class AP, Precision, and Recall."""

        y_true = np.asarray(y_true, dtype=np.int64)
        n_classes = len(class_names)
        proba = self.softmax_np(logits)
        y_pred = np.argmax(proba, axis=1)
        ap = self.per_class_ap(y_true, proba, n_classes)
        auc = self.per_class_auc(y_true, proba, n_classes)
        precision = precision_score(
            y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0,
        )
        recall = recall_score(
            y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0,
        )
        counts = np.bincount(y_true, minlength=n_classes)
        n_total = float(len(y_true))
        present = counts > 0
        n_classes_present = int(present.sum())
        if n_total and np.any(np.isnan(ap[present])):
            weighted_ap = float("nan")
        elif n_total:
            weighted_ap = float(np.dot(counts / n_total, np.nan_to_num(ap, nan=0.0)))
        else:
            weighted_ap = float("nan")
        per_class = []
        for k, name in enumerate(class_names):
            per_class.append({
                "class_index": k,
                "class_name": name,
                "ap": float(ap[k]),
                "auc": float(auc[k]),
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "n": int(counts[k]),
                "prevalence": float(counts[k] / n_total) if n_total else float("nan"),
            })
        return {
            "macro_ap": float(np.nanmean(ap)),
            "weighted_ap": weighted_ap,
            "macro_auc": float(np.nanmean(auc)),
            "n_test": int(len(y_true)),
            "n_classes_present": n_classes_present,
            "per_class": per_class,
        }

    def score_continuous(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Continuous tasks: R-squared, RMSE, MAE, and Spearman rho (write the last two only when requested by extra_metrics)."""

        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        rho = float(spearmanr(y_true, y_pred).correlation)
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse,
            "spearman": rho,
            "n_test": int(len(y_true)),
        }

    def build_delta_tables(self, metrics: pd.DataFrame, target_kind: str) -> pd.DataFrame:
        """Append primary-metric differences relative to emb* and morph_ext."""

        metric = "macro_ap" if target_kind == "discrete" else "r2"
        rows = []
        for seed in sorted(metrics["seed"].unique()):
            for head in metrics["head"].unique():
                for scope in metrics["scope"].unique():
                    sub = metrics[
                        (metrics["seed"] == seed)
                        & (metrics["head"] == head)
                        & (metrics["scope"] == scope)
                    ]
                    if sub.empty:
                        continue
                    by_input = {row["input"]: float(row[metric]) for _, row in sub.iterrows()}
                    for _, sel in sub.iterrows():
                        row = {
                            "seed": int(seed),
                            "head": head,
                            "scope": scope,
                            "input": sel["input"],
                            metric: float(sel[metric]),
                            "n_test": int(sel["n_test"]),
                        }
                        if INPUT_EMB in by_input:
                            row[f"delta_{metric}_vs_emb"] = float(sel[metric]) - by_input[INPUT_EMB]
                        if INPUT_MORPH in by_input:
                            row[f"delta_{metric}_vs_morph"] = float(sel[metric]) - by_input[INPUT_MORPH]
                        rows.append(row)
        return pd.DataFrame(rows)

    def summarize_metrics(
        self,
        metrics: pd.DataFrame,
        target_kind: str,
        extra_metrics: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Summarize default metrics across seeds using medians and means."""

        if target_kind == "discrete":
            metric_names = ["macro_ap", "weighted_ap"]
            primary = "macro_ap"
        else:
            metric_names = ["r2", "rmse"]
            primary = "r2"
        for name in extra_metrics:
            if name in metrics.columns and name not in metric_names:
                metric_names.append(name)
        grp = metrics.groupby(["input", "head", "scope"], sort=False)
        agg = {
            "n_seeds": (primary, "size"),
            "n_test": ("n_test", "first"),
        }
        for count_col in ("n_test_cells", "n_test_images", "n_test_fovs", "n_test_tracks"):
            if count_col in metrics.columns:
                agg[count_col] = (count_col, "first")
        for name in metric_names:
            if name not in metrics.columns:
                continue
            agg[f"{name}_median"] = (name, "median")
            agg[f"{name}_mean"] = (name, "mean")
            agg[f"{name}_std"] = (name, "std")
            agg[f"{name}_min"] = (name, "min")
            agg[f"{name}_max"] = (name, "max")
        if "n_classes_present" in metrics.columns:
            agg["n_classes_present"] = ("n_classes_present", "min")
        return grp.agg(**{k: v for k, v in agg.items()}).reset_index()

    def count_eval_units(self, obs: pd.DataFrame | None, mask: np.ndarray) -> dict[str, int]:
        """Report cell, image, FOV, and track counts side by side when fov_id / track_id are available."""

        if obs is None:
            return {}
        obs = obs.reset_index(drop=True)
        if "fov_id" not in obs.columns and "track_id" not in obs.columns:
            return {}
        sub = obs.loc[np.asarray(mask, dtype=bool)]
        out = {"n_test_cells": int(sub.shape[0])}
        if "image_name" in sub.columns:
            out["n_test_images"] = int(sub["image_name"].nunique())
        elif "image_id" in sub.columns:
            out["n_test_images"] = int(sub["image_id"].nunique())
        if "fov_id" in sub.columns:
            out["n_test_fovs"] = int(sub["fov_id"].nunique())
        if "track_id" in sub.columns:
            tids = pd.to_numeric(sub["track_id"], errors="coerce")
            out["n_test_tracks"] = int(tids[tids >= 0].nunique())
        return out

    def write_fov_time_means(
        self,
        obs: pd.DataFrame | None,
        y_true: np.ndarray,
        pred_records: list[dict[str, Any]],
        spec: dict[str, Any],
        h5ad_path: Path | None,
        run_name: str | None = None,
    ) -> None:
        """FOV-time mean observation table: average cell predictions per (fov_id, timepoint_index), then compute test R-squared.

        Skip if fov_id / timepoint_index are absent. Exclude from the stage D wide table.
        """

        if obs is None or not pred_records or spec.get("target_kind") != "continuous":
            return
        if h5ad_path is None:
            return
        obs = obs.reset_index(drop=True)
        if "fov_id" not in obs.columns or "timepoint_index" not in obs.columns:
            return
        test_mask = obs["split"].to_numpy() == "test"
        y_true = np.asarray(y_true, dtype=np.float64)
        fov_rows: list[dict[str, Any]] = []
        for rec in pred_records:
            pred = np.asarray(rec["pred"], dtype=np.float64).ravel()
            sub = obs.loc[test_mask, ["fov_id", "timepoint_index"]].copy()
            sub["y_true"] = y_true[test_mask]
            sub["y_pred"] = pred[test_mask]
            grouped = sub.groupby(["fov_id", "timepoint_index"], as_index=False)[["y_true", "y_pred"]].mean()
            met = self.score_continuous(grouped["y_true"].to_numpy(), grouped["y_pred"].to_numpy())
            fov_rows.append({
                "seed": int(rec["seed"]),
                "head": rec["head"],
                "input": rec["input"],
                "n_pairs": int(grouped.shape[0]),
                "r2": met["r2"],
                "rmse": met["rmse"],
                "mae": met["mae"],
                "spearman": met["spearman"],
            })
        if fov_rows:
            FeatureWriter.write_uns(
                h5ad_path, "type_prediction/fov_time_mean_metrics", pd.DataFrame(fov_rows), run_name=run_name)

    @staticmethod
    def softmax_np(logits: np.ndarray) -> np.ndarray:
        """Apply softmax along the last dimension."""

        z = np.asarray(logits, dtype=np.float64)
        z = z - np.max(z, axis=-1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=-1, keepdims=True)

    @staticmethod
    def per_class_ap(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute one-vs-rest Average Precision per class; return nan if the class is absent from the slice."""

        y_true = np.asarray(y_true, dtype=np.int64)
        ap = np.full(n_classes, np.nan, dtype=np.float64)
        for k in range(n_classes):
            mask = y_true == k
            if mask.any():
                ap[k] = float(average_precision_score(mask.astype(np.int64), proba[:, k]))
        return ap

    @staticmethod
    def per_class_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute one-vs-rest AUC per class; return nan if the slice lacks that class or negative examples."""

        y_true = np.asarray(y_true, dtype=np.int64)
        auc = np.full(n_classes, np.nan, dtype=np.float64)
        for k in range(n_classes):
            mask = y_true == k
            if mask.any() and (~mask).any():
                auc[k] = float(roc_auc_score(mask.astype(np.int64), proba[:, k]))
        return auc


class FeatureGroupEval:
    """Group-level morphology ablation and CCA; delegate training and scoring to HeadTrainer and MetricScorer.

    Ablate only canonical morph_ext columns: full-model reference + leave-one-group-out + leave-one-in.
    CCA: standardize embedding columns on train, then apply PCA64; truncate near-zero eigenvalues within each morphology group.
    Fit on train and report only on all test rows.
    """

    def __init__(self, prep: FeaturePrep, trainer: HeadTrainer, scorer: MetricScorer):
        self.prep = prep
        self.trainer = trainer
        self.scorer = scorer

    @classmethod
    def from_cell_table(
        cls,
        cell_table: dict[str, Any],
        n_mean: int,
        device: torch.device,
        bottleneck_dim: int,
        mlp_hidden: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        min_delta_ap: float,
        val_fraction: float,
        column_zscore: bool = True,
        feature_transforms: list[bool] | None = None,
        stratify_eval_key: str = "",
        meta_key: str = "gt_cell_type",
        target_kind: str = "discrete",
        meta_map: dict | None = None,
        meta_map_override: dict | None = None,
        type_name_to_index: dict[str, int] | None = None,
        type_index_to_name: dict[int, str] | None = None,
        obs_equals: dict[str, Any] | None = None,
        exclude_obs_in: dict[str, Any] | None = None,
        target_scale_max: float | None = None,
        lda_label_key: str = "",
        label_unit: str = "",
    ) -> FeatureGroupEval:
        """Build the group-level evaluator using the same FeaturePrep / HeadTrainer / MetricScorer as stage B."""

        prep = FeaturePrep(
            cell_table,
            n_mean=n_mean,
            column_zscore=column_zscore,
            feature_transforms=feature_transforms,
            stratify_eval_key=stratify_eval_key,
            meta_key=meta_key,
            target_kind=target_kind,
            meta_map=meta_map,
            meta_map_override=meta_map_override,
            type_name_to_index=type_name_to_index,
            type_index_to_name=type_index_to_name,
            obs_equals=obs_equals,
            exclude_obs_in=exclude_obs_in,
            target_scale_max=target_scale_max,
            lda_label_key=lda_label_key,
            label_unit=label_unit,
        ).prepare()
        trainer = HeadTrainer(
            device=device,
            bottleneck_dim=bottleneck_dim,
            mlp_hidden=mlp_hidden,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta_ap=min_delta_ap,
            val_fraction=val_fraction,
        )
        return cls(prep, trainer, MetricScorer())

    def run(
        self,
        h5ad_path: Path | str,
        run_name: str,
        mlp_seeds: tuple[int, ...],
        extra_metrics: tuple[str, ...],
    ) -> dict[str, Any]:
        """Write the group inventory, then run ablation and CCA."""

        groups = self.prep.morph_groups()
        self.write_group_inventory(h5ad_path, groups, run_name)
        self.run_ablation(h5ad_path, groups, mlp_seeds, extra_metrics, run_name)
        self.run_cca(h5ad_path, groups, run_name)
        return {
            "groups": list(groups.keys()),
            "analysis_role": "observational_not_deployment",
        }

    def run_ablation(
        self,
        h5ad_path: Path | str,
        groups: dict[str, list[str]],
        mlp_seeds: tuple[int, ...],
        extra_metrics: tuple[str, ...],
        run_name: str,
    ) -> None:
        """Retrain full-model, leave-one-group-out, and group-only variants; write metrics, deltas, and single-group metrics."""

        feature_cols = list(self.prep.cell_table["extended_morph_columns"])
        col_to_i = {c: i for i, c in enumerate(feature_cols)}
        # Cast morphology to float64 to fill nan, matching the legacy diagnose path; ablation inputs remain float32.
        X_full = np.nan_to_num(
            np.asarray(self.prep.X_morph, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0,
        ).astype(np.float32)
        variants: list[tuple[str, np.ndarray]] = [
            ("full", np.arange(len(feature_cols), dtype=np.int64)),
        ]
        for name in groups:
            drop_set = set(groups[name])
            variants.append((
                f"drop_{name}",
                np.array([col_to_i[c] for c in feature_cols if c not in drop_set], dtype=np.int64),
            ))
        for name in groups:
            variants.append((
                f"only_{name}",
                np.array([col_to_i[c] for c in groups[name]], dtype=np.int64),
            ))

        metric_parts = []
        per_class_parts = []
        train_mask = self.prep.train_mask
        for variant_name, keep_idx in variants:
            Xz = self.prep.column_standardize(X_full[:, keep_idx], train_mask)
            print(
                f"[C] ablate variant={variant_name} n_cols={Xz.shape[1]} head={HEAD_PROJ64}",
                flush=True,
            )
            metrics, per_class = self.trainer.evaluate_inputs(
                X_dict={variant_name: Xz},
                prep=self.prep,
                mlp_seeds=mlp_seeds,
                extra_metrics=extra_metrics,
                scorer=self.scorer,
                log_tag="[C] ablate",
                slot="variant",
                run_name=run_name,
            )
            metric_parts.append(metrics)
            if len(per_class):
                per_class_parts.append(per_class)

        metrics = pd.concat(metric_parts, ignore_index=True)
        FeatureWriter.write_uns(h5ad_path, "group_ablation/metrics", metrics, run_name=run_name)
        if per_class_parts:
            FeatureWriter.write_uns(
                h5ad_path,
                "group_ablation/metrics_per_class",
                pd.concat(per_class_parts, ignore_index=True), run_name=run_name)
        delta, only = self.build_ablation_tables(metrics, groups, self.prep.spec["target_kind"])
        FeatureWriter.write_uns(h5ad_path, "group_ablation/ablation_delta", delta, run_name=run_name)
        FeatureWriter.write_uns(h5ad_path, "group_ablation/ablation_only", only, run_name=run_name)

    def run_cca(self, h5ad_path: Path | str, groups: dict[str, list[str]], run_name: str) -> None:
        """Run CCA between embedding PCA64 and each morphology group after within-group PCA truncation; evaluate only all test rows."""

        feature_cols = list(self.prep.cell_table["extended_morph_columns"])
        col_to_i = {c: i for i, c in enumerate(feature_cols)}
        train_mask = self.prep.train_mask
        test_mask = self.prep.test_mask
        n_mean = self.prep.n_mean
        # Standardize both sides by training-set columns; reduce embeddings to at most 64 dimensions with PCA.
        X_e = np.nan_to_num(
            np.asarray(self.prep.X_emb[:, :n_mean], dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        X_m = np.nan_to_num(
            np.asarray(self.prep.X_morph, dtype=np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        Xe = StandardScaler().fit(X_e[train_mask]).transform(X_e)
        Xm = StandardScaler().fit(X_m[train_mask]).transform(X_m)
        n_train = int(train_mask.sum())
        n_pca = int(min(N_EMB_PCA, Xe.shape[1], n_train - 1))
        pca_e = PCA(n_components=n_pca, svd_solver="randomized", random_state=0)
        Ze_tr = pca_e.fit_transform(Xe[train_mask])
        Ze_te = pca_e.transform(Xe[test_mask])

        rows = []
        for group_name, gcols in groups.items():
            idx = np.array([col_to_i[c] for c in gcols], dtype=np.int64)
            Y_tr = Xm[train_mask][:, idx]
            Y_te = Xm[test_mask][:, idx]
            # Within-group PCA: discard eigenvalues below the threshold to avoid CCA degeneracy from Haralick collinearity.
            pca_y = PCA(svd_solver="full", random_state=0)
            pca_y.fit(Y_tr)
            n_y = int(np.sum(pca_y.explained_variance_ > PCA_EV_THR))
            n_y = int(min(n_y, Y_tr.shape[1], Ze_tr.shape[1], n_train - 2))
            Zy_tr = pca_y.transform(Y_tr)[:, :n_y]
            Zy_te = pca_y.transform(Y_te)[:, :n_y]
            group_var_kept = float(np.sum(pca_y.explained_variance_ratio_[:n_y]))
            # Fit CCA on train; compute per-axis test Pearson correlations and reconstruct the original group from embedding-side canonical variates.
            cca = CCA(n_components=n_y, scale=False, max_iter=1000)
            cca.fit(Ze_tr, Zy_tr)
            U_tr, _V_tr = cca.transform(Ze_tr, Zy_tr)
            U_te, V_te = cca.transform(Ze_te, Zy_te)
            rhos = [self.pearson_corr(U_te[:, k], V_te[:, k]) for k in range(n_y)]
            recon1 = self.morph_ss_explained(Y_tr, Y_te, U_tr[:, :1], U_te[:, :1])
            recon_all = self.morph_ss_explained(Y_tr, Y_te, U_tr, U_te)
            row = {
                "group": group_name,
                "n_group_cols": int(len(gcols)),
                "n_group_pcs": n_y,
                "group_pc_var_kept": group_var_kept,
                "n_emb_pcs": n_pca,
                "n_test": int(test_mask.sum()),
                "rho_1": rhos[0],
                "rho_2": rhos[1] if n_y > 1 else np.nan,
                "rho_3": rhos[2] if n_y > 2 else np.nan,
                "morph_ss_explained_r1": recon1,
                "morph_ss_explained_all": recon_all,
                "analysis_role": "observational_not_deployment",
            }
            rows.append(row)
            print(
                f"[C] cca {group_name} n_cols={len(gcols)} n_pcs={n_y} "
                f"rho_1={rhos[0]:.4f} ss_r1={recon1:.4f} ss_all={recon_all:.4f}",
                flush=True,
            )
        FeatureWriter.write_uns(h5ad_path, "group_cca/cca_by_group", pd.DataFrame(rows), run_name=run_name)

    def write_group_inventory(self, h5ad_path: Path | str, groups: dict[str, list[str]], run_name: str) -> None:
        """Write each group's column count for this task."""

        rows = [{"group": name, "n_cols": len(cols), "columns": ",".join(cols)} for name, cols in groups.items()]
        FeatureWriter.write_uns(h5ad_path, "group_inventory", pd.DataFrame(rows), run_name=run_name)

    def build_ablation_tables(
        self,
        metrics: pd.DataFrame,
        groups: dict[str, list[str]],
        target_kind: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Write leave-one-out deltas and leave-one-in single-group metrics together (use medians for MLP)."""

        metric_names = ["macro_ap", "weighted_ap"] if target_kind == "discrete" else ["r2", "rmse"]
        delta_rows = []
        only_rows = []
        for head, head_df in metrics.groupby("head", sort=False):
            for scope, scope_df in head_df.groupby("scope", sort=False):
                full_mask = scope_df["input"] == "full"
                n_test = int(scope_df.loc[full_mask, "n_test"].iloc[0])
                n_seeds_full = int(full_mask.sum())
                full_med = {
                    name: float(scope_df.loc[full_mask, name].median())
                    for name in metric_names
                }
                for group_name in groups:
                    drop_mask = scope_df["input"] == f"drop_{group_name}"
                    only_mask = scope_df["input"] == f"only_{group_name}"
                    base = {
                        "head": head,
                        "scope": scope,
                        "group": group_name,
                        "n_group_cols": len(groups[group_name]),
                        "n_test": n_test,
                        "n_seeds_full": n_seeds_full,
                        "analysis_role": "observational_not_deployment",
                    }
                    delta = {**base, "n_seeds_drop": int(drop_mask.sum())}
                    only = {**base, "n_seeds_only": int(only_mask.sum())}
                    for name in metric_names:
                        full_v = full_med[name]
                        drop_v = float(scope_df.loc[drop_mask, name].median())
                        delta[f"{name}_full"] = full_v
                        delta[f"{name}_drop"] = drop_v
                        delta[f"delta_{name}"] = full_v - drop_v
                        only[f"{name}_full"] = full_v
                        only[f"{name}_only"] = float(scope_df.loc[only_mask, name].median())
                    delta_rows.append(delta)
                    only_rows.append(only)
        return pd.DataFrame(delta_rows), pd.DataFrame(only_rows)

    def morph_ss_explained(
        self,
        Y_tr: np.ndarray,
        Y_te: np.ndarray,
        U_tr: np.ndarray,
        U_te: np.ndarray,
    ) -> float:
        """Fit Y~U by least squares on train and report the reconstructed fraction of the group's total sum of squares on test."""

        reg = LinearRegression()
        reg.fit(U_tr, Y_tr)
        pred = reg.predict(U_te)
        resid = ((Y_te - pred) ** 2).sum()
        center = ((Y_te - Y_te.mean(axis=0)) ** 2).sum()
        return float(1.0 - resid / center)

    def pearson_corr(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Pearson correlation after centering."""

        a = np.asarray(a, dtype=np.float64)
        a = a - a.mean()
        b = np.asarray(b, dtype=np.float64)
        b = b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
