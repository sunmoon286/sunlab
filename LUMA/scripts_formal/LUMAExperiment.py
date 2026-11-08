"""LUMA experiment orchestration: dataset building, training, inference, cell tables, tracking, feature analysis, and figure export.

The steps correspond to seven CellExperiment methods. Feature analysis runs in analyze_cell_features.
No command-line arguments are accepted; edit paths and hyperparameters only in the main() defaults.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import tifffile
import torch
from tqdm import tqdm

from scripts_formal.LUMACore import LUMATrainer
from scripts_formal.utils.elements_data import GridImageDataset
from scripts_formal.utils.elements_evaluation import CellEvaluator
from scripts_formal.utils.elements_feature_analyzer import (
    FeatureGroupEval,
    FeaturePrep,
    HEAD_PROJ64,
    HeadTrainer,
    resolve_torch_device,
)
from scripts_formal.utils.elements_morphology import build_feature_groups
from scripts_formal.utils.elements_result_io import AnnDataResultStore, FeatureReader, FeatureWriter
from scripts_formal.utils.elements_statistician import CellStatistician
from scripts_formal.utils.elements_tracker import CellTracker
from scripts_formal.utils.elements_visualizer import CellFeatureVisualizer, CellTrackVisualizer


def _resolve_luma_root() -> Path:
    """Resolve the install root: ``LUMA4_ROOT`` env var, else this package's parent."""

    env_root = os.environ.get("LUMA4_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


LUMA4_ROOT = _resolve_luma_root()


class CellExperiment:
    """Call library interfaces step by step without reimplementing training, inference, or analysis algorithms."""

    def __init__(
        self,
        trainer_params: dict[str, Any] | None = None,
        postprocessor_params: dict[str, Any] | None = None,
    ):
        """Store trainer and postprocessing defaults only; build the model during train / infer."""

        self.trainer_params = dict(trainer_params or {})
        self.postprocessor_params = dict(postprocessor_params or {})
        self.trainer: LUMATrainer | None = None
        self.statistician = CellStatistician()
        self.evaluator = CellEvaluator()
        self.tracker = CellTracker()

    def build_dataset(
        self,
        output_dir: str,
        image_records: list[dict[str, Any]] | None = None,
        image_path_template: str | list[str] | None = None,
        mask_path_template: str | list[str] | None = None,
        path_kw: list | None = None,
        mask_kw: list | None = None,
        image_names: list[str] | None = None,
        meta_map: dict | None = None,
        loading_options: dict[str, Any] | None = None,
        train_image_names: list[str] | tuple[str, ...] | None = None,
        overwrite: bool = True,
        **load_kwargs,
    ) -> str:
        """Build and save a GridImageDataset from raw images and masks.

        With overwrite=True, build in a sibling staging directory and replace
        data.h5 / metadata.pkl only on success, preserving the previous dataset on failure.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # overwrite=False appends directly to the destination; True stages writes before replacement.
        staging_path = None
        build_path = output_path
        if overwrite:
            staging_path = output_path / f".build_staging_{os.getpid()}"
            if staging_path.exists():
                shutil.rmtree(staging_path)
            staging_path.mkdir(parents=True, exist_ok=True)
            build_path = staging_path

        try:
            # Build the dataset and store the default loading options.
            dataset = GridImageDataset(data_storage_path=build_path, meta_map=dict(meta_map or {}))
            if loading_options:
                dataset.configure_default_loading_option(**loading_options)

            # Load images individually from records or in batches from path templates.
            if image_records:
                for record in image_records:
                    record_kwargs = dict(load_kwargs)
                    record_kwargs.update({
                        key: value for key, value in record.items()
                        if key not in {"image_paths", "mask_path", "image_name"}
                    })
                    dataset.load_grid_image(
                        image_paths=record["image_paths"],
                        mask_path=record["mask_path"],
                        image_name=record.get("image_name"),
                        **record_kwargs,
                    )
            else:
                dataset.get_dataset(
                    image_path_template=image_path_template,
                    mask_path_template=mask_path_template,
                    path_kw=path_kw,
                    mask_kw=mask_kw,
                    image_names=image_names,
                    **load_kwargs,
                )

            # Select training image names; use all images for training if none are provided.
            if train_image_names is None:
                train_image_names = list(dataset.grid_image.keys())
            dataset.get_merged_dataset(list(train_image_names))
            dataset.save()
        except Exception:
            if staging_path is not None:
                shutil.rmtree(staging_path, ignore_errors=True)
            raise

        n_train = len(dataset.train_names)
        n_test = len(dataset.test_names)
        if staging_path is not None:
            # Replace the final artifacts only after the entire build succeeds.
            for file_name in ("data.h5", "metadata.pkl"):
                os.replace(staging_path / file_name, output_path / file_name)
            shutil.rmtree(staging_path, ignore_errors=True)
            # Reload metadata.pkl to replace the stored staging path with output_path on disk.
            GridImageDataset.load(output_path)
        print(
            f"[build] {output_path} train={n_train} test={n_test}",
            flush=True,
        )
        return str(output_path)

    def train_model(
        self,
        dataset_path: str,
        sam_checkpoint_path: str,
        log_dir: str,
        title: str = "training_session",
        description: str = "",
        model_path: str | None = None,
        epochs: int = 100,
        num_workers: int = 0,
        batch_size: int = 4,
        accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        train_val_ratio: float = 0.8,
        optimizer_type: str = "adam",
        target_image_channel: list[int] | None = None,
        model_save_name: str = "best_model.pth",
        train_encoder: bool | list = False,
        encoder_learning_rate: float = 1e-5,
        use_amp: bool = True,
        use_bf16: bool = False,
        scheduler_T_max: int | None = None,
        load_model_path: str | None = None,
        load_typemap_upsampler: bool = True,
        load_components: list[str] | str | None = None,
        patience: int = 25,
        uncertainty_weight: bool = True,
        loss_type_hv: str = "cpdynamics_hv",
        hv_loss_weight: float = 2.0,
        class_weight: list[float] | None = None,
        **trainer_overrides,
    ) -> tuple[dict, str]:
        """Train SamUNetR on a Dataset and return the loss history and best weights path."""

        # Build or reload the trainer using this run's training parameters.
        if target_image_channel is None:
            target_image_channel = [2]
        self.configure_trainer(
            dataset_path=dataset_path,
            sam_checkpoint_path=sam_checkpoint_path,
            model_path=model_path,
            load_typemap_upsampler=load_typemap_upsampler,
            load_components=load_components,
            **trainer_overrides,
        )
        return self.trainer.train(
            title=title,
            description=description,
            log_dir=log_dir,
            epochs=epochs,
            num_workers=num_workers,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            learning_rate=learning_rate,
            train_val_ratio=train_val_ratio,
            optimizer_type=optimizer_type,
            target_image_channel=target_image_channel,
            model_save_name=model_save_name,
            train_encoder=train_encoder,
            encoder_learning_rate=encoder_learning_rate,
            use_amp=use_amp,
            use_bf16=use_bf16,
            scheduler_T_max=scheduler_T_max,
            load_model_path=load_model_path,
            load_typemap_upsampler=load_typemap_upsampler,
            load_components=load_components,
            patience=patience,
            uncertainty_weight=uncertainty_weight,
            loss_type_hv=loss_type_hv,
            hv_loss_weight=hv_loss_weight,
            class_weight=class_weight,
        )

    def infer(
        self,
        anndata_path: str,
        infer_source: str = "dataset",
        dataset_path: str = "",
        image_names: list[str] | tuple[str, ...] | None = None,
        image_records: list[dict[str, Any]] | None = None,
        sam_checkpoint_path: str = "",
        model_path: str = "",
        run_id: str | None = None,
        overwrite_anndata: bool = False,
        evaluate_flag: bool = False,
        type_mapping: dict[int, str] | None = None,
        target_stitch_channel: list[int] | None = None,
        stitch_batch_size: int = 2,
        num_instance_class: int = 1,
        input_z_size: int = 1,
        grid_size: tuple[int, int] = (1, 1),
        grid_target_size: tuple[int, int] = (1024, 1024),
        margin_ratio: float = 0.0,
        normalize: bool = True,
        object_size: int = 10,
        max_cpimage_size: int = 2048,
        flow_threshold: float = 0.5,
        mask_thres: float = 0.5,
        null_hypothesis_threshold: float = 0.15,
        iou_threshold: float = 0.5,
        eval_region: str | tuple | None = "auto",
        use_cyto_type: bool = True,
        **trainer_overrides,
    ) -> str:
        """Infer on the Dataset test split or standalone images (single time point / time series) and write one H5AD."""

        # Prepare the trainer and jobs first; fail early without touching results if the model is unavailable.
        if target_stitch_channel is None:
            target_stitch_channel = [2]
        self.configure_trainer(
            dataset_path=dataset_path or None,
            sam_checkpoint_path=sam_checkpoint_path,
            model_path=model_path,
            num_class=num_instance_class,
            input_z_size=input_z_size,
            **trainer_overrides,
        )
        jobs = self.collect_infer_jobs(
            infer_source=infer_source,
            dataset_path=dataset_path,
            image_names=image_names,
            image_records=image_records,
        )

        dest = Path(anndata_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # In overwrite mode, write to a sibling temporary file and replace the final H5AD only after all images succeed.
        staging_dest = None
        write_path = dest
        if overwrite_anndata:
            staging_dest = dest.with_name(f".{dest.stem}.staging_{os.getpid()}{dest.suffix}")
            staging_dest.unlink(missing_ok=True)
            write_path = staging_dest

        if run_id is None:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        type_mapping = dict(type_mapping or {})
        summary_records = []

        # Stitch predictions per image; optionally evaluate segmentation / in-place classification against GT.
        try:
            for job in tqdm(jobs, desc="infer"):
                image_id = AnnDataResultStore.sanitize_key(job["image_name"])
                self.trainer.postprocessor.stitch_prediction_map(
                    input_image=job["input_image"],
                    image_name=job["image_name"],
                    predict_instance=True,
                    num_instance_class=num_instance_class,
                    model_2_infer=self.trainer.model,
                    save_root=None,
                    anndata_path=str(write_path),
                    append_to_anndata=True,
                    image_id=image_id,
                    run_id=run_id,
                    overwrite_image=True,
                    target_stitch_channel=target_stitch_channel,
                    stitch_batch_size=stitch_batch_size,
                    object_size=object_size,
                    max_cpimage_size=max_cpimage_size,
                    flow_threshold=flow_threshold,
                    mask_thres=mask_thres,
                    null_hypothesis_threshold=null_hypothesis_threshold,
                    grid_size=grid_size,
                    grid_target_size=grid_target_size,
                    margin_ratio=margin_ratio,
                    normalize=normalize,
                    input_z_size=input_z_size,
                    meta_map=job.get("meta_map", {}),
                )
                if not evaluate_flag:
                    continue
                result = self.evaluator.run_evaluation(
                    adata_path=str(write_path),
                    image_id=image_id,
                    use_cyto_type=use_cyto_type,
                    null_hypothesis_threshold=null_hypothesis_threshold,
                    iou_threshold=iou_threshold,
                    eval_region=eval_region,
                    type_mapping=type_mapping,
                    res_name=job["image_name"],
                )
                if result:
                    summary_records.append(result["summary_record"])
        except Exception:
            if staging_dest is not None:
                staging_dest.unlink(missing_ok=True)
            raise

        if staging_dest is not None:
            os.replace(staging_dest, dest)
        if summary_records:
            csv_path = dest.with_name(dest.stem + "_evaluation_summary.csv")
            pd.DataFrame(summary_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[infer] evaluation csv={csv_path}", flush=True)
        print(f"[infer] h5ad={dest} images={len(jobs)}", flush=True)
        return str(dest)

    def extract_cell_table(
        self,
        source_kind: str,
        dataset_path: str = "",
        h5ad_path: str = "",
        instance_source: str = "gt",
        target_channel: list[int] | None = None,
        target_size: tuple[int, int] | None = None,
        train_image_names: tuple[str, ...] | list[str] = (),
        test_image_names: tuple[str, ...] | list[str] = (),
        extension_rate: float = 1.0,
        sam_checkpoint_path: str = "",
        backbone_type: str = "vit_l",
        batch_size: int = 2,
        use_bf16: bool = True,
        type_index_to_name: dict[int, str] | None = None,
        type_name_to_index: dict[str, int] | None = None,
        extra_config: dict[str, Any] | None = None,
        output_root: str = "",
        meta_map: dict | None = None,
        n_morph_workers: int = 7,
        max_cells_per_image: int | None = None,
        sample_seed: int = 0,
        attach_tracks: bool = True,
        device: str = "cpu",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Write a separate cell_table.h5ad from a Dataset, or extract and write back a cell table in a segmentation H5AD."""

        if target_channel is None:
            target_channel = [2]
        cell_table = self.statistician.extract_and_save_cell_table(
            source_kind=source_kind,
            dataset_path=dataset_path,
            h5ad_path=h5ad_path,
            instance_source=instance_source,
            target_channel=target_channel,
            target_size=target_size,
            train_image_names=train_image_names,
            test_image_names=test_image_names,
            extension_rate=extension_rate,
            sam_checkpoint_path=sam_checkpoint_path,
            backbone_type=backbone_type,
            batch_size=batch_size,
            use_bf16=use_bf16,
            type_index_to_name=type_index_to_name,
            type_name_to_index=type_name_to_index,
            extra_config=dict(extra_config or {}),
            output_root=output_root,
            meta_map=meta_map,
            n_morph_workers=n_morph_workers,
            max_cells_per_image=max_cells_per_image,
            sample_seed=sample_seed,
            attach_tracks=bool(attach_tracks) and max_cells_per_image is None,
            device=device,
            n_jobs=n_jobs,
        )
        dest = FeatureWriter.resolve_h5ad_path(source_kind, h5ad_path, output_root)
        print(f"[extract] h5ad={dest} n_cells={cell_table['obs'].shape[0]}", flush=True)
        return cell_table

    def track_cells(
        self,
        h5ad_path: str,
        config_path: str,
        **tracking_overrides,
    ) -> dict[str, Any]:
        """Run laptrack on an H5AD with an existing cell table and write tracks back to the same file."""

        result = self.tracker.track_anndata(
            config_path=config_path,
            anndata_path=h5ad_path,
            **tracking_overrides,
        )
        quality = self.statistician.summarize_track_quality(h5ad_path=h5ad_path)
        if len(quality["per_track"]):
            FeatureWriter.write_uns(h5ad_path, "tracking_quality", quality["per_track"])
            FeatureWriter.write_uns(h5ad_path, "tracking_quality_summary", quality["summary"])
        print(
            f"[track] h5ad={h5ad_path} n_tracks={result['n_tracks']}",
            flush=True,
        )
        return result

    def analyze_cell_features(
        self,
        source_kind: str = "dataset",
        data_name: str = "a4bad_cpdynamics_hv",
        dataset_path: str = "",
        h5ad_path: str = "",
        train_image_names: tuple[str, ...] = (),
        test_image_names: tuple[str, ...] = (),
        instance_source: str = "gt",
        target_channel: list[int] | None = None,
        target_size: tuple[int, int] | None = None,
        extension_rate: float = 1.0,
        sam_checkpoint_path: str = "",
        backbone_type: str = "vit_l",
        batch_size: int = 2,
        use_bf16: bool = True,
        type_index_to_name: dict[int, str] | None = None,
        type_name_to_index: dict[str, int] | None = None,
        meta_key: str = "gt_cell_type",
        target_kind: str = "discrete",
        meta_map: dict | None = None,
        obs_equals: dict[str, Any] | None = None,
        exclude_obs_in: dict[str, Any] | None = None,
        target_scale_max: float | None = None,
        run_extract_flag: bool = True,
        run_type_prediction_flag: bool = True,
        run_contrib_diagnosis_flag: bool = True,
        n_mean: int = 1024,
        heads: tuple[str, ...] = (HEAD_PROJ64,),
        mlp_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
        bottleneck_dim: int = 64,
        mlp_hidden: int = 256,
        mlp_batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta_ap: float = 1e-4,
        val_fraction: float = 0.1,
        column_zscore: bool = True,
        feature_transforms: list[bool] | None = None,
        stratify_eval_key: str = "",
        extra_metrics: tuple[str, ...] = (),
        output_root: str = "",
        run_visualization_flag: bool = True,
        lda_label_key: str = "",
        label_unit: str = "",
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        phate_knn: int = 15,
        phate_decay: int = 40,
        phate_t: int = 20,
        viz_random_state: int = 0,
        kde_n_levels: int = 6,
        viz_n_jobs: int = 1,
        device: str = "cpu",
        n_jobs: int = 1,
        n_morph_workers: int = 7,
        max_cells_per_image: int | None = None,
        sample_seed: int = 0,
        attach_tracks: bool = True,
    ) -> Path:
        """Optionally extract the main table, then predict, ablate, summarize, and visualize under uns/runs/<run_name>/."""

        # Resolve default channels, type names, filters, and output paths. H5AD input does not fall back to Dataset.
        if not dataset_path and source_kind == "dataset":
            dataset_path = str(LUMA4_ROOT / "datasets" / data_name)
        if not sam_checkpoint_path:
            sam_checkpoint_path = str(LUMA4_ROOT / "sam_checkpoints" / "sam_vit_l_0b3195.pth")
        if target_channel is None:
            target_channel = [2]
        if type_index_to_name is None:
            type_index_to_name = {0: "rpe", 1: "hesc"}
        if type_name_to_index is None:
            type_name_to_index = {"rpe": 0, "GFP-": 0, "hesc": 1, "GFP+": 1}
        if obs_equals is None:
            obs_equals = {}
        if exclude_obs_in is None:
            exclude_obs_in = {}
        if feature_transforms is None:
            feature_transforms = [False, False, False, False]
        stratify_eval_key = str(stratify_eval_key or "")
        lda_label_key = str(lda_label_key or "")
        if not lda_label_key:
            lda_label_key = stratify_eval_key
        if (
            not lda_label_key
            and target_kind == "continuous"
            and meta_key
            and meta_key != "gt_cell_type"
        ):
            lda_label_key = str(meta_key)
        label_unit = str(label_unit or "")
        type_index_to_name = dict(type_index_to_name)
        type_name_to_index = dict(type_name_to_index)
        if not output_root:
            output_root = str(
                LUMA4_ROOT / "embedding_results" / "dataset_result" / data_name / "cell_feature_analysis"
            )

        dest = FeatureWriter.resolve_h5ad_path(source_kind, h5ad_path, output_root)
        run_name = FeatureWriter.build_run_name(meta_key, target_kind, obs_equals, exclude_obs_in)
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"[analyze] h5ad={dest} run_name={run_name}", flush=True)

        # Stage A: store extraction configuration in the main cell table, not this run's prediction targets.
        if run_extract_flag:
            cell_table = self.extract_cell_table(
                source_kind=source_kind,
                dataset_path=dataset_path,
                h5ad_path=h5ad_path,
                instance_source=instance_source,
                target_channel=target_channel,
                target_size=target_size,
                train_image_names=train_image_names,
                test_image_names=test_image_names,
                extension_rate=extension_rate,
                sam_checkpoint_path=sam_checkpoint_path,
                backbone_type=backbone_type,
                batch_size=batch_size,
                use_bf16=use_bf16,
                type_index_to_name=type_index_to_name,
                type_name_to_index=type_name_to_index,
                extra_config={
                    "data_name": data_name,
                    "run_stamp": run_stamp,
                    "n_mean": int(n_mean),
                    "embedding_layout": "mean_1024",
                },
                output_root=output_root,
                meta_map=meta_map,
                n_morph_workers=n_morph_workers,
                max_cells_per_image=max_cells_per_image,
                sample_seed=sample_seed,
                attach_tracks=attach_tracks,
                device=device,
                n_jobs=n_jobs,
            )
        else:
            cell_table = FeatureReader.load_cell_table(dest)
            if not cell_table.get("feature_groups"):
                cell_table["feature_groups"] = build_feature_groups(
                    pd.DataFrame(columns=cell_table["extended_morph_columns"])
                )

        resolved_meta_map = FeatureReader.resolve_source_meta_map(
            source_kind=source_kind,
            dataset_path=dataset_path,
            h5ad_path=str(dest if source_kind == "h5ad" else h5ad_path),
            meta_map_override=meta_map,
        )
        prep_kwargs = {
            "n_mean": n_mean,
            "column_zscore": column_zscore,
            "feature_transforms": feature_transforms,
            "stratify_eval_key": stratify_eval_key,
            "meta_key": meta_key,
            "target_kind": target_kind,
            "meta_map": resolved_meta_map,
            "meta_map_override": meta_map,
            "type_name_to_index": type_name_to_index,
            "type_index_to_name": type_index_to_name,
            "obs_equals": obs_equals,
            "exclude_obs_in": exclude_obs_in,
            "target_scale_max": target_scale_max,
            "lda_label_key": lda_label_key,
            "label_unit": label_unit,
        }
        FeatureWriter.write_uns(
            dest,
            "run_config",
            {
                "data_name": data_name,
                "run_stamp": run_stamp,
                "run_name": run_name,
                "meta_key": meta_key,
                "target_kind": target_kind,
                "obs_equals": dict(obs_equals),
                "exclude_obs_in": {key: list(values) for key, values in dict(exclude_obs_in).items()},
                "target_scale_max": target_scale_max,
                "lda_label_key": lda_label_key,
                "label_unit": label_unit,
                "column_zscore": bool(column_zscore),
                "feature_transforms": [bool(v) for v in feature_transforms],
                "stratify_eval_key": str(stratify_eval_key or ""),
                "extra_metrics": list(extra_metrics),
                "n_mean": int(n_mean),
                "heads": list(heads),
                "mlp_seeds": list(mlp_seeds),
                "continuous_value_map": {},
                "device": str(device),
                "n_jobs": int(n_jobs),
                "viz_n_jobs": int(viz_n_jobs),
            },
            run_name=run_name,
        )
        FeatureWriter.write_run_index(dest, {
            "run_name": run_name,
            "data_name": data_name,
            "run_stamp": run_stamp,
            "meta_key": meta_key,
            "target_kind": target_kind,
            "n_cells_main": int(cell_table["obs"].shape[0]),
        })

        # Stages B / C / D / E / F.
        metrics = None
        if run_type_prediction_flag:
            metrics = self.predict_type(
                cell_table=cell_table,
                h5ad_path=dest,
                run_name=run_name,
                prep_kwargs=prep_kwargs,
                device=device,
                n_jobs=n_jobs,
                mlp_seeds=mlp_seeds,
                bottleneck_dim=bottleneck_dim,
                mlp_hidden=mlp_hidden,
                mlp_batch_size=mlp_batch_size,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                min_delta_ap=min_delta_ap,
                val_fraction=val_fraction,
                extra_metrics=extra_metrics,
            )
        if run_contrib_diagnosis_flag:
            self.diagnose_contrib(
                cell_table=cell_table,
                h5ad_path=dest,
                run_name=run_name,
                prep_kwargs=prep_kwargs,
                device=device,
                n_jobs=n_jobs,
                mlp_seeds=mlp_seeds,
                bottleneck_dim=bottleneck_dim,
                mlp_hidden=mlp_hidden,
                mlp_batch_size=mlp_batch_size,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
                patience=patience,
                min_delta_ap=min_delta_ap,
                val_fraction=val_fraction,
                extra_metrics=extra_metrics,
            )
        FeatureWriter.run(metrics=metrics, h5ad_path=dest, run_name=run_name)
        if run_visualization_flag:
            viz_table = CellFeatureVisualizer().run(
                cell_table=cell_table,
                h5ad_path=dest,
                run_name=run_name,
                n_mean=n_mean,
                mlp_batch_size=mlp_batch_size,
                column_zscore=column_zscore,
                feature_transforms=feature_transforms,
                meta_key=meta_key,
                target_kind=target_kind,
                meta_map=resolved_meta_map,
                type_name_to_index=type_name_to_index,
                type_index_to_name=type_index_to_name,
                obs_equals=obs_equals,
                exclude_obs_in=exclude_obs_in,
                target_scale_max=target_scale_max,
                lda_label_key=lda_label_key,
                label_unit=label_unit,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                phate_knn=phate_knn,
                phate_decay=phate_decay,
                phate_t=phate_t,
                viz_random_state=viz_random_state,
                kde_n_levels=kde_n_levels,
                viz_n_jobs=viz_n_jobs,
                device=device,
            )
            CellTrackVisualizer().run(
                cell_table=viz_table,
                h5ad_path=dest,
                run_name=run_name,
            )
        print(f"[analyze] done. h5ad={dest} run_name={run_name}", flush=True)
        return Path(dest)

    def export_figures(
        self,
        h5ad_path: str,
        export_dir: str,
        figure_format: str = "svg",
        dpi: int = FeatureWriter.FIGURE_DPI,
        keys: list[str] | None = None,
        run_name: str | None = None,
        meta_key: str = "",
        target_kind: str = "",
        obs_equals: dict[str, Any] | None = None,
        exclude_obs_in: dict[str, Any] | None = None,
    ) -> None:
        """Export uns/runs/<run_name>/visualization/figures to a directory."""

        if run_name is None:
            run_name = FeatureWriter.build_run_name(meta_key, target_kind, obs_equals, exclude_obs_in)
        FeatureWriter.export_figures(
            h5ad_path=h5ad_path,
            export_dir=export_dir,
            figure_format=figure_format,
            dpi=dpi,
            keys=keys,
            run_name=run_name,
        )
        print(f"[export] dir={export_dir} format={figure_format} run_name={run_name}", flush=True)

    def configure_trainer(
        self,
        dataset_path: str | None = None,
        sam_checkpoint_path: str | None = None,
        model_path: str | None = None,
        load_typemap_upsampler: bool = True,
        load_components: list[str] | str | None = None,
        **trainer_overrides,
    ) -> LUMATrainer:
        """Build or reload LUMATrainer and load weights as needed."""

        params = dict(self.trainer_params)
        params.update(trainer_overrides)
        if dataset_path is not None:
            params["dataset_path"] = dataset_path
        if sam_checkpoint_path:
            params["checkpoint"] = sam_checkpoint_path
        if params.get("dataset_path") in (None, ""):
            params["dataset_path"] = None
            params["init_model"] = False
        self.trainer = LUMATrainer(**params)
        if self.trainer.model is None:
            self.trainer.config_model()
        if self.postprocessor_params:
            self.trainer.config_postprocessor(**self.postprocessor_params)
        if model_path:
            self.trainer.load_model(
                model_path,
                load_typemap_upsampler=load_typemap_upsampler,
                load_components=load_components,
            )
        return self.trainer

    def collect_infer_jobs(
        self,
        infer_source: str,
        dataset_path: str,
        image_names: list[str] | tuple[str, ...] | None,
        image_records: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Collect Dataset test image names or standalone image records into inference jobs."""

        jobs = []
        if infer_source == "dataset":
            datagrid = GridImageDataset.load(dataset_path)
            names = list(image_names) if image_names else list(datagrid.test_names)
            for name in names:
                image_data = datagrid.grid_image[name]
                if hasattr(image_data, "patch_dataset") and image_data.patch_dataset is not None:
                    image_data.patch_dataset.apply_augmentations = False
                jobs.append({
                    "image_name": name,
                    "input_image": image_data,
                    "meta_map": getattr(datagrid, "meta_map", {}),
                })
            return jobs

        for record in image_records:
            image = record.get("image")
            if image is None:
                image = tifffile.imread(record["image_path"])
            jobs.append({
                "image_name": record["image_name"],
                "input_image": image,
                "meta_map": dict(record.get("meta_map") or {}),
            })
        return jobs

    def predict_type(
        self,
        cell_table: dict[str, Any],
        h5ad_path: Path,
        run_name: str,
        prep_kwargs: dict[str, Any],
        mlp_seeds: tuple[int, ...],
        bottleneck_dim: int,
        mlp_hidden: int,
        mlp_batch_size: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        min_delta_ap: float,
        val_fraction: float,
        extra_metrics: tuple[str, ...],
        device: str = "cpu",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Stage B: evaluate supervised prediction for three input groups."""

        device = resolve_torch_device(device, n_jobs)
        print(f"[B] device={device}", flush=True)
        prep = FeaturePrep(cell_table, **prep_kwargs).prepare()
        if prep.spec and prep.spec.get("target_kind") == "continuous":
            payload = FeatureReader.read_uns(h5ad_path, "run_config", run_name=run_name)
            payload["continuous_value_map"] = dict(prep.spec.get("continuous_value_map") or {})
            FeatureWriter.write_uns(h5ad_path, "run_config", payload, run_name=run_name)
        trainer = HeadTrainer(
            device=device,
            bottleneck_dim=bottleneck_dim,
            mlp_hidden=mlp_hidden,
            batch_size=mlp_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta_ap=min_delta_ap,
            val_fraction=val_fraction,
        )
        return trainer.run(
            prep=prep,
            h5ad_path=h5ad_path,
            run_name=run_name,
            mlp_seeds=mlp_seeds,
            extra_metrics=extra_metrics,
        )

    def diagnose_contrib(
        self,
        cell_table: dict[str, Any],
        h5ad_path: Path,
        run_name: str,
        prep_kwargs: dict[str, Any],
        mlp_seeds: tuple[int, ...],
        bottleneck_dim: int,
        mlp_hidden: int,
        mlp_batch_size: int,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        patience: int,
        min_delta_ap: float,
        val_fraction: float,
        extra_metrics: tuple[str, ...],
        device: str = "cpu",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Stage C: group-level ablation and group-level CCA."""

        device = resolve_torch_device(device, n_jobs)
        print(f"[C] device={device}", flush=True)
        return FeatureGroupEval.from_cell_table(
            cell_table=cell_table,
            device=device,
            bottleneck_dim=bottleneck_dim,
            mlp_hidden=mlp_hidden,
            batch_size=mlp_batch_size,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta_ap=min_delta_ap,
            val_fraction=val_fraction,
            **prep_kwargs,
        ).run(
            h5ad_path=h5ad_path,
            run_name=run_name,
            mlp_seeds=mlp_seeds,
            extra_metrics=extra_metrics,
        )


def main(
    run_build_dataset_flag: bool = False,
    run_train_flag: bool = False,
    run_infer_flag: bool = False,
    run_extract_flag: bool = False,
    run_track_flag: bool = False,
    run_analyze_flag: bool = False,
    run_export_flag: bool = False,
    dataset_path: str = str(LUMA4_ROOT / "datasets" / "a4bad_cpdynamics_hv"),
    sam_checkpoint_path: str = str(LUMA4_ROOT / "sam_checkpoints" / "sam_vit_l_0b3195.pth"),
    log_dir: str = str(LUMA4_ROOT / "training_logs"),
    model_path: str = "",
    anndata_path: str = "",
    output_root: str = "",
    tracking_config_path: str = str(
        LUMA4_ROOT / "scripts_formal" / "config" / "mdck2_timeseries_laptrack_config.json"
    ),
    export_dir: str = "",
    figure_format: str = "svg",
    infer_source: str = "dataset",
    extract_source_kind: str = "dataset",
    analyze_source_kind: str = "dataset",
    data_name: str = "a4bad_cpdynamics_hv",
    meta_key: str = "gt_cell_type",
    target_kind: str = "discrete",
    trainer_params: dict[str, Any] | None = None,
    postprocessor_params: dict[str, Any] | None = None,
) -> None:
    """Run seven steps in order using flags; all default to off. Edit these defaults, then enable the relevant flags."""

    if trainer_params is None:
        trainer_params = {
            "backbone_type": "vit_l",
            "num_class": 2,
            "use_prediction_head": True,
            "input_z_size": 1,
            "use_image_skip": True,
            "vit_adapt": "none",
        }
    if postprocessor_params is None:
        postprocessor_params = {}
    experiment = CellExperiment(
        trainer_params=trainer_params,
        postprocessor_params=postprocessor_params,
    )

    if run_build_dataset_flag:
        experiment.build_dataset(output_dir=dataset_path, image_records=[])
    if run_train_flag:
        experiment.train_model(
            dataset_path=dataset_path,
            sam_checkpoint_path=sam_checkpoint_path,
            log_dir=log_dir,
            model_path=model_path or None,
        )
    if run_infer_flag:
        experiment.infer(
            anndata_path=anndata_path,
            infer_source=infer_source,
            dataset_path=dataset_path,
            sam_checkpoint_path=sam_checkpoint_path,
            model_path=model_path,
        )
    if run_extract_flag:
        experiment.extract_cell_table(
            source_kind=extract_source_kind,
            dataset_path=dataset_path,
            h5ad_path=anndata_path,
            sam_checkpoint_path=sam_checkpoint_path,
            output_root=output_root,
        )
    if run_track_flag:
        experiment.track_cells(h5ad_path=anndata_path, config_path=tracking_config_path)
    if run_analyze_flag:
        experiment.analyze_cell_features(
            source_kind=analyze_source_kind,
            data_name=data_name,
            dataset_path=dataset_path,
            h5ad_path=anndata_path,
            sam_checkpoint_path=sam_checkpoint_path,
            output_root=output_root,
            meta_key=meta_key,
            target_kind=target_kind,
            run_extract_flag=False,
        )
    if run_export_flag:
        experiment.export_figures(
            h5ad_path=anndata_path or str(Path(output_root) / FeatureWriter.CELL_TABLE_NAME),
            export_dir=export_dir,
            figure_format=figure_format,
            meta_key=meta_key,
            target_kind=target_kind,
        )


if __name__ == "__main__":
    main()
