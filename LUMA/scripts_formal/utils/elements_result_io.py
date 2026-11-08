"""Read and write LUMA result H5AD files.

AnnDataResultStore: low-level AnnData / HDF5 nodes, /uns/luma/images, and in-place table updates.
FeatureReader: image records, main cell tables, and run-level analysis results from Dataset / H5AD.
FeatureWriter: main cell tables and appended metrics, coordinates, and SVG under uns/runs/<run_name>/.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import anndata as ad
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from scripts_formal.utils.elements_data import GridImageDataset, METAMAP

try:
    import cairosvg
except ImportError:
    cairosvg = None


# For continuous targets, extract exactly one ordinary decimal number per string if to_numeric fails.
_CONTINUOUS_LABEL_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)")

# Mark temporary image-record keys for incomplete writes so readers ignore leftover intermediate groups.
STAGING_KEY_MARKER = ".__staging__"


class AnnDataResultStore:
    """
    Store LUMA image-level results and evaluation records inside a single H5AD file.

    The class keeps AnnData's table schema valid by initializing the file through
    AnnData, then writes LUMA-specific image arrays below /uns/luma with AnnData's
    element writer so read_h5ad can still understand the custom groups.
    """

    def __init__(self, anndata_path: str, compression: str = "gzip") -> None:
        """Initialize a result store bound to one H5AD path."""
        self.anndata_path = anndata_path
        self.compression = compression
        self.initialize_store()

    def initialize_store(self) -> None:
        """Create a minimal legal H5AD file when the target path does not exist."""
        if os.path.exists(self.anndata_path):
            return

        output_dir = os.path.dirname(self.anndata_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        adata = ad.AnnData(X=np.empty((0, 0), dtype=np.float32))
        adata.uns["luma"] = {
            "project": {
                "schema_version": "0.1.0",
                "storage_policy": "image_records_in_uns_luma",
            }
        }
        adata.write_h5ad(self.anndata_path, compression=self.compression)

    def create_image_group(self, h5_file: h5py.File, image_id: str,
                           overwrite: bool = False) -> h5py.Group:
        """Create a staging group for one image record and return it for writing.

        The record is written under a temporary key and only published to
        ``/uns/luma/images/<image_id>`` by :meth:`commit_image_group`, so a failed
        write cannot destroy an already usable record.
        """
        image_key = self.sanitize_key(image_id)
        images_group = self.images_group(h5_file)

        if image_key in images_group and not overwrite:
            raise ValueError(f"Image record already exists: {image_key}")

        # Remove all stale staging groups for this image (including other process IDs), then create this run's staging group.
        for stale_key in self.list_staging_keys(images_group, image_key):
            del images_group[stale_key]
        image_group = images_group.create_group(self.staging_key(image_key))
        self.mark_dict_group(image_group)
        return image_group

    def commit_image_group(self, h5_file: h5py.File, image_id: str,
                           overwrite: bool = True) -> None:
        """Publish a staged image group, replacing the previous record only after a full write."""
        image_key = self.sanitize_key(image_id)
        images_group = self.images_group(h5_file)
        staging_key = self.staging_key(image_key)
        if staging_key not in images_group:
            raise ValueError(f"Staged image record not found: {staging_key}")

        h5_file.flush()
        if image_key in images_group:
            if not overwrite:
                del images_group[staging_key]
                raise ValueError(f"Image record already exists: {image_key}")
            del images_group[image_key]
        images_group.move(staging_key, image_key)
        h5_file.flush()

    def discard_image_group(self, h5_file: h5py.File, image_id: str) -> None:
        """Drop the staged record of a failed write, leaving any existing record untouched."""
        image_key = self.sanitize_key(image_id)
        images_group = self.images_group(h5_file)
        staging_key = self.staging_key(image_key)
        if staging_key in images_group:
            del images_group[staging_key]

    def images_group(self, h5_file: h5py.File) -> h5py.Group:
        """Return /uns/luma/images, creating the parent chain when missing."""
        uns_group = self.require_dict_group(h5_file, "uns")
        luma_group = self.require_dict_group(uns_group, "luma")
        return self.require_dict_group(luma_group, "images")

    @classmethod
    def staging_key(cls, image_key: str) -> str:
        """Return the temporary key used while one image record is being written."""
        return f"{image_key}{STAGING_KEY_MARKER}{os.getpid()}"

    @classmethod
    def list_staging_keys(cls, images_group: h5py.Group, image_key: str) -> List[str]:
        """List stale staging keys of one image record left by this or earlier processes."""
        prefix = f"{image_key}{STAGING_KEY_MARKER}"
        return [key for key in images_group.keys() if key.startswith(prefix)]

    @staticmethod
    def is_staging_key(image_key: Any) -> bool:
        """Report whether a /uns/luma/images key is an unfinished write."""
        return STAGING_KEY_MARKER in str(image_key)

    def append_image_result(self, image_id: str, result_bundle: Dict[str, Any],
                            overwrite: bool = False) -> None:
        """Append a complete image-level result bundle to /uns/luma/images."""
        with h5py.File(self.anndata_path, "a") as f:
            image_group = self.create_image_group(f, image_id, overwrite=overwrite)
            try:
                self.write_image_result_to_group(image_group, result_bundle)
            except Exception:
                self.discard_image_group(f, image_id)
                raise
            self.commit_image_group(f, image_id, overwrite=overwrite)

    def write_image_result_to_group(self, image_group: h5py.Group,
                                    result_bundle: Dict[str, Any]) -> None:
        """Write prediction, GT, embeddings, indexers, and metadata into one image group."""
        for key, value in result_bundle.items():
            self.write_value(image_group, key, value, overwrite=True)

    def append_evaluation_result(self, image_id: str, evaluation_result: Dict[str, Any],
                                 overwrite: bool = True) -> None:
        """Append an evaluation dictionary and optional instance-type maps to the H5AD file."""
        image_key = self.sanitize_key(image_id)
        clean_result, evaluation_maps = self.prepare_evaluation_result(evaluation_result)

        with h5py.File(self.anndata_path, "a") as f:
            uns_group = self.require_dict_group(f, "uns")
            luma_group = self.require_dict_group(uns_group, "luma")
            evaluations_group = self.require_dict_group(luma_group, "evaluations")

            if image_key in evaluations_group:
                if overwrite:
                    del evaluations_group[image_key]
                else:
                    raise ValueError(f"Evaluation record already exists: {image_key}")

            eval_group = evaluations_group.create_group(image_key)
            self.mark_dict_group(eval_group)
            self.write_value(eval_group, "result", clean_result, overwrite=True)

            if evaluation_maps:
                images_group = self.require_dict_group(luma_group, "images")
                image_group = images_group[image_key]
                maps_group = self.require_dict_group(image_group, "evaluation_maps")
                for map_name, map_array in evaluation_maps.items():
                    self.write_value(maps_group, map_name, map_array, overwrite=True)

    def write_image_fields(self, image_id: str, field_values: Dict[str, Any],
                           overwrite: bool = True) -> None:
        """Write selected slash-delimited fields under one image record without touching other fields."""
        image_key = self.sanitize_key(image_id)
        with h5py.File(self.anndata_path, "a") as f:
            uns_group = self.require_dict_group(f, "uns")
            luma_group = self.require_dict_group(uns_group, "luma")
            images_group = self.require_dict_group(luma_group, "images")
            if image_key in images_group:
                image_group = images_group[image_key]
            else:
                image_group = images_group.create_group(image_key)
                self.mark_dict_group(image_group)

            for field_path, value in field_values.items():
                path_parts = [self.sanitize_key(part) for part in str(field_path).strip("/").split("/")]
                parent_group = image_group
                for path_part in path_parts[:-1]:
                    parent_group = self.require_dict_group(parent_group, path_part)
                self.write_value(parent_group, path_parts[-1], value, overwrite=overwrite)

    def read_image_result(self, image_id: str,
                          field_paths: List[str] | Tuple[str, ...] | None = None) -> Dict[str, Any]:
        """Read a whole image record or selected slash-delimited fields from /uns/luma/images."""
        image_key = self.sanitize_key(image_id)
        with h5py.File(self.anndata_path, "r") as f:
            image_group = f["uns"]["luma"]["images"][image_key]
            if field_paths is None:
                return {key: ad.io.read_elem(image_group[key]) for key in image_group.keys()}

            selected_values = {}
            for field_path in field_paths:
                node = image_group
                for path_part in str(field_path).strip("/").split("/"):
                    node = node[self.sanitize_key(path_part)]
                selected_values[field_path] = ad.io.read_elem(node)
            return selected_values

    def read_table_elements(self, table_keys: Tuple[str, ...] = ('obs',),
                            luma_keys: Tuple[str, ...] = ()) -> Dict[str, Dict[str, Any]]:
        """Read selected AnnData table nodes and /uns/luma entries without loading image records."""
        table_values = {}
        luma_values = {}
        with h5py.File(self.anndata_path, "r") as f:
            for table_key in table_keys:
                key = self.sanitize_key(table_key)
                table_values[table_key] = ad.io.read_elem(f[key])
            if luma_keys:
                luma_group = f["uns"]["luma"]
                for luma_key in luma_keys:
                    key = self.sanitize_key(luma_key)
                    luma_values[luma_key] = ad.io.read_elem(luma_group[key])
        return {'tables': table_values, 'luma': luma_values}

    def write_table_elements(self, x_matrix: np.ndarray | None = None,
                             obs: pd.DataFrame | None = None,
                             var: pd.DataFrame | None = None,
                             obsm: Dict[str, np.ndarray] | None = None,
                             luma_uns: Dict[str, Any] | None = None) -> None:
        """Rewrite selected AnnData table nodes in place while preserving large image records."""
        with h5py.File(self.anndata_path, "a") as f:
            table_updates = {
                "X": x_matrix,
                "obs": obs.copy() if obs is not None else None,
                "var": var.copy() if var is not None else None,
                "obsm": obsm,
            }
            for root_key, value in table_updates.items():
                if value is None:
                    continue
                if root_key in f:
                    del f[root_key]
                ad.io.write_elem(f, root_key, value)

            if luma_uns is not None:
                uns_group = self.require_dict_group(f, "uns")
                luma_group = self.require_dict_group(uns_group, "luma")
                for key, value in luma_uns.items():
                    self.write_value(luma_group, key, value, overwrite=True)

    def list_image_ids(self) -> List[str]:
        """List all image IDs currently present in /uns/luma/images."""
        with h5py.File(self.anndata_path, "r") as f:
            if "uns" not in f or "luma" not in f["uns"] or "images" not in f["uns"]["luma"]:
                return []
            return [
                key for key in f["uns"]["luma"]["images"].keys()
                if not self.is_staging_key(key)
            ]

    def write_value(self, parent_group: h5py.Group, name: str, value: Any,
                    overwrite: bool = True) -> None:
        """Write one Python object as a schema-aware H5AD element."""
        key = self.sanitize_key(name)
        if key in parent_group:
            if overwrite:
                del parent_group[key]
            else:
                raise ValueError(f"Dataset or group already exists: {key}")

        value = self.prepare_value(value)
        if isinstance(value, dict):
            child_group = parent_group.create_group(key)
            self.mark_dict_group(child_group)
            for child_key, child_value in value.items():
                self.write_value(child_group, child_key, child_value, overwrite=True)
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.dtype.kind not in {"O", "U", "S"}:
            ad.io.write_elem(
                parent_group,
                key,
                value,
                dataset_kwargs={"compression": self.compression},
            )
        else:
            ad.io.write_elem(parent_group, key, value)

    def require_dict_group(self, parent_group: h5py.Group, name: str) -> h5py.Group:
        """Return an existing child group or create it as an AnnData-readable dict group."""
        key = self.sanitize_key(name)
        if key in parent_group:
            group = parent_group[key]
        else:
            group = parent_group.create_group(key)
        self.mark_dict_group(group)
        return group

    @staticmethod
    def mark_dict_group(group: h5py.Group) -> None:
        """Mark an HDF5 group with AnnData's dictionary encoding attributes."""
        group.attrs["encoding-type"] = "dict"
        group.attrs["encoding-version"] = "0.1.0"

    @staticmethod
    def sanitize_key(value: Any) -> str:
        """Convert an arbitrary image or field name into a safe HDF5 key."""
        text = str(value)
        for bad_char in ["/", "\\"]:
            text = text.replace(bad_char, "_")
        return text

    @staticmethod
    def prepare_value(value: Any) -> Any:
        """Normalize common NumPy and pandas values before writing them as H5AD elements."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Series):
            return value.to_frame(name=value.name if value.name is not None else "value")
        if isinstance(value, tuple):
            return np.asarray(value)
        if isinstance(value, list):
            if not value:
                return np.asarray([], dtype=np.float32)
            if all(isinstance(item, str) for item in value):
                return np.asarray(value, dtype=object)
            return np.asarray(value)
        return value

    @classmethod
    def prepare_evaluation_result(cls, evaluation_result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Split evaluation metrics from large evaluation maps before writing to H5AD."""
        clean_result = {}
        evaluation_maps = {}

        for key, value in evaluation_result.items():
            if key == "evaluation_figure":
                continue
            if key == "classification" and isinstance(value, dict):
                clean_classification = {}
                for class_key, class_value in value.items():
                    if class_key in {"gt_instance_type_map", "pred_instance_type_map"}:
                        evaluation_maps[class_key] = class_value
                    else:
                        clean_classification[class_key] = cls.prepare_serializable(class_value)
                clean_result[key] = clean_classification
            elif key == "summary_text" and isinstance(value, list):
                clean_result[key] = "\n".join(str(line) for line in value)
            else:
                clean_result[key] = cls.prepare_serializable(value)

        return clean_result, evaluation_maps

    @classmethod
    def prepare_serializable(cls, value: Any) -> Any:
        """Convert nested metrics into objects accepted by AnnData's element writer."""
        if isinstance(value, dict):
            return {str(k): cls.prepare_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls.prepare_serializable(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            return value
        if value is None:
            return "None"
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


class CompatUnpickler(pickle.Unpickler):
    """Support pickles written by numpy>=2 by redirecting numpy._core.* to local numpy.core.*."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class FeatureReader:
    """Read image records, arrays, metadata, and tracks from Dataset / H5AD."""

    @staticmethod
    def iter_image_records(
        source_kind: str,
        dataset_path: str,
        h5ad_path: str,
        instance_source: str,
        target_channel: list[int],
        target_size: tuple[int, int] | None,
        train_image_names: tuple[str, ...] | list[str],
        test_image_names: tuple[str, ...] | list[str],
    ) -> Iterator[dict[str, Any]]:
        """Yield one ImageRecord per image according to the source kind."""

        if source_kind == "dataset":
            yield from FeatureReader.iter_dataset_records(
                dataset_path=dataset_path,
                instance_source=instance_source,
                target_channel=target_channel,
                target_size=target_size,
            )
            return
        if source_kind == "h5ad":
            yield from FeatureReader.iter_h5ad_records(
                h5ad_path=h5ad_path,
                dataset_path=dataset_path,
                instance_source=instance_source,
                train_image_names=train_image_names,
                test_image_names=test_image_names,
            )
            return
        raise ValueError(f"unknown source_kind={source_kind!r}; expected 'dataset' or 'h5ad'.")

    @staticmethod
    def encode_gt_cell_type(
        obs: pd.DataFrame,
        type_name_to_index: dict[str, int] | None,
        type_index_to_name: dict[int, str] | None,
    ) -> pd.DataFrame:
        """Encode string instance types as integer gt_cell_type; do not write a separate gt_cell_type_name."""

        if "gt_cell_type" not in obs.columns:
            return obs
        obs = obs.copy()
        series = obs["gt_cell_type"]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            obs["gt_cell_type"] = numeric.astype(np.int64)
            return obs
        if not type_name_to_index:
            return obs
        names = series.astype(str)
        unknown = sorted(set(names) - set(type_name_to_index))
        if unknown:
            raise RuntimeError(
                f"gt_cell_type contains values not covered by type_name_to_index: {unknown}; "
                "for tasks without instance types, such as LIVECell / mixed, set type_name_to_index to {} "
                "and disable run_type_prediction_flag."
            )
        obs["gt_cell_type"] = names.map(type_name_to_index).astype(np.int64)
        print(
            f"[A] encoded gt_cell_type via type_name_to_index; "
            f"classes={sorted(set(obs['gt_cell_type'].tolist()))}",
            flush=True,
        )
        return obs

    @staticmethod
    def iter_dataset_records(
        dataset_path: str,
        instance_source: str,
        target_channel: list[int],
        target_size: tuple[int, int] | None,
    ) -> Iterator[dict[str, Any]]:
        """Enumerate images and patches from GridImageDataset metadata.pkl + data.h5."""

        if instance_source != "gt":
            raise RuntimeError(
                "Dataset input provides only GT instance maps; instance_source must be 'gt'."
            )
        dataset_root = Path(dataset_path)
        hdf5_path = dataset_root / "data.h5"
        grid_dataset = FeatureReader.load_grid_dataset(dataset_root / "metadata.pkl")
        split_by_image = FeatureReader.resolve_split_from_dataset(grid_dataset)

        for image_name, grid_image in grid_dataset.grid_image.items():
            grid_h, grid_w = tuple(int(v) for v in grid_image.grid_size)
            stitch_info = FeatureReader.normalize_stitch_info(dict(grid_image.stitch_info))
            patches = [
                {"patch_row": row, "patch_col": col}
                for row in range(grid_h)
                for col in range(grid_w)
            ]
            yield {
                "source_kind": "dataset",
                "image_name": image_name,
                "image_id": image_name,
                "split": split_by_image[image_name],
                "stitch_info": stitch_info,
                "instance_source": "gt",
                "target_channel": list(target_channel),
                "target_size": tuple(target_size),
                "hdf5_path": hdf5_path,
                "patches": patches,
                "meta_info": dict(getattr(grid_image, "meta_info", {}) or {}),
                "skip_reason": "",
            }

    @staticmethod
    def iter_h5ad_records(
        h5ad_path: str,
        dataset_path: str,
        instance_source: str,
        train_image_names: tuple[str, ...] | list[str],
        test_image_names: tuple[str, ...] | list[str],
    ) -> Iterator[dict[str, Any]]:
        """Enumerate images from /uns/luma/images in a LUMA evaluation H5AD.

        Read only attrs / stitch_info without loading all prediction maps and patch_embeddings into memory.
        Read arrays per image in extract_from_h5ad.
        """

        if instance_source not in ("gt", "pred"):
            raise ValueError(f"instance_source must be 'gt' or 'pred', got {instance_source!r}")
        store = AnnDataResultStore(h5ad_path)
        image_ids = store.list_image_ids()
        split_by_image = FeatureReader.resolve_split_for_h5ad(
            dataset_path=dataset_path,
            train_image_names=train_image_names,
            test_image_names=test_image_names,
        )
        instance_field = (
            "gt/gt_instance_map" if instance_source == "gt" else "prediction/pred_instance_map"
        )

        with h5py.File(h5ad_path, "r") as handle:
            images_group = handle["uns"]["luma"]["images"]
            for image_id in image_ids:
                image_key = AnnDataResultStore.sanitize_key(image_id)
                image_group = images_group[image_key]
                if not FeatureReader.h5_group_has_field(image_group, instance_field):
                    if instance_source == "gt":
                        print(
                            f"[read] skip {image_id}: missing {instance_field}",
                            flush=True,
                        )
                        yield {
                            "source_kind": "h5ad",
                            "image_name": image_id,
                            "image_id": image_id,
                            "split": "unknown",
                            "stitch_info": {},
                            "instance_source": instance_source,
                            "h5ad_path": h5ad_path,
                            "meta_info": {},
                            "skip_reason": f"missing {instance_field}",
                        }
                        continue
                    raise RuntimeError(f"{image_id} missing {instance_field}")
                attrs = dict(ad.io.read_elem(image_group["attrs"]) or {})
                stitch_info = {}
                if "stitch_info" in image_group:
                    stitch_info = dict(ad.io.read_elem(image_group["stitch_info"]) or {})
                image_name = str(attrs.get("image_name", image_id))
                split_key = image_name if image_name in split_by_image else image_id
                if split_key not in split_by_image:
                    raise RuntimeError(
                        f"H5AD image {image_name!r} (id={image_id!r}) is not in the train/test lists; "
                        "provide a companion Dataset or complete train_image_names / test_image_names in main()."
                    )
                yield {
                    "source_kind": "h5ad",
                    "image_name": image_name,
                    "image_id": image_id,
                    "split": split_by_image[split_key],
                    "stitch_info": FeatureReader.normalize_stitch_info(stitch_info),
                    "instance_source": instance_source,
                    "h5ad_path": h5ad_path,
                    "meta_info": dict(attrs.get("meta_info", {}) or {}),
                    "skip_reason": "",
                }

    @staticmethod
    def read_dataset_patch(
        h5_file: h5py.File,
        image_name: str,
        patch_row: int,
        patch_col: int,
        target_channel: list[int],
        target_size: tuple[int, int] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read one patch's intensity, GT instance map, and type_map.

        Preserve the original height and width when target_size is None. Return image_nchw (C, H, W) float32,
        instance_map (H, W) int64、type_map float32。
        """

        patch_group = h5_file[f"{image_name}/patch_{patch_row}_{patch_col}"]
        channel_arrays = []
        source_dtype = None
        for channel_idx in target_channel:
            channel_image = patch_group["raw_channels"][f"channel_{channel_idx}"]["image"][:]
            source_dtype = channel_image.dtype if source_dtype is None else source_dtype
            channel_arrays.append(channel_image)
        image_tensor = torch.from_numpy(np.stack(channel_arrays, axis=0)).float().unsqueeze(0)
        if image_tensor.shape[2] != 1:
            image_tensor = F.interpolate(
                image_tensor,
                size=(1, image_tensor.shape[-2], image_tensor.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        image_np = image_tensor.squeeze(0).numpy().astype(np.float32)
        if source_dtype is not None and np.issubdtype(source_dtype, np.integer):
            image_np = image_np / (float(np.iinfo(source_dtype).max) + 1e-6)

        instance_map = np.asarray(patch_group["mask"][0], dtype=np.int64)
        instance_out = instance_map
        if target_size is not None:
            instance_torch = torch.from_numpy(instance_map.astype(np.float64)).unsqueeze(0)
            instance_torch = TF.resize(
                instance_torch,
                list(target_size),
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
            instance_out = np.rint(instance_torch.squeeze(0).numpy()).astype(np.int64)

        type_map = np.asarray(patch_group["type_map"][:], dtype=np.float32)
        type_out = type_map
        if target_size is not None:
            type_torch = torch.from_numpy(type_map).float()
            type_torch = TF.resize(
                type_torch,
                list(target_size),
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
            type_out = type_torch.numpy().astype(np.float32)

        image_torch = torch.from_numpy(image_np).float()
        if image_torch.ndim == 3:
            image_torch = image_torch.unsqueeze(1)
        if target_size is not None:
            image_torch = F.interpolate(
                image_torch.unsqueeze(0),
                size=(image_torch.shape[1], *target_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
        image_nchw = image_torch[:, 0].contiguous()
        if image_nchw.shape[0] > 1:
            image_nchw = image_nchw.mean(dim=0, keepdim=True)
        return image_nchw.numpy().astype(np.float32), instance_out, type_out

    @staticmethod
    def read_dataset_instance_map(
        h5_file: h5py.File,
        image_name: str,
        patch_row: int,
        patch_col: int,
        target_size: tuple[int, int] | None,
    ) -> np.ndarray:
        """Read only the GT instance map. Do not resize when target_size is None."""

        patch_group = h5_file[f"{image_name}/patch_{patch_row}_{patch_col}"]
        instance_map = np.asarray(patch_group["mask"][0], dtype=np.int64)
        if target_size is None:
            return instance_map
        instance_torch = torch.from_numpy(instance_map.astype(np.float64)).unsqueeze(0)
        instance_torch = TF.resize(
            instance_torch,
            list(target_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
        return np.rint(instance_torch.squeeze(0).numpy()).astype(np.int64)

    @staticmethod
    def load_h5ad_extract_payload(image_group: h5py.Group, instance_source: str) -> dict[str, Any]:
        """Read only fields needed for extraction without also loading h_map / v_map and other arrays."""

        inst_path = (
            "gt/gt_instance_map" if instance_source == "gt" else "prediction/pred_instance_map"
        )
        type_path = (
            "gt/gt_type_cyto_map" if instance_source == "gt" else "prediction/pred_type_cyto_map"
        )
        gt_record: dict[str, Any] = {
            "gt_raw_image": FeatureReader.read_h5_elem_path(image_group, "gt/gt_raw_image"),
        }
        pred_record: dict[str, Any] = {}
        if instance_source == "gt":
            gt_record["gt_instance_map"] = FeatureReader.read_h5_elem_path(image_group, inst_path)
            type_map = FeatureReader.read_h5_elem_path(image_group, type_path)
            if type_map is not None:
                gt_record["gt_type_cyto_map"] = type_map
        else:
            pred_record["pred_instance_map"] = FeatureReader.read_h5_elem_path(image_group, inst_path)
            type_map = FeatureReader.read_h5_elem_path(image_group, type_path)
            if type_map is not None:
                pred_record["pred_type_cyto_map"] = type_map
        attrs = {}
        if "attrs" in image_group:
            attrs = dict(ad.io.read_elem(image_group["attrs"]) or {})
        embedding_indexer = None
        if "embedding_indexer" in image_group:
            embedding_indexer = ad.io.read_elem(image_group["embedding_indexer"])
        patch_embeddings = {}
        if "patch_embeddings" in image_group:
            patch_embeddings = ad.io.read_elem(image_group["patch_embeddings"]) or {}
        return {
            "gt": gt_record,
            "prediction": pred_record,
            "patch_embeddings": patch_embeddings,
            "embedding_indexer": embedding_indexer,
            "attrs": attrs,
        }

    @staticmethod
    def read_h5_elem_path(group: h5py.Group, field_path: str):
        """Read an AnnData element by slash-delimited path; return None if the path does not exist."""

        node = group
        for part in field_path.split("/"):
            key = AnnDataResultStore.sanitize_key(part)
            if key not in node:
                return None
            node = node[key]
        return ad.io.read_elem(node)

    @staticmethod
    def unpack_h5ad_arrays(
        payload: dict[str, Any],
        instance_source: str,
    ) -> dict[str, Any]:
        """Extract intensity, instances, types, and patch embeddings from a loaded H5AD image record."""

        gt_record = payload.get("gt", {}) or {}
        pred_record = payload.get("prediction", {}) or {}
        if instance_source == "gt":
            instance_map = np.asarray(gt_record["gt_instance_map"], dtype=np.int64)
            type_map = gt_record.get("gt_type_cyto_map", None)
        else:
            instance_map = np.asarray(pred_record["pred_instance_map"], dtype=np.int64)
            type_map = pred_record.get("pred_type_cyto_map", None)
        intensity = FeatureReader.squeeze_intensity(gt_record["gt_raw_image"])
        if type_map is not None:
            type_map = np.asarray(type_map)
        patch_embeddings = payload.get("patch_embeddings", {}) or {}
        embedding_indexer = payload.get("embedding_indexer", None)
        attrs = payload.get("attrs", {}) or {}
        return {
            "intensity": intensity,
            "instance_map": instance_map,
            "type_map": type_map,
            "patch_embeddings": patch_embeddings,
            "embedding_indexer": embedding_indexer,
            "attrs": attrs,
        }

    @staticmethod
    def squeeze_intensity(raw) -> np.ndarray:
        """Convert H5AD gt_raw_image to an (H, W) float32 grayscale plane."""

        arr = np.asarray(raw)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = arr.mean(axis=0)
        return arr.astype(np.float32)

    @staticmethod
    def load_grid_dataset(metadata_path: Path) -> GridImageDataset:
        """Load the GridImageDataset stored in metadata.pkl."""

        with open(metadata_path, "rb") as handle:
            return CompatUnpickler(handle).load()

    @staticmethod
    def resolve_split_from_dataset(grid_dataset: GridImageDataset) -> dict[str, str]:
        """Assign each image a split using the Dataset's image-level train/test name lists."""

        train_names = set(grid_dataset.train_test_names.get("train", []))
        test_names = set(grid_dataset.train_test_names.get("test", []))
        split_by_image = {}
        for image_name in grid_dataset.grid_image:
            if image_name in train_names:
                split_by_image[image_name] = "train"
            elif image_name in test_names:
                split_by_image[image_name] = "test"
            else:
                raise RuntimeError(f"image {image_name!r} is in neither train_names nor test_names")
        return split_by_image

    @staticmethod
    def resolve_split_for_h5ad(
        dataset_path: str,
        train_image_names: tuple[str, ...] | list[str],
        test_image_names: tuple[str, ...] | list[str],
    ) -> dict[str, str]:
        """Resolve H5AD splits from the companion Dataset first, otherwise from manually specified main() image name lists."""

        if dataset_path:
            grid_dataset = FeatureReader.load_grid_dataset(Path(dataset_path) / "metadata.pkl")
            return FeatureReader.resolve_split_from_dataset(grid_dataset)
        if train_image_names and test_image_names:
            split_by_image = {}
            for name in train_image_names:
                split_by_image[str(name)] = "train"
            for name in test_image_names:
                if str(name) in split_by_image:
                    raise RuntimeError(f"image {name!r} listed in both train and test")
                split_by_image[str(name)] = "test"
            return split_by_image
        if train_image_names and not test_image_names:
            return {str(name): "train" for name in train_image_names}
        raise RuntimeError(
            "H5AD input has no split: provide the corresponding GridImageDataset path, "
            "or specify train_image_names and test_image_names in main()."
        )

    @staticmethod
    def normalize_stitch_info(stitch_info: dict[str, Any]) -> dict[str, Any]:
        """Preserve original stitch_info, converting only existing pixel-dimension fields to integers."""

        out = dict(stitch_info)
        for key in ("patch_h", "patch_w", "margin_h", "margin_w", "original_h", "original_w"):
            if key in out and out[key] is not None and out[key] != "":
                out[key] = int(out[key])
        return out

    @staticmethod
    def h5_group_has_field(group: h5py.Group, field_path: str) -> bool:
        """Check whether a slash-delimited field path exists under the open image group."""

        node = group
        for part in field_path.split("/"):
            key = AnnDataResultStore.sanitize_key(part)
            if key not in node:
                return False
            node = node[key]
        return True

    @staticmethod
    def read_meta_map_from_dataset(dataset_path: str) -> dict:
        """Read meta_map from GridImageDataset.metadata.pkl."""

        grid_dataset = FeatureReader.load_grid_dataset(Path(dataset_path) / "metadata.pkl")
        return dict(getattr(grid_dataset, "meta_map", {}) or {})

    @staticmethod
    def read_meta_map_from_h5ad(h5ad_path: str) -> dict:
        """Read mappings from H5AD /uns/luma/meta_map; return an empty dict if absent."""

        with h5py.File(h5ad_path, "r") as handle:
            luma = handle["uns"]["luma"]
            if "meta_map" not in luma:
                return {}
            return dict(ad.io.read_elem(luma["meta_map"]) or {})

    @staticmethod
    def resolve_source_meta_map(
        source_kind: str,
        dataset_path: str,
        h5ad_path: str,
        meta_map_override: dict | None,
    ) -> dict:
        """Resolve this run's meta_map: use a nonempty main override, otherwise read from dataset / H5AD."""

        if meta_map_override:
            return dict(meta_map_override)
        if source_kind == "dataset" and dataset_path:
            return FeatureReader.read_meta_map_from_dataset(dataset_path)
        if source_kind == "h5ad" and h5ad_path:
            loaded = FeatureReader.read_meta_map_from_h5ad(h5ad_path)
            if loaded:
                return loaded
            return {}
        return {}

    @staticmethod
    def attach_image_meta_to_obs(
        obs: pd.DataFrame,
        meta_by_image: dict[str, dict],
        meta_map: dict | None = None,
    ) -> pd.DataFrame:
        """Write image-level meta_info and source meta_map keys as obs columns without overwriting existing or supervision columns."""

        if obs.empty:
            return obs
        obs = obs.copy()
        skip_keys = {"gt_cell_type", "target", "target_name"}
        keys = []
        for info in meta_by_image.values():
            for key in info:
                if key not in keys:
                    keys.append(key)
        for key in keys:
            if key in obs.columns or key in skip_keys:
                continue
            obs[key] = [
                meta_by_image.get(str(name), {}).get(key)
                for name in obs["image_name"].to_numpy()
            ]
        if meta_map:
            for key in meta_map:
                if key in obs.columns or key in skip_keys or key == "batch_name":
                    continue
                obs[key] = FeatureReader.encode_raw_from_meta_map(obs["image_name"], key, meta_map)
                print(
                    f"[A] meta_map {key} counts={obs[key].value_counts().to_dict()}",
                    flush=True,
                )
        return obs

    @staticmethod
    def fill_obs_from_meta_map(
        obs: pd.DataFrame,
        keys: list[str] | tuple[str, ...],
        meta_map: dict | None,
    ) -> pd.DataFrame:
        """Write requested meta keys to obs; leave existing columns, gt_cell_type, and empty keys unchanged. Skip keys absent from source meta_map.

        Match requested keys case-insensitively to canonical obs / meta_map names, e.g. modality -> Modality.
        """

        if obs.empty or not meta_map:
            return obs
        obs = obs.copy()
        skip_keys = {"", "gt_cell_type", "target", "target_name"}
        for key in keys:
            key = FeatureReader.resolve_obs_key(obs, str(key or ""), meta_map)
            if key in skip_keys or key in obs.columns or key not in meta_map:
                continue
            obs[key] = FeatureReader.encode_raw_from_meta_map(obs["image_name"], key, meta_map)
            print(
                f"[B] filled obs {key} from meta_map counts={obs[key].value_counts().to_dict()}",
                flush=True,
            )
        return obs

    @staticmethod
    def encode_raw_from_meta_map(image_names: pd.Series, meta_key: str, meta_map: dict) -> pd.Series:
        """Encode an image name using one meta_map key without modifying the caller's dictionary."""

        if meta_key == "batch_name":
            raise RuntimeError(
                "When using batch_name as a prediction target, read the existing meta_info / obs column; "
                "do not modify the batch_name dictionary during analysis."
            )
        inner = meta_map[meta_key]
        if isinstance(inner, dict):
            inner = dict(inner)
        mapper = {meta_key: inner}
        values = [
            METAMAP.build_meta_info_from_meta_map(str(name), mapper)[meta_key]
            for name in image_names.to_numpy()
        ]
        return pd.Series(values, index=image_names.index)

    @staticmethod
    def decode_target_name(raw_value, meta_key: str, meta_map: dict, type_index_to_name: dict | None) -> str:
        """Translate a raw target value into a name keyword."""

        if meta_key == "gt_cell_type" and type_index_to_name:
            try:
                return str(type_index_to_name.get(int(raw_value), raw_value))
            except (TypeError, ValueError):
                return str(raw_value)
        if meta_key in (meta_map or {}):
            return METAMAP.decode_mapper_value(meta_map[meta_key], raw_value)
        return "" if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)) else str(raw_value)

    @staticmethod
    def resolve_obs_key(
        obs: pd.DataFrame,
        key: str,
        meta_map: dict | None = None,
    ) -> str:
        """Match a requested meta key to its unique case variant in obs / meta_map."""

        key = str(key or "")
        if not key or key in obs.columns or key in (meta_map or {}):
            return key
        lowered = key.lower()
        column_hits = [name for name in obs.columns if str(name).lower() == lowered]
        if len(column_hits) == 1:
            return str(column_hits[0])
        map_hits = [name for name in (meta_map or {}) if str(name).lower() == lowered]
        if len(map_hits) == 1:
            return str(map_hits[0])
        return key

    @staticmethod
    def decode_gt_cell_type(series: pd.Series, type_index_to_name: dict) -> pd.Series:
        """Translate integer gt_cell_type values into type names using type_index_to_name."""

        names = {int(index): str(name) for index, name in type_index_to_name.items()}
        return series.map(lambda value: names[int(value)])

    @staticmethod
    def resolve_prediction_target(
        obs: pd.DataFrame,
        meta_key: str,
        target_kind: str,
        meta_map: dict,
        meta_map_override: dict | None,
        type_name_to_index: dict[str, int] | None,
        type_index_to_name: dict[int, str] | None,
    ) -> tuple[pd.DataFrame, dict]:
        """Write obs['target'] / obs['target_name'] and return the class-label mapping."""

        obs = obs.copy()
        # Use the majority-vote column for instance types; re-encode image names for mapping overrides, otherwise prefer existing obs columns.
        if meta_key == "gt_cell_type":
            obs = FeatureReader.encode_gt_cell_type(obs, type_name_to_index, type_index_to_name)
            raw = obs["gt_cell_type"]
        elif meta_map_override:
            raw = FeatureReader.encode_raw_from_meta_map(obs["image_name"], meta_key, meta_map)
        elif meta_key in obs.columns:
            raw = obs[meta_key]
        else:
            raw = FeatureReader.encode_raw_from_meta_map(obs["image_name"], meta_key, meta_map)

        # Continuous targets: use numeric values directly; extract numbers from strings only for training y without changing the original obs column.
        if target_kind == "continuous":
            numeric = pd.to_numeric(raw, errors="coerce")
            value_map: dict[str, float] = {}
            if numeric.notna().all():
                target = numeric.to_numpy(dtype=np.float64)
            else:
                labels = raw.astype(str)
                used: dict[float, str] = {}
                for label in pd.unique(labels):
                    hits = _CONTINUOUS_LABEL_NUMBER.findall(label)
                    if len(hits) != 1:
                        raise RuntimeError(
                            f"Continuous target {meta_key!r} value {label!r} "
                            f"does not yield exactly one number (matches: {hits})"
                        )
                    number = float(hits[0])
                    if number in used:
                        raise RuntimeError(
                            f"Continuous target {meta_key!r} values {label!r} and {used[number]!r} "
                            f"yield the same number {number}"
                        )
                    used[number] = label
                    value_map[label] = number
                target = labels.map(value_map).to_numpy(dtype=np.float64)
            names = [
                FeatureReader.decode_target_name(
                    value, meta_key, meta_map, type_index_to_name,
                ) or str(value)
                for value in raw.to_numpy()
            ]
            obs["target"] = target
            obs["target_name"] = names
            return obs, {
                "n_classes": 0,
                "class_index_to_name": {},
                "class_index_to_raw": {},
                "continuous_value_map": value_map,
            }

        if target_kind != "discrete":
            raise ValueError(f"target_kind must be 'discrete' or 'continuous'; got {target_kind!r}")

        # Discrete targets: preserve existing 0..K-1 values; otherwise map sorted unique values to consecutive integers.
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.notna().all():
            raw_ints = numeric.astype(np.int64).to_numpy()
            unique_raw = np.unique(raw_ints)
            if unique_raw.min() == 0 and unique_raw.max() == unique_raw.size - 1:
                encoded = raw_ints
                index_to_raw = {int(i): int(i) for i in unique_raw}
            else:
                index_to_raw = {i: int(v) for i, v in enumerate(unique_raw)}
                raw_to_index = {int(v): i for i, v in enumerate(unique_raw)}
                encoded = np.array([raw_to_index[int(v)] for v in raw_ints], dtype=np.int64)
        else:
            names_raw = raw.astype(str).to_numpy()
            unique_names = sorted(set(names_raw.tolist()))
            name_to_index = {name: i for i, name in enumerate(unique_names)}
            encoded = np.array([name_to_index[name] for name in names_raw], dtype=np.int64)
            index_to_raw = {i: name for i, name in enumerate(unique_names)}

        n_classes = int(encoded.max()) + 1 if encoded.size else 0
        if n_classes < 2:
            raise RuntimeError(f"Discrete target {meta_key!r} requires at least 2 classes; got n_classes={n_classes}")
        class_index_to_name = {}
        for idx in range(n_classes):
            raw_val = index_to_raw[idx]
            class_index_to_name[idx] = FeatureReader.decode_target_name(
                raw_val, meta_key, meta_map, type_index_to_name,
            ) or str(raw_val)
        obs["target"] = encoded
        obs["target_name"] = [class_index_to_name[int(i)] for i in encoded]
        return obs, {
            "n_classes": n_classes,
            "class_index_to_name": {str(k): v for k, v in class_index_to_name.items()},
            "class_index_to_raw": {str(k): v for k, v in index_to_raw.items()},
        }

    @staticmethod
    def image_id_matches_fov(image_id: str, fov_id: str) -> bool:
        """Match image keys to this FOV by exact equality or the `{fov}__` prefix (avoid matching Point1 to Point10)."""

        image_id = str(image_id)
        fov_id = str(fov_id)
        return image_id == fov_id or image_id.startswith(fov_id + "__")

    @staticmethod
    def expand_h5ad_split_names_by_fov(
        h5ad_path: str,
        train_fovs: tuple[str, ...] | list[str],
        test_fovs: tuple[str, ...] | list[str],
    ) -> tuple[list[str], list[str]]:
        """Expand H5AD image_id values into train / test image name lists using FOV lists."""

        train_fovs = [str(v) for v in train_fovs]
        test_fovs = [str(v) for v in test_fovs]
        overlap = set(train_fovs) & set(test_fovs)
        if overlap:
            raise RuntimeError(f"train/test FOV overlap: {sorted(overlap)}")
        store = AnnDataResultStore(h5ad_path)
        image_ids = [str(v) for v in store.list_image_ids()]
        train_names: list[str] = []
        test_names: list[str] = []
        unmatched: list[str] = []
        for image_id in image_ids:
            if any(FeatureReader.image_id_matches_fov(image_id, fov) for fov in train_fovs):
                train_names.append(image_id)
            elif any(FeatureReader.image_id_matches_fov(image_id, fov) for fov in test_fovs):
                test_names.append(image_id)
            else:
                unmatched.append(image_id)
        if unmatched:
            raise RuntimeError(
                f"H5AD contains {len(unmatched)} images outside the provided FOV lists, e.g. {unmatched[:5]}"
            )
        if not train_names or not test_names:
            raise RuntimeError(
                f"FOV expansion produced train={len(train_names)} test={len(test_names)}; both splits must be nonempty"
            )
        print(
            f"[split] train_fovs={len(train_fovs)} test_fovs={len(test_fovs)} "
            f"train_images={len(train_names)} test_images={len(test_names)}",
            flush=True,
        )
        return train_names, test_names

    @staticmethod
    def attach_h5ad_track_columns(obs: pd.DataFrame, h5ad_path: str) -> pd.DataFrame:
        """Attach track_* from source H5AD by (image_id, instance_id); never join by obs_name."""

        if "image_id" not in obs.columns or "instance_id" not in obs.columns:
            raise RuntimeError("Attaching tracks requires image_id and instance_id in obs")
        with h5py.File(h5ad_path, "r") as handle:
            luma = handle["uns"]["luma"]
            if "tracking" not in luma:
                print("[A] H5AD has no /uns/luma/tracking; skipping track attachment", flush=True)
                return obs
            src_obs = ad.io.read_elem(handle["obs"])
            summary = ad.io.read_elem(luma["tracking"]["track_summary"])

        src = src_obs.reset_index(drop=True)
        src["image_id"] = src["image_id"].astype(str)
        src["instance_id"] = pd.to_numeric(src["instance_id"], errors="raise").astype(np.int64)
        src_cols = ["image_id", "instance_id", "track_id", "track_order"]
        src = src[src_cols].drop_duplicates(["image_id", "instance_id"], keep="first")

        left = obs.reset_index(drop=True)
        left["image_id"] = left["image_id"].astype(str)
        left["instance_id"] = pd.to_numeric(left["instance_id"], errors="raise").astype(np.int64)
        merged = left.merge(src, on=["image_id", "instance_id"], how="left")
        merged["track_id"] = pd.to_numeric(merged["track_id"], errors="coerce").fillna(-1).astype(np.int64)
        merged["track_order"] = pd.to_numeric(merged["track_order"], errors="coerce").fillna(-1).astype(np.int64)

        summary = summary.reset_index(drop=True)
        keep_sum = [c for c in ("track_id", "track_length", "duration_frames", "n_gap_nodes") if c in summary.columns]
        summary = summary[keep_sum].drop_duplicates("track_id", keep="first")
        summary["track_id"] = pd.to_numeric(summary["track_id"], errors="raise").astype(np.int64)
        merged = merged.merge(summary, on="track_id", how="left")

        n_assigned = int((merged["track_id"] >= 0).sum())
        print(
            f"[A] attached tracks n_assigned={n_assigned}/{int(merged.shape[0])} "
            f"n_unique_tracks={int(merged.loc[merged['track_id'] >= 0, 'track_id'].nunique())}",
            flush=True,
        )
        if "obs_name" in merged.columns:
            merged = merged.set_index("obs_name", drop=False)
        return merged

    @staticmethod
    def load_cell_table(h5ad_path: Path | str) -> dict[str, Any]:
        """Restore the analysis dictionary from the main cell table's H5AD; the caller rebuilds morphology groups by column name."""

        adata = ad.read_h5ad(h5ad_path)
        obs = adata.obs.copy()
        if "obs_name" in obs.columns:
            obs = obs.reset_index(drop=True)
        else:
            obs = obs.reset_index()
        X_sam = np.asarray(adata.obsm["X_sam"], dtype=np.float32)
        morph = np.asarray(adata.obsm["X_morph"], dtype=np.float32)
        morph_raw = np.asarray(adata.obsm["X_morph_raw"], dtype=np.float32)
        morph_columns = [str(c) for c in adata.uns["morph_columns"]]
        morph_raw_columns = [str(c) for c in adata.uns["morph_raw_columns"]]
        embedding_columns = [str(c) for c in adata.var_names]
        config_key = "cell_table_config" if "cell_table_config" in adata.uns else "cell_feature_config"
        return {
            "obs": obs,
            "embeddings": X_sam,
            "X_sam": X_sam,
            "embedding_columns": embedding_columns,
            "extended_morph_raw": morph_raw,
            "extended_morph_raw_columns": morph_raw_columns,
            "extended_morph": morph,
            "extended_morph_columns": morph_columns,
            "config": dict(adata.uns[config_key]),
        }

    @staticmethod
    def read_uns(h5ad_path: Path | str, key_path: str, run_name: str | None = None):
        """Read an uns subtree by slash-delimited path without loading X."""

        key_path = FeatureWriter.run_key(run_name, key_path) if run_name else key_path
        with h5py.File(h5ad_path, "r") as handle:
            return ad.io.read_elem(FeatureReader._resolve_uns_node(handle, key_path))

    @staticmethod
    def has_uns(h5ad_path: Path | str, key_path: str, run_name: str | None = None) -> bool:
        """Check whether a slash-delimited uns path exists."""

        key_path = FeatureWriter.run_key(run_name, key_path) if run_name else key_path
        with h5py.File(h5ad_path, "r") as handle:
            node = handle["uns"]
            for part in str(key_path).split("/"):
                if not part:
                    continue
                key = AnnDataResultStore.sanitize_key(part)
                if key not in node:
                    return False
                node = node[key]
            return True

    @staticmethod
    def read_obsm(h5ad_path: Path | str, key: str) -> np.ndarray:
        """Read one top-level obsm array without loading X."""

        key = AnnDataResultStore.sanitize_key(key)
        with h5py.File(h5ad_path, "r") as handle:
            return np.asarray(ad.io.read_elem(handle["obsm"][key]))

    @staticmethod
    def has_obsm(h5ad_path: Path | str, key: str) -> bool:
        """Check whether top-level obsm contains the key."""

        key = AnnDataResultStore.sanitize_key(key)
        with h5py.File(h5ad_path, "r") as handle:
            return "obsm" in handle and key in handle["obsm"]

    @staticmethod
    def read_run_coords(h5ad_path: Path | str, run_name: str, tag: str) -> dict[str, np.ndarray]:
        """Read dimensionality-reduction coordinates saved by an analysis, keyed by reducer name."""

        payload = FeatureReader.read_uns(h5ad_path, f"coords/{tag}", run_name=run_name)
        coords = {}
        for reducer, array in dict(payload).items():
            if reducer == "obs_name":
                continue
            coords[str(reducer)] = np.asarray(array, dtype=np.float64)
        return coords

    @staticmethod
    def read_figure(h5ad_path: Path | str, key: str, run_name: str | None = None) -> bytes:
        """Read the raw bytes of a stored SVG."""

        arr = FeatureReader.read_uns(h5ad_path, f"visualization/figures/{key}", run_name=run_name)
        return np.asarray(arr, dtype=np.uint8).tobytes()

    @staticmethod
    def read_seed0(h5ad_path: Path | str, tag: str, run_name: str | None = None) -> dict[str, Any]:
        """Read seed 0 BottleneckMLP weights and hyperparameters and reconstruct the training payload."""

        payload = FeatureReader.read_uns(
            h5ad_path, f"type_prediction/proj64_seed0/{tag}", run_name=run_name,
        )
        state = payload["state_dict"]
        return {
            "state_dict": {str(k): np.asarray(v) for k, v in dict(state).items()},
            "in_dim": int(np.asarray(payload["in_dim"]).reshape(-1)[0]),
            "bottleneck_dim": int(np.asarray(payload["bottleneck_dim"]).reshape(-1)[0]),
            "mlp_hidden": int(np.asarray(payload["mlp_hidden"]).reshape(-1)[0]),
            "out_dim": int(np.asarray(payload["out_dim"]).reshape(-1)[0]),
            "input_name": str(np.asarray(payload["input_name"]).reshape(-1)[0]),
            "seed": 0,
        }

    @staticmethod
    def _resolve_uns_node(handle: h5py.File, key_path: str):
        """Open a slash-delimited node path under uns."""

        node = handle["uns"]
        for part in str(key_path).split("/"):
            if not part:
                continue
            node = node[AnnDataResultStore.sanitize_key(part)]
        return node


class FeatureWriter:
    """Write the main cell table and append analysis results to uns/runs/<run_name>/."""

    CELL_TABLE_NAME = "cell_table.h5ad"
    INPUT_EMB = "emb*"
    INPUT_MORPH = "morph_ext"
    INPUT_CONCAT = "emb*||morph_ext"
    FIGURE_DPI = 200

    @staticmethod
    def resolve_h5ad_path(source_kind: str, h5ad_path: str, output_root: Path | str) -> Path:
        """Write H5AD input back to its source; write Dataset input to output_root/cell_table.h5ad."""

        if source_kind == "h5ad":
            return Path(h5ad_path)
        return Path(output_root) / FeatureWriter.CELL_TABLE_NAME

    @staticmethod
    def build_run_name(
        meta_key: str,
        target_kind: str,
        obs_equals: dict[str, Any] | None = None,
        exclude_obs_in: dict[str, Any] | None = None,
    ) -> str:
        """Build a stable run_name from the target key, task kind, and filter conditions."""

        parts = []
        for key, value in sorted(dict(obs_equals or {}).items()):
            parts.append(f"{key}={value}")
        for key, values in sorted(dict(exclude_obs_in or {}).items()):
            joined = ",".join(str(item) for item in list(values))
            parts.append(f"{key}!={joined}")
        filt = "|".join(parts) if parts else "all"
        return f"{meta_key}|{target_kind}|{filt}"

    @staticmethod
    def run_key(run_name: str, key_path: str) -> str:
        """Prefix slash-delimited uns paths with runs/<run_name>/."""

        return f"runs/{run_name}/{key_path}"

    @staticmethod
    def write_cell_table(
        h5ad_path: Path | str,
        cell_table: dict[str, Any],
        skipped: list[dict[str, Any]],
        in_place: bool = False,
    ) -> Path:
        """Write the main cell table. With in_place, replace only X/obs/var/obsm and preserve /uns/luma."""

        groups_rows = []
        for gname, cols in cell_table["feature_groups"].items():
            for col in cols:
                groups_rows.append({"feature": col, "group": gname})
        skipped_rows = [
            {"image_id": r["image_id"], "image_name": r["image_name"], "skip_reason": r["skip_reason"]}
            for r in skipped
        ]
        h5ad_path = Path(h5ad_path)
        uns_payload = {
            "morph_feature_groups": pd.DataFrame(groups_rows),
            "skipped_images": pd.DataFrame(
                skipped_rows, columns=["image_id", "image_name", "skip_reason"],
            ),
            "morph_raw_columns": list(cell_table["extended_morph_raw_columns"]),
            "morph_columns": list(cell_table["extended_morph_columns"]),
            "cell_table_config": json.loads(json.dumps(cell_table["config"], default=str)),
        }
        if in_place:
            obs, var, X_sam, obsm = FeatureWriter.cell_table_arrays(cell_table)
            store = AnnDataResultStore(str(h5ad_path))
            store.write_table_elements(x_matrix=X_sam, obs=obs, var=var, obsm=obsm)
            for key, value in uns_payload.items():
                FeatureWriter.write_uns(h5ad_path, key, value)
        else:
            adata = FeatureWriter.build_cell_anndata(cell_table)
            for key, value in uns_payload.items():
                adata.uns[key] = value
            h5ad_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write_h5ad(h5ad_path, compression="gzip")
        print(f"[write] {h5ad_path} n_cells={cell_table['obs'].shape[0]} in_place={in_place}", flush=True)
        return h5ad_path

    @staticmethod
    def cell_table_arrays(cell_table: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
        """Convert the cell table dictionary into AnnData obs / var / X / obsm."""

        obs = cell_table["obs"].copy()
        if "obs_name" in obs.columns:
            obs = obs.set_index("obs_name", drop=False)
        X_sam = np.asarray(cell_table["X_sam"], dtype=np.float32)
        var = pd.DataFrame(
            {
                "feature_id": cell_table["embedding_columns"],
                "feature_source": ["sam"] * len(cell_table["embedding_columns"]),
            },
            index=cell_table["embedding_columns"],
        )
        obsm = {
            "X_sam": X_sam,
            "X_morph_raw": np.asarray(cell_table["extended_morph_raw"], dtype=np.float32),
            "X_morph": np.asarray(cell_table["extended_morph"], dtype=np.float32),
        }
        return obs, var, X_sam, obsm

    @staticmethod
    def build_cell_anndata(cell_table: dict[str, Any]) -> ad.AnnData:
        """Create an AnnData representation of the cell table: X=X_sam, with morphology in obsm."""

        obs, var, X_sam, obsm = FeatureWriter.cell_table_arrays(cell_table)
        adata = ad.AnnData(X=X_sam, obs=obs, var=var)
        for key, array in obsm.items():
            adata.obsm[key] = array
        return adata

    @staticmethod
    def write_uns(
        h5ad_path: Path | str,
        key_path: str,
        value: Any,
        run_name: str | None = None,
    ) -> None:
        """Append or overwrite an uns subtree in an existing H5AD."""

        if run_name:
            key_path = FeatureWriter.run_key(run_name, key_path)
        h5ad_path = Path(h5ad_path)
        parts = [AnnDataResultStore.sanitize_key(p) for p in str(key_path).split("/") if p]
        store = AnnDataResultStore(str(h5ad_path))
        with h5py.File(h5ad_path, "a") as handle:
            parent = handle["uns"]
            for part in parts[:-1]:
                parent = store.require_dict_group(parent, part)
            store.write_value(parent, parts[-1], value, overwrite=True)
        if isinstance(value, pd.DataFrame):
            FeatureWriter.log_table(f"uns/{key_path}", value)
            return
        print(f"[write] uns/{key_path}", flush=True)

    @staticmethod
    def write_run_coords(
        h5ad_path: Path | str,
        run_name: str,
        tag: str,
        coords: dict[str, np.ndarray],
        obs_names: np.ndarray | list[str],
    ) -> None:
        """Write an analysis run's dimensionality-reduction coordinates to uns/runs/<run_name>/coords/<tag>."""

        payload = {"obs_name": np.asarray(obs_names, dtype=object)}
        for reducer, array in coords.items():
            payload[str(reducer)] = np.asarray(array, dtype=np.float64)
        FeatureWriter.write_uns(h5ad_path, f"coords/{tag}", payload, run_name=run_name)

    @staticmethod
    def write_run_index(h5ad_path: Path | str, run_row: dict[str, Any]) -> None:
        """Register this analysis in uns/run_index; overwrite runs with the same name."""

        row = pd.DataFrame([run_row])
        if FeatureReader.has_uns(h5ad_path, "run_index"):
            current = FeatureReader.read_uns(h5ad_path, "run_index")
            current = current.loc[current["run_name"].astype(str) != str(run_row["run_name"])]
            row = pd.concat([current, row], ignore_index=True)
        FeatureWriter.write_uns(h5ad_path, "run_index", row)

    @staticmethod
    def write_figure(
        h5ad_path: Path | str,
        key: str,
        fig,
        run_name: str | None = None,
    ) -> None:
        """Save a Figure as SVG bytes under visualization/figures/<key>."""

        buf = BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
        FeatureWriter.write_uns(h5ad_path, f"visualization/figures/{key}", svg, run_name=run_name)

    @staticmethod
    def write_seed0(
        h5ad_path: Path | str,
        stem: str,
        payload: dict[str, Any],
        run_name: str | None = None,
    ) -> None:
        """Write seed 0 BottleneckMLP weights as an uns array dictionary."""

        packed = {
            "in_dim": int(payload["in_dim"]),
            "bottleneck_dim": int(payload["bottleneck_dim"]),
            "mlp_hidden": int(payload["mlp_hidden"]),
            "out_dim": int(payload["out_dim"]),
            "input_name": str(payload["input_name"]),
            "seed": 0,
            "state_dict": {
                str(name): np.asarray(weight.detach().cpu()) if hasattr(weight, "detach") else np.asarray(weight)
                for name, weight in dict(payload["state_dict"]).items()
            },
        }
        FeatureWriter.write_uns(
            h5ad_path, f"type_prediction/proj64_seed0/{stem}", packed, run_name=run_name,
        )

    @staticmethod
    def export_figures(
        h5ad_path: Path | str,
        export_dir: Path | str,
        figure_format: str = "svg",
        dpi: int = FIGURE_DPI,
        keys: list[str] | None = None,
        run_name: str | None = None,
    ) -> None:
        """Export visualization/figures to a folder; figure_format is svg or png."""

        if figure_format not in ("svg", "png"):
            raise ValueError(f"figure_format must be 'svg' or 'png'; got {figure_format!r}")
        if figure_format == "png" and cairosvg is None:
            raise RuntimeError(
                "PNG export requires cairosvg. Install it in monai_yulin: pip install cairosvg"
            )
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        figures = FeatureReader.read_uns(h5ad_path, "visualization/figures", run_name=run_name)
        names = list(keys) if keys is not None else list(figures.keys())
        for name in names:
            raw = np.asarray(figures[name], dtype=np.uint8).tobytes()
            if figure_format == "svg":
                (export_dir / f"{name}.svg").write_bytes(raw)
            else:
                cairosvg.svg2png(
                    bytestring=raw,
                    write_to=str(export_dir / f"{name}.png"),
                    dpi=int(dpi),
                )
            print(f"[export] {export_dir / name}.{figure_format}", flush=True)

    @staticmethod
    def log_table(name: str, df: pd.DataFrame, max_print: int = 80) -> None:
        """Log a preview of the written table."""

        print(f"[write] {name}  rows={len(df)}", flush=True)
        if len(df) <= max_print:
            print(df.to_string(index=False), flush=True)
            return
        print(df.head(16).to_string(index=False), flush=True)
        print(f"  ... {len(df) - 16} more rows omitted from log", flush=True)

    @staticmethod
    def run(metrics: pd.DataFrame | None, h5ad_path: Path | str, run_name: str) -> None:
        """Stage D: write wide and three-input comparison tables to uns/runs/<run_name>/summary."""

        if metrics is None or not len(metrics):
            print("[D] no prediction metrics; skip summary tables", flush=True)
            return
        target_kind = str(metrics["target_kind"].iloc[0]) if "target_kind" in metrics.columns else "discrete"
        default_names = ["macro_ap", "weighted_ap"] if target_kind == "discrete" else ["r2", "rmse"]
        metric_names = [c for c in default_names if c in metrics.columns]
        FeatureWriter.write_uns(
            h5ad_path, "summary/wide", FeatureWriter.build_wide_table(metrics, metric_names),
            run_name=run_name,
        )
        FeatureWriter.write_uns(
            h5ad_path,
            "summary/input_comparison",
            FeatureWriter.build_input_comparison(metrics, metric_names),
            run_name=run_name,
        )


    @staticmethod
    def build_wide_table(metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
        """Wide table: rows = input x head, columns = slice x default metric."""

        work = metrics.copy()
        work["row_label"] = work["input"].astype(str) + " | " + work["head"].astype(str)
        values = list(metric_names) + ["n_test", "n_test_cells", "n_test_images", "n_test_fovs", "n_test_tracks"]
        values = [c for c in values if c in work.columns]
        wide = work.pivot_table(
            index="row_label",
            columns="scope",
            values=values,
            aggfunc="mean",
        )
        wide.columns = [f"{metric}_{scope}" for metric, scope in wide.columns]
        return wide.reset_index()


    @staticmethod
    def build_input_comparison(metrics: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
        """Compare medians for the three primary inputs under the same head, slice, and metric."""

        rows = []
        for head in metrics["head"].unique():
            for scope in metrics["scope"].unique():
                sub = metrics[(metrics["head"] == head) & (metrics["scope"] == scope)]
                if sub.empty:
                    continue
                for metric in metric_names:
                    by_input = sub.groupby("input")[metric].median()
                    emb = float(by_input[FeatureWriter.INPUT_EMB]) if FeatureWriter.INPUT_EMB in by_input.index else np.nan
                    morph = float(by_input[FeatureWriter.INPUT_MORPH]) if FeatureWriter.INPUT_MORPH in by_input.index else np.nan
                    concat = float(by_input[FeatureWriter.INPUT_CONCAT]) if FeatureWriter.INPUT_CONCAT in by_input.index else np.nan
                    rows.append({
                        "head": head,
                        "scope": scope,
                        "metric": metric,
                        f"{metric}_emb": emb,
                        f"{metric}_morph_ext": morph,
                        f"{metric}_concat": concat,
                        "delta_concat_vs_emb": concat - emb if np.isfinite(concat) and np.isfinite(emb) else np.nan,
                        "delta_concat_vs_morph": concat - morph if np.isfinite(concat) and np.isfinite(morph) else np.nan,
                        "delta_morph_vs_emb": morph - emb if np.isfinite(morph) and np.isfinite(emb) else np.nan,
                    })
        return pd.DataFrame(rows)

