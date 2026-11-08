"""Extract single-cell tables from Dataset or LUMA result H5AD and summarize tracking quality.

The main cell table contains SAM mean embeddings and 136 canonical morphology columns. Do not store analysis targets or filters in it.
Write H5AD input back in place; write Dataset input to a separate cell_table.h5ad.
"""

from __future__ import annotations

import gc
import os
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.measure import regionprops
from tqdm import tqdm

from scripts_formal.utils.elements_morphology import (
    all_feature_names,
    build_feature_groups,
    compute_morph_matrix,
)
from scripts_formal.utils.elements_result_io import AnnDataResultStore, FeatureReader, FeatureWriter
from scripts_formal.utils.elements_vit import retrieve_sam_encoder

SAM_INPUT_SIZE = (1024, 1024)
SAM_AFTER_THREADS = 1
_BLAS_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextmanager
def sam_vit_forward_threads(
    n_forward: int,
    n_after: int = SAM_AFTER_THREADS,
):
    """Use n_forward threads for SAM ViT forward passes; restore n_after on exit.

    Morphology uses a thread pool; nested OpenMP would compete for CPU cores. Reduce threads to 1 after forward passes.
    """

    for name in _BLAS_THREAD_ENV:
        os.environ[name] = str(int(n_forward))
    torch.set_num_threads(int(n_forward))
    try:
        yield
    finally:
        for name in _BLAS_THREAD_ENV:
            os.environ[name] = str(int(n_after))
        torch.set_num_threads(int(n_after))


def resolve_torch_device(device: str, n_jobs: int) -> torch.device:
    """Resolve the device for table extraction / classification / regression. Use n_jobs compute threads on cpu."""

    name = str(device).strip().lower()
    if name == "cpu":
        for key in _BLAS_THREAD_ENV:
            os.environ[key] = str(int(n_jobs))
        torch.set_num_threads(int(n_jobs))
        resolved = torch.device("cpu")
        print(f"[device] {resolved} n_jobs={int(n_jobs)}", flush=True)
        return resolved
    if name == "cuda":
        resolved = torch.device("cuda")
        print(f"[device] {resolved}", flush=True)
        return resolved
    raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")


class CellStatistician:
    """Extract single-cell tables and summarize track_length and other quality statistics when tracks are available."""

    def extract_and_save_cell_table(
        self,
        source_kind: str,
        dataset_path: str,
        h5ad_path: str,
        instance_source: str,
        target_channel: list[int],
        target_size: tuple[int, int] | None,
        train_image_names: tuple[str, ...] | list[str],
        test_image_names: tuple[str, ...] | list[str],
        extension_rate: float,
        sam_checkpoint_path: str,
        backbone_type: str,
        batch_size: int,
        use_bf16: bool,
        type_index_to_name: dict[int, str] | None,
        type_name_to_index: dict[str, int] | None,
        extra_config: dict[str, Any],
        output_root: str = "",
        meta_map: dict | None = None,
        n_morph_workers: int = 7,
        max_cells_per_image: int | None = None,
        sample_seed: int = 0,
        attach_tracks: bool = True,
        device: str = "cpu",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Read source -> extract features -> write the main cell table; do not persist analysis targets or filters."""

        if max_cells_per_image is not None and int(max_cells_per_image) <= 0:
            raise ValueError("max_cells_per_image must be a positive integer or None")
        if max_cells_per_image is not None and attach_tracks:
            raise ValueError("max_cells_per_image and attach_tracks are mutually exclusive")

        resolved_meta_map = FeatureReader.resolve_source_meta_map(
            source_kind=source_kind,
            dataset_path=dataset_path,
            h5ad_path=h5ad_path,
            meta_map_override=meta_map,
        )
        records = list(FeatureReader.iter_image_records(
            source_kind=source_kind,
            dataset_path=dataset_path,
            h5ad_path=h5ad_path,
            instance_source=instance_source,
            target_channel=target_channel,
            target_size=target_size,
            train_image_names=train_image_names,
            test_image_names=test_image_names,
        ))
        skipped = [r for r in records if r.get("skip_reason")]
        usable = [r for r in records if not r.get("skip_reason")]
        print(f"[A] images usable={len(usable)} skipped={len(skipped)}", flush=True)

        if max_cells_per_image is not None:
            multi = [
                r["image_name"] for r in usable
                if len(r.get("patches") or []) > 1
            ]
            if multi:
                raise ValueError(
                    "max_cells_per_image supports only one patch per image; multi-patch images include "
                    + ", ".join(str(name) for name in multi[:5])
                )

        if source_kind == "dataset":
            torch_device = resolve_torch_device(device, n_jobs)
            encoder = self.build_sam_encoder(
                backbone_type, sam_checkpoint_path, torch_device, n_jobs,
            )
            cell_table = self.extract_from_dataset(
                records=usable,
                encoder=encoder,
                device=torch_device,
                extension_rate=extension_rate,
                batch_size=batch_size,
                use_bf16=use_bf16,
                type_index_to_name=type_index_to_name,
                n_morph_workers=n_morph_workers,
                max_cells_per_image=max_cells_per_image,
                sample_seed=sample_seed,
                n_jobs=n_jobs,
            )
            del encoder
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
        elif source_kind == "h5ad":
            cell_table = self.extract_from_h5ad(
                records=usable,
                extension_rate=extension_rate,
                type_index_to_name=type_index_to_name,
                target_size=target_size,
                n_morph_workers=n_morph_workers,
                max_cells_per_image=max_cells_per_image,
                sample_seed=sample_seed,
            )
        else:
            raise ValueError(f"unknown source_kind={source_kind!r}")

        n_images_usable = int(len(usable))
        n_images_skipped = int(len(skipped))
        meta_by_image = {
            str(record["image_name"]): dict(record.get("meta_info", {}) or {})
            for record in usable
        }
        cell_table["obs"] = FeatureReader.attach_image_meta_to_obs(
            cell_table["obs"],
            meta_by_image,
            resolved_meta_map,
        )
        if source_kind == "h5ad" and attach_tracks:
            cell_table["obs"] = FeatureReader.attach_h5ad_track_columns(
                cell_table["obs"], h5ad_path,
            )
        cell_table["obs"] = FeatureReader.encode_gt_cell_type(
            cell_table["obs"].reset_index(drop=True),
            type_name_to_index,
            type_index_to_name,
        )
        if "image_name" in cell_table["obs"].columns:
            n_images_usable = int(cell_table["obs"]["image_name"].nunique())

        feature_cols = list(cell_table["extended_morph_columns"])
        cell_table["feature_groups"] = build_feature_groups(pd.DataFrame(columns=feature_cols))
        print(f"[A] morph columns={len(feature_cols)}", flush=True)

        dest = FeatureWriter.resolve_h5ad_path(source_kind, h5ad_path, output_root)
        config = dict(extra_config)
        config.update({
            "source_kind": source_kind,
            "dataset_path": dataset_path,
            "h5ad_path": str(h5ad_path or dest),
            "instance_source": instance_source,
            "target_channel": list(target_channel),
            "target_size": None if target_size is None else list(target_size),
            "extension_rate": float(extension_rate),
            "extension_median_scope": "image_valid_region",
            "embedding_layout": "mean_1024",
            "n_cells": int(cell_table["obs"].shape[0]),
            "n_features": int(len(cell_table["extended_morph_columns"])),
            "n_features_raw": int(len(cell_table["extended_morph_raw_columns"])),
            "length_features_scaled": True,
            "n_morph_workers": int(n_morph_workers),
            "n_images_usable": n_images_usable,
            "n_images_skipped": n_images_skipped,
            "type_index_to_name": {str(k): v for k, v in (type_index_to_name or {}).items()},
            "type_name_to_index": dict(type_name_to_index or {}),
            "meta_map_keys": list(resolved_meta_map.keys()),
            "max_cells_per_image": max_cells_per_image,
            "sample_seed": int(sample_seed),
            "attach_tracks": bool(attach_tracks and source_kind == "h5ad"),
        })
        if source_kind == "h5ad":
            config["embedding_origin"] = "h5ad_patch_embeddings_frozen_sam"
        cell_table["config"] = config
        FeatureWriter.write_cell_table(dest, cell_table, skipped, in_place=(source_kind == "h5ad"))

        if source_kind == "h5ad" and attach_tracks:
            quality = self.summarize_track_quality(h5ad_path=str(dest), obs=cell_table["obs"])
            if len(quality["per_track"]):
                FeatureWriter.write_uns(dest, "tracking_quality", quality["per_track"])
                FeatureWriter.write_uns(dest, "tracking_quality_summary", quality["summary"])
        return cell_table

    @staticmethod
    def apply_obs_equals_filter(
        cell_table: dict[str, Any],
        obs_equals: dict[str, Any] | None,
        meta_map: dict | None = None,
        type_index_to_name: dict | None = None,
    ) -> dict[str, Any]:
        """Filter cells by obs column equality and subset embedding and morphology matrices in sync.

        Match column names case-insensitively. Translate gt_cell_type to type names before string comparison;
        compare other columns in their native types and fail directly on type mismatch.
        """

        if not obs_equals:
            return cell_table
        obs = cell_table["obs"].reset_index(drop=True)
        n_before = int(obs.shape[0])
        keep = np.ones(n_before, dtype=bool)
        for column, value in obs_equals.items():
            keep &= CellStatistician.obs_filter_mask(
                obs, column, [value], meta_map, type_index_to_name,
            )
        n_keep = int(keep.sum())
        print(f"[A] obs_equals={dict(obs_equals)} keep={n_keep}/{n_before}", flush=True)
        return CellStatistician.slice_cell_table(cell_table, keep)


    @staticmethod
    def apply_obs_in_exclude(
        cell_table: dict[str, Any],
        exclude_obs_in: dict[str, Any] | None,
        meta_map: dict | None = None,
        type_index_to_name: dict | None = None,
    ) -> dict[str, Any]:
        """Drop cells whose obs column values belong to the given set and subset matrices in sync.

        Use the same column-name and value matching rules as apply_obs_equals_filter.
        """

        if not exclude_obs_in:
            return cell_table
        obs = cell_table["obs"].reset_index(drop=True)
        n_before = int(obs.shape[0])
        keep = np.ones(n_before, dtype=bool)
        for column, values in exclude_obs_in.items():
            keep &= ~CellStatistician.obs_filter_mask(
                obs, column, list(values), meta_map, type_index_to_name,
            )
        n_keep = int(keep.sum())
        dumped = {k: list(v) for k, v in dict(exclude_obs_in).items()}
        print(f"[A] exclude_obs_in={dumped} keep={n_keep}/{n_before}", flush=True)
        return CellStatistician.slice_cell_table(cell_table, keep)

    @staticmethod
    def obs_filter_mask(
        obs: pd.DataFrame,
        column: str,
        values: list[Any],
        meta_map: dict | None,
        type_index_to_name: dict | None,
    ) -> np.ndarray:
        """Build a row mask matching the requested value; translate gt_cell_type to names without converting other columns."""

        resolved = FeatureReader.resolve_obs_key(obs, column, meta_map)
        series = obs[resolved]
        if resolved == "gt_cell_type":
            series = FeatureReader.decode_gt_cell_type(series, type_index_to_name)
        stored = next(value for value in series if pd.notna(value))
        stored_is_str = isinstance(stored, str)
        seen = list(pd.unique(series.dropna()))
        for want in values:
            if isinstance(want, str) != stored_is_str:
                raise RuntimeError(
                    f"Filter {resolved}={want!r} is incompatible with the column's native type; column values={seen}"
                )
        if stored_is_str:
            return series.isin(values).to_numpy()
        return pd.to_numeric(series, errors="raise").isin(
            [float(want) for want in values]
        ).to_numpy()


    @staticmethod
    def slice_cell_table(cell_table: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
        """Subset obs and row-aligned matrices using a boolean mask."""

        obs = cell_table["obs"].reset_index(drop=True)
        obs = obs.loc[keep].reset_index(drop=True)
        if "obs_name" in obs.columns:
            obs = obs.set_index("obs_name", drop=False)
        cell_table["obs"] = obs
        for key in ("embeddings", "X_sam", "extended_morph_raw", "extended_morph"):
            if key in cell_table and cell_table[key] is not None:
                matrix = np.asarray(cell_table[key])
                if matrix.shape[0] == keep.shape[0]:
                    cell_table[key] = matrix[keep]
        return cell_table


    def extract_from_dataset(self, 
        records: list[dict[str, Any]],
        encoder: torch.nn.Module,
        device: torch.device,
        extension_rate: float,
        batch_size: int,
        use_bf16: bool,
        type_index_to_name: dict[int, str] | None,
        n_morph_workers: int = 7,
        max_cells_per_image: int | None = None,
        sample_seed: int = 0,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Encode Dataset patches in batches and collect valid-region cells."""

        morph_columns = all_feature_names()
        if not records:
            print("[A] extract_from_dataset received empty records; returning an empty cell table", flush=True)
            return self.assemble_cell_table([], [], [], morph_columns)

        # Resolve shared fields from the first record and reuse them for logging and subsequent batch reads.
        hdf5_path = records[0]["hdf5_path"]
        stitch_by_image = {r["image_name"]: r["stitch_info"] for r in records}
        split_by_image = {r["image_name"]: r["split"] for r in records}
        raw_target_size = records[0]["target_size"]
        target_size = None if raw_target_size is None else tuple(raw_target_size)
        target_channel = list(records[0]["target_channel"])

        print(
            f"[A] morph workers={int(n_morph_workers)} sam_forward_threads={int(n_jobs)} "
            f"sam_input={SAM_INPUT_SIZE} morph_target_size={target_size}",
            flush=True,
        )
        # Flatten each image's patch list into an encoding job queue.
        patch_jobs = []
        for record in records:
            for patch in record["patches"]:
                patch_jobs.append({
                    "image_name": record["image_name"],
                    "patch_row": patch["patch_row"],
                    "patch_col": patch["patch_col"],
                })

        image_median_diams, image_mean_equiv_radii = self.image_geometry_scales(
            plane="patch",
            patch_jobs=patch_jobs,
            hdf5_path=hdf5_path,
            stitch_by_image=stitch_by_image,
            target_size=target_size,
        )

        obs_rows: list[dict[str, Any]] = []
        embedding_chunks: list[np.ndarray] = []
        morph_chunks: list[np.ndarray] = []

        with h5py.File(hdf5_path, "r") as h5_file:
            pbar = tqdm(range(0, len(patch_jobs), batch_size), desc="dataset patches")
            for batch_start in pbar:
                batch_jobs = patch_jobs[batch_start:batch_start + batch_size]
                batch_images = []
                batch_morph_images = []
                batch_masks = []
                batch_type_maps = []
                for job in batch_jobs:
                    image_nchw, instance_map, type_map = FeatureReader.read_dataset_patch(
                        h5_file=h5_file,
                        image_name=job["image_name"],
                        patch_row=job["patch_row"],
                        patch_col=job["patch_col"],
                        target_channel=target_channel,
                        target_size=target_size,
                    )
                    batch_images.append(torch.from_numpy(sam_input_nchw(image_nchw)))
                    batch_morph_images.append(np.asarray(image_nchw[0], dtype=np.float32))
                    batch_masks.append(instance_map)
                    batch_type_maps.append(type_map)
                image_batch = torch.stack(batch_images, dim=0).to(device, non_blocking=True)
                embed_batch = self.encode_patch_embeddings(
                    encoder, image_batch, use_bf16, n_jobs,
                )

                for batch_idx, job in enumerate(batch_jobs):
                    image_name = job["image_name"]
                    image_hw = batch_morph_images[batch_idx]
                    morph_size = target_size if target_size is not None else tuple(batch_masks[batch_idx].shape)
                    candidates = self.collect_valid_cells(
                        instance_map=batch_masks[batch_idx],
                        type_map=batch_type_maps[batch_idx],
                        intensity_image=image_hw,
                        image_name=image_name,
                        image_id=image_name,
                        split=split_by_image[image_name],
                        stitch_info=stitch_by_image[image_name],
                        extension_rate=extension_rate,
                        median_bbox_diag=image_median_diams[image_name],
                        type_index_to_name=type_index_to_name,
                        plane="patch",
                        target_size=morph_size,
                        patch_row=job["patch_row"],
                        patch_col=job["patch_col"],
                        embedding_hwc=embed_batch[batch_idx],
                    )
                    if max_cells_per_image is not None:
                        candidates = self.sample_candidates_per_image(
                            candidates, image_name, max_cells_per_image, sample_seed,
                        )
                    if not candidates:
                        continue
                    morph_mat = compute_morph_matrix(
                        image_hw, batch_masks[batch_idx], candidates, morph_columns,
                        image_mean_equiv_radius=image_mean_equiv_radii[image_name],
                        n_workers=n_morph_workers,
                    )
                    obs_rows.extend(cell["obs"] for cell in candidates)
                    embedding_chunks.append(
                        np.stack([cell["embedding"] for cell in candidates], axis=0).astype(np.float32)
                    )
                    morph_chunks.append(morph_mat)
                del image_batch, embed_batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                pbar.set_postfix(cells=len(obs_rows))

        return self.assemble_cell_table(obs_rows, embedding_chunks, morph_chunks, morph_columns)


    def extract_from_h5ad(self, 
        records: list[dict[str, Any]],
        extension_rate: float,
        type_index_to_name: dict[int, str] | None,
        target_size: tuple[int, int] | None = None,
        n_morph_workers: int = 7,
        max_cells_per_image: int | None = None,
        sample_seed: int = 0,
    ) -> dict[str, Any]:
        """Re-pool embeddings using GT/pred instance maps for each H5AD image and compute extended morphology.

        Use stored patch embeddings, independent of target_size.
        When target_size is not None, resize morphology images so the patch side length matches it; otherwise do not resize.
        Do not read adata.X / obsm['X_sam']. Discard each image after processing.
        """

        print(f"[A] morph workers={int(n_morph_workers)} morph_target_size={target_size}", flush=True)
        morph_columns = all_feature_names()
        if not records:
            return self.assemble_cell_table([], [], [], morph_columns)
        obs_rows: list[dict[str, Any]] = []
        embedding_chunks: list[np.ndarray] = []
        morph_chunks: list[np.ndarray] = []
        h5ad_path = str(records[0]["h5ad_path"])

        with h5py.File(h5ad_path, "r") as handle:
            images_group = handle["uns"]["luma"]["images"]
            for record in tqdm(records, desc="h5ad images"):
                image_key = AnnDataResultStore.sanitize_key(record["image_id"])
                payload = FeatureReader.load_h5ad_extract_payload(images_group[image_key], record["instance_source"])
                arrays = FeatureReader.unpack_h5ad_arrays(payload, record["instance_source"])
                intensity = arrays["intensity"]
                instance_map = arrays["instance_map"]
                type_map = arrays["type_map"]
                stitch_info = record["stitch_info"]
                image_shape = instance_map.shape
                median_diag, mean_r = self.image_geometry_scales(
                    plane="stitched",
                    instance_map=instance_map,
                    stitch_info=stitch_info,
                    frame_shape=image_shape,
                )
                candidates = self.collect_valid_cells(
                    instance_map=instance_map,
                    type_map=type_map,
                    intensity_image=intensity,
                    image_name=record["image_name"],
                    image_id=record["image_id"],
                    split=record["split"],
                    stitch_info=stitch_info,
                    extension_rate=extension_rate,
                    median_bbox_diag=median_diag,
                    type_index_to_name=type_index_to_name,
                    plane="stitched",
                    embedding_indexer=arrays["embedding_indexer"],
                )
                if max_cells_per_image is not None:
                    candidates = self.sample_candidates_per_image(
                        candidates, record["image_name"], max_cells_per_image, sample_seed,
                    )
                if not candidates:
                    del payload, arrays, intensity, instance_map, type_map
                    continue
                ext_bboxes = np.array([cell["ext_bbox"] for cell in candidates], dtype=np.int64)
                global_map = self.build_global_embedding_map_from_patches(
                    arrays["patch_embeddings"], stitch_info,
                )
                integral = self.build_integral_embedding_map(global_map)
                pooled = self.pool_embeddings_by_integral_bbox(
                    integral, ext_bboxes, image_shape, global_map.shape[:2],
                )
                for cell, vector in zip(candidates, pooled):
                    cell["embedding"] = vector.astype(np.float32)
                    cell["obs"]["embed_region_n"] = -1
                morph_image = intensity
                morph_instance = instance_map
                morph_cells = candidates
                morph_radius = mean_r
                if target_size is not None:
                    scale_y = float(target_size[0]) / float(stitch_info["patch_h"])
                    scale_x = float(target_size[1]) / float(stitch_info["patch_w"])
                    morph_shape = (
                        int(round(image_shape[0] * scale_y)),
                        int(round(image_shape[1] * scale_x)),
                    )
                    morph_image = resize_plane(intensity, morph_shape, nearest=False)
                    morph_instance = resize_plane(instance_map, morph_shape, nearest=True)
                    morph_cells = scale_candidate_geometry(candidates, scale_y, scale_x, morph_shape)
                    morph_radius = float(mean_r) * scale_y if np.isfinite(mean_r) else mean_r
                morph_mat = compute_morph_matrix(
                    morph_image, morph_instance, morph_cells, morph_columns,
                    image_mean_equiv_radius=morph_radius,
                    n_workers=n_morph_workers,
                )
                obs_rows.extend(cell["obs"] for cell in candidates)
                embedding_chunks.append(pooled.astype(np.float32))
                morph_chunks.append(morph_mat)
                del payload, arrays, intensity, instance_map, type_map, global_map, integral
                gc.collect()

        return self.assemble_cell_table(obs_rows, embedding_chunks, morph_chunks, morph_columns)



    def collect_valid_cells(self, 
        instance_map: np.ndarray,
        type_map: np.ndarray | None,
        intensity_image: np.ndarray,
        image_name: str,
        image_id: str,
        split: str,
        stitch_info: dict[str, int],
        extension_rate: float,
        median_bbox_diag: float,
        type_index_to_name: dict[int, str] | None,
        plane: str,
        target_size: tuple[int, int] | None = None,
        patch_row: int = -1,
        patch_col: int = -1,
        embedding_hwc: np.ndarray | None = None,
        embedding_indexer: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """List cells with centroids inside the valid region; pool embeddings immediately when `plane="patch"`."""

        # Instance geometry and extended bounding boxes.
        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids != 0]
        if instance_ids.size == 0:
            return []
        image_shape = instance_map.shape
        props = {int(prop.label): prop for prop in regionprops(instance_map, intensity_image=intensity_image)}
        bbox_list = np.array([props[int(inst_id)].bbox for inst_id in instance_ids], dtype=np.int64)
        extended_bboxes = self.compute_extended_bboxes(
            bbox_list, image_shape, extension_rate, median_bbox_diag=median_bbox_diag,
        )

        # Majority-vote types; use -1 without type_map. Route three-channel maps through majority_cell_types_index.
        if type_map is None:
            cell_types = np.full(instance_ids.shape, -1, dtype=np.int64)
        else:
            cell_types, _ = self.majority_cell_types_index(type_map, instance_map, instance_ids)

        # Patch plane: map encoding dimensions back to native stitched coordinates and record the embedding grid size.
        if plane == "patch":
            native_h = stitch_info.get("patch_h", image_shape[0]) or image_shape[0]
            native_w = stitch_info.get("patch_w", image_shape[1]) or image_shape[1]
            margin_h = stitch_info.get("margin_h", 0)
            margin_w = stitch_info.get("margin_w", 0)
            stride_h = max(native_h - margin_h, 1)
            stride_w = max(native_w - margin_w, 1)
            scale_y = native_h / float(target_size[0])
            scale_x = native_w / float(target_size[1])
            embed_shape = embedding_hwc.shape[:2]

        # Filter instances by valid region and write obs; pool patch embeddings immediately, leaving stitched pooling to the caller's integral images.
        candidates: list[dict[str, Any]] = []
        for idx, inst_id in enumerate(instance_ids):
            prop = props[int(inst_id)]
            centroid_y, centroid_x = float(prop.centroid[0]), float(prop.centroid[1])
            if not self.cell_centroid_in_valid_region(
                centroid_y, centroid_x, stitch_info, plane,
                target_size if plane == "patch" else image_shape,
            ):
                continue
            ext_bbox = extended_bboxes[idx]
            if plane == "patch":
                pr, pc = int(patch_row), int(patch_col)
                global_y = pr * stride_h + centroid_y * scale_y
                global_x = pc * stride_w + centroid_x * scale_x
                gy1, gx1, gy2, gx2 = self.map_bbox_to_embedding_bbox(
                    ext_bbox, image_shape, embed_shape,
                )
                region = embedding_hwc[gy1:gy2, gx1:gx2, :]
                region_h = max(gy2 - gy1, 1)
                region_w = max(gx2 - gx1, 1)
                embed_vec = region.mean(axis=(0, 1)).astype(np.float32)
            else:
                if embedding_indexer is None:
                    pr, pc = -1, -1
                else:
                    indexer = np.asarray(embedding_indexer)
                    row = int(np.clip(centroid_y, 0, image_shape[0] - 1))
                    col = int(np.clip(centroid_x, 0, image_shape[1] - 1))
                    pr, pc = int(indexer[0, row, col]), int(indexer[1, row, col])
                global_y, global_x = centroid_y, centroid_x
                embed_vec, region_h, region_w = None, -1, -1
            obs = self.base_obs_row(
                image_name=image_name,
                image_id=image_id,
                instance_id=int(inst_id),
                patch_row=pr,
                patch_col=pc,
                split=split,
                centroid_y=centroid_y,
                centroid_x=centroid_x,
                global_y=float(global_y),
                global_x=float(global_x),
                bbox=bbox_list[idx],
                ext_bbox=ext_bbox,
                median_bbox_diag=median_bbox_diag,
                embed_region_h=int(region_h),
                embed_region_w=int(region_w),
                type_index=int(cell_types[idx]),
                type_index_to_name=type_index_to_name,
            )
            cell = {
                "obs": obs,
                "instance_id": int(inst_id),
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "ext_bbox": ext_bbox,
            }
            if embed_vec is not None:
                cell["embedding"] = embed_vec
            candidates.append(cell)
        return candidates


    def assemble_cell_table(self, 
        obs_rows: list[dict[str, Any]],
        embedding_chunks: list[np.ndarray],
        morph_chunks: list[np.ndarray],
        morph_columns: list[str],
    ) -> dict[str, Any]:
        """Assemble obs, mean embeddings, and the canonical morphology column matrix."""

        n_morph = len(morph_columns)
        if not obs_rows:
            obs = pd.DataFrame()
            embeddings = np.empty((0, 1024), dtype=np.float32)
            extended_morph = np.empty((0, n_morph), dtype=np.float32)
            n_embed = 1024
        else:
            obs = pd.DataFrame(obs_rows)
            obs = obs.set_index("obs_name", drop=False)
            embeddings = np.concatenate(embedding_chunks, axis=0).astype(np.float32)
            extended_morph = np.concatenate(morph_chunks, axis=0).astype(np.float32)
            n_embed = int(embeddings.shape[1])
        embedding_columns = [f"sam_mean_{i:04d}" for i in range(n_embed)]
        return {
            "obs": obs,
            "embeddings": embeddings,
            "X_sam": embeddings,
            "embedding_columns": embedding_columns,
            "extended_morph_raw": extended_morph,
            "extended_morph_raw_columns": list(morph_columns),
            "extended_morph": extended_morph,
            "extended_morph_columns": list(morph_columns),
            "feature_groups": build_feature_groups(pd.DataFrame(
                np.zeros((0, n_morph)), columns=morph_columns,
            )),
        }



    def image_geometry_scales(self, 
        plane: str,
        instance_map: np.ndarray | None = None,
        stitch_info: dict[str, int] | None = None,
        frame_shape: tuple[int, int] | None = None,
        patch_jobs: list[dict[str, Any]] | None = None,
        hdf5_path: Path | None = None,
        stitch_by_image: dict[str, dict[str, int]] | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[float, float] | tuple[dict[str, float], dict[str, float]]:
        """Compute median bbox diagonals and mean equivalent radii for valid-region cells.

        Single image: pass `instance_map` / `stitch_info` / `frame_shape` and return two scalars.
        Dataset: pass `patch_jobs` / `hdf5_path` / `stitch_by_image` / `target_size`,
        combine diagonals and areas across patches of each `image_name`, then summarize into two dictionaries.
        """

        # Use one record for a single image; for Dataset, open HDF5 and read instance maps per patch without retaining all masks.
        diags_by_image: dict[str, list[float]] = {}
        areas_by_image: dict[str, list[float]] = {}
        h5_file = h5py.File(hdf5_path, "r") if patch_jobs is not None else None
        jobs = patch_jobs if patch_jobs is not None else [None]
        try:
            iterator = (
                tqdm(jobs, desc="Scanning image-level geometry scales")
                if patch_jobs is not None
                else jobs
            )
            for job in iterator:
                if job is None:
                    image_name, inst, stitch, pl, frame = (
                        "", instance_map, stitch_info, plane, frame_shape,
                    )
                else:
                    image_name = job["image_name"]
                    inst = FeatureReader.read_dataset_instance_map(
                        h5_file, image_name, job["patch_row"], job["patch_col"], target_size,
                    )
                    stitch = stitch_by_image[image_name]
                    pl, frame = "patch", target_size
                instance_ids = np.unique(inst)
                instance_ids = instance_ids[instance_ids != 0]
                if instance_ids.size == 0:
                    continue
                props = {int(prop.label): prop for prop in regionprops(inst)}
                for inst_id in instance_ids:
                    prop = props[int(inst_id)]
                    centroid_y, centroid_x = float(prop.centroid[0]), float(prop.centroid[1])
                    if not self.cell_centroid_in_valid_region(
                        centroid_y, centroid_x, stitch, pl, frame,
                    ):
                        continue
                    y1, x1, y2, x2 = prop.bbox
                    diags_by_image.setdefault(image_name, []).append(float(np.hypot(y2 - y1, x2 - x1)))
                    areas_by_image.setdefault(image_name, []).append(float(prop.area))
        finally:
            if h5_file is not None:
                h5_file.close()

        # Compute medians / mean equivalent radii per image; for Dataset images without valid cells, use 0 and nan.
        median_diams: dict[str, float] = {}
        mean_radii: dict[str, float] = {}
        names = list(diags_by_image)
        if patch_jobs is not None:
            names = list(dict.fromkeys([job["image_name"] for job in patch_jobs] + names))
        elif not names:
            names = [""]
        for image_name in names:
            diags = np.asarray(diags_by_image.get(image_name, []), dtype=np.float64)
            diags = diags[diags > 0] if diags.size else diags
            median_diams[image_name] = float(np.median(diags)) if diags.size > 0 else 0.0
            areas = np.asarray(areas_by_image.get(image_name, []), dtype=np.float64).ravel()
            ok = np.isfinite(areas) & (areas > 0)
            mean_radii[image_name] = (
                float(np.sqrt(areas[ok] / np.pi).mean()) if np.any(ok) else np.nan
            )
        if patch_jobs is None:
            key = names[0]
            return median_diams[key], mean_radii[key]
        return median_diams, mean_radii


    def cell_centroid_in_valid_region(self, 
        centroid_y: float,
        centroid_x: float,
        stitch_info: dict[str, int],
        plane: str,
        frame_shape: tuple[int, int],
    ) -> bool:
        """Return True when a centroid lies within the stitch-valid region.

        patch: centroids use encoded `frame_shape`; rescale to native patch coordinates, then remove the inner margin.
        stitched: centroids already use stitched-image pixel coordinates; remove the outer image margin left by edge patches.
        """

        margin_h = stitch_info.get("margin_h", 0)
        margin_w = stitch_info.get("margin_w", 0)
        if plane == "patch":
            canvas_h = stitch_info.get("patch_h", frame_shape[0]) or frame_shape[0]
            canvas_w = stitch_info.get("patch_w", frame_shape[1]) or frame_shape[1]
            cy = centroid_y * (canvas_h / float(frame_shape[0]))
            cx = centroid_x * (canvas_w / float(frame_shape[1]))
        else:
            canvas_h, canvas_w = frame_shape
            cy, cx = centroid_y, centroid_x
        return (
            (margin_h // 2) <= cy < (canvas_h - (margin_h - margin_h // 2))
            and (margin_w // 2) <= cx < (canvas_w - (margin_w - margin_w // 2))
        )


    def compute_extended_bboxes(self, 
        bbox_list: np.ndarray,
        image_shape: tuple[int, int],
        extension_rate: float,
        median_bbox_diag: float | None = None,
    ) -> np.ndarray:
        """Extend each instance bbox by the median diagonal times extension_rate."""

        if median_bbox_diag is None:
            diam_list = np.sqrt(np.sum((bbox_list[:, 2:] - bbox_list[:, :2]) ** 2, axis=1))
            median_bbox_diag = (
                float(np.median(diam_list[diam_list > 0])) if np.any(diam_list > 0) else 0.0
            )
        bbox_extension = float(median_bbox_diag) * extension_rate
        return np.clip(
            bbox_list + np.array([-bbox_extension, -bbox_extension, bbox_extension, bbox_extension]),
            a_min=[0, 0, 0, 0],
            a_max=[image_shape[0], image_shape[1], image_shape[0], image_shape[1]],
        ).astype(np.int64)


    def map_bbox_to_embedding_bbox(self, 
        bbox: np.ndarray,
        image_shape: tuple[int, int],
        embedding_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        """Map image-space bounding boxes to the low-resolution embedding grid."""

        image_h, image_w = image_shape
        embed_h, embed_w = embedding_shape
        y1, x1, y2, x2 = [int(v) for v in bbox]
        gy1 = max(0, int(np.floor(y1 * embed_h / image_h)))
        gx1 = max(0, int(np.floor(x1 * embed_w / image_w)))
        gy2 = min(embed_h, int(np.ceil(y2 * embed_h / image_h)))
        gx2 = min(embed_w, int(np.ceil(x2 * embed_w / image_w)))
        if gy2 <= gy1:
            gy2 = min(embed_h, gy1 + 1)
            gy1 = max(0, gy2 - 1)
        if gx2 <= gx1:
            gx2 = min(embed_w, gx1 + 1)
            gx1 = max(0, gx2 - 1)
        return gy1, gx1, gy2, gx2


    def majority_cell_types_channel(self, 
        type_map: np.ndarray,
        instance_map: np.ndarray,
        instance_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Majority-vote Dataset type_map of shape (n_types, H, W) by summing over instance pixels."""

        n_types = int(type_map.shape[0])
        sorted_ids = np.sort(instance_ids)
        counts = np.zeros((sorted_ids.size, n_types), dtype=np.float64)
        for type_idx in range(n_types):
            counts[:, type_idx] = ndimage.sum_labels(
                type_map[type_idx],
                labels=instance_map,
                index=sorted_ids,
            )
        totals = counts.sum(axis=1, keepdims=True)
        scores = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        majority = np.argmax(scores, axis=1).astype(np.int64)
        id_to_row = {int(inst_id): row for row, inst_id in enumerate(sorted_ids)}
        row_index = np.array([id_to_row[int(inst_id)] for inst_id in instance_ids], dtype=np.int64)
        return majority[row_index], scores[row_index].astype(np.float32)


    def majority_cell_types_index(self, 
        type_map: np.ndarray,
        instance_map: np.ndarray,
        instance_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Majority-vote instance pixels when the H5AD type map contains (H, W) class indexes."""

        type_map = np.asarray(type_map)
        if type_map.ndim == 3:
            return self.majority_cell_types_channel(type_map, instance_map, instance_ids)
        n_types = int(type_map.max()) + 1
        sorted_ids = np.sort(instance_ids)
        counts = np.zeros((sorted_ids.size, n_types), dtype=np.float64)
        for class_id in range(n_types):
            counts[:, class_id] = ndimage.sum_labels(
                (type_map == class_id).astype(np.uint8),
                labels=instance_map,
                index=sorted_ids,
            )
        totals = counts.sum(axis=1, keepdims=True)
        scores = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        majority = np.argmax(scores, axis=1).astype(np.int64)
        id_to_row = {int(inst_id): row for row, inst_id in enumerate(sorted_ids)}
        row_index = np.array([id_to_row[int(inst_id)] for inst_id in instance_ids], dtype=np.int64)
        return majority[row_index], scores[row_index].astype(np.float32)


    def base_obs_row(self, 
        image_name: str,
        image_id: str,
        instance_id: int,
        patch_row: int,
        patch_col: int,
        split: str,
        centroid_y: float,
        centroid_x: float,
        global_y: float,
        global_x: float,
        bbox: np.ndarray,
        ext_bbox: np.ndarray,
        median_bbox_diag: float,
        embed_region_h: int,
        embed_region_w: int,
        type_index: int,
        type_index_to_name: dict[int, str] | None,
    ) -> dict[str, Any]:
        """Assemble one row of cell table obs."""

        gt_cell_type = np.nan
        if type_index >= 0:
            gt_cell_type = int(type_index)
        obs = {
            "obs_name": f"{image_name}__p{patch_row:02d}_{patch_col:02d}__inst_{int(instance_id):06d}",
            "image_name": image_name,
            "image_id": image_id,
            "patch_row": int(patch_row),
            "patch_col": int(patch_col),
            "split": split,
            "instance_id": int(instance_id),
            "gt_cell_type": gt_cell_type,
            "in_valid_region": True,
            "centroid_y": centroid_y,
            "centroid_x": centroid_x,
            "global_y": float(global_y),
            "global_x": float(global_x),
            "bbox_y1": int(bbox[0]),
            "bbox_x1": int(bbox[1]),
            "bbox_y2": int(bbox[2]),
            "bbox_x2": int(bbox[3]),
            "ext_bbox_y1": int(ext_bbox[0]),
            "ext_bbox_x1": int(ext_bbox[1]),
            "ext_bbox_y2": int(ext_bbox[2]),
            "ext_bbox_x2": int(ext_bbox[3]),
            "image_median_bbox_diag": float(median_bbox_diag),
            "embed_region_h": int(embed_region_h),
            "embed_region_w": int(embed_region_w),
            "embed_region_n": int(embed_region_h * embed_region_w) if embed_region_h > 0 else -1,
        }
        return obs


    def build_sam_encoder(
        self,
        backbone_type: str,
        sam_checkpoint_path: str,
        device: torch.device,
        n_jobs: int,
    ) -> torch.nn.Module:
        """Load the official SAM ViT, remove its neck, freeze weights, and move it to the device."""

        encoder = retrieve_sam_encoder(model_type=backbone_type, checkpoint=sam_checkpoint_path)
        encoder.eval()
        encoder.requires_grad_(False)
        encoder.to(device)
        print(
            f"[A] loaded neck-stripped SAM encoder ({backbone_type}) on {device}; "
            f"forward threads={int(n_jobs)}, then {SAM_AFTER_THREADS}",
            flush=True,
        )
        return encoder


    def encode_patch_embeddings(self, 
        encoder: torch.nn.Module,
        image_batch: torch.Tensor,
        use_bf16: bool,
        n_jobs: int,
    ) -> np.ndarray:
        """Run SAM ViT without its neck and return CPU float32 embeddings of shape (B, H, W, C)."""

        x_rgb = torch.repeat_interleave(image_batch, 3, dim=1)
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_amp = bool(image_batch.is_cuda)
        with torch.no_grad():
            with sam_vit_forward_threads(n_forward=int(n_jobs)):
                with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                    tokens = encoder(x_rgb, get_hidden_state=[])
                if isinstance(tokens, tuple):
                    tokens = tokens[0]
                tokens = tokens.float().detach().cpu()
        return tokens.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)

    def sample_candidates_per_image(
        self,
        candidates: list[dict[str, Any]],
        image_name: str,
        max_cells_per_image: int,
        sample_seed: int,
    ) -> list[dict[str, Any]]:
        """Sample valid-region cells per image without replacement; call before morphology extraction."""

        n_keep = int(max_cells_per_image)
        n_have = int(len(candidates))
        if n_have <= n_keep:
            return candidates
        seed = (int(sample_seed) + (zlib.crc32(str(image_name).encode()) % 1_000_003)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(n_have, size=n_keep, replace=False))
        print(
            f"[A] sample {image_name}: {n_keep}/{n_have} seed={seed}",
            flush=True,
        )
        return [candidates[int(i)] for i in chosen]

    def summarize_track_quality(
        self,
        h5ad_path: str | None = None,
        obs: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Summarize each track's length, frame span, and gaps, and report overall distributions."""

        per_track = None
        if h5ad_path:
            with h5py.File(h5ad_path, "r") as handle:
                luma = handle["uns"]["luma"] if "uns" in handle and "luma" in handle["uns"] else None
                if luma is not None and "tracking" in luma and "track_summary" in luma["tracking"]:
                    per_track = ad.io.read_elem(luma["tracking"]["track_summary"])
        if per_track is None:
            if obs is None:
                store = AnnDataResultStore(h5ad_path)
                obs = store.read_table_elements(table_keys=("obs",))["tables"]["obs"]
            work = obs.reset_index(drop=True)
            if "track_id" not in work.columns:
                empty = pd.DataFrame(columns=[
                    "track_id", "n_cells", "track_length", "duration_frames", "n_gap_nodes",
                ])
                return {"per_track": empty, "summary": pd.DataFrame()}
            work["track_id"] = pd.to_numeric(work["track_id"], errors="coerce").fillna(-1).astype(np.int64)
            assigned = work.loc[work["track_id"] >= 0]
            rows = []
            for track_id, group in assigned.groupby("track_id", sort=True):
                row = {"track_id": int(track_id), "n_cells": int(len(group))}
                if "track_length" in group.columns and pd.notna(group["track_length"].iloc[0]):
                    row["track_length"] = int(group["track_length"].iloc[0])
                elif "track_order" in group.columns:
                    row["track_length"] = int(
                        pd.to_numeric(group["track_order"], errors="coerce").max()
                        - pd.to_numeric(group["track_order"], errors="coerce").min()
                        + 1
                    )
                else:
                    row["track_length"] = int(len(group))
                if "duration_frames" in group.columns and pd.notna(group["duration_frames"].iloc[0]):
                    row["duration_frames"] = int(group["duration_frames"].iloc[0])
                else:
                    row["duration_frames"] = -1
                if "n_gap_nodes" in group.columns and pd.notna(group["n_gap_nodes"].iloc[0]):
                    row["n_gap_nodes"] = int(group["n_gap_nodes"].iloc[0])
                else:
                    row["n_gap_nodes"] = 0
                rows.append(row)
            per_track = pd.DataFrame(rows)

        per_track = per_track.reset_index(drop=True)
        if not len(per_track):
            return {"per_track": per_track, "summary": pd.DataFrame()}
        n_cells = 0
        n_unassigned = 0
        if obs is not None and "track_id" in obs.columns:
            tid = pd.to_numeric(obs["track_id"], errors="coerce").fillna(-1)
            n_cells = int(len(tid))
            n_unassigned = int((tid < 0).sum())
        length = pd.to_numeric(per_track["track_length"], errors="coerce")
        duration = (
            pd.to_numeric(per_track["duration_frames"], errors="coerce")
            if "duration_frames" in per_track.columns else pd.Series(dtype=float)
        )
        gaps = (
            pd.to_numeric(per_track["n_gap_nodes"], errors="coerce")
            if "n_gap_nodes" in per_track.columns else pd.Series(dtype=float)
        )
        summary = pd.DataFrame([{
            "n_tracks": int(len(per_track)),
            "n_assigned_cells": int(n_cells - n_unassigned) if n_cells else -1,
            "n_unassigned_cells": int(n_unassigned) if n_cells else -1,
            "fraction_assigned": float((n_cells - n_unassigned) / n_cells) if n_cells else np.nan,
            "track_length_mean": float(length.mean()),
            "track_length_median": float(length.median()),
            "track_length_min": float(length.min()),
            "track_length_max": float(length.max()),
            "duration_frames_mean": float(duration.mean()) if len(duration) else np.nan,
            "n_gap_nodes_mean": float(gaps.mean()) if len(gaps) else np.nan,
        }])
        return {"per_track": per_track, "summary": summary}

    @staticmethod
    def patch_name_to_coord(patch_name: str) -> Tuple[int, int]:
        """Parse patch_<row>_<col> into integer grid coordinates."""

        _, row, col = patch_name.split("_")
        return int(row), int(col)

    def build_global_embedding_map_from_patches(self, patch_embeddings: Dict[str, np.ndarray],
                                                stitch_info: Dict[str, Any]) -> np.ndarray:
        """Build one global low-resolution SAM embedding map from per-patch embeddings."""
        patch_names = sorted(patch_embeddings.keys(), key=self.patch_name_to_coord)
        coords = [self.patch_name_to_coord(name) for name in patch_names]
        grid_rows = max(coord[0] for coord in coords) + 1
        grid_cols = max(coord[1] for coord in coords) + 1

        first_embedding = patch_embeddings[patch_names[0]]
        channels, embed_h, embed_w = first_embedding.shape
        patch_h = int(stitch_info.get('patch_h', embed_h))
        patch_w = int(stitch_info.get('patch_w', embed_w))
        margin_h = int(stitch_info.get('margin_h', 0))
        margin_w = int(stitch_info.get('margin_w', 0))

        # Convert pixel-space margins to embedding-space margins with the same split rule.
        emb_hmh = int(round((margin_h // 2) / max(patch_h, 1) * embed_h))
        emb_hcmh = int(round((margin_h - margin_h // 2) / max(patch_h, 1) * embed_h))
        emb_hmw = int(round((margin_w // 2) / max(patch_w, 1) * embed_w))
        emb_hcmw = int(round((margin_w - margin_w // 2) / max(patch_w, 1) * embed_w))
        valid_embed_h = embed_h - emb_hmh - emb_hcmh
        valid_embed_w = embed_w - emb_hmw - emb_hcmw
        actual_embed_h = valid_embed_h * grid_rows + emb_hmh + emb_hcmh
        actual_embed_w = valid_embed_w * grid_cols + emb_hmw + emb_hcmw

        global_sum = np.zeros((actual_embed_h, actual_embed_w, channels), dtype=np.float32)
        global_count = np.zeros((actual_embed_h, actual_embed_w), dtype=np.float32)

        # Place only the valid, de-margined region of each patch into the global map.
        for patch_name in tqdm(patch_names, desc="Building global SAM map"):
            row, col = self.patch_name_to_coord(patch_name)
            patch_embedding = patch_embeddings[patch_name].astype(np.float32).transpose(1, 2, 0)
            source_h = slice(emb_hmh, embed_h - emb_hcmh if emb_hcmh > 0 else None)
            source_w = slice(emb_hmw, embed_w - emb_hcmw if emb_hcmw > 0 else None)
            target_h = slice(row * valid_embed_h + emb_hmh, (row + 1) * valid_embed_h + emb_hmh)
            target_w = slice(col * valid_embed_w + emb_hmw, (col + 1) * valid_embed_w + emb_hmw)
            global_sum[target_h, target_w] += patch_embedding[source_h, source_w]
            global_count[target_h, target_w] += 1

        return global_sum / np.maximum(global_count[..., None], 1.0)

    def build_integral_embedding_map(self, global_embedding_map: np.ndarray) -> np.ndarray:
        """Build a padded 2D integral image for fast rectangular embedding pooling."""
        integral = global_embedding_map.cumsum(axis=0).cumsum(axis=1)
        return np.pad(integral, ((1, 0), (1, 0), (0, 0)), mode='constant')

    def pool_embeddings_by_integral_bbox(self, integral_embedding_map: np.ndarray,
                                         bbox_list: np.ndarray,
                                         image_shape: Tuple[int, int],
                                         embedding_shape: Tuple[int, int]) -> np.ndarray:
        """Pool one mean embedding for each image-space bbox using the integral map."""
        pooled_embeddings = []
        for bbox in tqdm(bbox_list, desc="Pooling SAM embeddings"):
            gy1, gx1, gy2, gx2 = self.map_bbox_to_embedding_bbox(bbox, image_shape, embedding_shape)
            region_sum = (
                integral_embedding_map[gy2, gx2]
                - integral_embedding_map[gy1, gx2]
                - integral_embedding_map[gy2, gx1]
                + integral_embedding_map[gy1, gx1]
            )
            region_area = max((gy2 - gy1) * (gx2 - gx1), 1)
            pooled_embeddings.append(region_sum / region_area)
        return np.stack(pooled_embeddings, axis=0).astype(np.float32)


def sam_input_nchw(image_nchw: np.ndarray) -> np.ndarray:
    """Morphology images may use other sizes; the copy sent to SAM is always 1024x1024."""

    image = np.asarray(image_nchw, dtype=np.float32)
    if image.ndim == 2:
        image = image[None, ...]
    if tuple(image.shape[-2:]) == SAM_INPUT_SIZE:
        return np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).unsqueeze(0)
    tensor = F.interpolate(tensor, size=SAM_INPUT_SIZE, mode="bilinear", align_corners=False)
    return tensor.squeeze(0).contiguous().numpy().astype(np.float32)


def resize_plane(image_hw: np.ndarray, size: tuple[int, int], nearest: bool) -> np.ndarray:
    """Resize an HxW image: nearest-neighbor interpolation and integer rounding for instance maps, bilinear for grayscale."""

    values = np.asarray(image_hw)
    tensor = torch.from_numpy(values.astype(np.float32)).view(1, 1, *values.shape[-2:])
    if nearest:
        tensor = F.interpolate(tensor, size=size, mode="nearest")
        return np.rint(tensor[0, 0].numpy()).astype(np.int64)
    tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return tensor[0, 0].contiguous().numpy().astype(np.float32)


def scale_candidate_geometry(
    candidates: list[dict[str, Any]],
    scale_y: float,
    scale_x: float,
    morph_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """Rescale stitched-image centroids and extended boxes to morphology-image coordinates."""

    height, width = morph_shape
    scaled = []
    for cell in candidates:
        box = np.rint(
            np.asarray(cell["ext_bbox"], dtype=np.float64)
            * np.array([scale_y, scale_x, scale_y, scale_x])
        ).astype(np.int64)
        y1 = int(np.clip(box[0], 0, height - 1))
        x1 = int(np.clip(box[1], 0, width - 1))
        y2 = int(np.clip(box[2], y1 + 1, height))
        x2 = int(np.clip(box[3], x1 + 1, width))
        scaled.append({
            "instance_id": cell["instance_id"],
            "centroid_y": float(cell["centroid_y"]) * scale_y,
            "centroid_x": float(cell["centroid_x"]) * scale_x,
            "ext_bbox": np.array([y1, x1, y2, x2], dtype=np.int64),
        })
    return scaled
