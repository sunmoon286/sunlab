import colorsys
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad
import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd
import phate
import torch
import umap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde, spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

from scripts_formal.utils.elements_feature_analyzer import (
    BottleneckMLP,
    FeaturePrep,
    INPUT_EMB,
    INPUT_MORPH,
    resolve_torch_device,
)
from scripts_formal.utils.elements_result_io import AnnDataResultStore, FeatureReader, FeatureWriter

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _require_path(value: str | None, name: str) -> str:
    """Reject unset paths instead of falling back to a historical dataset location."""

    if not value:
        raise ValueError(
            f"{name} is required; pass an explicit path to the LUMA result H5AD or output directory."
        )
    return str(value)


def main(
    adata_path: str | None = None,
    output_dir: str | None = None,
    export_mode: str = 'time_series',
    segmentation_panel_long_side: int = 1280,
    segmentation_export_visual: bool = True,
    segmentation_export_raw: bool = False,
) -> dict:
    """Run the selected LUMA visualization workflow."""
    # Both the input result file and the output directory are caller-supplied.
    adata_path = _require_path(adata_path, 'adata_path')
    output_dir = _require_path(output_dir, 'output_dir')

    # Select the configured visualization workflow from the main function defaults.
    normalized_mode = export_mode.lower().strip()
    if normalized_mode in {'time_series', 'timeseries', 'tracking'}:
        visualizer = CellTimeSeriesVisualizer()
        return visualizer.export_visuals(
            adata_path=adata_path,
            output_dir=output_dir,
        )

    # Export per-image segmentation diagnostic panels without reading the full AnnData object.
    if normalized_mode in {'segmentation', 'seg'}:
        visualizer = CellSegmentationVisualizer()
        return visualizer.export_segmentation_panels(
            adata_path=adata_path,
            output_dir=output_dir,
            panel_long_side=segmentation_panel_long_side,
            export_visual=segmentation_export_visual,
            export_raw=segmentation_export_raw,
        )

    raise ValueError(f'Unsupported export_mode: {export_mode}')

class CellTimeSeriesVisualizer:
    """
    Export raw-image, segmentation and tracking visualizations from LUMA AnnData.
    """

    def export_visuals(
        self,
        adata_path: str | None = None,
        output_dir: str | None = None,
        raw_channel_indices: tuple[int, ...] = (0,),
        raw_channel_axis: int | None = None,
        draw_instance_contours: bool = True,
        instance_contour_width: int | None = None,
        draw_tracks: bool = True,
        track_tail_frames: int = 10,
        tail_alpha_min: float = 0.15,
        fps: int = 6,
        video_name: str = 'time_series_tracks.mp4',
        track_graph_name: str = 'track_graph.png',
        save_frame_pngs: bool = False,
    ) -> dict:
        """Export the fixed time-series video and whole-track graph."""
        # Load only table-level data and prepare output paths.
        adata_path = _require_path(adata_path, 'adata_path')
        output_dir = _require_path(output_dir, 'output_dir')
        tables = self.load_visualization_tables(adata_path)
        os.makedirs(output_dir, exist_ok=True)

        # Export a transparent track graph first, then the single-overlay video.
        track_graph_path = self.export_track_graph(
            adata_path=adata_path,
            output_dir=output_dir,
            tables=tables,
            track_graph_name=track_graph_name,
        )
        video_path, frame_dir = self.export_video(
            adata_path=adata_path,
            output_dir=output_dir,
            tables=tables,
            raw_channel_indices=raw_channel_indices,
            raw_channel_axis=raw_channel_axis,
            draw_instance_contours=draw_instance_contours,
            instance_contour_width=instance_contour_width,
            draw_tracks=draw_tracks,
            track_tail_frames=track_tail_frames,
            tail_alpha_min=tail_alpha_min,
            fps=fps,
            video_name=video_name,
            save_frame_pngs=save_frame_pngs,
        )

        return {
            'video_path': video_path,
            'track_graph_path': track_graph_path,
            'frame_dir': frame_dir,
            'n_frames': int(tables['frame_table'].shape[0]),
            'n_tracks': int(tables['track_summary'].shape[0]),
            'n_gap_nodes': int(tables['gap_nodes'].shape[0]),
        }

    def export_video(
        self,
        adata_path: str,
        output_dir: str,
        tables: Dict[str, pd.DataFrame],
        raw_channel_indices: tuple[int, ...],
        raw_channel_axis: int | None,
        draw_instance_contours: bool,
        instance_contour_width: int | None,
        draw_tracks: bool,
        track_tail_frames: int,
        tail_alpha_min: float,
        fps: int,
        video_name: str,
        save_frame_pngs: bool,
    ) -> Tuple[str, str]:
        """Render every frame as a single raw-image overlay and write one MP4 video."""
        video_path = os.path.join(output_dir, video_name)
        frame_dir = os.path.join(output_dir, 'frames') if save_frame_pngs else ''
        if save_frame_pngs:
            os.makedirs(frame_dir, exist_ok=True)

        # Stream image arrays from H5AD and initialize the video writer from the first frame.
        writer = None
        with h5py.File(adata_path, 'r') as h5_file:
            for frame_index, frame_row in tables['frame_table'].reset_index(drop=True).iterrows():
                image_id = str(frame_row['image_id'])
                current_t = int(frame_row['timepoint_index'])
                raw_image, instance_map = self.read_frame_arrays(h5_file, image_id)
                frame_rgb = self.render_frame(
                    raw_image=raw_image,
                    instance_map=instance_map,
                    image_id=image_id,
                    current_t=current_t,
                    tables=tables,
                    raw_channel_indices=raw_channel_indices,
                    raw_channel_axis=raw_channel_axis,
                    draw_instance_contours=draw_instance_contours,
                    instance_contour_width=instance_contour_width,
                    draw_tracks=draw_tracks,
                    track_tail_frames=track_tail_frames,
                    tail_alpha_min=tail_alpha_min,
                )

                if writer is None:
                    frame_height, frame_width = frame_rgb.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                if save_frame_pngs:
                    frame_path = os.path.join(frame_dir, f'frame_{frame_index:04d}.png')
                    cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if writer is not None:
            writer.release()
        return video_path, frame_dir

    def export_track_graph(
        self,
        adata_path: str,
        output_dir: str,
        tables: Dict[str, pd.DataFrame],
        track_graph_name: str,
    ) -> str:
        """Export one transparent whole-track graph at the original image pixel size."""
        first_image_id = str(tables['frame_table'].iloc[0]['image_id'])
        with h5py.File(adata_path, 'r') as h5_file:
            instance_map = ad.io.read_elem(h5_file['uns']['luma']['images'][first_image_id]['prediction']['pred_instance_map'])
        image_height, image_width = int(instance_map.shape[-2]), int(instance_map.shape[-1])
        track_graph_path = os.path.join(output_dir, track_graph_name)
        dpi = 100

        # Build a transparent canvas whose saved pixel dimensions match the raw image.
        fig = plt.figure(figsize=(image_width / dpi, image_height / dpi), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.axis('off')

        # Draw optimized tracks and their start/end markers without any image background.
        node_table = tables['node_table']
        for track_id, track_nodes in node_table.groupby('track_id', sort=True):
            if int(track_id) < 0:
                continue
            ordered_nodes = track_nodes.sort_values(['track_order', 't'])
            color = self.get_track_color(int(track_id))
            ax.plot(ordered_nodes['x'], ordered_nodes['y'], color=color, linewidth=1.5, alpha=0.85)
            ax.scatter(ordered_nodes['x'].iloc[0], ordered_nodes['y'].iloc[0], color=[color], s=12, marker='o', linewidths=0)
            ax.scatter(ordered_nodes['x'].iloc[-1], ordered_nodes['y'].iloc[-1], color=[color], s=16, marker='s', linewidths=0)

        # Draw lineage edges and gap nodes as graph nodes on the same transparent canvas.
        for _, lineage_edge in tables['lineage_edges'].iterrows():
            parent_nodes = node_table[node_table['track_id'] == int(lineage_edge['parent_track_id'])].sort_values('track_order')
            child_nodes = node_table[node_table['track_id'] == int(lineage_edge['child_track_id'])].sort_values('track_order')
            if parent_nodes.empty or child_nodes.empty:
                continue
            ax.plot(
                [parent_nodes['x'].iloc[-1], child_nodes['x'].iloc[0]],
                [parent_nodes['y'].iloc[-1], child_nodes['y'].iloc[0]],
                color=self.get_track_color(int(lineage_edge['child_track_id'])),
                linewidth=1.0,
                linestyle='--',
                alpha=0.55,
            )

        gap_nodes = node_table[node_table['is_gap']]
        if not gap_nodes.empty:
            gap_colors = [self.get_track_color(int(track_id)) for track_id in gap_nodes['track_id']]
            ax.scatter(gap_nodes['x'], gap_nodes['y'], color=gap_colors, s=12, marker='o', linewidths=0, alpha=0.55)

        fig.savefig(track_graph_path, transparent=True, dpi=dpi, pad_inches=0)
        plt.close(fig)
        return track_graph_path

    def load_visualization_tables(self, adata_path: str) -> Dict[str, pd.DataFrame]:
        """Load obs and /uns/luma/tracking tables needed by the visualizer."""
        table_data = AnnDataResultStore(adata_path).read_table_elements(
            table_keys=('obs',),
            luma_keys=('tracking',),
        )
        obs = table_data['tables']['obs'].copy()
        obs['obs_name'] = obs.index.astype(str)
        tracking = table_data['luma']['tracking']
        track_summary = tracking['track_summary'].copy()
        obs_graph = tracking['obs_graph'].copy()
        lineage_edges = tracking['lineage_edges'].copy()
        gap_nodes = tracking['gap_nodes'].copy()

        # Build frame and node tables once so rendering functions can stay small.
        frame_table = obs[['image_id', 'timepoint_index']].drop_duplicates().sort_values(['timepoint_index', 'image_id'])
        node_table = self.build_node_table(obs, gap_nodes)
        return {
            'obs': obs,
            'track_summary': track_summary,
            'obs_graph': obs_graph,
            'lineage_edges': lineage_edges,
            'gap_nodes': gap_nodes,
            'frame_table': frame_table.reset_index(drop=True),
            'node_table': node_table.reset_index(drop=True),
        }

    def render_frame(
        self,
        raw_image: np.ndarray,
        instance_map: np.ndarray,
        image_id: str,
        current_t: int,
        tables: Dict[str, pd.DataFrame],
        raw_channel_indices: tuple[int, ...],
        raw_channel_axis: int | None,
        draw_instance_contours: bool,
        instance_contour_width: int | None,
        draw_tracks: bool,
        track_tail_frames: int,
        tail_alpha_min: float,
    ) -> np.ndarray:
        """Render one video frame as raw image plus segmentation and tracking overlays."""
        display_image = self.compose_raw_image(raw_image, raw_channel_indices, raw_channel_axis)
        image_height, image_width = instance_map.shape[-2], instance_map.shape[-1]
        dpi = 100

        # Draw the raw image on a full-frame axis with pixel coordinates.
        fig = plt.figure(figsize=(image_width / dpi, image_height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(display_image, cmap='gray' if display_image.ndim == 2 else None, vmin=0, vmax=1)
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)
        ax.axis('off')

        # Overlay instance contours, track tails, real nodes, and gap nodes in that order.
        current_obs = tables['obs'][tables['obs']['image_id'].astype(str) == image_id]
        if draw_instance_contours:
            self.draw_instance_contours(ax, instance_map, current_obs, instance_contour_width)
        if draw_tracks:
            self.draw_track_tail(ax, tables['node_table'], current_t, track_tail_frames, tail_alpha_min)
            self.draw_current_nodes(ax, current_obs)
            self.draw_gap_nodes(ax, tables['gap_nodes'], current_t)

        fig.canvas.draw()
        frame_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return frame_rgb

    def draw_instance_contours(
        self,
        ax: Any,
        instance_map: np.ndarray,
        current_obs: pd.DataFrame,
        instance_contour_width: int | None,
    ) -> None:
        """Draw per-instance contours rather than filled instance masks."""
        instance_map = np.asarray(instance_map)
        image_height, image_width = instance_map.shape[-2], instance_map.shape[-1]
        contour_width = instance_contour_width if instance_contour_width is not None else max(1, round(max(image_height, image_width) / 400))
        overlay = np.zeros((image_height, image_width, 4), dtype=np.uint8)
        instance_to_track = dict(zip(current_obs['instance_id'].astype(int), current_obs['track_id'].astype(int)))

        # Use cv2 contours so the requested contour width is interpreted in image pixels.
        instance_ids = np.unique(instance_map)
        for instance_id in instance_ids[instance_ids != 0]:
            instance_mask = (instance_map == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            track_id = int(instance_to_track.get(int(instance_id), -1))
            if track_id >= 0:
                color = self.get_track_color(track_id)
            else:
                hue = (int(instance_id) * 0.7548776662466927) % 1.0
                color = colorsys.hsv_to_rgb(hue, 0.55, 0.9)
            rgba_color = tuple(int(channel * 255) for channel in color) + (230,)
            cv2.drawContours(overlay, contours, -1, rgba_color, thickness=int(contour_width))

        ax.imshow(overlay.astype(np.float32) / 255.0)

    def draw_track_tail(
        self,
        ax: Any,
        node_table: pd.DataFrame,
        current_t: int,
        track_tail_frames: int,
        tail_alpha_min: float,
    ) -> None:
        """Draw recent track segments ending at the current frame."""
        start_t = current_t - track_tail_frames + 1
        tail_nodes = node_table[(node_table['track_id'] >= 0) & (node_table['t'] >= start_t) & (node_table['t'] <= current_t)]

        for track_id, track_nodes in tail_nodes.groupby('track_id', sort=True):
            ordered_nodes = track_nodes.sort_values(['track_order', 't'])
            color = self.get_track_color(int(track_id))
            for node_index in range(max(ordered_nodes.shape[0] - 1, 0)):
                source_node = ordered_nodes.iloc[node_index]
                target_node = ordered_nodes.iloc[node_index + 1]
                segment_age = current_t - int(target_node['t'])
                alpha = max(tail_alpha_min, 1.0 - segment_age / max(track_tail_frames, 1))
                ax.plot(
                    [source_node['x'], target_node['x']],
                    [source_node['y'], target_node['y']],
                    color=color,
                    linewidth=1.5,
                    alpha=alpha,
                )

    def draw_current_nodes(self, ax: Any, current_obs: pd.DataFrame) -> None:
        """Draw real observations in the current frame as colored track nodes."""
        if current_obs.empty:
            return
        node_colors = [self.get_track_color(int(track_id)) if int(track_id) >= 0 else (0.75, 0.75, 0.75) for track_id in current_obs['track_id']]
        ax.scatter(
            current_obs['centroid_x'],
            current_obs['centroid_y'],
            color=node_colors,
            s=18,
            marker='o',
            linewidths=0.6,
            edgecolors='white',
            alpha=0.95,
        )

    def draw_gap_nodes(self, ax: Any, gap_nodes: pd.DataFrame, current_t: int) -> None:
        """Draw interpolated gap nodes at the current frame without contours."""
        current_gap_nodes = gap_nodes[gap_nodes['t'].astype(int) == int(current_t)]
        if current_gap_nodes.empty:
            return
        node_colors = [self.get_track_color(int(track_id)) for track_id in current_gap_nodes['track_id']]
        ax.scatter(
            current_gap_nodes['x'],
            current_gap_nodes['y'],
            color=node_colors,
            s=18,
            marker='o',
            linewidths=0.6,
            edgecolors='white',
            alpha=0.55,
        )

    def compose_raw_image(
        self,
        raw_image: np.ndarray,
        raw_channel_indices: tuple[int, ...],
        raw_channel_axis: int | None,
    ) -> np.ndarray:
        """Select display channels from a raw image and normalize them to 0-1."""
        raw_array = np.asarray(raw_image)
        raw_array = np.squeeze(raw_array)
        if raw_array.ndim == 2:
            return self.normalize_image(raw_array)

        if raw_channel_axis is not None:
            raw_array = np.moveaxis(raw_array, raw_channel_axis, -1)
        elif raw_array.shape[0] in {1, 3, 4}:
            raw_array = np.moveaxis(raw_array, 0, -1)

        if len(raw_channel_indices) == 1:
            return self.normalize_image(raw_array[..., raw_channel_indices[0]])

        channels = [self.normalize_image(raw_array[..., channel_index]) for channel_index in raw_channel_indices[:3]]
        return np.stack(channels, axis=-1)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Percentile-normalize one image channel for display only."""
        image = image.astype(np.float32)
        lower, upper = np.percentile(image, [1, 99])
        return np.clip((image - lower) / max(upper - lower, 1.0e-6), 0.0, 1.0)

    def get_track_color(self, track_id: int) -> Tuple[float, float, float]:
        """Map a track ID to a stable RGB color."""
        if track_id < 0:
            return 0.75, 0.75, 0.75
        hue = (int(track_id) * 0.618033988749895) % 1.0
        return colorsys.hsv_to_rgb(hue, 0.72, 0.95)

    def build_node_table(self, obs: pd.DataFrame, gap_nodes: pd.DataFrame) -> pd.DataFrame:
        """Combine real observation nodes and gap nodes into one plotting table."""
        real_nodes = pd.DataFrame({
            'track_id': obs['track_id'].astype(int),
            'track_order': obs['track_order'].astype(int),
            't': obs['timepoint_index'].astype(int),
            'x': obs['centroid_x'].astype(float),
            'y': obs['centroid_y'].astype(float),
            'is_gap': np.zeros(obs.shape[0], dtype=bool),
            'obs_name': obs['obs_name'].astype(str),
        })
        real_nodes = real_nodes[real_nodes['track_id'] >= 0]

        gap_table = gap_nodes.copy()
        if gap_table.empty:
            return real_nodes.reset_index(drop=True)
        gap_table = pd.DataFrame({
            'track_id': gap_table['track_id'].astype(int),
            'track_order': gap_table['track_order'].astype(float),
            't': gap_table['t'].astype(int),
            'x': gap_table['x'].astype(float),
            'y': gap_table['y'].astype(float),
            'is_gap': np.ones(gap_table.shape[0], dtype=bool),
            'obs_name': np.full(gap_table.shape[0], '', dtype=object),
        })
        return pd.concat([real_nodes, gap_table], axis=0, ignore_index=True)

    def read_frame_arrays(self, h5_file: h5py.File, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read one raw image and prediction instance map from /uns/luma/images."""
        image_group = h5_file['uns']['luma']['images'][image_id]
        raw_image = ad.io.read_elem(image_group['gt']['gt_raw_image'])
        instance_map = ad.io.read_elem(image_group['prediction']['pred_instance_map'])
        return raw_image, instance_map


class CellSegmentationVisualizer(CellTimeSeriesVisualizer):
    """
    Export per-image segmentation diagnostic panels from LUMA H5AD image records.
    """

    def export_segmentation_panels(
        self,
        adata_path: str | None = None,
        output_dir: str | None = None,
        image_ids: tuple[str, ...] | None = None,
        raw_channel_indices: tuple[int, ...] = (0,),
        raw_channel_axis: int | None = None,
        panel_long_side: int = 1280,
        label_color: Tuple[int, int, int] = (255, 255, 255),
        export_visual: bool = True,
        export_raw: bool = False,
    ) -> dict:
        """Export segmentation visual panels and optional uint16 raw map images per image record."""
        # Prepare one output subdirectory for each image ID while streaming image records one at a time.
        adata_path = _require_path(adata_path, 'adata_path')
        output_dir = _require_path(output_dir, 'output_dir')
        os.makedirs(output_dir, exist_ok=True)
        panel_paths = []
        raw_image_paths = {}
        image_dirs = {}
        with h5py.File(adata_path, 'r') as h5_file:
            image_group = h5_file['uns']['luma']['images']
            selected_image_ids = (
                sorted(key for key in image_group.keys() if not AnnDataResultStore.is_staging_key(key))
                if image_ids is None else list(image_ids)
            )
            for image_id in selected_image_ids:
                image_output_dir = os.path.join(output_dir, self.safe_filename(str(image_id)))
                os.makedirs(image_output_dir, exist_ok=True)
                image_dirs[str(image_id)] = image_output_dir
                if not export_visual and not export_raw:
                    continue

                image_arrays = self.read_segmentation_arrays(h5_file, str(image_id))
                if export_visual:
                    panel_image = self.render_segmentation_panel(
                        image_arrays=image_arrays,
                        raw_channel_indices=raw_channel_indices,
                        raw_channel_axis=raw_channel_axis,
                        panel_long_side=panel_long_side,
                        label_color=label_color,
                    )
                    panel_path = os.path.join(image_output_dir, 'segmentation_panel.png')
                    cv2.imwrite(panel_path, cv2.cvtColor(panel_image, cv2.COLOR_RGB2BGR))
                    panel_paths.append(panel_path)
                if export_raw:
                    raw_image_paths[str(image_id)] = self.export_raw_segmentation_images(image_arrays, image_output_dir)

        return {
            'output_dir': output_dir,
            'image_dirs': image_dirs,
            'panel_paths': panel_paths,
            'raw_image_paths': raw_image_paths,
            'n_images': len(image_dirs),
            'n_visual_panels': len(panel_paths),
            'n_image_dirs': len(image_dirs),
            'n_raw_images': sum(len(paths) for paths in raw_image_paths.values()),
        }

    def read_segmentation_arrays(self, h5_file: h5py.File, image_id: str) -> Dict[str, np.ndarray]:
        """Read the raw, GT, prediction and stitched map arrays for one image record."""
        image_group = h5_file['uns']['luma']['images'][image_id]
        gt_raw_image = ad.io.read_elem(image_group['gt']['gt_raw_image'])
        pred_mask = ad.io.read_elem(image_group['prediction']['mask'])
        pred_h_map = ad.io.read_elem(image_group['prediction']['h_map'])
        pred_v_map = ad.io.read_elem(image_group['prediction']['v_map'])
        return {
            'raw_image': gt_raw_image,
            'gt_raw_image': gt_raw_image,
            'gt_instance_map': ad.io.read_elem(image_group['gt']['gt_instance_map']),
            'pred_instance_map': ad.io.read_elem(image_group['prediction']['pred_instance_map']),
            'pred_mask': pred_mask,
            'mask': pred_mask,
            'pred_h_map': pred_h_map,
            'h_map': pred_h_map,
            'pred_v_map': pred_v_map,
            'v_map': pred_v_map,
            'pred_type_cyto_map': ad.io.read_elem(image_group['prediction']['pred_type_cyto_map']),
            'gt_type_cyto_map': ad.io.read_elem(image_group['gt']['gt_type_cyto_map']),
        }

    def render_segmentation_panel(
        self,
        image_arrays: Dict[str, np.ndarray],
        raw_channel_indices: tuple[int, ...],
        raw_channel_axis: int | None,
        panel_long_side: int,
        label_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Render one 2-row by 3-column segmentation summary image."""
        # Prepare the raw display image and overlay instance contours on separate copies.
        raw_display = self.compose_raw_image(image_arrays['raw_image'], raw_channel_indices, raw_channel_axis)
        raw_rgb = self.to_rgb_uint8(raw_display)
        gt_overlay = self.render_contour_overlay(raw_rgb, image_arrays['gt_instance_map'], contour_color=(80, 255, 140))
        pred_overlay = self.render_contour_overlay(raw_rgb, image_arrays['pred_instance_map'], contour_color=(255, 190, 70))

        # Prepare scalar prediction maps for compact visual inspection.
        panels = [
            self.resize_and_label(raw_rgb, 'Stitched raw image', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
            self.resize_and_label(gt_overlay, 'Ground truth instances', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
            self.resize_and_label(pred_overlay, 'Predicted instances', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
            self.resize_and_label(self.render_scalar_map(image_arrays['pred_mask'], signed=False), 'Predicted mask', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
            self.resize_and_label(self.render_scalar_map(image_arrays['pred_h_map'], signed=True), 'Predicted hmap', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
            self.resize_and_label(self.render_scalar_map(image_arrays['pred_v_map'], signed=True), 'Predicted vmap', panel_long_side, label_color, interpolation=cv2.INTER_AREA),
        ]

        # Combine the six fixed-size panels into the requested 2x3 diagnostic layout.
        top_row = np.concatenate(panels[:3], axis=1)
        bottom_row = np.concatenate(panels[3:], axis=1)
        return np.concatenate([top_row, bottom_row], axis=0)

    def render_contour_overlay(
        self,
        raw_rgb: np.ndarray,
        instance_map: np.ndarray,
        contour_color: Tuple[int, int, int],
        contour_width: int | None = None,
    ) -> np.ndarray:
        """Overlay instance contours on a raw RGB image."""
        overlay = raw_rgb.copy()
        instance_array = np.squeeze(np.asarray(instance_map))
        image_height, image_width = instance_array.shape[-2], instance_array.shape[-1]
        width = contour_width if contour_width is not None else max(1, round(max(image_height, image_width) / 400))

        # Draw every non-background instance ID as a contour in image-pixel coordinates.
        instance_ids = np.unique(instance_array)
        for instance_id in instance_ids[instance_ids != 0]:
            instance_mask = (instance_array == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, contour_color, thickness=int(width))
        return overlay

    def export_raw_segmentation_images(self, image_arrays: Dict[str, np.ndarray], image_output_dir: str) -> Dict[str, str]:
        """Write selected segmentation arrays as display-normalized uint16 TIFF images."""
        # Convert continuous maps and label maps with rules suited to visual inspection.
        raw_specs = {
            'mask': ('mask', 'probability'),
            'h_map': ('h_map', 'signed'),
            'v_map': ('v_map', 'signed'),
            'pred_type_cyto_map': ('pred_type_cyto_map', 'label'),
            'gt_type_cyto_map': ('gt_type_cyto_map', 'label'),
            'gt_raw_image': ('gt_raw_image', 'robust'),
            'pred_instance_map': ('pred_instance_map', 'label'),
            'gt_instance_map': ('gt_instance_map', 'label'),
        }
        image_paths = {}
        for output_name, (array_key, conversion_mode) in raw_specs.items():
            uint16_image = self.convert_array_to_uint16_image(image_arrays[array_key], conversion_mode)
            image_path = os.path.join(image_output_dir, f'{output_name}.tif')
            self.write_uint16_image(image_path, uint16_image)
            image_paths[output_name] = image_path
        return image_paths

    def convert_array_to_uint16_image(self, image_array: np.ndarray, conversion_mode: str) -> np.ndarray:
        """Convert one stored array into a viewable uint16 image according to its value semantics."""
        array = np.squeeze(np.asarray(image_array))
        if conversion_mode == 'probability':
            return self.scale_unit_array_to_uint16(array)
        if conversion_mode == 'signed':
            return self.scale_signed_array_to_uint16(array)
        if conversion_mode == 'label':
            return self.scale_label_array_to_uint16(array)
        if conversion_mode == 'robust':
            return self.scale_robust_array_to_uint16(array)
        raise ValueError(f'Unsupported raw image conversion mode: {conversion_mode}')

    def scale_unit_array_to_uint16(self, image_array: np.ndarray) -> np.ndarray:
        """Scale a probability-like array from 0-1 into the full uint16 range."""
        clipped_array = np.clip(np.asarray(image_array, dtype=np.float32), 0.0, 1.0)
        return np.round(clipped_array * 65535.0).astype(np.uint16)

    def scale_signed_array_to_uint16(self, image_array: np.ndarray) -> np.ndarray:
        """Scale a signed map symmetrically around zero into uint16 for display."""
        signed_array = np.asarray(image_array, dtype=np.float32)
        finite_values = signed_array[np.isfinite(signed_array)]
        scale = max(float(np.percentile(np.abs(finite_values), 99)), 1.0e-6)
        normalized_array = np.clip((signed_array + scale) / (2.0 * scale), 0.0, 1.0)
        return np.round(normalized_array * 65535.0).astype(np.uint16)

    def scale_label_array_to_uint16(self, image_array: np.ndarray) -> np.ndarray:
        """Scale integer label IDs into uint16 display range without preserving original label IDs."""
        label_array = self.prepare_label_array(image_array)
        max_label = float(np.max(label_array))
        if max_label <= 0:
            return np.zeros(label_array.shape, dtype=np.uint16)
        normalized_array = np.clip(label_array.astype(np.float32) / max_label, 0.0, 1.0)
        return np.round(normalized_array * 65535.0).astype(np.uint16)

    def scale_robust_array_to_uint16(self, image_array: np.ndarray) -> np.ndarray:
        """Percentile-scale a raw image array into uint16 for convenient viewing."""
        display_array = self.prepare_display_image_array(image_array)
        if display_array.ndim == 3:
            channels = [self.scale_robust_channel_to_uint16(display_array[..., channel_index]) for channel_index in range(display_array.shape[-1])]
            return np.stack(channels, axis=-1)
        return self.scale_robust_channel_to_uint16(display_array)

    def scale_robust_channel_to_uint16(self, image_channel: np.ndarray) -> np.ndarray:
        """Scale one numeric image channel using 1st and 99th percentiles."""
        channel = np.asarray(image_channel, dtype=np.float32)
        finite_values = channel[np.isfinite(channel)]
        lower, upper = np.percentile(finite_values, [1, 99])
        if upper <= lower:
            return np.zeros(channel.shape, dtype=np.uint16)
        normalized_channel = np.clip((channel - lower) / (upper - lower), 0.0, 1.0)
        return np.round(normalized_channel * 65535.0).astype(np.uint16)

    def prepare_label_array(self, image_array: np.ndarray) -> np.ndarray:
        """Convert 2D label maps or one-hot class maps into a 2D label array."""
        label_array = np.squeeze(np.asarray(image_array))
        if label_array.ndim == 3 and label_array.shape[0] <= 16:
            label_array = np.argmax(label_array, axis=0)
        elif label_array.ndim == 3 and label_array.shape[-1] <= 16:
            label_array = np.argmax(label_array, axis=-1)
        return label_array.astype(np.uint32)

    def prepare_display_image_array(self, image_array: np.ndarray) -> np.ndarray:
        """Put grayscale or channel-first raw images into a cv2-writable display layout."""
        display_array = np.squeeze(np.asarray(image_array))
        if display_array.ndim == 3 and display_array.shape[0] in {1, 3, 4}:
            display_array = np.moveaxis(display_array, 0, -1)
        if display_array.ndim == 3 and display_array.shape[-1] > 4:
            display_array = display_array[..., 0]
        return display_array

    def write_uint16_image(self, image_path: str, image_array: np.ndarray) -> None:
        """Write one uint16 grayscale or RGB image with OpenCV channel ordering."""
        output_array = np.asarray(image_array)
        if output_array.ndim == 3 and output_array.shape[-1] == 3:
            output_array = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
        elif output_array.ndim == 3 and output_array.shape[-1] == 4:
            output_array = cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(image_path, output_array)

    def render_scalar_map(self, map_array: np.ndarray, signed: bool) -> np.ndarray:
        """Convert one prediction scalar map into an RGB visualization panel."""
        scalar_array = np.squeeze(np.asarray(map_array)).astype(np.float32)
        if signed:
            scale = max(float(np.nanpercentile(np.abs(scalar_array), 99)), 1.0e-6)
            normalized = np.clip((scalar_array + scale) / (2.0 * scale), 0.0, 1.0)
            color_panel = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            return cv2.cvtColor(color_panel, cv2.COLOR_BGR2RGB)

        normalized = np.clip(scalar_array, 0.0, 1.0)
        gray_panel = (normalized * 255).astype(np.uint8)
        return np.repeat(gray_panel[..., None], 3, axis=-1)

    def to_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert a normalized grayscale or RGB display image to uint8 RGB."""
        image_array = np.asarray(image)
        if image_array.ndim == 2:
            image_array = np.repeat(image_array[..., None], 3, axis=-1)
        return (np.clip(image_array, 0.0, 1.0) * 255).astype(np.uint8)

    def resize_and_label(
        self,
        image: np.ndarray,
        label: str,
        panel_long_side: int,
        label_color: Tuple[int, int, int],
        interpolation: int,
    ) -> np.ndarray:
        """Resize one panel by long side and draw its name at the top-left corner."""
        image_array = np.asarray(image)
        image_height, image_width = image_array.shape[:2]
        resize_scale = float(panel_long_side) / float(max(image_height, image_width))
        resized_size = (max(1, round(image_width * resize_scale)), max(1, round(image_height * resize_scale)))
        resized_image = cv2.resize(image_array, resized_size, interpolation=interpolation)

        # Draw a dark outline under the requested label color so white labels stay readable.
        text_scale = max(0.7, min(1.4, panel_long_side / 900.0))
        text_thickness = max(2, round(panel_long_side / 640))
        text_origin = (max(12, round(panel_long_side * 0.02)), max(24, round(panel_long_side * 0.045)))
        cv2.putText(resized_image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 0), text_thickness + 3, cv2.LINE_AA)
        cv2.putText(resized_image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, text_scale, label_color, text_thickness, cv2.LINE_AA)
        return resized_image

    @staticmethod
    def safe_filename(value: str) -> str:
        """Convert one image ID into a filename-safe stem."""
        safe_text = str(value)
        for bad_char in ['/', '\\', ' ', ':']:
            safe_text = safe_text.replace(bad_char, '_')
        return safe_text


if __name__ == '__main__':
    main()

VIZ_INPUTS = ((INPUT_EMB, "emb"), (INPUT_MORPH, "morph"))
KDE_GRID_SIZE = 120
SCATTER_POINT_SIZE = 6
SCATTER_ALPHA = 0.45


def run_visualization(
    cell_table: dict[str, Any],
    h5ad_path: Path | str,
    run_name: str,
    n_mean: int,
    mlp_batch_size: int,
    column_zscore: bool,
    feature_transforms: list[bool],
    meta_key: str,
    target_kind: str,
    meta_map: dict | None = None,
    type_name_to_index: dict[str, int] | None = None,
    type_index_to_name: dict[int, str] | None = None,
    obs_equals: dict[str, Any] | None = None,
    exclude_obs_in: dict[str, Any] | None = None,
    target_scale_max: float | None = None,
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
) -> dict[str, Any]:
    """Stage E entry: preprocess -> reduce and plot emb* / morph_ext separately -> write coordinates and SVG to uns/runs."""

    configure_matplotlib()
    # Classification / regression n_jobs does not control reduction. Set torch threads to 1; UMAP / PHATE use viz_n_jobs.
    device = resolve_torch_device(device, 1)
    print(f"[E] device={device} viz_n_jobs={int(viz_n_jobs)}", flush=True)
    prep = FeaturePrep(
        cell_table,
        n_mean=n_mean,
        column_zscore=column_zscore,
        feature_transforms=feature_transforms,
        meta_key=meta_key,
        target_kind=target_kind,
        meta_map=meta_map,
        type_name_to_index=type_name_to_index,
        type_index_to_name=type_index_to_name,
        obs_equals=obs_equals,
        exclude_obs_in=exclude_obs_in,
        target_scale_max=target_scale_max,
        lda_label_key=lda_label_key,
        label_unit=label_unit,
    ).prepare()
    spec = prep.spec
    cell_table = prep.cell_table
    obs = cell_table["obs"].reset_index(drop=True)
    train_mask = prep.train_mask
    test_mask = prep.test_mask
    lda_pack = build_lda_label_pack(obs, spec)
    print(
        f"[E] target_kind={spec['target_kind']} n_train={int(train_mask.sum())} "
        f"n_test={int(test_mask.sum())} lda_dim={lda_pack['n_components']}",
        flush=True,
    )

    # Match stage B defaults: apply per-image z-score / row L2 before column standardization; default to column standardization only.
    X_emb, X_morph = prep.apply_feature_transforms()
    X_dict = prep.build_input_matrices(X_emb, X_morph, train_mask)
    feature_names = {
        INPUT_EMB: list(cell_table["embedding_columns"][:n_mean]),
        INPUT_MORPH: list(cell_table["extended_morph_columns"]),
    }
    geometry_rows: list[dict[str, Any]] = []

    for input_name, tag in VIZ_INPUTS:
        print(f"[E] input={input_name}", flush=True)
        have_proj = FeatureReader.has_uns(
            h5ad_path, f"type_prediction/proj64_seed0/{tag}", run_name=run_name,
        )
        coords = fit_reducers_on_input(
            X=X_dict[input_name],
            train_mask=train_mask,
            test_mask=test_mask,
            lda_y=lda_pack["y_code"],
            n_lda_components=lda_pack["n_components"],
            target_kind=spec["target_kind"],
            have_proj=have_proj,
            h5ad_path=h5ad_path,
            tag=tag,
            batch_size=mlp_batch_size,
            device=device,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            phate_knn=phate_knn,
            phate_decay=phate_decay,
            phate_t=phate_t,
            viz_random_state=viz_random_state,
            viz_n_jobs=viz_n_jobs,
            run_name=run_name,
        )
        FeatureWriter.write_run_coords(
            h5ad_path, run_name, tag, coords, obs["obs_name"].to_numpy() if "obs_name" in obs.columns else obs.index.to_numpy(),
        )
        fig, lda_limits, panel_limits = draw_scatter_figure(
            coords=coords,
            test_mask=test_mask,
            display=lda_pack["display"],
            label_order=lda_pack["display_order"],
            colors=lda_pack["colors"],
        )
        FeatureWriter.write_figure(h5ad_path, f"scatter_{tag}", fig, run_name=run_name)
        if panel_limits:
            FeatureWriter.write_uns(h5ad_path, f"visualization/scatter_limits/{tag}", panel_limits, run_name=run_name)
        if spec["target_kind"] == "continuous":
            geometry_rows.append(
                lda_geometry_row(
                    input_name=input_name,
                    z_test=coords["lda"][test_mask],
                    display_test=lda_pack["display"][test_mask],
                    label_order=lda_pack["display_order"],
                    day_numeric=lda_pack["day_numeric"][test_mask],
                    y_time=pd.to_numeric(obs["target"], errors="raise").to_numpy(dtype=np.float64)[test_mask],
                    lda_label_key=str(spec.get("lda_label_key") or ""),
                )
            )
        # For continuous tasks with 2D LDA, save a separate density plot without overlaying scatter points.
        if spec["target_kind"] == "continuous" and lda_pack["n_components"] >= 2:
            fig = draw_lda_density_figure(
                z_test=coords["lda"][test_mask],
                display_test=lda_pack["display"][test_mask],
                label_order=lda_pack["display_order"],
                colors=lda_pack["colors"],
                xlim=lda_limits[0],
                ylim=lda_limits[1],
                kde_n_levels=kde_n_levels,
            )
            FeatureWriter.write_figure(h5ad_path, f"lda_kde_{tag}", fig, run_name=run_name)
        fig = draw_condition_heatmap(
            X=X_dict[input_name],
            obs=obs,
            test_mask=test_mask,
            spec=spec,
            lda_pack=lda_pack,
            feature_names=feature_names[input_name],
            input_name=input_name,
        )
        FeatureWriter.write_figure(h5ad_path, f"heatmap_{tag}", fig, run_name=run_name)
        del coords
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if geometry_rows:
        FeatureWriter.write_uns(h5ad_path, "visualization/lda_geometry", pd.DataFrame(geometry_rows), run_name=run_name)
    print(f"[E] wrote figures and coords into {h5ad_path}", flush=True)
    return cell_table


def configure_matplotlib() -> None:
    """Set fonts and minus-sign rendering for the headless backend. Use only Western fonts in figures."""

    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })


def load_saved_coords(
    h5ad_path: Path,
    tag: str,
    run_name: str,
) -> dict[str, np.ndarray]:
    """Read coordinates from uns/runs/<run_name>/coords/<tag>, aligned with this run's filtered rows."""

    if not FeatureReader.has_uns(h5ad_path, f"coords/{tag}", run_name=run_name):
        return {}
    return FeatureReader.read_run_coords(h5ad_path, run_name, tag)


def redraw_saved_visualization(
    h5ad_path: Path | str,
    run_name: str,
    n_mean: int,
    column_zscore: bool,
    feature_transforms: list[bool],
    meta_key: str,
    target_kind: str,
    mlp_batch_size: int = 256,
    meta_map: dict | None = None,
    type_name_to_index: dict[str, int] | None = None,
    type_index_to_name: dict[int, str] | None = None,
    obs_equals: dict[str, Any] | None = None,
    exclude_obs_in: dict[str, Any] | None = None,
    target_scale_max: float | None = None,
    lda_label_key: str = "",
    label_unit: str = "",
    kde_n_levels: int = 6,
) -> None:
    """Overwrite SVG using saved run coordinates without refitting reducers."""

    configure_matplotlib()
    cell_table = FeatureReader.load_cell_table(h5ad_path)
    prep = FeaturePrep(
        cell_table,
        n_mean=n_mean,
        column_zscore=column_zscore,
        feature_transforms=feature_transforms,
        meta_key=meta_key,
        target_kind=target_kind,
        meta_map=meta_map,
        type_name_to_index=type_name_to_index,
        type_index_to_name=type_index_to_name,
        obs_equals=obs_equals,
        exclude_obs_in=exclude_obs_in,
        target_scale_max=target_scale_max,
        lda_label_key=lda_label_key,
        label_unit=label_unit,
    ).prepare()
    spec = prep.spec
    obs = prep.cell_table["obs"].reset_index(drop=True)
    train_mask = prep.train_mask
    test_mask = prep.test_mask
    lda_pack = build_lda_label_pack(obs, spec)
    X_dict = prep.build_input_matrices()
    feature_names = {
        INPUT_EMB: list(prep.cell_table["embedding_columns"][:n_mean]),
        INPUT_MORPH: list(prep.cell_table["extended_morph_columns"]),
    }
    for input_name, tag in VIZ_INPUTS:
        coords = load_saved_coords(h5ad_path, tag, run_name)
        if not coords:
            continue
        fig, lda_limits, panel_limits = draw_scatter_figure(
            coords=coords,
            test_mask=test_mask,
            display=lda_pack["display"],
            label_order=lda_pack["display_order"],
            colors=lda_pack["colors"],
        )
        FeatureWriter.write_figure(h5ad_path, f"scatter_{tag}", fig, run_name=run_name)
        if panel_limits:
            FeatureWriter.write_uns(
                h5ad_path, f"visualization/scatter_limits/{tag}", panel_limits, run_name=run_name,
            )
        if spec["target_kind"] == "continuous" and lda_pack["n_components"] >= 2:
            fig = draw_lda_density_figure(
                z_test=coords["lda"][test_mask],
                display_test=lda_pack["display"][test_mask],
                label_order=lda_pack["display_order"],
                colors=lda_pack["colors"],
                xlim=lda_limits[0],
                ylim=lda_limits[1],
                kde_n_levels=kde_n_levels,
            )
            FeatureWriter.write_figure(h5ad_path, f"lda_kde_{tag}", fig, run_name=run_name)
        fig = draw_condition_heatmap(
            X=X_dict[input_name],
            obs=obs,
            test_mask=test_mask,
            spec=spec,
            lda_pack=lda_pack,
            feature_names=feature_names[input_name],
            input_name=input_name,
        )
        FeatureWriter.write_figure(h5ad_path, f"heatmap_{tag}", fig, run_name=run_name)
    print(f"[E] redrew figures into {h5ad_path} run={run_name}", flush=True)



def build_lda_label_pack(obs: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Build integer encodings for LDA, display names for plotting, and colors.

    Discrete: use target as classes. Continuous: read the supplied lda_label_key;
    fail if the column is absent rather than falling back to each unique continuous target.
    """

    train_mask = obs["split"].to_numpy() == "train"
    if spec["target_kind"] == "discrete":
        y_code = pd.to_numeric(obs["target"], errors="raise").to_numpy(dtype=np.int64)
        display_order = list(spec["class_names"])
        display = np.asarray([display_order[int(v)] for v in y_code], dtype=object)
        day_numeric = np.full(len(obs), np.nan, dtype=np.float64)
        sequential = False
    else:
        lda_label_key = str(spec.get("lda_label_key") or "")
        if not lda_label_key:
            raise RuntimeError(
                "Continuous-task LDA requires lda_label_key or stratify_eval_key / meta_key; "
                "it does not fall back to each unique target."
            )
        if lda_label_key not in obs.columns:
            raise RuntimeError(
                f"Continuous-task LDA requires obs column {lda_label_key!r}; "
                "a missing column causes failure without falling back to each unique target."
            )
        numeric = pd.to_numeric(obs[lda_label_key], errors="coerce")
        if numeric.notna().all():
            days = numeric.to_numpy(dtype=np.float64)
            levels = np.array(sorted(np.unique(days)))
            code_map = {float(level): i for i, level in enumerate(levels)}
            y_code = np.array([code_map[float(v)] for v in days], dtype=np.int64)
            unit = str(spec.get("label_unit") or "")
            display_order = [format_time_label(level, unit) for level in levels]
            display = np.asarray([format_time_label(v, unit) for v in days], dtype=object)
            day_numeric = days
        else:
            labels = obs[lda_label_key].astype(str).to_numpy()
            display_order = sorted(set(labels.tolist()))
            code_map = {lab: i for i, lab in enumerate(display_order)}
            y_code = np.array([code_map[str(v)] for v in labels], dtype=np.int64)
            display = np.asarray(labels, dtype=object)
            parsed = np.full(len(labels), np.nan, dtype=np.float64)
            for i, lab in enumerate(labels):
                text = str(lab)
                if text.startswith("t") and "-" in text[1:]:
                    left, right = text[1:].split("-", 1)
                    parsed[i] = 0.5 * (float(left) + float(right))
            if np.isfinite(parsed).all():
                day_numeric = parsed
            elif "timepoint_index" in obs.columns:
                day_numeric = pd.to_numeric(
                    obs["timepoint_index"], errors="raise",
                ).to_numpy(dtype=np.float64)
            else:
                day_numeric = np.array(
                    [code_map[str(v)] for v in labels], dtype=np.float64,
                )
        sequential = True
    n_train_classes = int(np.unique(y_code[train_mask]).size)
    n_components = min(2, n_train_classes - 1)
    colors = colors_for_labels(display_order, sequential=sequential)
    return {
        "y_code": y_code,
        "display": display,
        "display_order": display_order,
        "day_numeric": day_numeric,
        "n_train_classes": n_train_classes,
        "n_components": n_components,
        "colors": colors,
        "lda_label_key": str(spec.get("lda_label_key") or ""),
    }


def format_time_label(value: float | int, unit: str) -> str:
    """Format display names for continuous values. Prefix numbers with unit unchanged; an empty unit yields only numbers."""

    fv = float(value)
    number = str(int(fv)) if fv.is_integer() else str(fv)
    return f"{unit}{number}"


def colors_for_labels(label_order: list[str], sequential: bool) -> dict[str, tuple]:
    """Use tab10/tab20 for discrete classes and viridis in temporal order for continuous time points."""

    n = len(label_order)
    if sequential and n > 1:
        cmap = plt.cm.viridis
        return {lab: cmap(i / (n - 1)) for i, lab in enumerate(label_order)}
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return {lab: cmap(i % cmap.N) for i, lab in enumerate(label_order)}


def fit_reducers_on_input(
    X: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    lda_y: np.ndarray,
    n_lda_components: int,
    target_kind: str,
    have_proj: bool,
    h5ad_path: Path,
    tag: str,
    batch_size: int,
    device: torch.device,
    umap_n_neighbors: int,
    umap_min_dist: float,
    phate_knn: int,
    phate_decay: int,
    phate_t: int,
    viz_random_state: int,
    viz_n_jobs: int,
    run_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Fit UMAP / LDA / optional Proj64-PCA / continuous-task PHATE on train and store coordinates for train and test."""

    X = np.asarray(X, dtype=np.float32)
    X_train = X[train_mask]
    X_test = X[test_mask]
    n_rows = int(X.shape[0])
    coords: dict[str, np.ndarray] = {}

    # UMAP: unsupervised with a fixed neighbor count, not tuned to test appearance. Use embedding_ for train and transform for test.
    print("[E] fit UMAP", flush=True)
    umap_model = umap.UMAP(
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        n_components=2,
        metric="euclidean",
        random_state=int(viz_random_state),
        n_jobs=int(viz_n_jobs),
    )
    umap_model.fit(X_train)
    z_umap = np.full((n_rows, 2), np.nan, dtype=np.float64)
    z_umap[train_mask] = np.asarray(umap_model.embedding_, dtype=np.float64)
    z_umap[test_mask] = np.asarray(umap_model.transform(X_test), dtype=np.float64)
    coords["umap"] = z_umap

    # Shrinkage LDA: eigen + auto shrinkage avoids inverting the 1024D sample covariance; retain at most two axes.
    print("[E] fit LDA", flush=True)
    lda = LinearDiscriminantAnalysis(
        solver="eigen",
        shrinkage="auto",
        n_components=int(n_lda_components),
    )
    lda.fit(X_train, lda_y[train_mask])
    z_lda_all = np.asarray(lda.transform(X), dtype=np.float64)
    if z_lda_all.ndim == 1:
        z_lda_all = z_lda_all.reshape(-1, 1)
    z_lda_all = z_lda_all[:, : int(n_lda_components)]
    coords["lda"] = z_lda_all

    if have_proj:
        print(f"[E] Proj64-PCA from uns/type_prediction/proj64_seed0/{tag}", flush=True)
        bottleneck = load_proj64_bottleneck(h5ad_path, tag, X, batch_size, device, run_name=run_name)
        n_comp = int(min(2, bottleneck.shape[1], int(train_mask.sum()) - 1))
        pca = PCA(n_components=n_comp, random_state=int(viz_random_state))
        pca.fit(bottleneck[train_mask])
        z_proj = np.full((n_rows, n_comp), np.nan, dtype=np.float64)
        z_proj[train_mask] = np.asarray(pca.transform(bottleneck[train_mask]), dtype=np.float64)
        z_proj[test_mask] = np.asarray(pca.transform(bottleneck[test_mask]), dtype=np.float64)
        coords["proj64_pca"] = z_proj

    # Use PHATE only for continuous-time tasks; project only test rows.
    if target_kind == "continuous":
        print("[E] fit PHATE", flush=True)
        phate_model = phate.PHATE(
            n_components=2,
            knn=int(phate_knn),
            decay=int(phate_decay),
            t=int(phate_t),
            knn_dist="euclidean",
            n_jobs=int(viz_n_jobs),
            random_state=int(viz_random_state),
            verbose=0,
        )
        phate_model.fit(X_train)
        z_phate = np.full((n_rows, 2), np.nan, dtype=np.float64)
        z_phate[train_mask] = np.asarray(phate_model.transform(X_train), dtype=np.float64)
        z_phate[test_mask] = np.asarray(phate_model.transform(X_test), dtype=np.float64)
        coords["phate"] = z_phate
    return coords


def load_proj64_bottleneck(
    h5ad_path: Path,
    tag: str,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    run_name: str | None = None,
) -> np.ndarray:
    """Load the seed 0 BottleneckMLP and return the 64-dimensional output after Linear+ReLU."""

    payload = FeatureReader.read_seed0(h5ad_path, tag, run_name=run_name)
    model = BottleneckMLP(
        int(payload["in_dim"]),
        int(payload["bottleneck_dim"]),
        int(payload["mlp_hidden"]),
        out_dim=int(payload["out_dim"]),
    )
    model.load_state_dict({
        name: torch.from_numpy(np.asarray(weight))
        for name, weight in payload["state_dict"].items()
    })
    model.to(device)
    bottleneck = model.extract_proj(np.asarray(X, dtype=np.float32), batch_size, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bottleneck



def load_scatter_limits(h5ad_path: Path, tag: str, run_name: str) -> dict[str, dict[str, list[float]]]:
    """Read stage E scatter-panel axis limits for reuse in stage F."""

    key = f"visualization/scatter_limits/{tag}"
    if not FeatureReader.has_uns(h5ad_path, key, run_name=run_name):
        return {}
    loaded = FeatureReader.read_uns(h5ad_path, key, run_name=run_name)
    return dict(loaded)


def draw_scatter_figure(
    coords: dict[str, np.ndarray],
    test_mask: np.ndarray,
    display: np.ndarray,
    label_order: list[str],
    colors: dict[str, tuple],
) -> tuple:
    """Draw scatter plots; use per-class kernel density for 1D LDA and only cell points for 2D. Return fig, LDA limits, and panel limits."""

    panel_keys = [key for key in ("umap", "lda", "proj64_pca", "phate") if key in coords]
    n_panel = len(panel_keys)
    n_col = n_panel if n_panel <= 3 else 2
    n_row = 1 if n_panel <= 3 else 2
    fig, axes = plt.subplots(n_row, n_col, figsize=(5.2 * n_col, 4.8 * n_row))
    axes_flat = np.atleast_1d(axes).ravel()
    lda_limits = (None, None)
    panel_limits: dict[str, dict[str, list[float]]] = {}
    titles = {
        "umap": "UMAP (unsupervised; fit on train, project test)",
        "lda": "LDA (supervised shrunk discriminant; axes from train, plot is test)",
        "proj64_pca": "Proj64-PCA (seed-0 bottleneck; PCA on train, project test)",
        "phate": "PHATE (unsupervised diffusion potential; fit on train, project test)",
    }
    display_test = display[test_mask]
    for i, key in enumerate(panel_keys):
        ax = axes_flat[i]
        z_test = coords[key][test_mask]
        if key == "lda" and z_test.shape[1] == 1:
            draw_lda_1d_density(ax, z_test[:, 0], display_test, label_order, colors)
        else:
            draw_colored_scatter(ax, z_test, display_test, label_order, colors)
            if key == "lda":
                lo, hi = padded_limits(z_test)
                lda_limits = ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])))
                ax.set_xlim(lda_limits[0])
                ax.set_ylim(lda_limits[1])
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("dim 1" if key != "lda" else "LD1")
        if not (key == "lda" and z_test.shape[1] == 1):
            ax.set_ylabel("dim 2" if key != "lda" else "LD2")
        ax.legend(markerscale=2.4, fontsize=7, frameon=False, loc="best")
        if z_test.ndim == 2 and z_test.shape[1] >= 2:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            panel_limits[key] = {
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
            }
    for j in range(n_panel, len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    return fig, lda_limits, panel_limits


def draw_colored_scatter(
    ax,
    z: np.ndarray,
    display: np.ndarray,
    label_order: list[str],
    colors: dict[str, tuple],
) -> None:
    """Draw test scatter points colored by display name without class means or text labels."""

    for lab in label_order:
        mask = display == lab
        if not mask.any():
            continue
        ax.scatter(
            z[mask, 0],
            z[mask, 1],
            s=SCATTER_POINT_SIZE,
            alpha=SCATTER_ALPHA,
            c=[colors[lab]],
            label=str(lab),
            linewidths=0,
            rasterized=True,
        )


def draw_lda_1d_density(
    ax,
    z: np.ndarray,
    display: np.ndarray,
    label_order: list[str],
    colors: dict[str, tuple],
) -> None:
    """For K=2, LDA has only LD1: draw per-class Gaussian kernel density using Scott's bandwidth."""

    z = np.asarray(z, dtype=np.float64).ravel()
    lo, hi = padded_limits(z.reshape(-1, 1))
    grid = np.linspace(float(lo[0]), float(hi[0]), 512)
    ymax = 0.0
    dens_by_lab = []
    for lab in label_order:
        mask = display == lab
        if int(mask.sum()) < 2:
            continue
        dens = gaussian_kde(z[mask], bw_method="scott")(grid)
        dens_by_lab.append((lab, mask, dens))
        ymax = max(ymax, float(dens.max()))
    rug_y = -0.04 * max(ymax, 1e-8)
    for lab, mask, dens in dens_by_lab:
        ax.plot(grid, dens, color=colors[lab], label=str(lab), linewidth=1.8)
        ax.plot(
            z[mask],
            np.full(int(mask.sum()), rug_y),
            "|",
            color=colors[lab],
            alpha=0.35,
            markersize=4,
        )
    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylabel("density")


def draw_lda_density_figure(
    z_test: np.ndarray,
    display_test: np.ndarray,
    label_order: list[str],
    colors: dict[str, tuple],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    kde_n_levels: int,
):
    """Continuous tasks with 2D LDA: draw all-test density contours and dashed class-mean lines without cell points."""

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    kde = gaussian_kde(z_test.T, bw_method="scott")
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], KDE_GRID_SIZE),
        np.linspace(ylim[0], ylim[1], KDE_GRID_SIZE),
    )
    dens = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, dens, levels=int(kde_n_levels), colors="0.35", linewidths=1.0)
    mean_x = []
    mean_y = []
    for lab in label_order:
        mask = display_test == lab
        if not mask.any():
            continue
        mx = float(z_test[mask, 0].mean())
        my = float(z_test[mask, 1].mean())
        mean_x.append(mx)
        mean_y.append(my)
        ax.scatter([mx], [my], s=70, c=[colors[lab]], edgecolors="k", zorder=3)
        ax.annotate(str(lab), (mx, my), textcoords="offset points", xytext=(7, 7), fontsize=9)
    ax.plot(mean_x, mean_y, linestyle="--", color="k", linewidth=1.3, zorder=2, label="class means (time order)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_title("LDA density (test; KDE contours + class-mean polyline, no cells)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    return fig


def lda_geometry_row(
    input_name: str,
    z_test: np.ndarray,
    display_test: np.ndarray,
    label_order: list[str],
    day_numeric: np.ndarray,
    y_time: np.ndarray,
    lda_label_key: str = "",
) -> dict[str, Any]:
    """Continuous tasks: test Spearman correlation of LD1 / polyline arc length with time, plus adjacent vs nonadjacent class-mean distances."""

    ld1 = z_test[:, 0]
    if lda_label_key == "time_bin":
        spearman_label_col = "spearman_ld1_vs_time_bin"
        spearman_arc_col = "spearman_polyline_arclength_vs_time_bin"
    else:
        spearman_label_col = "spearman_ld1_vs_growth_day"
        spearman_arc_col = "spearman_polyline_arclength_vs_growth_day"
    row = {
        "input": input_name,
        "n_lda_components": int(z_test.shape[1]),
        "n_test": int(len(z_test)),
        "n_time_levels": int(len(label_order)),
        spearman_label_col: float(spearmanr(ld1, day_numeric).correlation),
        "spearman_ld1_vs_y": float(spearmanr(ld1, y_time).correlation),
    }
    means = []
    present = []
    for lab in label_order:
        mask = display_test == lab
        if not mask.any():
            continue
        means.append(z_test[mask].mean(axis=0))
        present.append(lab)
    means = np.asarray(means, dtype=np.float64)
    adj, nonadj = adjacent_and_nonadjacent_distances(means)
    row["mean_adjacent_class_mean_dist"] = adj
    row["mean_nonadjacent_class_mean_dist"] = nonadj
    if np.isfinite(adj) and np.isfinite(nonadj) and nonadj > 0:
        row["adjacent_over_nonadjacent"] = float(adj / nonadj)
    else:
        row["adjacent_over_nonadjacent"] = float("nan")
    if z_test.shape[1] >= 2 and len(means) >= 2:
        arc = project_onto_polyline(z_test, means)
        row[spearman_arc_col] = float(spearmanr(arc, day_numeric).correlation)
    else:
        row[spearman_arc_col] = float("nan")
    return row


def adjacent_and_nonadjacent_distances(means: np.ndarray) -> tuple[float, float]:
    """Given class means in temporal order, compute mean Euclidean distances for adjacent and nonadjacent pairs."""

    if len(means) < 2:
        return float("nan"), float("nan")
    adj = []
    nonadj = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            dist = float(np.linalg.norm(means[i] - means[j]))
            if j == i + 1:
                adj.append(dist)
            else:
                nonadj.append(dist)
    mean_adj = float(np.mean(adj)) if adj else float("nan")
    mean_non = float(np.mean(nonadj)) if nonadj else float("nan")
    return mean_adj, mean_non


def project_onto_polyline(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Find each point's nearest point on the class-mean polyline and its arc length from the first time point."""

    seg = np.diff(vertices, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = np.empty(len(points), dtype=np.float64)
    for i, point in enumerate(points):
        best_d = np.inf
        best_s = 0.0
        for s in range(len(seg)):
            length = float(seg_len[s])
            start = vertices[s]
            if length == 0:
                t = 0.0
                q = start
            else:
                t = float(np.clip(np.dot(point - start, seg[s]) / (length * length), 0.0, 1.0))
                q = start + t * seg[s]
            dist = float(np.linalg.norm(point - q))
            if dist < best_d:
                best_d = dist
                best_s = float(cum[s] + t * length)
        out[i] = best_s
    return out


def draw_condition_heatmap(
    X: np.ndarray,
    obs: pd.DataFrame,
    test_mask: np.ndarray,
    spec: dict[str, Any],
    lda_pack: dict[str, Any],
    feature_names: list[str],
    input_name: str,
):
    """Keep columns in natural class or time order without column clustering; use test condition means, standardize rows, then cluster rows."""

    column_labels = heatmap_column_labels(obs, spec, lda_pack)
    column_order = heatmap_column_order(spec, lda_pack)
    facets = heatmap_facets(obs, spec)
    if facets is None:
        fig, ax = plt.subplots(figsize=heatmap_figsize(X.shape[1], len(column_order)))
        plot_one_heatmap(
            ax, X[test_mask], column_labels[test_mask], column_order, feature_names,
        )
        ax.set_title(f"{input_name} condition-mean heatmap (test)", fontsize=10)
        fig.tight_layout()
        return fig

    names = list(facets.keys())
    n_facet = len(names)
    fig, axes = plt.subplots(
        1, n_facet, figsize=(4.6 * n_facet, heatmap_figsize(X.shape[1], len(column_order))[1]),
    )
    axes_list = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes_list, names):
        mask = test_mask & facets[name]
        plot_one_heatmap(ax, X[mask], column_labels[mask], column_order, feature_names)
        ax.set_title(str(name), fontsize=10)
    fig.suptitle(f"{input_name} condition-mean heatmap (test, faceted by cell type)", fontsize=11)
    fig.tight_layout()
    return fig


def heatmap_column_labels(
    obs: pd.DataFrame,
    spec: dict[str, Any],
    lda_pack: dict[str, Any],
) -> np.ndarray:
    """Heatmap column labels: class names for discrete tasks; label_unit plus numbers for continuous tasks, matching LDA."""

    if spec["target_kind"] == "discrete":
        names = list(spec["class_names"])
        y = pd.to_numeric(obs["target"], errors="raise").to_numpy(dtype=np.int64)
        return np.asarray([names[int(v)] for v in y], dtype=object)
    return lda_pack["display"]


def heatmap_column_order(spec: dict[str, Any], lda_pack: dict[str, Any]) -> list[str]:
    """Column order: class_names for discrete tasks; ascending numeric values for continuous tasks."""

    if spec["target_kind"] == "discrete":
        return list(spec["class_names"])
    return list(lda_pack["display_order"])


def heatmap_facets(obs: pd.DataFrame, spec: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """Facet by gt_cell_type when multiple cell types exist and columns do not already represent cell types."""

    if "gt_cell_type" not in obs.columns:
        return None
    types = pd.unique(obs["gt_cell_type"].dropna())
    if len(types) <= 1:
        return None
    if spec["target_kind"] == "discrete" and spec["meta_key"] == "gt_cell_type":
        return None
    series = FeatureReader.decode_gt_cell_type(
        obs["gt_cell_type"], spec["type_index_to_name"],
    )
    return {name: (series.to_numpy() == name) for name in sorted(pd.unique(series))}


def plot_one_heatmap(
    ax,
    X: np.ndarray,
    column_labels: np.ndarray,
    column_order: list[str],
    feature_names: list[str],
) -> None:
    """Compute test means per column -> row z-score -> Ward row clustering -> imshow."""

    used = []
    cols = []
    for lab in column_order:
        mask = column_labels == lab
        if not mask.any():
            continue
        cols.append(np.mean(X[mask], axis=0))
        used.append(str(lab))
    matrix = np.column_stack(cols).astype(np.float64)
    mu = matrix.mean(axis=1, keepdims=True)
    sd = matrix.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    zmat = (matrix - mu) / sd
    names = list(feature_names)
    if zmat.shape[0] >= 2:
        order = leaves_list(linkage(pdist(zmat, metric="euclidean"), method="ward"))
        zmat = zmat[order]
        names = [names[i] for i in order]
    vmax = float(np.nanmax(np.abs(zmat)))
    if vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        zmat, aspect="auto", interpolation="nearest", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, rasterized=True,
    )
    ax.set_xticks(np.arange(len(used)))
    ax.set_xticklabels(used, rotation=45, ha="right", fontsize=8)
    if len(names) <= 200:
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=4)
    else:
        ax.set_yticks([])
    ax.set_ylabel("features (row-clustered)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def heatmap_figsize(n_features: int, n_cols: int) -> tuple[float, float]:
    """Size the heatmap canvas narrower for fewer columns and taller for more rows, with an upper bound to avoid oversized figures."""

    width = max(5.5, 0.55 * max(n_cols, 2) + 2.2)
    height = min(22.0, max(6.0, 0.018 * n_features + 2.5))
    return (width, height)


def padded_limits(z: np.ndarray, pad_frac: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Pad coordinate ranges so scatter and density plots share the same axis limits."""

    z = np.asarray(z, dtype=np.float64)
    lo = z.min(axis=0)
    hi = z.max(axis=0)
    span = hi - lo
    span = np.where(span == 0.0, 1.0, span)
    return lo - pad_frac * span, hi + pad_frac * span

MIN_TRACK_LENGTH = 10
MAX_DT = 2
N_LONGEST_PER_FOV = 6
FLOW_GRID_N = 20
MIN_EDGES_PER_CELL = 8
TRACK_BG_SIZE = 3
TRACK_BG_ALPHA = 0.18
TRACK_LINE_WIDTH = 1.15
REDUCERS = ("umap", "lda", "proj64_pca", "phate")


def run_track_visualization(
    cell_table: dict[str, Any],
    h5ad_path: Path | str,
    run_name: str,
) -> None:
    """Stage F entry: longest tracks per FOV -> polylines -> displacement field -> example tracks -> coordinate table."""

    configure_matplotlib()
    obs = cell_table["obs"].reset_index(drop=True)
    if "track_id" not in obs.columns:
        print("[F] skip: obs has no track_id", flush=True)
        return
    if "fov_id" not in obs.columns or "timepoint_index" not in obs.columns:
        print("[F] skip: obs is missing fov_id or timepoint_index", flush=True)
        return
    test_mask = obs["split"].to_numpy() == "test"

    # Select the n tracks with the most observations per test FOV; both input groups share this list.
    sample_table = sample_track_table(obs, test_mask)
    FeatureWriter.write_uns(h5ad_path, "visualization/track_sample", sample_table, run_name=run_name)
    example_ids = [
        (row["fov_id"], int(row["track_id"]))
        for _, row in sample_table.iterrows()
    ]
    print(f"[F] longest_tracks={int(sample_table.shape[0])} n_per_fov={N_LONGEST_PER_FOV}", flush=True)

    flow_rows: list[dict[str, Any]] = []
    track_coord_parts: list[pd.DataFrame] = []
    for input_name, tag in VIZ_INPUTS:
        coords = load_saved_coords(h5ad_path, tag, run_name)
        limits = load_scatter_limits(h5ad_path, tag, run_name)
        track_coord_parts.append(build_track_coord_table(tag, obs, test_mask, coords))
        for reducer in REDUCERS:
            if reducer not in coords:
                continue
            z = np.asarray(coords[reducer], dtype=np.float64)
            if z.ndim != 2 or z.shape[1] < 2:
                continue
            xlim, ylim = resolve_axis_limits(z[test_mask], limits.get(reducer))
            fig = draw_scatter_tracks(
                z=z,
                obs=obs,
                test_mask=test_mask,
                sample_table=sample_table,
                xlim=xlim,
                ylim=ylim,
                reducer=reducer,
                tag=tag,
            )
            FeatureWriter.write_figure(h5ad_path, f"scatter_tracks_{tag}_{reducer}", fig, run_name=run_name)
            fig, geometry = draw_flow_field(
                z=z,
                obs=obs,
                test_mask=test_mask,
                xlim=xlim,
                ylim=ylim,
                input_name=input_name,
                reducer=reducer,
            )
            FeatureWriter.write_figure(h5ad_path, f"flow_{tag}_{reducer}", fig, run_name=run_name)
            flow_rows.append(geometry)
            fig = draw_track_examples(
                z=z,
                obs=obs,
                example_ids=example_ids,
                reducer=reducer,
                tag=tag,
            )
            FeatureWriter.write_figure(h5ad_path, f"track_examples_{tag}_{reducer}", fig, run_name=run_name)
        del coords
    if track_coord_parts:
        FeatureWriter.write_uns(
            h5ad_path, "visualization/track_coords", pd.concat(track_coord_parts, ignore_index=True), run_name=run_name)
    if flow_rows:
        FeatureWriter.write_uns(h5ad_path, "visualization/track_flow_geometry", pd.DataFrame(flow_rows), run_name=run_name)
    print(f"[F] wrote tracks into {h5ad_path}", flush=True)


def sample_track_table(obs: pd.DataFrame, test_mask: np.ndarray) -> pd.DataFrame:
    """Select the n eligible tracks with the most observations per test FOV; break ties by ascending track_id."""

    eligible = collect_eligible_tracks(obs, test_mask)
    if eligible.empty:
        return pd.DataFrame(columns=["fov_id", "track_id", "n_obs"])
    parts = []
    for _fov_id, block in eligible.groupby("fov_id", sort=True):
        taken = block.sort_values(
            ["n_obs", "track_id"], ascending=[False, True],
        ).head(N_LONGEST_PER_FOV)
        parts.append(taken)
    return pd.concat(parts, ignore_index=True)[["fov_id", "track_id", "n_obs"]].copy()


def collect_eligible_tracks(obs: pd.DataFrame, test_mask: np.ndarray) -> pd.DataFrame:
    """Select test tracks with track_id>=0 and at least 10 observations."""

    sub = obs.loc[test_mask].copy()
    sub["track_id"] = pd.to_numeric(sub["track_id"], errors="coerce")
    sub = sub.loc[sub["track_id"] >= 0]
    if sub.empty:
        return pd.DataFrame(columns=["fov_id", "track_id", "n_obs"])
    rows = []
    for (fov_id, track_id), group in sub.groupby(["fov_id", "track_id"], sort=True):
        n_obs = int(group.shape[0])
        if n_obs < MIN_TRACK_LENGTH:
            continue
        rows.append({
            "fov_id": fov_id,
            "track_id": int(track_id),
            "n_obs": n_obs,
        })
    return pd.DataFrame(rows)


def build_track_coord_table(
    tag: str,
    obs: pd.DataFrame,
    test_mask: np.ndarray,
    coords: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a coordinate table for test rows with track_id>=0 for later line-width changes / resampling."""

    tids = pd.to_numeric(obs["track_id"], errors="coerce").to_numpy()
    keep = test_mask & np.isfinite(tids) & (tids >= 0)
    parts = []
    for reducer, z in coords.items():
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] < 2:
            continue
        parts.append(pd.DataFrame({
            "obs_name": obs["obs_name"].to_numpy()[keep],
            "fov_id": obs["fov_id"].to_numpy()[keep],
            "track_id": tids[keep].astype(np.int64),
            "timepoint_index": pd.to_numeric(obs["timepoint_index"], errors="raise").to_numpy()[keep],
            "x": z[keep, 0],
            "y": z[keep, 1],
            "split": obs["split"].to_numpy()[keep],
            "tag": tag,
            "reducer": reducer,
        }))
    if not parts:
        return pd.DataFrame(columns=[
            "obs_name", "fov_id", "track_id", "timepoint_index",
            "x", "y", "split", "tag", "reducer",
        ])
    return pd.concat(parts, ignore_index=True)


def resolve_axis_limits(
    z_test: np.ndarray,
    saved: dict[str, list[float]] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Prefer stage E axis limits; if absent, recompute using the same pad_frac."""

    finite = np.isfinite(z_test).all(axis=1)
    z_use = z_test[finite]
    if saved and "xlim" in saved and "ylim" in saved:
        xlim = (float(saved["xlim"][0]), float(saved["xlim"][1]))
        ylim = (float(saved["ylim"][0]), float(saved["ylim"][1]))
        return xlim, ylim
    lo, hi = padded_limits(z_use)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def draw_scatter_tracks(
    z: np.ndarray,
    obs: pd.DataFrame,
    test_mask: np.ndarray,
    sample_table: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    reducer: str,
    tag: str,
):
    """Use faint all-test points as the background and time-colored polylines of the longest tracks per FOV as the foreground."""

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    display, label_order, colors = track_background_labels(obs)
    z_test = z[test_mask]
    finite = np.isfinite(z_test).all(axis=1)
    display_test = display[test_mask]
    for lab in label_order:
        mask = finite & (display_test == lab)
        if not mask.any():
            continue
        ax.scatter(
            z_test[mask, 0],
            z_test[mask, 1],
            s=TRACK_BG_SIZE,
            alpha=TRACK_BG_ALPHA,
            c=[colors[lab]],
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    cmap = plt.cm.viridis
    t_all = pd.to_numeric(obs["timepoint_index"], errors="raise").to_numpy(dtype=np.float64)
    t_max = 59.0
    segments, seg_colors, arrows = build_sampled_polylines(
        z, obs, sample_table, cmap, t_max,
    )
    if segments:
        lc = LineCollection(
            segments, colors=seg_colors, linewidths=TRACK_LINE_WIDTH, zorder=3,
        )
        ax.add_collection(lc)
    for x0, y0, x1, y1, color in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.0, "mutation_scale": 9},
            zorder=4,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=t_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="timepoint_index")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("dim 1" if reducer != "lda" else "LD1")
    ax.set_ylabel("dim 2" if reducer != "lda" else "LD2")
    ax.set_title(
        f"{tag} {reducer}: held-out FOVs; longest {N_LONGEST_PER_FOV} tracks/FOV; not class-mean polyline",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def track_background_labels(obs: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, tuple]]:
    """Color the background by time_bin when available, otherwise use six bins of continuous time."""

    if "time_bin" in obs.columns:
        display = obs["time_bin"].astype(str).to_numpy()
        order = sorted(pd.unique(display).tolist())
    else:
        t = pd.to_numeric(obs["timepoint_index"], errors="raise").to_numpy(dtype=np.int64)
        display = np.asarray([f"t{int(v):02d}" for v in t], dtype=object)
        order = sorted(pd.unique(display).tolist())
    colors = colors_for_labels(order, sequential=True)
    return display, order, colors


def build_sampled_polylines(
    z: np.ndarray,
    obs: pd.DataFrame,
    sample_table: pd.DataFrame,
    cmap,
    t_max: float,
) -> tuple[list[np.ndarray], list[tuple], list[tuple]]:
    """Connect listed tracks as time-ordered polylines; omit edges with delta_t>2."""

    segments: list[np.ndarray] = []
    seg_colors: list[tuple] = []
    arrows: list[tuple] = []
    if sample_table.empty:
        return segments, seg_colors, arrows
    sample_keys = set(
        zip(sample_table["fov_id"].to_numpy(), sample_table["track_id"].astype(int).to_numpy())
    )
    for (fov_id, track_id), group in group_test_tracks(obs, z):
        if (fov_id, int(track_id)) not in sample_keys:
            continue
        pts = group[["x", "y", "t"]].to_numpy(dtype=np.float64)
        last_seg = None
        for i in range(pts.shape[0] - 1):
            dt = pts[i + 1, 2] - pts[i, 2]
            if dt < 1 or dt > MAX_DT:
                if last_seg is not None:
                    arrows.append(last_seg)
                    last_seg = None
                continue
            seg = pts[i:i + 2, :2]
            if not np.isfinite(seg).all():
                continue
            color = cmap(float(pts[i + 1, 2]) / t_max)
            segments.append(seg)
            seg_colors.append(color)
            last_seg = (float(seg[0, 0]), float(seg[0, 1]), float(seg[1, 0]), float(seg[1, 1]), color)
        if last_seg is not None:
            arrows.append(last_seg)
    return segments, seg_colors, arrows


def group_test_tracks(obs: pd.DataFrame, z: np.ndarray):
    """List test observations with finite coordinates by FOV and track_id, sorted by time."""

    tids = pd.to_numeric(obs["track_id"], errors="coerce")
    keep = (obs["split"].to_numpy() == "test") & tids.notna() & (tids >= 0)
    work = pd.DataFrame({
        "fov_id": obs["fov_id"].to_numpy()[keep],
        "track_id": tids.to_numpy()[keep].astype(np.int64),
        "t": pd.to_numeric(obs["timepoint_index"], errors="raise").to_numpy()[keep],
        "x": z[keep, 0],
        "y": z[keep, 1],
    })
    work = work.loc[np.isfinite(work[["x", "y"]]).all(axis=1)]
    work = work.sort_values(["fov_id", "track_id", "t"])
    return work.groupby(["fov_id", "track_id"], sort=True)


def draw_flow_field(
    z: np.ndarray,
    obs: pd.DataFrame,
    test_mask: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    input_name: str,
    reducer: str,
) -> tuple:
    """Bin all eligible adjacent pairs into a 20x20 grid to average the displacement field. Return fig and geometry rows."""

    edges = collect_track_edges(z, obs)
    geometry = compute_flow_geometry(z, obs, test_mask, edges)
    geometry["input"] = input_name
    geometry["reducer"] = reducer
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    z_test = z[test_mask]
    finite = np.isfinite(z_test).all(axis=1)
    ax.scatter(
        z_test[finite, 0], z_test[finite, 1],
        s=TRACK_BG_SIZE, alpha=0.12, c="0.55", linewidths=0, rasterized=True, zorder=1,
    )
    grid = build_flow_grid(edges, xlim, ylim)
    if not grid.empty:
        mag = grid["mag"].to_numpy(dtype=np.float64)
        ax.quiver(
            grid["x"].to_numpy(dtype=np.float64), grid["y"].to_numpy(dtype=np.float64),
            grid["u"].to_numpy(dtype=np.float64), grid["v"].to_numpy(dtype=np.float64), mag,
            cmap="viridis", angles="xy", scale_units="xy",
            width=0.0035, zorder=3,
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=float(mag.min()), vmax=float(mag.max())),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="mean |dz/dt|")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("dim 1" if reducer != "lda" else "LD1")
    ax.set_ylabel("dim 2" if reducer != "lda" else "LD2")
    ax.set_title(
        f"{reducer}: population displacement field (test tracks, 20x20, min {MIN_EDGES_PER_CELL} edges)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig, geometry


def build_flow_grid(
    edges: list[dict[str, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> pd.DataFrame:
    """Bin eligible adjacent pairs into a FLOW_GRID_N x FLOW_GRID_N grid and average displacement; one row per grid cell.

    Align grid boundaries with axis limits: divide xlim / ylim linearly into FLOW_GRID_N cells along x/y.
    Omit cells with fewer than MIN_EDGES_PER_CELL edges. Columns x/y are cell centers; u/v are mean delta_z/delta_t within each cell.
    """

    columns = [
        "gx", "gy", "x_lo", "x_hi", "y_lo", "y_hi", "x", "y", "u", "v", "mag", "n_edges",
    ]
    if not edges:
        return pd.DataFrame(columns=columns)
    xs = np.array([e["x0"] for e in edges], dtype=np.float64)
    ys = np.array([e["y0"] for e in edges], dtype=np.float64)
    us = np.array([e["u"] for e in edges], dtype=np.float64)
    vs = np.array([e["v"] for e in edges], dtype=np.float64)
    x_edges = np.linspace(xlim[0], xlim[1], FLOW_GRID_N + 1)
    y_edges = np.linspace(ylim[0], ylim[1], FLOW_GRID_N + 1)
    xi = np.clip(np.digitize(xs, x_edges) - 1, 0, FLOW_GRID_N - 1)
    yi = np.clip(np.digitize(ys, y_edges) - 1, 0, FLOW_GRID_N - 1)
    rows: list[dict[str, Any]] = []
    for gx in range(FLOW_GRID_N):
        for gy in range(FLOW_GRID_N):
            sel = (xi == gx) & (yi == gy)
            n_e = int(sel.sum())
            if n_e < MIN_EDGES_PER_CELL:
                continue
            x_lo = float(x_edges[gx])
            x_hi = float(x_edges[gx + 1])
            y_lo = float(y_edges[gy])
            y_hi = float(y_edges[gy + 1])
            u = float(us[sel].mean())
            v = float(vs[sel].mean())
            rows.append({
                "gx": gx,
                "gy": gy,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "x": 0.5 * (x_lo + x_hi),
                "y": 0.5 * (y_lo + y_hi),
                "u": u,
                "v": v,
                "mag": float(np.hypot(u, v)),
                "n_edges": n_e,
            })
    return pd.DataFrame(rows, columns=columns)


def collect_track_edges(z: np.ndarray, obs: pd.DataFrame) -> list[dict[str, float]]:
    """Collect all eligible adjacent test pairs with 1<=delta_t<=2. u,v are delta_z/delta_t; include fov_id / track_id / start and end frames."""

    edges: list[dict[str, float]] = []
    for (fov_id, track_id), group in group_test_tracks(obs, z):
        pts = group[["x", "y", "t"]].to_numpy(dtype=np.float64)
        for i in range(pts.shape[0] - 1):
            dt = pts[i + 1, 2] - pts[i, 2]
            if dt < 1 or dt > MAX_DT:
                continue
            dx = pts[i + 1, 0] - pts[i, 0]
            dy = pts[i + 1, 1] - pts[i, 1]
            if not np.isfinite([dx, dy, dt]).all():
                continue
            edges.append({
                "fov_id": str(fov_id),
                "track_id": int(track_id),
                "t0": float(pts[i, 2]),
                "t1": float(pts[i + 1, 2]),
                "x0": float(pts[i, 0]),
                "y0": float(pts[i, 1]),
                "dx": float(dx),
                "dy": float(dy),
                "dt": float(dt),
                "u": float(dx / dt),
                "v": float(dy / dt),
            })
    return edges


def compute_flow_geometry(
    z: np.ndarray,
    obs: pd.DataFrame,
    test_mask: np.ndarray,
    edges: list[dict[str, float]],
) -> dict[str, Any]:
    """Displacement-field geometry: median step length, track persistence, and alignment with the local time gradient."""

    if not edges:
        return {
            "n_edges": 0,
            "median_step_norm": float("nan"),
            "median_persistence": float("nan"),
            "spearman_step_align_time": float("nan"),
        }
    step_norm = np.hypot(
        np.array([e["u"] for e in edges], dtype=np.float64),
        np.array([e["v"] for e in edges], dtype=np.float64),
    )
    persist = []
    for (_fov_id, _track_id), group in group_test_tracks(obs, z):
        if int(group.shape[0]) < MIN_TRACK_LENGTH:
            continue
        pts = group[["x", "y", "t"]].to_numpy(dtype=np.float64)
        arc = 0.0
        for i in range(pts.shape[0] - 1):
            dt = pts[i + 1, 2] - pts[i, 2]
            if dt < 1 or dt > MAX_DT:
                continue
            arc += float(np.hypot(pts[i + 1, 0] - pts[i, 0], pts[i + 1, 1] - pts[i, 1]))
        net = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        if arc > 0:
            persist.append(net / arc)
    z_test = z[test_mask]
    t_test = pd.to_numeric(obs["timepoint_index"], errors="raise").to_numpy(dtype=np.float64)[test_mask]
    finite = np.isfinite(z_test).all(axis=1) & np.isfinite(t_test)
    align = float("nan")
    if int(finite.sum()) >= 3:
        model = LinearRegression()
        model.fit(z_test[finite], t_test[finite])
        grad = np.asarray(model.coef_, dtype=np.float64)
        gnorm = float(np.linalg.norm(grad))
        if gnorm > 0:
            ghat = grad / gnorm
            cosines = []
            for edge in edges:
                vec = np.array([edge["dx"], edge["dy"]], dtype=np.float64)
                nrm = float(np.linalg.norm(vec))
                if nrm == 0:
                    continue
                cosines.append(float(np.dot(vec, ghat) / nrm))
            if cosines:
                align = float(np.median(cosines))
    return {
        "n_edges": int(len(edges)),
        "median_step_norm": float(np.median(step_norm)),
        "median_persistence": float(np.median(persist)) if persist else float("nan"),
        "spearman_step_align_time": align,
    }


def draw_track_examples(
    z: np.ndarray,
    obs: pd.DataFrame,
    example_ids: list[tuple[Any, int]],
    reducer: str,
    tag: str,
):
    """Draw small multiples for the same longest tracks, with points colored by timepoint_index."""

    n = max(len(example_ids), 1)
    n_col = 4
    n_row = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.4 * n_col, 3.1 * n_row))
    axes_flat = np.atleast_1d(axes).ravel()
    tracks = {key: group for key, group in group_test_tracks(obs, z)}
    cmap = plt.cm.viridis
    t_max = 59.0
    for i, ax in enumerate(axes_flat):
        if i >= len(example_ids):
            ax.axis("off")
            continue
        fov_id, track_id = example_ids[i]
        group = tracks.get((fov_id, track_id))
        ax.set_title(f"{fov_id} track {track_id}", fontsize=8)
        if group is None or group.empty:
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        pts = group[["x", "y", "t"]].to_numpy(dtype=np.float64)
        for j in range(pts.shape[0] - 1):
            dt = pts[j + 1, 2] - pts[j, 2]
            if dt < 1 or dt > MAX_DT:
                continue
            ax.plot(
                pts[j:j + 2, 0], pts[j:j + 2, 1],
                color=cmap(float(pts[j + 1, 2]) / t_max), linewidth=1.4,
            )
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=pts[:, 2], cmap=cmap, vmin=0.0, vmax=t_max,
            s=12, zorder=3, rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{tag} {reducer}: longest {N_LONGEST_PER_FOV} tracks per FOV", fontsize=11)
    fig.tight_layout()
    return fig


class CellFeatureVisualizer:
    """Held-out dimensionality reduction and scatter, density, and heatmap plots for emb* / morph_ext."""

    def run(self, **kwargs):
        """Stage E entry point."""

        return run_visualization(**kwargs)


class CellTrackVisualizer:
    """Draw single-cell trajectories and population displacement fields in stage E coordinates."""

    def run(self, **kwargs):
        """Stage F entry point."""

        return run_track_visualization(**kwargs)

