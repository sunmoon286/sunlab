import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import h5py
import laptrack
import numpy as np
import pandas as pd

from scripts_formal.utils.elements_result_io import AnnDataResultStore, FeatureReader


# Packaged default tracking config: <package>/config/laptrack_config.json.
DEFAULT_TRACKING_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / 'config' / 'laptrack_config.json'
)


def main(
    config_path: str = DEFAULT_TRACKING_CONFIG_PATH,
) -> dict:
    """Run the fixed LUMA time-series tracking workflow."""
    # Run the tracking workflow with script-level defaults only.
    tracker = CellTracker()
    return tracker.track_anndata(config_path=config_path)


class CellTracker:
    """
    Track LUMA AnnData cell observations with laptrack and write tracking results.
    """

    def track_anndata(
        self,
        config_path: str = DEFAULT_TRACKING_CONFIG_PATH,
        **tracking_overrides,
    ) -> dict:
        """Run laptrack on one LUMA AnnData file, one FOV at a time when fov_id is present."""
        # Load JSON parameters and let explicit function kwargs override the config file.
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config_params = json.load(config_file)
        tracking_params = self.merge_tracking_params(config_params, tracking_overrides)
        adata_path = tracking_params.get('anndata_path')
        if not adata_path:
            raise ValueError(
                "tracking needs an input H5AD: pass anndata_path=... (CellExperiment.track_cells "
                "does this via h5ad_path) or set 'anndata_path' in the tracking config."
            )
        fov_column = str(tracking_params.get('fov_column', 'fov_id'))

        # Load only the AnnData observation table; FOV membership comes from obs and /uns/luma/fovs.
        result_store = AnnDataResultStore(adata_path)
        if FeatureReader.has_uns(adata_path, "cell_table_config"):
            cfg = FeatureReader.read_uns(adata_path, "cell_table_config")
            if cfg.get("max_cells_per_image") not in (None, "", "None"):
                raise RuntimeError("Tracking cannot run on a sampled cell table")
        obs = result_store.read_table_elements(table_keys=('obs',))['tables']['obs']
        fov_ids = self.resolve_fov_ids(adata_path=adata_path, obs=obs, fov_column=fov_column)

        # Track each FOV independently, then offset identifiers so track_id stays unique in obs.
        fov_results: List[Dict[str, pd.DataFrame]] = []
        track_id_offset = 0
        tree_id_offset = 0
        for fov_id in fov_ids:
            fov_obs = obs if fov_id is None else obs[obs[fov_column].astype(str) == str(fov_id)]
            if fov_obs.empty:
                continue
            tracking_df, point_mapping, weighted_feature_columns = self.prepare_tracking_dataframe(
                obs=fov_obs,
                tracking_params=tracking_params,
            )
            track_df, split_df, merge_df = self.run_laptrack(
                tracking_df=tracking_df,
                weighted_feature_columns=weighted_feature_columns,
                tracking_params=tracking_params,
            )
            fov_result = self.extract_tracking_result(
                track_df=track_df,
                split_df=split_df,
                merge_df=merge_df,
                point_mapping=point_mapping,
            )
            if fov_id is not None:
                fov_result = self.attach_fov_id(fov_result, str(fov_id))
            fov_result = self.offset_track_identifiers(
                fov_result, track_id_offset=track_id_offset, tree_id_offset=tree_id_offset,
            )
            fov_results.append(fov_result)
            track_id_offset, tree_id_offset = self.next_identifier_offsets(
                fov_result, track_id_offset, tree_id_offset,
            )
            n_tracks = int(fov_result['track_summary'].shape[0])
            print(f'Tracked FOV {fov_id}: {n_tracks} tracks, {int(fov_obs.shape[0])} cells.')

        tracking_result = self.concat_tracking_results(fov_results)
        self.write_tracking_to_anndata(
            adata_path=adata_path,
            obs=obs,
            tracking_result=tracking_result,
            tracking_params=tracking_params,
            config_path=config_path,
            tracking_overrides=tracking_overrides,
            fov_ids=[fov_id for fov_id in fov_ids if fov_id is not None],
            fov_column=fov_column,
        )

        return {
            'adata_path': adata_path,
            'config_path': config_path,
            'fov_ids': [fov_id for fov_id in fov_ids if fov_id is not None],
            'n_tracks': int(tracking_result['track_summary'].shape[0]),
            'n_tracked_observations': int(tracking_result['obs_assignments'].shape[0]),
            'n_gap_nodes': int(tracking_result['gap_nodes'].shape[0]),
        }


    def merge_tracking_params(self, config_params: Dict[str, Any], tracking_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge JSON tracking parameters with explicit keyword overrides."""
        params = dict(config_params)
        cost_keys = {
            'track_start_cost', 'track_end_cost', 'segment_start_cost', 'segment_end_cost',
            'no_splitting_cost', 'no_merging_cost', 'alternative_cost_factor',
        }
        laptrack_costs = dict(params.get('laptrack_costs', {}))

        # Allow both laptrack_costs={...} and direct track_start_cost=... style overrides.
        for key, value in tracking_overrides.items():
            if key == 'laptrack_costs':
                laptrack_costs.update(value)
            elif key in cost_keys:
                laptrack_costs[key] = value
            else:
                params[key] = value
        params['laptrack_costs'] = laptrack_costs
        return params

    def prepare_tracking_dataframe(
        self,
        obs: pd.DataFrame,
        tracking_params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Build the laptrack input DataFrame, point mapping, and weighted feature list."""
        obs = obs.copy()
        point_ids = np.arange(obs.shape[0], dtype=np.int64)
        tracking_df = pd.DataFrame({
            'laptrack_point_id': point_ids,
            'obs_name': obs.index.astype(str).to_numpy(),
            'obs_index': np.arange(obs.shape[0], dtype=np.int64),
            'frame': obs[tracking_params['time_column']].to_numpy(dtype=np.int64),
            'x': obs[tracking_params['x_column']].to_numpy(dtype=np.float64),
            'y': obs[tracking_params['y_column']].to_numpy(dtype=np.float64),
        }).set_index('laptrack_point_id', drop=True)
        point_mapping = tracking_df[['obs_name', 'obs_index']].reset_index()

        # Normalize optional numeric features before using them as additional coordinates.
        normalized_features = self.normalize_visual_features(
            obs=obs,
            visual_feature_columns=tuple(tracking_params.get('visual_feature_columns', [])),
            visual_feature_normalization=str(tracking_params.get('visual_feature_normalization', 'zscore')),
            visual_feature_clip_percentiles=tuple(tracking_params.get('visual_feature_clip_percentiles', [1.0, 99.0])),
        )
        weighted_feature_columns = []
        feature_weight = float(tracking_params.get('visual_feature_weight', 1.0))
        for feature_column in normalized_features.columns:
            weighted_column = f'weighted_{feature_column}'
            tracking_df[weighted_column] = normalized_features[feature_column].to_numpy(dtype=np.float64) * feature_weight
            weighted_feature_columns.append(weighted_column)

        return tracking_df, point_mapping, weighted_feature_columns

    def run_laptrack(
        self,
        tracking_df: pd.DataFrame,
        weighted_feature_columns: List[str],
        tracking_params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run one laptrack prediction and return point, split, and merge tables."""
        distance_metric = str(tracking_params['distance_metric'])
        distance_power = 2 if distance_metric == 'sqeuclidean' else 1
        laptrack_costs = {
            key: value
            for key, value in tracking_params.get('laptrack_costs', {}).items()
            if value is not None
        }

        # Build the tracker from config distances expressed in pixels.
        tracker = laptrack.LapTrack(
            metric=distance_metric,
            cutoff=float(tracking_params['max_linking_distance_px']) ** distance_power,
            gap_closing_metric=distance_metric,
            gap_closing_cutoff=float(tracking_params['gap_closing_distance_px']) ** distance_power,
            gap_closing_max_frame_count=int(tracking_params['gap_closing_max_frame_count']),
            splitting_metric=distance_metric,
            splitting_cutoff=float(tracking_params['splitting_distance_px']) ** distance_power if tracking_params['allow_splitting'] else False,
            merging_metric=distance_metric,
            merging_cutoff=float(tracking_params['merging_distance_px']) ** distance_power if tracking_params['allow_merging'] else False,
            **laptrack_costs,
        )
        return tracker.predict_dataframe(
            tracking_df,
            coordinate_cols=['x', 'y'] + weighted_feature_columns,
            frame_col='frame',
            only_coordinate_cols=False,
            index_offset=0,
        )

    def extract_tracking_result(
        self,
        track_df: pd.DataFrame,
        split_df: pd.DataFrame,
        merge_df: pd.DataFrame,
        point_mapping: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Convert laptrack outputs into AnnData-friendly tracking tables."""
        if track_df.empty:
            return self.empty_tracking_result(point_mapping)

        # Join laptrack point assignments back to AnnData observation names.
        all_nodes = track_df.copy()
        all_nodes.index.name = 'laptrack_point_id'
        all_nodes = all_nodes.reset_index()
        all_nodes = all_nodes.merge(point_mapping, on='laptrack_point_id', how='left', suffixes=('', '_mapped'))
        if 'obs_name_mapped' in all_nodes:
            all_nodes['obs_name'] = all_nodes['obs_name_mapped'].fillna(all_nodes['obs_name'])
        if 'obs_index_mapped' in all_nodes:
            all_nodes['obs_index'] = all_nodes['obs_index_mapped'].fillna(all_nodes['obs_index'])
        all_nodes = all_nodes.drop(columns=[column for column in ['obs_name_mapped', 'obs_index_mapped'] if column in all_nodes])
        all_nodes = all_nodes.rename(columns={'frame': 't'})
        all_nodes = all_nodes.sort_values(['track_id', 't', 'laptrack_point_id']).reset_index(drop=True)
        all_nodes['track_order'] = all_nodes.groupby('track_id').cumcount().astype(np.int64)
        all_nodes['track_id'] = all_nodes['track_id'].astype(np.int64)
        all_nodes['tree_id'] = all_nodes['tree_id'].astype(np.int64)

        # Build the persistent tables consumed by AnnData and the visualizer.
        obs_assignments = all_nodes[['obs_name', 'track_id', 'track_order']].copy()
        split_events = split_df.copy().reset_index(drop=True)
        merge_events = merge_df.copy().reset_index(drop=True)
        track_summary = self.build_track_summary(all_nodes, split_events, merge_events)
        lineage_edges = self.build_lineage_edges(track_summary, split_events, merge_events)
        obs_graph = self.build_obs_graph(all_nodes, lineage_edges)
        gap_nodes = self.build_gap_nodes(all_nodes)

        return {
            'obs_assignments': obs_assignments.reset_index(drop=True),
            'track_summary': track_summary.reset_index(drop=True),
            'obs_graph': obs_graph.reset_index(drop=True),
            'lineage_edges': lineage_edges.reset_index(drop=True),
            'split_events': split_events.reset_index(drop=True),
            'merge_events': merge_events.reset_index(drop=True),
            'gap_nodes': gap_nodes.reset_index(drop=True),
            'point_mapping': point_mapping.reset_index(drop=True),
        }

    def write_tracking_to_anndata(
        self,
        adata_path: str,
        obs: pd.DataFrame,
        tracking_result: Dict[str, pd.DataFrame],
        tracking_params: Dict[str, Any],
        config_path: str,
        tracking_overrides: Dict[str, Any],
        fov_ids: Optional[List[str]] = None,
        fov_column: str = 'fov_id',
    ) -> pd.DataFrame:
        """Overwrite the current AnnData tracking fields and /uns/luma/tracking result."""
        redundant_obs_columns = [
            'track_parent_id', 'track_root_id', 'track_generation', 'track_fate',
            'track_t', 'track_length', 'track_start_t', 'track_end_t', 'tracking_run_id',
        ]
        obs = obs.drop(columns=[column for column in redundant_obs_columns if column in obs]).copy()

        # Store only the final track ID and each node order inside the optimized track.
        obs['track_id'] = np.full(obs.shape[0], -1, dtype=np.int64)
        obs['track_order'] = np.full(obs.shape[0], -1, dtype=np.int64)
        obs_assignments = tracking_result['obs_assignments'].set_index('obs_name')
        assigned_obs = obs.index.intersection(obs_assignments.index)
        obs.loc[assigned_obs, 'track_id'] = obs_assignments.loc[assigned_obs, 'track_id'].to_numpy(dtype=np.int64)
        obs.loc[assigned_obs, 'track_order'] = obs_assignments.loc[assigned_obs, 'track_order'].to_numpy(dtype=np.int64)

        # Replace the whole tracking dict so the H5AD file contains one current result only.
        tracking_record = {
            'params': self.prepare_serializable(tracking_params),
            'tracking_backend': 'laptrack',
            'laptrack_version': laptrack.__version__,
            'track_summary': tracking_result['track_summary'],
            'obs_graph': tracking_result['obs_graph'],
            'lineage_edges': tracking_result['lineage_edges'],
            'split_events': tracking_result['split_events'],
            'merge_events': tracking_result['merge_events'],
            'gap_nodes': tracking_result['gap_nodes'],
            'point_mapping': tracking_result['point_mapping'],
            'config_path': config_path,
            'tracking_overrides': self.prepare_serializable(tracking_overrides),
            'created_at': datetime.now().isoformat(timespec='seconds'),
        }
        luma_uns: Dict[str, Any] = {'tracking': tracking_record}
        if fov_ids:
            luma_uns['fovs'] = {
                'fov_column': fov_column,
                'fov_ids': list(fov_ids),
            }
        AnnDataResultStore(adata_path).write_table_elements(
            obs=obs,
            luma_uns=luma_uns,
        )
        return obs

    def resolve_fov_ids(
        self,
        adata_path: str,
        obs: pd.DataFrame,
        fov_column: str,
    ) -> List[Optional[str]]:
        """Return FOV identifiers from /uns/luma/fovs, else from obs, else a single-pass sentinel."""

        if fov_column not in obs.columns:
            return [None]

        with h5py.File(adata_path, 'r') as h5_file:
            luma_group = h5_file['uns']['luma']
            if 'fovs' in luma_group:
                fov_record = ad.io.read_elem(luma_group['fovs'])
                stored_ids = [str(fov_id) for fov_id in np.asarray(fov_record['fov_ids']).tolist()]
                if stored_ids:
                    return stored_ids

        return list(dict.fromkeys(obs[fov_column].astype(str).tolist()))

    def attach_fov_id(
        self,
        tracking_result: Dict[str, pd.DataFrame],
        fov_id: str,
    ) -> Dict[str, pd.DataFrame]:
        """Copy fov_id onto every tracking table produced for one FOV."""

        tagged_result = {}
        for table_key, table in tracking_result.items():
            tagged = table.copy()
            tagged['fov_id'] = fov_id
            tagged_result[table_key] = tagged
        return tagged_result

    def offset_track_identifiers(
        self,
        tracking_result: Dict[str, pd.DataFrame],
        track_id_offset: int,
        tree_id_offset: int,
    ) -> Dict[str, pd.DataFrame]:
        """Shift per-FOV laptrack IDs so concatenated results remain unique."""

        shifted = {key: table.copy() for key, table in tracking_result.items()}
        track_columns = {
            'obs_assignments': ('track_id',),
            'gap_nodes': ('track_id',),
            'obs_graph': ('source_track_id', 'target_track_id'),
            'lineage_edges': ('parent_track_id', 'child_track_id', 'root_track_id'),
            'split_events': ('parent_track_id', 'child_track_id'),
            'merge_events': ('parent_track_id', 'child_track_id'),
        }
        for table_key, columns in track_columns.items():
            table = shifted[table_key]
            if table.empty:
                continue
            for column in columns:
                table[column] = table[column].astype(np.int64) + track_id_offset

        summary = shifted['track_summary']
        if not summary.empty:
            summary['track_id'] = summary['track_id'].astype(np.int64) + track_id_offset
            summary['tree_id'] = summary['tree_id'].astype(np.int64) + tree_id_offset
            summary['root_id'] = summary['root_id'].astype(np.int64) + track_id_offset
            parent_ids = summary['parent_id'].astype(np.int64).to_numpy()
            parent_ids = np.where(parent_ids >= 0, parent_ids + track_id_offset, parent_ids)
            summary['parent_id'] = parent_ids
        return shifted

    def next_identifier_offsets(
        self,
        tracking_result: Dict[str, pd.DataFrame],
        track_id_offset: int,
        tree_id_offset: int,
    ) -> Tuple[int, int]:
        """Return the next unused track_id and tree_id after one FOV result."""

        summary = tracking_result['track_summary']
        if summary.empty:
            return track_id_offset, tree_id_offset
        return (
            int(summary['track_id'].max()) + 1,
            int(summary['tree_id'].max()) + 1,
        )

    def concat_tracking_results(
        self,
        fov_results: List[Dict[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """Concatenate per-FOV tracking tables into one AnnData tracking payload."""

        if not fov_results:
            return self.empty_tracking_result(pd.DataFrame(columns=['obs_name', 'obs_index']))
        merged = {}
        for table_key in fov_results[0]:
            merged[table_key] = pd.concat(
                [result[table_key] for result in fov_results],
                axis=0,
                ignore_index=True,
            )
        return merged

    def build_track_summary(
        self,
        all_nodes: pd.DataFrame,
        split_events: pd.DataFrame,
        merge_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Summarize each optimized track into one row of lineage and motion metrics."""
        parent_by_child = {}
        for event_table in [split_events, merge_events]:
            for _, event_row in event_table.iterrows():
                child_id = int(event_row['child_track_id'])
                if child_id not in parent_by_child:
                    parent_by_child[child_id] = int(event_row['parent_track_id'])
        parent_tracks = set(split_events.get('parent_track_id', pd.Series(dtype=np.int64)).astype(int).tolist())
        parent_tracks.update(merge_events.get('parent_track_id', pd.Series(dtype=np.int64)).astype(int).tolist())

        summary_rows = []
        for track_id, track_nodes in all_nodes.groupby('track_id', sort=True):
            ordered_nodes = track_nodes.sort_values(['track_order', 't'])
            coordinates = ordered_nodes[['x', 'y']].to_numpy(dtype=np.float64)
            segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1) if coordinates.shape[0] > 1 else np.array([])
            displacement = float(np.linalg.norm(coordinates[-1] - coordinates[0])) if coordinates.shape[0] > 1 else 0.0
            duration_frames = int(ordered_nodes['t'].max() - ordered_nodes['t'].min() + 1)
            parent_id = int(parent_by_child.get(int(track_id), -1))
            root_id = int(track_id)
            generation = 0
            while root_id in parent_by_child:
                root_id = int(parent_by_child[root_id])
                generation += 1
            fate = 'branch' if int(track_id) in parent_tracks else 'end'

            summary_rows.append({
                'track_id': int(track_id),
                'tree_id': int(ordered_nodes['tree_id'].iloc[0]),
                'parent_id': parent_id,
                'root_id': root_id,
                'generation': generation,
                'fate': fate,
                'start_t': int(ordered_nodes['t'].min()),
                'end_t': int(ordered_nodes['t'].max()),
                'duration_frames': duration_frames,
                'track_length': int(ordered_nodes.shape[0]),
                'n_observed_nodes': int(ordered_nodes.shape[0]),
                'n_gap_nodes': int(np.maximum(np.diff(ordered_nodes['t'].to_numpy(dtype=np.int64)) - 1, 0).sum()) if ordered_nodes.shape[0] > 1 else 0,
                'start_obs_name': str(ordered_nodes['obs_name'].iloc[0]),
                'end_obs_name': str(ordered_nodes['obs_name'].iloc[-1]),
                'mean_speed_px_per_frame': float(segment_lengths.sum() / max(duration_frames - 1, 1)),
                'net_displacement_px': displacement,
                'path_length_px': float(segment_lengths.sum()),
            })

        return pd.DataFrame(summary_rows, columns=[
            'track_id', 'tree_id', 'parent_id', 'root_id', 'generation', 'fate', 'start_t', 'end_t',
            'duration_frames', 'track_length', 'n_observed_nodes', 'n_gap_nodes',
            'start_obs_name', 'end_obs_name', 'mean_speed_px_per_frame',
            'net_displacement_px', 'path_length_px',
        ])
    
    def build_lineage_edges(
        self,
        track_summary: pd.DataFrame,
        split_events: pd.DataFrame,
        merge_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a track-level split and merge edge table from laptrack event tables."""
        summary_by_track = track_summary.set_index('track_id')
        edge_rows = []
        for event_table, edge_type in [(split_events, 'split'), (merge_events, 'merge')]:
            for _, event_row in event_table.iterrows():
                parent_id = int(event_row['parent_track_id'])
                child_id = int(event_row['child_track_id'])
                if parent_id not in summary_by_track.index or child_id not in summary_by_track.index:
                    continue
                parent_summary = summary_by_track.loc[parent_id]
                child_summary = summary_by_track.loc[child_id]
                edge_rows.append({
                    'parent_track_id': parent_id,
                    'child_track_id': child_id,
                    'root_track_id': int(child_summary['root_id']),
                    'parent_end_t': int(parent_summary['end_t']),
                    'child_start_t': int(child_summary['start_t']),
                    'generation': int(child_summary['generation']),
                    'edge_type': edge_type,
                })
        return pd.DataFrame(edge_rows, columns=[
            'parent_track_id', 'child_track_id', 'root_track_id', 'parent_end_t',
            'child_start_t', 'generation', 'edge_type',
        ])
    
    def build_obs_graph(self, all_nodes: pd.DataFrame, lineage_edges: pd.DataFrame) -> pd.DataFrame:
        """Build an observation-level directed edge table for track and lineage links."""
        edge_rows = []

        # Add temporal edges between adjacent real observations within each optimized track.
        for _, track_nodes in all_nodes.groupby('track_id', sort=True):
            ordered_nodes = track_nodes.sort_values(['track_order', 't'])
            for source_index in range(max(ordered_nodes.shape[0] - 1, 0)):
                source_node = ordered_nodes.iloc[source_index]
                target_node = ordered_nodes.iloc[source_index + 1]
                edge_rows.append(self.build_obs_edge_row(source_node, target_node, 'track'))

        # Add split and merge edges from the final parent observation to the first child observation.
        for _, lineage_edge in lineage_edges.iterrows():
            parent_nodes = all_nodes[all_nodes['track_id'] == int(lineage_edge['parent_track_id'])].sort_values('track_order')
            child_nodes = all_nodes[all_nodes['track_id'] == int(lineage_edge['child_track_id'])].sort_values('track_order')
            if parent_nodes.empty or child_nodes.empty:
                continue
            edge_rows.append(self.build_obs_edge_row(parent_nodes.iloc[-1], child_nodes.iloc[0], str(lineage_edge['edge_type'])))

        return pd.DataFrame(edge_rows, columns=[
            'source_obs_name', 'target_obs_name', 'source_obs_index', 'target_obs_index',
            'source_track_id', 'target_track_id', 'edge_type', 'source_t', 'target_t', 'dt', 'distance_px',
        ])

    def build_gap_nodes(self, all_nodes: pd.DataFrame) -> pd.DataFrame:
        """Interpolate optional gap nodes between tracked observations separated by missing frames."""
        gap_rows = []
        for _, track_nodes in all_nodes.groupby('track_id', sort=True):
            ordered_nodes = track_nodes.sort_values(['track_order', 't'])
            for source_index in range(max(ordered_nodes.shape[0] - 1, 0)):
                source_node = ordered_nodes.iloc[source_index]
                target_node = ordered_nodes.iloc[source_index + 1]
                dt = int(target_node['t'] - source_node['t'])
                if dt <= 1:
                    continue
                for gap_step in range(1, dt):
                    ratio = gap_step / dt
                    gap_rows.append({
                        'track_id': int(source_node['track_id']),
                        'track_order': float(source_node['track_order']) + ratio,
                        'source_obs_name': str(source_node['obs_name']),
                        'target_obs_name': str(target_node['obs_name']),
                        't': int(source_node['t'] + gap_step),
                        'x': float(source_node['x'] + (target_node['x'] - source_node['x']) * ratio),
                        'y': float(source_node['y'] + (target_node['y'] - source_node['y']) * ratio),
                        'alpha': 0.45,
                    })
        return pd.DataFrame(gap_rows, columns=[
            'track_id', 'track_order', 'source_obs_name', 'target_obs_name', 't', 'x', 'y', 'alpha',
        ])

    def normalize_visual_features(
        self,
        obs: pd.DataFrame,
        visual_feature_columns: tuple[str, ...],
        visual_feature_normalization: str,
        visual_feature_clip_percentiles: tuple[float, float],
    ) -> pd.DataFrame:
        """Normalize optional numeric visual features before laptrack receives them."""
        normalized = pd.DataFrame(index=obs.index)
        lower_percentile, upper_percentile = visual_feature_clip_percentiles

        for feature_column in visual_feature_columns:
            values = pd.to_numeric(obs[feature_column], errors='coerce').to_numpy(dtype=np.float64)
            lower, upper = np.nanpercentile(values, [lower_percentile, upper_percentile])
            clipped = np.clip(values, lower, upper)
            median_value = np.nanmedian(clipped)
            clipped = np.nan_to_num(clipped, nan=median_value)

            if visual_feature_normalization == 'minmax':
                denominator = np.max(clipped) - np.min(clipped)
                normalized[feature_column] = (clipped - np.min(clipped)) / denominator if denominator > 0 else np.zeros_like(clipped)
            elif visual_feature_normalization == 'none':
                normalized[feature_column] = clipped
            else:
                standard_deviation = np.std(clipped)
                normalized[feature_column] = (clipped - np.mean(clipped)) / standard_deviation if standard_deviation > 0 else np.zeros_like(clipped)

        return normalized

    def empty_tracking_result(self, point_mapping: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Return empty tracking tables with the expected schema."""
        return {
            'obs_assignments': pd.DataFrame(columns=['obs_name', 'track_id', 'track_order']),
            'track_summary': pd.DataFrame(columns=[
                'track_id', 'tree_id', 'parent_id', 'root_id', 'generation', 'fate', 'start_t', 'end_t',
                'duration_frames', 'track_length', 'n_observed_nodes', 'n_gap_nodes',
                'start_obs_name', 'end_obs_name', 'mean_speed_px_per_frame',
                'net_displacement_px', 'path_length_px',
            ]),
            'obs_graph': pd.DataFrame(columns=[
                'source_obs_name', 'target_obs_name', 'source_obs_index', 'target_obs_index',
                'source_track_id', 'target_track_id', 'edge_type', 'source_t', 'target_t', 'dt', 'distance_px',
            ]),
            'lineage_edges': pd.DataFrame(columns=[
                'parent_track_id', 'child_track_id', 'root_track_id', 'parent_end_t',
                'child_start_t', 'generation', 'edge_type',
            ]),
            'split_events': pd.DataFrame(columns=['parent_track_id', 'child_track_id']),
            'merge_events': pd.DataFrame(columns=['parent_track_id', 'child_track_id']),
            'gap_nodes': pd.DataFrame(columns=[
                'track_id', 'track_order', 'source_obs_name', 'target_obs_name', 't', 'x', 'y', 'alpha',
            ]),
            'point_mapping': point_mapping.reset_index(drop=True),
        }

    def prepare_serializable(self, value: Any) -> Any:
        """Convert config values into AnnData-writable scalars and containers."""
        if isinstance(value, dict):
            return {str(key): self.prepare_serializable(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [self.prepare_serializable(item) for item in value]
        if isinstance(value, list):
            return [self.prepare_serializable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if value is None:
            return 'None'
        return value

    def build_obs_edge_row(self, source_node: pd.Series, target_node: pd.Series, edge_type: str) -> Dict[str, Any]:
        """Build one observation-level edge row from two real tracking nodes."""
        distance_px = float(np.linalg.norm(
            np.array([target_node['x'], target_node['y']], dtype=np.float64)
            - np.array([source_node['x'], source_node['y']], dtype=np.float64)
        ))
        return {
            'source_obs_name': str(source_node['obs_name']),
            'target_obs_name': str(target_node['obs_name']),
            'source_obs_index': int(source_node['obs_index']),
            'target_obs_index': int(target_node['obs_index']),
            'source_track_id': int(source_node['track_id']),
            'target_track_id': int(target_node['track_id']),
            'edge_type': edge_type,
            'source_t': int(source_node['t']),
            'target_t': int(target_node['t']),
            'dt': int(target_node['t'] - source_node['t']),
            'distance_px': distance_px,
        }


if __name__ == '__main__':
    main()