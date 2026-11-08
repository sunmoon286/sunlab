import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from scipy import ndimage, sparse
from skimage.io import imread
from skimage.measure import regionprops
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score

# AnnDataResultStore is imported lazily inside run_evaluation so this module stays
# importable in environments without anndata (e.g. conda env test_microsam, used by
# the CellPose / microSAM baseline scripts that only need in-memory map scoring).


class CellEvaluator:
    """
    Evaluate LUMA segmentation and classification results from AnnData image records or in-memory maps.
    """

    def run_evaluation(
        self,
        adata_path: Optional[str] = None,
        image_id: Optional[str] = None,
        gt_inst_map: Optional[np.ndarray] = None,
        pred_inst_map: Optional[np.ndarray] = None,
        gt_type_map: Optional[np.ndarray] = None,
        pred_type_map: Optional[np.ndarray] = None,
        use_cyto_type: bool = True,
        null_hypothesis_threshold: float = 0.8,
        iou_threshold: float = 0.5,
        res_name: str = None,
        eval_region: Tuple[int, int, int, int]|None = None,
        type_mapping: Optional[Dict[int, str]] = {},
        **kwargs
    ) -> Dict[str, any]:
        """
        Runs a comprehensive evaluation for both segmentation and classification.

        Writes metrics back to the AnnData store whenever adata_path and image_id are provided.
        When use_cyto_type is False, type maps are optional and only segmentation is evaluated.

        Args:
            adata_path (str|None): LUMA H5AD result store.
            image_id (str|None): Image key under /uns/luma/images when using adata_path.
            gt_inst_map (np.ndarray|None): Manual GT instance map.
            pred_inst_map (np.ndarray|None): Manual predicted instance map.
            gt_type_map (np.ndarray|None): Manual GT type map.
            pred_type_map (np.ndarray|None): Manual predicted type map.
            use_cyto_type (bool): If True, require type maps and evaluate classification.
            null_hypothesis_threshold (float): Threshold for assigning background type.
            iou_threshold (float): IoU threshold for a segmentation true positive.
            res_name (str|None): Name used in reports.
            eval_region (tuple|None): Optional y1/x1/y2/x2 crop, or 'auto' for stitch margins.
            type_mapping (dict): Mapping from integer type IDs to display names.

        Returns:
            Dict[str, any]: Segmentation, classification, and summary outputs.
        """
        stitch_info = None
        if adata_path and image_id:
            from scripts_formal.utils.elements_result_io import AnnDataResultStore

            image_record = AnnDataResultStore(adata_path).read_image_result(image_id)
            gt_inst_map = image_record['gt']['gt_instance_map']
            pred_inst_map = image_record['prediction']['pred_instance_map']
            gt_type_map = image_record['gt']['gt_type_cyto_map']
            pred_type_map = image_record['prediction']['pred_type_cyto_map']
            stitch_info = image_record.get('stitch_info', None)
            if res_name is None:
                res_name = image_id
        else:
            res_name = 'manual_input_results' if res_name is None else res_name

        # Instance maps are always required; type maps are required only when evaluating classification.
        required_maps = [gt_inst_map, pred_inst_map]
        if use_cyto_type:
            required_maps.extend([gt_type_map, pred_type_map])
        if any(m is None for m in required_maps):
            print("Error: One or more required maps were not provided.")
            return {}

        # Optionally crop the evaluation region before computing metrics.
        if eval_region:
            if type(eval_region) in [list, tuple] and len(eval_region) == 4:
                y1, x1, y2, x2 = eval_region
                gt_inst_map = gt_inst_map[y1:y2, x1:x2]
                pred_inst_map = pred_inst_map[y1:y2, x1:x2]
                if gt_type_map is not None:
                    gt_type_map = gt_type_map[y1:y2, x1:x2]
                if pred_type_map is not None:
                    pred_type_map = pred_type_map[y1:y2, x1:x2]
            elif eval_region == 'auto' and stitch_info:
                mh, mw = stitch_info.get('margin_h', 0), stitch_info.get('margin_w', 0)
                y1, y2 = mh // 2, mh - mh // 2
                x1, x2 = mw // 2, mw - mw // 2
                y2_slice = None if y2 == 0 else -y2
                x2_slice = None if x2 == 0 else -x2
                gt_inst_map = gt_inst_map[y1:y2_slice, x1:x2_slice]
                pred_inst_map = pred_inst_map[y1:y2_slice, x1:x2_slice]
                if gt_type_map is not None:
                    gt_type_map = gt_type_map[y1:y2_slice, x1:x2_slice]
                if pred_type_map is not None:
                    pred_type_map = pred_type_map[y1:y2_slice, x1:x2_slice]

        # Evaluate segmentation first; classification only when cyto-type maps are in use.
        seg_results, matched_pairs = self.evaluate_segmentation(
            gt_inst_map, pred_inst_map, iou_threshold, verbose=False
        )
        class_results = None
        if use_cyto_type:
            class_results = self.evaluate_classification(
                matched_pairs,
                gt_inst_map, pred_inst_map,
                gt_type_map, pred_type_map,
                null_hypothesis_threshold,
                type_mapping=type_mapping
            )

        # Build text summary and flat summary record for downstream reports.
        summary_text_lines = [f"--- Unified Evaluation Summary of {res_name}---", "\n"]
        summary_text_lines.append("Segmentation Metrics:")
        for k, v in seg_results.items():
            line = f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}"
            summary_text_lines.append(line)

        if use_cyto_type:
            summary_text_lines.append("\nClassification Metrics (on matched instances):")
            if class_results:
                self.append_iou_summary(summary_text_lines, class_results)
                self.append_ranking_summary(summary_text_lines, class_results)
                report_str = class_results['report'].to_string()
                summary_text_lines.append(report_str)
            else:
                summary_text_lines.append("  - No matched instances to evaluate classification.")

        summary_record = self.build_summary_record(res_name, seg_results, class_results, type_mapping)
        total_results = {
            "segmentation": seg_results,
            "classification": class_results,
            "matched_pairs": matched_pairs,
            "summary_text": summary_text_lines,
            "summary_record": summary_record
        }

        # Write metrics into the AnnData store whenever an image record path is provided.
        if adata_path and image_id:
            from scripts_formal.utils.elements_result_io import AnnDataResultStore

            AnnDataResultStore(adata_path).append_evaluation_result(image_id, total_results, overwrite=True)

        return total_results

    def run_evaluation_batch(
        self,
        adata_path: Optional[str] = None,
        image_id: Optional[List[str]] = None,
        image_dir: Optional[str] = None,
        gt_inst_prefix: str = 'gt_inst_',
        pred_inst_prefix: str = 'pred_inst_',
        gt_type_prefix: Optional[str] = 'gt_type_',
        pred_type_prefix: Optional[str] = 'pred_type_',
        use_cyto_type: bool = True,
        null_hypothesis_threshold: float = 0.8,
        iou_threshold: float = 0.5,
        eval_region: Tuple[int, int, int, int]|None = None,
        type_mapping: Optional[Dict[int, str]] = {},
        res_dir: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Batch-evaluate multiple images via AnnData image IDs or a directory of map files.

        Mode A: provide adata_path and a list of image_id.
        Mode B: provide image_dir and four matcher strings (gt_inst_prefix / pred_inst_prefix /
        gt_type_prefix / pred_type_prefix). Each string only tags file role (not a full filename);
        files containing that string form one sorted list, then the four lists are paired
        one-to-one by index into eval groups. res_name is derived from the GT instance filename
        by removing the matcher string.

        When res_dir is set, writes:
          - evaluation_summary.csv: one row per image from summary_record
          - evaluation_summaries.json: per-image summary_record and summary_text

        Returns:
            Dict with summary_table (DataFrame) and per_image_summaries (dict).
        """
        # Build the per-image evaluation job list for either AnnData or directory mode.
        eval_jobs = []
        if adata_path is not None:
            for one_id in image_id:
                eval_jobs.append({
                    'mode': 'adata',
                    'image_id': one_id,
                    'res_name': one_id,
                })
        else:
            eval_jobs = self._build_map_eval_jobs(
                image_dir=image_dir,
                gt_inst_token=gt_inst_prefix,
                pred_inst_token=pred_inst_prefix,
                gt_type_token=gt_type_prefix,
                pred_type_token=pred_type_prefix,
                use_cyto_type=use_cyto_type,
            )

        # Run evaluation for each image and collect summary outputs.
        summary_records = []
        per_image_summaries = {}
        for job in tqdm(eval_jobs, desc="Batch evaluation"):
            if job['mode'] == 'adata':
                result = self.run_evaluation(
                    adata_path=adata_path,
                    image_id=job['image_id'],
                    use_cyto_type=use_cyto_type,
                    null_hypothesis_threshold=null_hypothesis_threshold,
                    iou_threshold=iou_threshold,
                    eval_region=eval_region,
                    type_mapping=type_mapping,
                    res_name=job['res_name'],
                )
            else:
                gt_inst_map = np.squeeze(imread(job['gt_inst_path']))
                pred_inst_map = np.squeeze(imread(job['pred_inst_path']))
                gt_type_map = None
                pred_type_map = None
                if use_cyto_type:
                    gt_type_map = np.squeeze(imread(job['gt_type_path']))
                    pred_type_map = np.squeeze(imread(job['pred_type_path']))
                result = self.run_evaluation(
                    gt_inst_map=gt_inst_map,
                    pred_inst_map=pred_inst_map,
                    gt_type_map=gt_type_map,
                    pred_type_map=pred_type_map,
                    use_cyto_type=use_cyto_type,
                    null_hypothesis_threshold=null_hypothesis_threshold,
                    iou_threshold=iou_threshold,
                    eval_region=eval_region,
                    type_mapping=type_mapping,
                    res_name=job['res_name'],
                )

            if not result:
                continue
            summary_records.append(result['summary_record'])
            per_image_summaries[job['res_name']] = {
                'summary_record': result['summary_record'],
                'summary_text': result['summary_text'],
            }

        summary_table = pd.DataFrame(summary_records)

        # Optionally persist the aggregated table and merged per-image summaries.
        if res_dir is not None:
            os.makedirs(res_dir, exist_ok=True)
            csv_path = os.path.join(res_dir, 'evaluation_summary.csv')
            json_path = os.path.join(res_dir, 'evaluation_summaries.json')
            summary_table.to_csv(csv_path, index=False)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(per_image_summaries, f, ensure_ascii=False, indent=2)

        return {
            'summary_table': summary_table,
            'per_image_summaries': per_image_summaries,
        }

    def _build_map_eval_jobs(
        self,
        image_dir: str,
        gt_inst_token: str,
        pred_inst_token: str,
        gt_type_token: Optional[str],
        pred_type_token: Optional[str],
        use_cyto_type: bool,
    ) -> List[Dict[str, str]]:
        """Build eval groups by pairing four matcher-token file lists one-to-one by index."""
        dir_files = os.listdir(image_dir)
        gt_inst_files = sorted([fname for fname in dir_files if gt_inst_token in fname])
        pred_inst_files = sorted([fname for fname in dir_files if pred_inst_token in fname])
        if len(gt_inst_files) != len(pred_inst_files):
            raise ValueError(
                f"GT/pred instance file counts differ: {len(gt_inst_files)} vs {len(pred_inst_files)}"
            )
        gt_type_files = None
        pred_type_files = None
        if use_cyto_type:
            gt_type_files = sorted([fname for fname in dir_files if gt_type_token in fname])
            pred_type_files = sorted([fname for fname in dir_files if pred_type_token in fname])
            if len(gt_type_files) != len(gt_inst_files) or len(pred_type_files) != len(gt_inst_files):
                raise ValueError(
                    "Type map file counts must match instance map file counts: "
                    f"gt_inst={len(gt_inst_files)}, pred_inst={len(pred_inst_files)}, "
                    f"gt_type={len(gt_type_files)}, pred_type={len(pred_type_files)}"
                )

        eval_jobs = []
        for i, gt_inst_fname in enumerate(gt_inst_files):
            name, _ext = os.path.splitext(gt_inst_fname)
            res_name = name.replace(gt_inst_token, '', 1).strip('_-. ') or f"image_{i:04d}"
            job = {
                'mode': 'maps',
                'res_name': res_name,
                'gt_inst_path': os.path.join(image_dir, gt_inst_fname),
                'pred_inst_path': os.path.join(image_dir, pred_inst_files[i]),
            }
            if use_cyto_type:
                job['gt_type_path'] = os.path.join(image_dir, gt_type_files[i])
                job['pred_type_path'] = os.path.join(image_dir, pred_type_files[i])
            eval_jobs.append(job)
        return eval_jobs

    def compute_ranking_metrics(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_classes: int,
    ) -> Tuple[any, any]:
        """Score ranking quality: binary ROC-AUC, or multi-class one-vs-rest AP then macro / weighted means.

        Multi-class AP averages only over classes present in y_true, so a missing rare class
        (e.g. unlabeled on hesc_cellcycle) does not abort the metric.
        """

        roc_auc_val = "N/A"
        average_precision_val = "N/A"
        present_classes = np.unique(y_true)
        if present_classes.size < 2:
            return "Undefined (Single Class)", "Undefined (Single Class)"

        try:
            if n_classes == 2:
                if y_score.shape[1] >= 2:
                    roc_auc_val = float(roc_auc_score(y_true, y_score[:, 1]))
                else:
                    roc_auc_val = "Error (Score Shape)"
                return roc_auc_val, average_precision_val

            per_class_ap = np.full(n_classes, np.nan, dtype=np.float64)
            for class_idx in range(n_classes):
                class_mask = y_true == class_idx
                if class_mask.any():
                    per_class_ap[class_idx] = float(
                        average_precision_score(class_mask.astype(np.int64), y_score[:, class_idx])
                    )
            counts = np.bincount(y_true.astype(np.int64), minlength=n_classes)
            n_total = float(len(y_true))
            present = ~np.isnan(per_class_ap)
            if not present.any():
                return roc_auc_val, "Undefined (Single Class)"
            average_precision_val = {
                'macro': float(np.nanmean(per_class_ap)),
                'weighted': float(np.dot(counts / n_total, np.nan_to_num(per_class_ap, nan=0.0))),
                'per_class': {
                    int(class_idx): float(per_class_ap[class_idx])
                    for class_idx in np.where(present)[0]
                },
            }
            return roc_auc_val, average_precision_val
        except Exception as exc:
            print(f"Warning: ranking metric calculation failed: {exc}")
            return "Error", "Error"

    def extract_ranking_summary_fields(self, class_results: Dict[str, any]) -> Dict[str, str]:
        """Flatten ROC-AUC or multi-class AP into CSV-style summary fields."""

        ranking_fields: Dict[str, str] = {}
        average_precision = class_results.get('average_precision')
        if isinstance(average_precision, dict) and average_precision.get('macro') is not None:
            ranking_fields['Classification AP'] = f"{average_precision['macro']:.4f}"
        roc_auc_val = class_results.get('roc_auc')
        if isinstance(roc_auc_val, (int, float)):
            ranking_fields['Classification AUC'] = f"{roc_auc_val:.4f}"
        return ranking_fields

    def append_ranking_summary(self, summary_text_lines: list, class_results: Dict[str, any]) -> None:
        """Append binary ROC-AUC or multi-class macro AP to the text summary."""

        average_precision = class_results.get('average_precision')
        if isinstance(average_precision, dict) and average_precision.get('macro') is not None:
            summary_text_lines.append(f"  - macro AP: {average_precision['macro']:.4f}")
            return
        roc_auc_val = class_results.get('roc_auc')
        if isinstance(roc_auc_val, (int, float)):
            summary_text_lines.append(f"  - ROC AUC: {roc_auc_val:.4f}")

    def append_iou_summary(self, summary_text_lines: list, class_results: Dict[str, any]) -> None:
        """Append pixel-wise and instance-wise IoU details to the text summary."""
        if 'pixel_iou_metrics' in class_results:
            summary_text_lines.append("  Pixel-wise Type Map IoU:")
            iou_metrics = class_results['pixel_iou_metrics']
            if 'mIoU' in iou_metrics:
                summary_text_lines.append(f"    - Mean IoU: {iou_metrics['mIoU']:.4f}")
            for k, v in iou_metrics.items():
                if k != 'mIoU':
                    class_id = k.split('_')[1]
                    summary_text_lines.append(f"    - IoU for class {class_id}: {v:.4f}")
        if 'instance_type_iou_metrics' in class_results:
            summary_text_lines.append("  Instance-wise Type Map IoU:")
            inst_iou_metrics = class_results['instance_type_iou_metrics']
            if 'mIoU' in inst_iou_metrics:
                summary_text_lines.append(f"    - Mean IoU: {inst_iou_metrics['mIoU']:.4f}")
            for k, v in inst_iou_metrics.items():
                if k != 'mIoU':
                    class_id = k.split('_')[1]
                    summary_text_lines.append(f"    - IoU for class {class_id}: {v:.4f}")

    def build_summary_record(self, res_name: str, seg_results: Dict[str, any],
                             class_results: Optional[Dict[str, any]],
                             type_mapping: Dict[int, str]) -> Dict[str, any]:
        """Build a flat dictionary of key evaluation metrics for CSV-style summaries."""
        summary_record = {'Dataset': res_name}
        if class_results:
            gt_counts = class_results.get('gt_type_counts', {})
            total_cells = class_results.get('total_cells', 0)
            for type_name, count in gt_counts.items():
                summary_record[f"{type_name} Count"] = count
                summary_record[f"{type_name} Percentage"] = f"{(count / total_cells * 100):.3f}%" if total_cells > 0 else "0.000%"
            summary_record['Total GT Cells'] = class_results.get('total_gt_cells', 0)
            summary_record['Total Detected Cells'] = class_results.get('total_cells', 0)

        summary_record['Segmentation Precision'] = f"{seg_results.get('Precision', 0):.4f}"
        summary_record['Segmentation Recall'] = f"{seg_results.get('Recall', 0):.4f}"
        summary_record['Segmentation AJI'] = f"{seg_results.get('AJI', 0):.4f}"
        summary_record['Segmentation PQ'] = f"{seg_results.get('PQ', 0):.4f}"
        summary_record['Segmentation DQ'] = f"{seg_results.get('DQ', 0):.4f}"
        summary_record['Segmentation SQ'] = f"{seg_results.get('SQ', 0):.4f}"

        if class_results and 'report' in class_results:
            report_df = class_results['report']
            if not report_df.empty and type_mapping:
                for type_name in type_mapping.values():
                    if type_name in report_df.index:
                        summary_record[f'{type_name} Precision'] = f"{report_df.loc[type_name, 'precision']:.4f}"
                        summary_record[f'{type_name} Recall'] = f"{report_df.loc[type_name, 'recall']:.4f}"
                        summary_record[f'{type_name} F1'] = f"{report_df.loc[type_name, 'f1-score']:.4f}"
                    else:
                        summary_record[f'{type_name} Precision'] = "0.0000"
                        summary_record[f'{type_name} Recall'] = "0.0000"
                        summary_record[f'{type_name} F1'] = "0.0000"
            ranking_fields = self.extract_ranking_summary_fields(class_results)
            summary_record.update(ranking_fields)
        return summary_record

    def evaluate_classification(
        self,
        matched_pairs: Dict[int, int],
        gt_inst_map: np.ndarray,
        pred_inst_map: np.ndarray,
        gt_type_map: np.ndarray,
        pred_type_map: np.ndarray,
        null_hypothesis_threshold: float = 0.8,
        type_mapping: Dict[int, str] = {}
    ) -> Optional[Dict[str, any]]:
        """Evaluate cell classification on matched GT/predicted instances."""

        def get_type_counts(inst_map, type_map, labels):
            """Count per-class pixels for every instance label."""
            counts = []
            sorted_labels = np.sort(labels)
            all_labels, pixel_counts = np.unique(inst_map[inst_map > 0], return_counts=True)
            total_pixels = np.zeros(all_labels.max() + 1, dtype=int)
            total_pixels[all_labels] = pixel_counts
            for i in range(n_classes):
                weights = (type_map == i).astype(np.uint8)
                class_counts = ndimage.sum_labels(weights, labels=inst_map, index=sorted_labels)
                counts.append(class_counts)
            counts_matrix = np.vstack(counts).T
            counts_dict = {label: counts_matrix[i] for i, label in enumerate(sorted_labels)}
            total_pixels_dict = {label: total_pixels[label] for label in sorted_labels}
            return counts_dict, total_pixels_dict

        if not matched_pairs:
            print("No matched instances provided for classification evaluation.")
            return None

        if type_mapping:
            n_classes = len(type_mapping)
            class_names = [type_mapping[i] for i in sorted(type_mapping.keys())]
        else:
            n_classes = int(max(gt_type_map.max(), pred_type_map.max())) + 1
            class_names = [str(i) for i in range(n_classes)]

        all_gt_labels = np.unique(gt_inst_map)
        all_gt_labels = all_gt_labels[all_gt_labels > 0]
        all_pred_labels = np.unique(pred_inst_map)
        all_pred_labels = all_pred_labels[all_pred_labels > 0]

        gt_counts_dict, _ = get_type_counts(gt_inst_map, gt_type_map.astype(int), all_gt_labels)
        pred_counts_dict, pred_total_pixels_dict = get_type_counts(pred_inst_map, pred_type_map.astype(int), all_pred_labels)

        # Generate GT and predicted instance-type maps for IoU diagnostics.
        max_gt_id = np.max(gt_inst_map)
        gt_lut = np.zeros(max_gt_id + 1, dtype=np.int64)
        for lbl in all_gt_labels:
            gt_lut[lbl] = np.argmax(gt_counts_dict[lbl])
        gt_instance_type_map = gt_lut[gt_inst_map]

        max_pred_id = np.max(pred_inst_map)
        pred_lut = np.zeros(max_pred_id + 1, dtype=np.int64)
        for lbl in all_pred_labels:
            counts = pred_counts_dict[lbl]
            total = pred_total_pixels_dict[lbl]
            if total == 0:
                continue
            scores = counts / total
            if scores[0] > null_hypothesis_threshold:
                pred_type = 0
            else:
                pred_type = np.argmax(scores[1:]) + 1 if n_classes > 1 else 0
            pred_lut[lbl] = pred_type
        pred_instance_type_map = pred_lut[pred_inst_map]

        instance_type_iou_metrics = self.calculate_iou_metrics(gt_instance_type_map, pred_instance_type_map, n_classes)

        # Build matched-instance classification arrays.
        y_true, y_pred, y_score = [], [], []
        for gt_label, pred_label in tqdm(matched_pairs.items(), desc="Evaluating Classification on Matches"):
            y_true.append(np.argmax(gt_counts_dict[gt_label]))
            pred_counts = pred_counts_dict.get(pred_label)
            if pred_counts is None:
                y_pred.append(0)
                y_score.append(np.zeros(n_classes))
                continue
            total_pixels = pred_total_pixels_dict[pred_label]
            scores = pred_counts / total_pixels if total_pixels > 0 else np.zeros(n_classes)
            y_score.append(scores)
            if scores[0] > null_hypothesis_threshold:
                y_pred.append(0)
            else:
                y_pred.append(np.argmax(scores[1:]) + 1 if n_classes > 1 else 0)

        if not y_true:
            return None
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)

        # Binary ranking uses ROC-AUC; multi-class ranking uses one-vs-rest AP then macro / weighted means.
        roc_auc_val, average_precision_val = self.compute_ranking_metrics(y_true, y_score, n_classes)

        gt_type_counts = {}
        if type_mapping:
            for type_id, type_name in type_mapping.items():
                gt_type_counts[type_name] = int(np.sum(y_true == type_id))
        else:
            for i in range(n_classes):
                gt_type_counts[str(i)] = int(np.sum(y_true == i))

        logit_to_name = {i: type_mapping.get(i, str(i)) for i in range(n_classes)} if type_mapping else {i: str(i) for i in range(n_classes)}
        y_true_remapped = np.array([logit_to_name.get(label, str(label)) for label in y_true])
        y_pred_remapped = np.array([logit_to_name.get(label, str(label)) for label in y_pred])

        report_dict = classification_report(
            y_true=y_true_remapped, 
            y_pred=y_pred_remapped, 
            output_dict=True, 
            zero_division=0,
            labels=class_names)
        report_df = pd.DataFrame(report_dict).transpose()
        cm = confusion_matrix(y_true_remapped, y_pred_remapped, labels=class_names)
        pixel_iou_metrics = self.calculate_iou_metrics(gt_type_map, pred_type_map, n_classes)

        return {
            "report": report_df,
            "confusion_matrix": cm,
            "class_names": class_names,
            "roc_auc": roc_auc_val,
            "average_precision": average_precision_val,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "pixel_iou_metrics": pixel_iou_metrics,
            "instance_type_iou_metrics": instance_type_iou_metrics,
            "gt_instance_type_map": gt_instance_type_map,
            "pred_instance_type_map": pred_instance_type_map,
            "gt_type_counts": gt_type_counts,
            "total_cells": len(y_true),
            'total_gt_cells': np.unique(gt_inst_map).size - 1
        }

    def calculate_iou_metrics(self, gt_map: np.ndarray, pred_map: np.ndarray,
                              num_classes: int) -> Dict[str, float]:
        """Calculate per-class and mean IoU for two integer label maps."""
        gt_flat = gt_map.flatten().astype(int)
        pred_flat = pred_map.flatten().astype(int)
        pixel_cm = sparse.coo_matrix(
            (np.ones(gt_flat.shape[0]), (gt_flat, pred_flat)),
            shape=(num_classes, num_classes),
            dtype=np.int64
        ).toarray()
        intersection = np.diag(pixel_cm)
        ground_truth_set = pixel_cm.sum(axis=1)
        predicted_set = pixel_cm.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        iou_per_class = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        gt_present_classes = np.where(ground_truth_set > 0)[0]
        metrics = {}
        if len(gt_present_classes) > 0:
            metrics['mIoU'] = np.mean(iou_per_class[gt_present_classes])
            for i in gt_present_classes:
                metrics[f'class_{i}_IoU'] = iou_per_class[i]
        else:
            metrics['mIoU'] = 0.0
        return metrics

    def evaluate_segmentation(self, gt_inst_map:np.ndarray, pred_inst_map:np.ndarray,
                              iou_threshold: float = 0.5,
                              verbose: bool = True) -> Tuple[Dict[str, float], Dict[int, int]]:
        """Evaluate instance segmentation: TP/FP/FN, Precision/Recall, AJI, and PQ/DQ/SQ."""
        if verbose: print(f"\n--- Evaluating Segmentation at IoU Threshold: {iou_threshold} (Efficient Method) ---")

        # Collect instance labels and areas from GT and prediction maps.
        gt_props = regionprops(gt_inst_map)
        pred_props = regionprops(pred_inst_map)
        gt_labels = np.array([prop.label for prop in gt_props])
        pred_labels = np.array([prop.label for prop in pred_props])

        if len(gt_labels) == 0 and len(pred_labels) == 0:
            if verbose: print("  - No instances found in both GT and prediction maps.")
            seg_results = {
                "TP": 0, "FP": 0, "FN": 0,
                "Precision": 1.0, "Recall": 1.0,
                "AJI": 1.0, "PQ": 1.0, "DQ": 1.0, "SQ": 1.0,
            }
            return seg_results, {}
        if len(gt_labels) == 0 or len(pred_labels) == 0:
            tp = 0
            fp = len(pred_labels)
            fn = len(gt_labels)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            seg_results = {
                "TP": tp, "FP": fp, "FN": fn,
                "Precision": precision, "Recall": recall,
                "AJI": 0.0, "PQ": 0.0, "DQ": 0.0, "SQ": 0.0,
            }
            for k, v in seg_results.items():
                if verbose: print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
            return seg_results, {}

        gt_areas = np.array([prop.area for prop in gt_props])
        pred_areas = np.array([prop.area for prop in pred_props])
        valid_pixels = (gt_inst_map > 0) & (pred_inst_map > 0)
        gt_flat = gt_inst_map[valid_pixels]
        pred_flat = pred_inst_map[valid_pixels]
        max_gt_label = np.max(gt_labels) if len(gt_labels) > 0 else 0
        max_pred_label = np.max(pred_labels) if len(pred_labels) > 0 else 0
        intersection_matrix_sparse = sparse.coo_matrix(
            (np.ones(len(gt_flat)), (gt_flat, pred_flat)),
            shape=(max_gt_label + 1, max_pred_label + 1)
        ).toarray()
        intersection_matrix = intersection_matrix_sparse[gt_labels, :][:, pred_labels]

        # Convert intersections to IoU and greedily match predictions to GT instances.
        gt_areas_broadcast = np.expand_dims(gt_areas, axis=1)
        pred_areas_broadcast = np.expand_dims(pred_areas, axis=0)
        union_matrix = gt_areas_broadcast + pred_areas_broadcast - intersection_matrix
        iou_matrix = np.divide(
            intersection_matrix, union_matrix,
            out=np.zeros_like(intersection_matrix, dtype=float),
            where=union_matrix != 0
        )

        # PQ matching: one-to-one pairs with IoU >= threshold.
        hits = iou_matrix >= iou_threshold
        matched_pairs = {}
        matched_ious = []
        matched_pred_indices = set()
        for i in range(len(gt_labels)):
            hit_indices = np.where(hits[i, :])[0]
            if len(hit_indices) > 0:
                best_match_j = -1
                max_iou = -1
                for j in hit_indices:
                    if j not in matched_pred_indices and iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        best_match_j = j
                if best_match_j != -1:
                    matched_pairs[gt_labels[i]] = pred_labels[best_match_j]
                    matched_pred_indices.add(best_match_j)
                    matched_ious.append(max_iou)

        tp = len(matched_pairs)
        fp = len(pred_labels) - tp
        fn = len(gt_labels) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
        pq = dq * sq

        # AJI matching (Kumar et al.): each GT takes the unused pred with max Jaccard.
        # Matching order is greedy by GT index (the definition used by this implementation).
        # Denominator = matched unions + unmatched GT areas + unmatched pred areas.
        aji_used_pred = set()
        aji_inter_sum = 0.0
        aji_union_sum = 0.0
        for i in range(len(gt_labels)):
            best_j = -1
            best_iou = 0.0
            for j in range(len(pred_labels)):
                if j in aji_used_pred or intersection_matrix[i, j] == 0:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                aji_used_pred.add(best_j)
                aji_inter_sum += intersection_matrix[i, best_j]
                aji_union_sum += union_matrix[i, best_j]
            else:
                # Unmatched GT is paired with an empty prediction, so its union is its own area.
                aji_union_sum += gt_areas[i]
        for j in range(len(pred_labels)):
            if j not in aji_used_pred:
                aji_union_sum += pred_areas[j]
        aji = float(aji_inter_sum / aji_union_sum) if aji_union_sum > 0 else 0.0

        seg_results = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "AJI": aji,
            "PQ": pq,
            "DQ": dq,
            "SQ": sq,
        }
        for k, v in seg_results.items():
            if verbose:
                print(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")
        return seg_results, matched_pairs