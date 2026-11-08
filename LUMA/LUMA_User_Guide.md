# LUMA User Guide (General Edition)

**Purpose of this document**: a product-level guide for deployment, integration, and secondary development. It is not tied to any example dataset, directory layout, or task script; every path is written as a placeholder.

**Reading guide**: new users start with chapter 2 and workflows A/B in chapter 5; integrators and secondary developers read chapters 3, 4, and 8; tuning and troubleshooting are in chapters 6, 9, and 10.

---

## Contents

1. [Product overview](#1-product-overview)
2. [Quick start](#2-quick-start)
3. [Core concepts and data model](#3-core-concepts-and-data-model)
4. [API reference](#4-api-reference)
5. [Workflow examples](#5-workflow-examples)
6. [Training and inference internals](#6-training-and-inference-internals)
7. [Cell representations and analysis](#7-cell-representations-and-analysis)
8. [Result storage and export](#8-result-storage-and-export)
9. [Deployment, performance, and reproducibility](#9-deployment-performance-and-reproducibility)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limitations](#11-known-limitations)
- [Appendix A: Parameter quick reference](#appendix-a-parameter-quick-reference)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: Configuration templates](#appendix-c-configuration-templates)
- [Appendix D: Corrections relative to the Chinese edition](#appendix-d-corrections-relative-to-the-chinese-edition)

---

## 1. Product overview

### 1.1 Positioning

LUMA is a cell-analysis system built for **large microscopy images**. It chains "whole-image segmentation + cell-type recognition + single-cell representation + downstream analysis" into one reusable pipeline. It is aimed at the following situations:

- images far larger than a single-GPU memory budget (typically 2048x2048 and above, possibly multi-channel and multi-Z);
- you need **instance segmentation**, **in-situ cell type**, and **single-cell representations** from the same run;
- you need supervised classification/regression, morphological group-level attribution, dimensionality-reduction visualization, and trajectory analysis on top of one representation;
- you need to transfer an existing model to new data, a new modality, or a new laboratory.

### 1.2 Capability boundaries

| Supported | Not supported |
|---|---|
| Grid-based segmentation of large images, instance recovery, in-situ type prediction | General object detection / segmentation of non-microscopy images |
| Building a supervised dataset from images + instance masks and training on it | Training without any annotation |
| Inference on standalone large images, single-image batches, and time series | Automatic image-directory discovery and inference (image records must be given explicitly) |
| Extracting the SAM representation (1024-D with `vit_l`) and 136 morphological columns | End-to-end differentiable morphological learning |
| Supervised single-cell classification/regression, group-level ablation, CCA, UMAP/LDA/PHATE | A productized pure-clustering workflow for unlabeled data (a metadata column must be named as the "target") |
| Multi-frame tracking and trajectory visualization on top of laptrack | Tracking-error benchmarks (for example TRA) |

### 1.3 System composition

> The original document embeds an architecture figure named `LUMA4_architecture_overview.png`. That asset is **not present** in `scripts_formal/`, so the pipeline is described in text here.

Main chain: input image -> grid split -> input adaptation (2D degeneration or volume adapter) -> SAM ViT encoder (frozen / fully fine-tuned / LoRA) -> two UNETR decoders -> mask / HV / type heads -> half-margin cropping and stitching -> CP-dynamics instance recovery -> H5AD result.

Downstream chain: H5AD image record -> cell table (SAM representation + morphology) -> tracking -> supervised analysis -> visualization.

### 1.4 Inputs and outputs

| Stage | Input | Output |
|---|---|---|
| Dataset build | image files (single-channel / multi-channel / multi-Z) + 2D instance mask + metadata mapping | `GridImageDataset` (`data.h5` + `metadata.pkl`) |
| Training | Dataset + SAM weights (optionally a pretrained checkpoint) | `training_logs/<title>_<timestamp>/best_model.pth` and related files |
| Inference | the Dataset's test images, or standalone image records + model weights | result H5AD (`/uns/luma/images/...`), optionally an evaluation CSV |
| Cell table | Dataset (GT instances) or result H5AD (GT/predicted instances) | `cell_table.h5ad` (`X_sam`, `X_morph`, `obs`) |
| Analysis | cell-table H5AD + target column + filters | metrics, coordinates, and figures under `uns/runs/<run_name>/` |
| Export | analysis H5AD | SVG / PNG figures (representation or CSV export does not exist; see [Sec. 11](#11-known-limitations)) |

### 1.5 Terminology

- **Dataset**: the on-disk dataset managed by `GridImageDataset`, holding images, patches, GT masks, type maps, and the split.
- **Patch / Grid**: a sub-image after splitting a large image on a grid; adjacent patches overlap by a margin, and inference crops the half margin so the patches stitch back seamlessly.
- **Cell table**: an AnnData table with one row per cell, holding the SAM representation and 136 morphological columns.
- **Run**: the result set of one analysis configuration (target + kind + filters); the directory name is `{meta_key}|{target_kind}|{filter}`.
- **Representation**: in this document, the per-cell `X_sam` (1024-D with `vit_l`) and `X_morph` (136 columns) matrices.

---

## 2. Quick start

### 2.1 Runtime environment

| Item | Requirement |
|---|---|
| Python | 3.10+ (use whatever the existing environment provides) |
| Deep learning | PyTorch, MONAI, timm; SAM weight files |
| Images and tables | `tifffile`, `scikit-image`, `opencv-python`, `pandas`, `anndata`, `h5py` |
| Analysis | `scikit-learn`, `umap-learn`, `phate`; PNG export needs `cairosvg` |
| Tracking | `laptrack` |
| Hardware | a GPU is recommended for training/inference; morphology, dimensionality reduction, and tracking run on CPU |

After installation, run one **import and weight self-check**:

```python
import torch, anndata, monai, timm, laptrack
from scripts_formal.LUMAExperiment import CellExperiment

print("torch cuda:", torch.cuda.is_available())
print("CellExperiment ok:", CellExperiment)
```

### 2.2 Deployment checklist

Before pointing LUMA at new data, confirm each item:

1. **Install root**: `LUMA4_ROOT` is resolved as the environment variable `LUMA4_ROOT` when it is set, otherwise as the parent of the `scripts_formal` package. All default paths (SAM weights, datasets, output directories) are derived from it, so either set the variable or install the package in place.
2. **SAM weights**: the weight matching `backbone_type` exists (`vit_h` / `vit_l` / `vit_b`); the packaged defaults look under `<LUMA4_ROOT>/sam_checkpoints/`.
3. **Channel convention**: decide which biological channel is used for training/inference (`target_image_channel` / `target_stitch_channel` / `target_channel`); do not rely on the default index.
4. **Input dimensionality**: use `input_z_size=1` for 2D data; for multi-Z data use `input_z_size>1` and keep training and inference consistent.
5. **Class count**: `num_class` / `num_instance_class` must equal the number of type-head channels (baseline included) and must match the checkpoint.
6. **Preprocessing**: the dataset's `normalize` and `rescale` must match the values used at inference time. Note that `infer` does not expose `rescale` (it is fixed at 1.0 for standalone images), so build datasets with `rescale=1.0` if you want to reproduce that path.
7. **Grid geometry**: the Dataset entry reads `stitch_info` from inside the file; the standalone-image entry must be given the training-time `grid_size` / `grid_target_size` / `margin_ratio` explicitly.
8. **Split**: `build_dataset` puts every image into train and leaves test empty by default; if you want to infer, pass `train_image_names` explicitly so the remaining images become test.
9. **Post-processing**: the default post-processor filters instances by major-axis length (`conduct_size_filter=True`, `min_major_axis_length` derived from `min_mask_size=3000`, about 61.8 px). If your cells are smaller, set `postprocessor_params={"conduct_size_filter": False}` on `CellExperiment` (see [Sec. 6.5](#65-cp-dynamics-instance-recovery)).

### 2.3 Minimal example: inference on one large image

The example below uses only `CellExperiment`. The model must already be trained, and the training configuration must match the inference configuration.

```python
import tifffile
from scripts_formal.LUMAExperiment import CellExperiment

image = tifffile.imread("/data/images/large_001.tif")   # (Z,H,W) or (H,W)

exp = CellExperiment(trainer_params={
    "backbone_type": "vit_l",
    "num_class": 1,                # only the baseline channel when there is no type supervision
    "use_prediction_head": True,
    "input_z_size": 1,
    "use_image_skip": True,
    "vit_adapt": "none",
})

exp.infer(
    anndata_path="/data/results/large_001.h5ad",
    infer_source="images",
    image_records=[{
        "image_name": "large_001",
        "image": image,
        "meta_map": {},            # optional: image-level metadata
    }],
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    model_path="/data/models/best_model.pth",
    overwrite_anndata=True,
    grid_size=(2, 2),
    grid_target_size=(1024, 1024),
    margin_ratio=0.1,
    input_z_size=1,
    normalize=False,               # must match dataset building / training
    target_stitch_channel=[0],
    num_instance_class=1,
)
```

Product: `/data/results/large_001.h5ad`, with `/uns/luma/images/large_001/` containing mask, HV, type map, instance map, patch embeddings, and the stitching geometry.

See chapter 5 for the complete workflows.

---

## 3. Core concepts and data model

### 3.1 Images and channel conventions

Every biological channel is stored in memory and on disk as `(Z, H, W)`:

- 2D image: `Z=1`;
- one multi-channel file: set `image_axis_order` to `cyx` / `yxc` / `czyx`;
- several single-channel files: pass a list as `image_paths`, one channel per file;
- a single 3D array from one file is interpreted as `ZYX` (not `CYX`) in `auto` mode, unless the last dimension happens to be 1/3/4 (then it is read as `YXC`); if it really is `(C,H,W)`, write `image_axis_order="cyx"` explicitly.

The mask must be a **single-channel 2D instance label image** with the same H/W as the image; label 0 is background and positive integers are instance IDs.

### 3.2 Grid, patch, and margin

The whole image is split into `grid_size=(n_rows, n_cols)` patches, each carrying a `margin_ratio` overlap:

```
patch_h = H / ((1 - margin_ratio_h) * n_rows + margin_ratio_h)
margin_h = patch_h * margin_ratio_h
adjacent patch step = patch_h - margin_h
```

`patch_h` / `margin_h` are truncated to integers, so the stitched canvas (`(patch_h - margin_h) * n_rows + margin_h`) can differ from the source by a few pixels; when it does, the image is resized to the exact stitched size before splitting.

At inference every patch prediction is first resized back to the native patch size, then the half margin is cropped away, and the patch is added (`+=`) into the full image on the grid; adjacent valid regions are complementary, so **no weighted blending is needed**.

`stitch_info` has six fields: `patch_h`, `patch_w`, `margin_h`, `margin_w`, `original_h`, `original_w`. Both evaluation and cell-table extraction treat the center region left after removing the half margin as the **valid region**.

With `grid_size="auto"` the grid is estimated from `field_target_size`.

### 3.3 Dataset: `GridImageDataset`

Four-level structure:

```
GridImageDataset                      on-disk container (metadata.pkl + data.h5)
 |-- grid_image: dict[str, GridImage] one record per image
 |    |-- patch_dataset: GridDataSet  patch-level torch Dataset
 |    `-- sc_dataset: scDataSet       single-cell crop Dataset
 `-- merged_dataset['patch'|'sc']     Dataset merged over the training images
GridImageLite                         pure in-memory inference variant, writes nothing
```

On-disk layout:

```
<dataset_dir>/
  data.h5
  metadata.pkl
```

Each image group inside `data.h5`:

```
/<image_name>/
  attrs: meta_info, grid_size, cyto_type_class_weights,
         input_z_size, image_channel_count, channel_shapes,
         channel_axis_orders, channel_z_sizes, channel_source_paths
  sc_stack/                          optional: single-cell crops
  patch_<i>_<j>/
    raw_channels/channel_<c>/image  (Z, patch_h, patch_w)
    mask                            (1, H, W) instance labels
    hv_map                          (4, H, W): background/foreground + X/Y flow
    type_map                        (1+L, H, W): baseline + L type channels
```

Dataset-build defaults (`GridImageDataset.default_loading_option`, overridable through `loading_options`):

| Parameter | Default | Meaning |
|---|---|---|
| `grid_size` | `(2, 2)` | grid rows/columns; may be `"auto"` |
| `grid_target_size` | `(1024, 1024)` | patch size fed to the model |
| `field_target_size` | `(1536, 1536)` | target field of view when `grid_size="auto"` |
| `margin_ratio` | `0.1` | overlap ratio between adjacent patches |
| `rescale` | `1.0` | whole-image scaling factor |
| `normalize` | `False` | whether to z-score the whole image into [0,1] |
| `input_z_size` | `10` | fixed Z depth on disk / for training; set `1` for 2D data |
| `generate_classification` | `{}` | rules that generate the type map |
| `type_mode` | `"inst_wise"` | instance-level or signal-level types |
| `cell_target_size` | `(224, 224)` | single-cell crop size |
| `extension_rate` | `1.0` | bbox expansion ratio for single cells |
| `object_attri_limit` | `{"axis_major_length": (85, None)}` | drop too-small instances while building |
| `defined_attri_func` | `None` | extra `regionprops` properties used by `object_attri_limit` |
| `delete_edges` | `False` | erase mask edges before splitting |
| `image_axis_order` | `"auto"` | how channel/Z axes are interpreted |

`GridImageDataset` always injects an empty `batch_name` mapper into `meta_map`; it is filled in build order (see [Sec. 3.7](#37-metadata-meta_info-and-meta_map)).

### 3.4 Inference result H5AD

An inference result is one H5AD whose image-level arrays live under `/uns/luma/images/<image_id>/`:

```
attrs/                image_id, run_id, image_name, n_types,
                      grid_rows, grid_cols, meta_info
prediction/           mask, h_map, v_map, pred_type_cyto_map,
                      pred_instance_map
gt/                   gt_instance_map, gt_type_cyto_map, gt_raw_image
embedding_indexer     (4, H, W): pixel -> patch row/col + token row/col
stitch_info           six-field stitching geometry
postprocess_params    thresholds and CP-dynamics settings
model_info            model_class
patch_embeddings/     patch_<i>_<j>: frozen-SAM patch embedding
```

`/uns/luma/meta_map` is written while stitching. Note that every image overwrites it, so after a batch inference it holds **only the last image's** mapping; the authoritative per-image metadata is each record's `attrs.meta_info`. Evaluation results live in `/uns/luma/evaluations/<image_id>/result`.

### 3.5 Cell table

The cell table is a standard AnnData:

| Node | Content |
|---|---|
| `X` / `obsm["X_sam"]` | SAM mean-pooled representation (`1024` columns with `vit_l`) |
| `obsm["X_morph"]` | 136 canonical morphological columns (equivalent-radius normalization already applied at extraction) |
| `obsm["X_morph_raw"]` | same matrix as `X_morph` (compatibility key) |
| `obs` | cell-level metadata, indexed by `obs_name`, formatted `{image_name}__p{rr}_{cc}__inst_{id:06d}` |
| `var` | `feature_id`, `feature_source="sam"` |
| `uns["morph_columns"]` / `uns["morph_raw_columns"]` | canonical column names |
| `uns["morph_feature_groups"]` | morphological column to physical quantity group |
| `uns["cell_table_config"]` | extraction configuration (the prediction target is not stored here) |
| `uns["skipped_images"]` | skip log |
| `uns["run_index"]` | registered analysis runs |

Base `obs` columns: `obs_name`, `image_name`, `image_id`, `patch_row`, `patch_col`, `split`, `instance_id`, `gt_cell_type`, `in_valid_region`, `centroid_y/x`, `global_y/x`, `bbox_y1/x1/y2/x2`, `ext_bbox_y1/x1/y2/x2`, `image_median_bbox_diag`, `embed_region_h/w/n`. The keys of each image's `meta_info` and of `meta_map` are written as columns of the same name.

> `gt_cell_type` is the **instance-level majority vote of the type map that belongs to the selected instance source**: the GT type when `instance_source="gt"`, the model's predicted type when `instance_source="pred"`. The Dataset-entry GT type map is generated from the intensity thresholds in `generate_classification` (see [Sec. 11](#11-known-limitations), item 6). The column is stored as an **integer class index**; filters such as `obs_equals` / `exclude_obs_in` compare it by decoding through `type_index_to_name`, so `{"gt_cell_type": ("untyped",)}` works even though `obs` holds integers.

### 3.6 Analysis runs

One analysis maps to a stable directory name:

```
run_name = "{meta_key}|{target_kind}|{filter}"
filter   = sorted obs_equals("key=value") and exclude_obs_in("key!=a,b") joined by "|"; "all" when empty
```

Everything for that run is written under `uns/runs/<run_name>/`, stage by stage:

| Key | Stage |
|---|---|
| `run_config` | orchestration |
| `type_prediction/{metrics, metrics_per_class, delta_metrics, metrics_summary, fov_time_mean_metrics, proj64_seed0/...}` | B |
| `group_inventory`, `group_ablation/{metrics, metrics_per_class, ablation_delta, ablation_only}`, `group_cca/cca_by_group` | C |
| `summary/{wide, input_comparison}` | D |
| `coords/{emb,morph}`, `visualization/figures/*`, `visualization/scatter_limits/*`, `visualization/lda_geometry` | E |
| `visualization/{track_sample, track_coords, track_flow_geometry}` plus the trajectory figures | F |

The same cell table can carry several runs (different targets, different filters) without overwriting each other.

### 3.7 Metadata: `meta_info` and `meta_map`

| | `meta_map` | `meta_info` |
|---|---|---|
| Level | dataset-level mapping rules | image-level encoded key/value pairs |
| Stored in | `GridImageDataset.meta_map`; H5AD `/uns/luma/meta_map` | each `GridImage.meta_info`; H5 `attrs`; H5AD image `attrs.meta_info` |
| Purpose | generate the encoding from `image_name` | the encoded result, written into same-named `obs` columns in stage A |

Every key of `meta_map` is an **output column name**; its value can be:

1. **A dictionary mapping**: `{substring or full image_name: value}`, resolved by longest match; values may be strings, integers, or floats.
2. **A registered function name**: a string starting with `METAMAP_`, which calls the encode/decode pair in `metamap_function_pairs.py`.
3. `batch_name`: a special key numbered automatically in build order; the user does not supply it.

Example (turning a CSV into per-image metadata):

```python
import pandas as pd

meta = pd.read_csv("/data/metadata.csv").set_index("image_name")
meta_map = {col: meta[col].to_dict() for col in meta.columns}
# result looks like {"condition": {"sample_01": "control", ...},
#                   "time_h":   {"sample_01": 0.0, ...}}
```

> Caveat: every image must hit one key of a dictionary mapping; otherwise dataset building raises `KeyError`.

### 3.8 Data splits

| Level | Definition | Stored in |
|---|---|---|
| Image-level train/test | `get_merged_dataset(train_list)`: the given list is train, everything else is test | `metadata.pkl` |
| Patch-level train/val | during training the merged patch Dataset is split by `random_split(train_val_ratio=0.8)` | current DataLoader only |
| Cell-table split | stage A writes the image-level split into `obs["split"]` with values `"train"` / `"test"` | main-table `obs` |
| MLP fit/val | re-split inside train by `val_fraction` | current training only |

When the H5AD entry has no Dataset path, at least `train_image_names` must be supplied; giving only `test_image_names`, or neither, makes stage A fail (there is no "test-only" entry point yet; see [Sec. 11](#11-known-limitations)).

### 3.9 Compatibility contract

When applying a checkpoint to new data, the following must match the training run, otherwise the results are not interpretable:

| Item | Impact |
|---|---|
| `backbone_type` | SAM ViT variant; bound to the weight shapes |
| `input_z_size` | 1 means 2D degeneration; >1 enables the volume adapter |
| `num_class` | number of type-head channels (baseline included) |
| `use_prediction_head`, `use_image_skip`, `vit_adapt`, `lora_rank` | model structure and weight key names |
| `target_image_channel` / `target_stitch_channel` | input channel selection |
| `normalize`, `rescale` | pixel distribution |
| `grid_size`, `grid_target_size`, `margin_ratio` | patch geometry and stitching |
| `generate_classification` and the type index mapping | semantics of `gt_cell_type` |
| `image_axis_order` | channel/Z axis interpretation |
| post-processor settings (`min_mask_size`, `conduct_size_filter`, ...) | which instances survive |

---

## 4. API reference

### 4.1 `CellExperiment` overview

`CellExperiment` is the only recommended product-level entry point. It stores parameters and calls the library implementation; it does not re-implement any algorithm.

| Method | Purpose | Main product |
|---|---|---|
| `build_dataset` | build a `GridImageDataset` from images + masks | `data.h5`, `metadata.pkl` |
| `train_model` | train SamUNetR | `best_model.pth`, `backbone_2d.pth`, `adapter_3d.pth`, logs |
| `infer` | inference on Dataset test images or standalone images | result H5AD; optionally an evaluation CSV |
| `extract_cell_table` | stage A: extract the cell table | `cell_table.h5ad`, or an in-place table swap |
| `track_cells` | laptrack tracking | `obs.track_id`, `/uns/luma/tracking` |
| `analyze_cell_features` | stages B to F | `uns/runs/<run_name>/` |
| `export_figures` | export stored figures | SVG / PNG |

Utility methods: `configure_trainer`, `collect_infer_jobs`, `predict_type`, `diagnose_contrib`.

> `LUMAExperiment.main()` is only a skeleton that calls the seven methods behind seven boolean switches; all switches default to off, and its `build` branch passes an empty image-record list, so it is not usable as a product entry point. When integrating, call the `CellExperiment` methods directly and pass every path and parameter explicitly.

### 4.2 `build_dataset`

```python
build_dataset(
    output_dir: str,
    image_records: list[dict] | None = None,
    image_path_template: str | list[str] | None = None,
    mask_path_template: str | list[str] | None = None,
    path_kw: list | None = None,
    mask_kw: list | None = None,
    image_names: list[str] | None = None,
    meta_map: dict | None = None,
    loading_options: dict | None = None,
    train_image_names: list[str] | tuple[str, ...] | None = None,
    overwrite: bool = True,
    **load_kwargs,
) -> str
```

Two ways to build a dataset:

**Option 1: one record per image (recommended)**

```python
image_records=[{
    "image_name": "sample_01",
    "image_paths": ["/data/images/sample_01_ch0.tif",
                    "/data/images/sample_01_ch1.tif"],
    "mask_path": "/data/masks/sample_01.tif",
}]
```

**Option 2: path templates in bulk**

```python
image_path_template="/data/images/{}.tif"
mask_path_template="/data/masks/{}.tif"
path_kw=[("sample_01",), ("sample_02",)]
image_names=["sample_01", "sample_02"]
```

Key parameters:

| Parameter | Meaning |
|---|---|
| `loading_options` | overrides `GridImageDataset.default_loading_option`; only known keys are accepted, an unknown key raises `KeyError` |
| `train_image_names` | image-level train list; **when omitted every image goes to train and test stays empty** |
| `overwrite` | `True` rebuilds `data.h5` / `metadata.pkl` by first building in a staging directory inside the same folder and replacing the files only on success, so a failure keeps the previous version; `False` writes directly into the target directory (append semantics) |
| `**load_kwargs` | per-image overrides of `loading_options` (both options support this) |

Requirements: every image needs a 2D instance mask, and image and mask must share the same H/W.

> There is no `image_dir` / glob parameter on `build_dataset`. However, `image_paths` accepts a directory path, which `GridImage` expands into the `.png` / `.jpg` / `.jpeg` / `.tif` / `.tiff` files it contains; the mask stays a single file per image.

### 4.3 `train_model`

```python
train_model(
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
) -> tuple[dict, str]
```

`target_image_channel` falls back to `[2]` when not given.

Three training modes:

| Mode | Configuration | Trainable parameters |
|---|---|---|
| Frozen backbone | `vit_adapt="none"`, `train_encoder=False` | decoders + three heads + volume adapter (the adapter only exists when `input_z_size>1`) |
| Full / partial fine-tuning | `vit_adapt="none"`, `train_encoder=True` or a list of block indices | the above + the unfrozen ViT blocks |
| LoRA | `vit_adapt="lora"`, `lora_rank=4` | LoRA branches only + decoders + three heads + adapter; `train_encoder` is ignored and the ViT stays frozen |

Model selection and early stopping: the model is saved as `model_save_name` whenever the validation total loss improves, and training stops after `patience` consecutive epochs without improvement. When no validation split is available at all, the model is saved every epoch and the returned path is `*_final.pth`.

Product directory `training_logs/<title>_<timestamp>/`:

- `best_model.pth`: the complete state dict;
- `backbone_2d.pth`: the `vit` + `upsampler` + `predictors` components (no volume adapter);
- `adapter_3d.pth`: written when a volume adapter exists;
- `log.txt`, `loss.png`.

### 4.4 `infer`

```python
infer(
    anndata_path: str,
    infer_source: str = "dataset",          # "dataset" | anything else (standalone images)
    dataset_path: str = "",
    image_names: list[str] | tuple[str, ...] | None = None,
    image_records: list[dict] | None = None,
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
) -> str
```

`target_stitch_channel` falls back to `[2]` when not given. `num_instance_class` is forwarded as the trainer's `num_class`, so it must equal the type-head channel count of the checkpoint.

Inference sources:

| Entry | Input | Behaviour |
|---|---|---|
| `infer_source="dataset"` | `dataset_path` (+ optional `image_names`) | reads the Dataset's images and GT; when `image_names` is empty the test names are used |
| anything else | `image_records` | each record supplies `image_name` plus `image` (an ndarray) or `image_path` (read with `tifffile`) |

Fields of a standalone-image record:

| Field | Required | Meaning |
|---|---|---|
| `image_name` | yes | unique image name; converted into the H5AD key through `sanitize_key` |
| `image` | one of the two | ndarray; `(C,Z,H,W)` (multi-channel) or `(Z,H,W)` (single-channel) recommended |
| `image_path` | one of the two | a single image file path, read internally with `tifffile.imread` |
| `meta_map` | no | metadata mapping for this image, written to `/uns/luma/meta_map` and the image's `attrs.meta_info` |

Behaviour of standalone-image inference:

1. the ndarray is wrapped in `GridImageLite` (nothing written to disk, no GT, no augmentation);
2. the grid is cut by `grid_size` / `grid_target_size` / `margin_ratio` (`rescale` is fixed at 1.0 because `infer` does not expose it);
3. patches are forwarded batch by batch; for LoRA models the frozen-ViT features are written to `patch_embeddings`;
4. patches are cropped by the half margin and stitched back into the full image; the background channel of the type map is clamped to `null_hypothesis_threshold` and then `argmax` is applied;
5. CP-dynamics recovers instances;
6. the H5AD is written.

Key thresholds:

| Parameter | Default | Meaning |
|---|---|---|
| `target_stitch_channel` | `[2]` | biological channel used for inference and stitching |
| `stitch_batch_size` | 2 | patches per forward batch |
| `object_size` | 10 | minimum instance size in CP-dynamics |
| `max_cpimage_size` | 2048 | long-edge cap of the dynamics canvas |
| `flow_threshold` | 0.5 | flow-field consistency QC |
| `mask_thres` | 0.5 | cell-probability threshold |
| `null_hypothesis_threshold` | 0.15 | lower clamp of the type background channel |
| `evaluate_flag` | `False` | evaluation is only possible on the Dataset entry, which has GT |
| `eval_region` | `"auto"` | crop the stitching margin, then evaluate |

> `normalize` defaults to `True` in `infer` while dataset building defaults to `False`. For standalone-image inference always set it explicitly to the training-time value.

> The default post-processor also removes instances whose major axis is shorter than `sqrt(min_mask_size/pi)*2` (about 61.8 px with `min_mask_size=3000`). Configure it through `CellExperiment(postprocessor_params={...})`; see [Sec. 6.5](#65-cp-dynamics-instance-recovery).

### 4.5 `extract_cell_table`

```python
extract_cell_table(
    source_kind: str,                    # "dataset" | "h5ad"
    dataset_path: str = "",
    h5ad_path: str = "",
    instance_source: str = "gt",        # the dataset entry supports "gt" only
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
    extra_config: dict | None = None,
    output_root: str = "",
    meta_map: dict | None = None,
    n_morph_workers: int = 7,
    max_cells_per_image: int | None = None,
    sample_seed: int = 0,
    attach_tracks: bool = True,
    device: str = "cpu",
    n_jobs: int = 1,
) -> dict
```

`target_channel` falls back to `[2]` when not given.

Two entries:

| Entry | Data source | Embedding source | Instance source | Table write mode |
|---|---|---|---|---|
| `"dataset"` | intensity images, GT instance maps, and `type_map` from `data.h5` | re-encodes patches with the frozen SAM ViT | GT only | writes the whole file `output_root/cell_table.h5ad` |
| `"h5ad"` | instance maps, intensity images, and `patch_embeddings` from `/uns/luma/images/<image_id>` | concatenates the stored patch embeddings and pools them with an integral image | GT or pred | `write_table_elements` swaps the table in place, keeping `/uns/luma` |

Common rules:

- **Valid-region filter**: only cells whose centroid falls inside the stitch valid region are kept.
- **Extended bbox**: each bbox is expanded by the median bbox diagonal of the image's valid-region cells times `extension_rate`; it is used for representation pooling and neighbourhood morphology.
- **Sampling**: `max_cells_per_image` samples per image without replacement after the valid-region filter and before morphology; the Dataset entry requires a single patch per image (a multi-patch image raises immediately), the H5AD entry samples over the whole image; it is mutually exclusive with `attach_tracks` (`CellExperiment` switches `attach_tracks` off automatically when sampling is on).
- **Attach tracks**: `attach_tracks=True` only applies to the H5AD entry and copies existing columns such as `track_id` into `obs`; when the file already contains tracks, `uns/tracking_quality` and `uns/tracking_quality_summary` are refreshed as well.
- **Parallel morphology**: `n_morph_workers` is passed to the morphology thread pool.

> The H5AD entry requires an H5AD produced by LUMA inference that contains `patch_embeddings`. If an external model output lacks these fields, stage A cannot work.

### 4.6 `track_cells`

```python
track_cells(h5ad_path: str, config_path: str, **tracking_overrides) -> dict
```

- the input H5AD must already contain a cell table in `obs` (run `infer` + `extract_cell_table` first);
- a sampled cell table (`cell_table_config.max_cells_per_image` non-empty) is rejected;
- when `fov_column` exists (default `fov_id`) tracking runs FOV by FOV, otherwise once over the whole table;
- the result is written back into the same H5AD's `obs` (`track_id`, `track_order`) and `/uns/luma/tracking/`, and `uns/tracking_quality` / `uns/tracking_quality_summary` are appended;
- `tracking_overrides` override same-named JSON fields; `laptrack_costs` works both as a dictionary and as flat keys.

`config_path` is **required** here, even though `CellTracker.track_anndata` and `elements_tracker.main()` default it to the packaged `config/laptrack_config.json`. `LUMAExperiment.main()` defaults its tracking switch to `config/mdck2_timeseries_laptrack_config.json`.

Configuration fields are listed in [Appendix C](#appendix-c-configuration-templates).

### 4.7 `analyze_cell_features`

```python
analyze_cell_features(
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
    obs_equals: dict | None = None,
    exclude_obs_in: dict | None = None,
    target_scale_max: float | None = None,
    run_extract_flag: bool = True,
    run_type_prediction_flag: bool = True,
    run_contrib_diagnosis_flag: bool = True,
    n_mean: int = 1024,
    heads: tuple[str, ...] = ("proj64_mlp",),
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
) -> Path
```

Behaviour:

1. if `run_extract_flag=True`, stage A runs first; otherwise `cell_table.h5ad` is read directly;
2. `run_config` and `run_index` are written;
3. stage B (`run_type_prediction_flag`): supervised classification/regression on the three input sets;
4. stage C (`run_contrib_diagnosis_flag`): morphological group-level ablation and CCA;
5. stage D: summary tables (when B produced metrics);
6. stages E/F (`run_visualization_flag`): dimensionality reduction, heatmaps, trajectory projection.

**Project-specific defaults still present in the code.** The signature above is the real one, so several parameters silently fall back to project-specific values when left empty. Override them explicitly in a new deployment:

| Parameter | Effective value when empty |
|---|---|
| `dataset_path` | `<LUMA4_ROOT>/datasets/<data_name>` (dataset entry only; the H5AD entry never falls back) |
| `sam_checkpoint_path` | `<LUMA4_ROOT>/sam_checkpoints/sam_vit_l_0b3195.pth` |
| `target_channel` | `[2]` |
| `type_index_to_name` | `{0: "rpe", 1: "hesc"}` |
| `type_name_to_index` | `{"rpe": 0, "GFP-": 0, "hesc": 1, "GFP+": 1}` |
| `output_root` | `<LUMA4_ROOT>/embedding_results/dataset_result/<data_name>/cell_feature_analysis` (used by the dataset entry; the H5AD entry writes in place) |
| `lda_label_key` | `stratify_eval_key`, falling back to `meta_key` for continuous targets other than `gt_cell_type` |
| `data_name` | `"a4bad_cpdynamics_hv"` (also used in the paths above) |

The target is specified by three items:

| Parameter | Meaning |
|---|---|
| `meta_key` | target column name. For instance-level types use the reserved name `"gt_cell_type"`; for image-level attributes use a `meta_map` key |
| `target_kind` | must be given explicitly as `"discrete"` or `"continuous"` |
| `meta_map` | optional; empty when the source already carries the mapping, supply it when it is missing or must be overridden |

> `type_index_to_name` and `type_name_to_index` are needed whenever the target is `gt_cell_type`, because the stored column holds integer class indices: the name map is used to decode, filter, and stratify.

### 4.8 `export_figures`

```python
export_figures(
    h5ad_path: str,
    export_dir: str,
    figure_format: str = "svg",          # "svg" | "png"
    dpi: int = 200,
    keys: list[str] | None = None,
    run_name: str | None = None,
    meta_key: str = "",
    target_kind: str = "",
    obs_equals: dict | None = None,
    exclude_obs_in: dict | None = None,
) -> None
```

- it only reads the stored SVG bytes; it never re-runs a reduction and never redraws;
- `figure_format="png"` rasterizes with `cairosvg`;
- when `run_name` is omitted it is recomputed from `meta_key` / `target_kind` / the filters.

### 4.9 Utility methods

| Method | Purpose |
|---|---|
| `configure_trainer(...)` | build or reload `LUMATrainer`, loading weights when asked |
| `collect_infer_jobs(...)` | turn Dataset test names or standalone image records into an inference job list |
| `predict_type(...)` | thin stage-B wrapper (you must build `cell_table` and `prep_kwargs` yourself) |
| `diagnose_contrib(...)` | thin stage-C wrapper |

---

## 5. Workflow examples

### 5.0 Choosing a workflow

| Your starting point | Recommended workflow | Training needed | GT mask needed |
|---|---|---|---|
| Several large images + masks + metadata, from scratch | A | yes | yes |
| A trained model and only large images; need segmentation/type/representation | B | no | no |
| A trained model; batch inference over several standalone images | C | no | no |
| A dataset with segmentation GT; only need cell representations and analysis | D | no | yes |
| An existing LUMA result H5AD; want representation analysis | E | no | no (pred instances) |
| Time series data needing tracking and trajectory analysis | F | no | no |
| Transfer an existing model to new data/modality | G | optional fine-tuning | yes (for fine-tuning) |

### 5.1 Workflow A: from large images + masks + metadata to a complete analysis

**Applies to**: building a dataset from scratch, training the main model, inferring, extracting the cell table, and running representation analysis.

**Prerequisites**:

- every image has a 2D instance mask;
- the channel index used for training is known;
- image-level metadata (condition, time, batch, ...) is known;
- it is decided which images are train and which are test.

**Step 1: define the metadata**

```python
import pandas as pd

meta = pd.read_csv("/data/my_project/metadata.csv").set_index("image_name")
# split is the dataset-split column; it is not cell-table metadata. Convert the others to per-image maps.
meta_map = {col: meta[col].to_dict() for col in meta.columns if col != "split"}
train_images = meta.index[meta["split"] == "train"].tolist()
```

`metadata.csv` needs at least an `image_name` column; the remaining columns become `obs` columns of the cell table.

**Step 2: build the dataset**

```python
from scripts_formal.LUMAExperiment import CellExperiment

DATASET = "/data/my_project/dataset"

exp = CellExperiment(trainer_params={
    "backbone_type": "vit_l",
    "num_class": 3,                 # 1 baseline + 2 supervised types
    "use_prediction_head": True,
    "input_z_size": 1,              # 2D
    "use_image_skip": True,
    "vit_adapt": "none",
})

exp.build_dataset(
    output_dir=DATASET,
    image_records=[
        {
            "image_name": name,
            "image_paths": [f"/data/my_project/images/{name}.tif"],
            "mask_path": f"/data/my_project/masks/{name}.tif",
        }
        for name in meta.index
    ],
    meta_map=meta_map,
    loading_options={
        "grid_size": (2, 2),
        "grid_target_size": (1024, 1024),
        "margin_ratio": 0.1,
        "input_z_size": 1,
        "image_axis_order": "auto",
        "normalize": False,
        "generate_classification": {
            "channel": [0, 1],      # one type per biological channel
            "threshold": ["auto", "auto"],
        },
        "type_mode": "inst_wise",
    },
    train_image_names=train_images,  # the remaining images become test
)
```

**Step 3: train**

```python
history, best_model = exp.train_model(
    dataset_path=DATASET,
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    log_dir="/data/my_project/training_logs",
    title="my_project",
    epochs=100,
    batch_size=1,
    accumulation_steps=8,
    learning_rate=5e-5,
    encoder_learning_rate=1e-6,
    target_image_channel=[0],
    train_encoder=True,
    loss_type_hv="cpdynamics_hv",
    hv_loss_weight=1.0,
    patience=20,
)
```

**Step 4: inference + evaluation**

```python
RESULT = "/data/my_project/result.h5ad"

exp.infer(
    anndata_path=RESULT,
    infer_source="dataset",
    dataset_path=DATASET,
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    model_path=best_model,
    num_instance_class=3,
    target_stitch_channel=[0],
    input_z_size=1,
    evaluate_flag=True,
    overwrite_anndata=True,
)
```

Products:

- `RESULT`: the image records;
- `result_evaluation_summary.csv` next to `RESULT` (per-image segmentation/classification metrics).

**Step 5: extract the cell table**

```python
exp.extract_cell_table(
    source_kind="h5ad",
    h5ad_path=RESULT,
    dataset_path=DATASET,          # used to recover the split
    instance_source="pred",        # predicted instances; use "gt" for GT instances
    target_channel=[0],
    type_index_to_name={0: "untyped", 1: "type_a", 2: "type_b"},
    type_name_to_index={"untyped": 0, "type_a": 1, "type_b": 2},
    attach_tracks=False,
    device="cuda",
    n_jobs=4,
)
```

**Step 6: representation analysis**

> Semantic warning: with `instance_source="pred"` in step 5, `gt_cell_type` holds the model's predicted type, so classifying by it is a self-consistency check ("can the representation reproduce the segmentation/classifier output?"). For a biological interpretation against **manual GT types**, re-extract with `instance_source="gt"` (the H5AD must carry GT maps), or merge an external type table into `obs` yourself.

Example 1: discrete classification by cell type.

```python
exp.analyze_cell_features(
    source_kind="h5ad",
    data_name="my_project",
    h5ad_path=RESULT,
    dataset_path=DATASET,
    run_extract_flag=False,
    instance_source="pred",
    target_channel=[0],
    meta_key="gt_cell_type",
    target_kind="discrete",
    exclude_obs_in={"gt_cell_type": ("untyped",)},
    type_index_to_name={0: "untyped", 1: "type_a", 2: "type_b"},
    type_name_to_index={"untyped": 0, "type_a": 1, "type_b": 2},
    stratify_eval_key="condition",
    device="cuda",
    n_jobs=4,
)
```

Example 2: regression against an image-level metadata column (same cell table, another run).

```python
exp.analyze_cell_features(
    source_kind="h5ad",
    data_name="my_project",
    h5ad_path=RESULT,
    dataset_path=DATASET,
    run_extract_flag=False,
    instance_source="pred",
    target_channel=[0],
    meta_key="time_h",
    target_kind="continuous",
    target_scale_max=48.0,
    lda_label_key="time_h",
    label_unit="h",
    extra_metrics=("mae", "spearman"),
    device="cuda",
    n_jobs=4,
)
```

**Step 7: export the figures**

```python
exp.export_figures(
    h5ad_path=RESULT,
    export_dir="/data/my_project/figures",
    figure_format="png",
    run_name="gt_cell_type|discrete|gt_cell_type!=untyped",  # or the name built from your real filters
)
```

`run_name` can be computed with `FeatureWriter.build_run_name`; if you are unsure, omit it first, because the analysis log prints the actual value.

**Workflow A checkpoints**:

- do `train_names` / `test_names` match the expectation after building;
- does the validation loss decrease in the training log, and did early stopping trigger;
- does the number of images in the result H5AD equal the number of test images;
- are `n_cells` and the `obs["split"]` distribution of the cell table plausible;
- are `target_kind`, `continuous_value_map`, and `obs_equals` in `run_config` correct.

### 5.2 Workflow B: a trained model, GridImageLite inference directly on a large image

**Applies to**: the model is trained; the new data has images only, no masks; only segmentation/type/representation are needed.

**Prerequisites**:

- the model weights match the training configuration (see Sec. 3.9);
- the index of each biological channel is known;
- it is known whether the data is 2D or multi-Z.

**Example**:

```python
import tifffile
from scripts_formal.LUMAExperiment import CellExperiment

image = tifffile.imread("/data/new/large_001.tif")   # (Z,H,W)

exp = CellExperiment(trainer_params={
    "backbone_type": "vit_l",
    "num_class": 1,
    "use_prediction_head": True,
    "input_z_size": 1,
    "use_image_skip": True,
    "vit_adapt": "none",
})

exp.infer(
    anndata_path="/data/new/results.h5ad",
    infer_source="images",
    image_records=[{
        "image_name": "large_001",
        "image": image,
        "meta_map": {"condition": {"large_001": "treated"}},
    }],
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    model_path="/data/models/best_model.pth",
    overwrite_anndata=True,
    # grid and preprocessing must match training
    grid_size=(2, 2),
    grid_target_size=(1024, 1024),
    margin_ratio=0.1,
    normalize=False,
    input_z_size=1,
    target_stitch_channel=[0],
    num_instance_class=1,
    # post-processing
    object_size=10,
    flow_threshold=0.5,
    mask_thres=0.5,
    null_hypothesis_threshold=0.15,
)
```

**Multi-channel input**:

- single file `(C,Z,H,W)`: pass it straight to `image`;
- several single-channel files: read them separately, then `np.stack` into `(C,Z,H,W)`;
- single file `(C,H,W)` with C>4: `infer` does not expose `image_axis_order`, so convert it to `(C,1,H,W)` before passing it in (see [Sec. 11](#11-known-limitations)).

**Product**: `/data/new/results.h5ad`. If you also need a cell table:

```python
exp.extract_cell_table(
    source_kind="h5ad",
    h5ad_path="/data/new/results.h5ad",
    instance_source="pred",
    target_channel=[0],
    train_image_names=["large_001"],   # pure inference has no real train/test; mark everything as train first
    test_image_names=(),
    attach_tracks=False,
    device="cuda",
    n_jobs=4,
)
```

> A pure inference scenario has no real train/test, and the `split` column only exists to satisfy the downstream contract; to run stage B you must prepare a labelled train/test split separately.

### 5.3 Workflow C: batch inference over several standalone images

Same as workflow B, only with several entries in `image_records`:

```python
records = []
for name in ["img_001", "img_002", "img_003"]:
    records.append({
        "image_name": name,
        "image": tifffile.imread(f"/data/new/{name}.tif"),
        "meta_map": {"condition": {name: "control"}},
    })

exp.infer(
    anndata_path="/data/new/batch.h5ad",
    infer_source="images",
    image_records=records,
    ...
)
```

Key points:

- all images go into **one H5AD**, distinguished by `image_id`;
- `image_name` must be unique and must not contain `/` or `\` (they are replaced by `_`);
- every image may carry its own `meta_map`, but remember that `/uns/luma/meta_map` only keeps the last one written while each image keeps its own `attrs.meta_info`;
- if channels or sizes differ between images, `target_stitch_channel` and `grid_size` must cover all of them;
- per-image evaluation needs GT, and the standalone-image entry does not support GT, see [Sec. 11](#11-known-limitations).

### 5.4 Workflow D: a dataset with segmentation GT, extract representations and analyze

**Applies to**: a `GridImageDataset` already exists (built by `build_dataset` or a compatible tool) and you only want to extract representations from GT instances, without training or inference.

**Property**: stage A re-encodes patches with the **frozen SAM ViT** from the Dataset entry and does not need the main-model checkpoint, so no training is required.

```python
from scripts_formal.LUMAExperiment import CellExperiment

DATASET = "/data/my_project/dataset"
OUTPUT_ROOT = "/data/my_project/representation"

exp = CellExperiment()

# 1) extract the cell table (GT instances)
exp.extract_cell_table(
    source_kind="dataset",
    dataset_path=DATASET,
    instance_source="gt",
    target_channel=[0],
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    backbone_type="vit_l",
    type_index_to_name={0: "untyped", 1: "type_a", 2: "type_b"},
    type_name_to_index={"untyped": 0, "type_a": 1, "type_b": 2},
    output_root=OUTPUT_ROOT,
    extension_rate=1.0,
    n_morph_workers=7,
    device="cuda",
    n_jobs=4,
)

# 2) analyze directly (do not re-extract)
exp.analyze_cell_features(
    source_kind="dataset",
    data_name="my_project",
    dataset_path=DATASET,
    output_root=OUTPUT_ROOT,
    run_extract_flag=False,
    instance_source="gt",
    target_channel=[0],
    type_index_to_name={0: "untyped", 1: "type_a", 2: "type_b"},
    type_name_to_index={"untyped": 0, "type_a": 1, "type_b": 2},
    meta_key="gt_cell_type",
    target_kind="discrete",
    exclude_obs_in={"gt_cell_type": ("untyped",)},
    device="cuda",
    n_jobs=4,
)
```

Products:

- `OUTPUT_ROOT/cell_table.h5ad`: `X_sam`, `X_morph` (136 columns), `obs`;
- `uns/runs/<run_name>/` inside that file: stages B to F.

> `output_root` must be identical in `extract_cell_table` and `analyze_cell_features`, otherwise the latter cannot find the main table.

### 5.5 Workflow E: an existing result H5AD, representation analysis

**Applies to**: inference is already done (a LUMA-generated H5AD) and you want a cell table plus downstream analysis.

```python
RESULT = "/data/my_project/result.h5ad"
DATASET = "/data/my_project/dataset"     # may be empty, but then give train/test lists
OUTPUT_ROOT = "/data/my_project/representation"

exp.extract_cell_table(
    source_kind="h5ad",
    h5ad_path=RESULT,
    dataset_path=DATASET,
    instance_source="pred",
    target_channel=[0],
    type_index_to_name={0: "untyped", 1: "type_a", 2: "type_b"},
    type_name_to_index={"untyped": 0, "type_a": 1, "type_b": 2},
    attach_tracks=True,        # attaches tracks automatically when the H5AD already has them
    device="cuda",
    n_jobs=4,
)

exp.analyze_cell_features(
    source_kind="h5ad",
    data_name="my_project",
    h5ad_path=RESULT,
    dataset_path=DATASET,
    run_extract_flag=False,
    instance_source="pred",
    target_channel=[0],
    meta_key="time_h",
    target_kind="continuous",
    device="cuda",
    n_jobs=4,
)
```

Key points:

- on the H5AD entry, `train_image_names` / `test_image_names` are used only when `dataset_path` is empty;
- when `dataset_path` is empty and only train is given, every listed image is marked train; when both are given they must be disjoint and must cover every image in the H5AD;
- `instance_source="gt"` needs `gt/gt_instance_map` inside the H5AD; an inference-only H5AD without GT can only use `"pred"`.

### 5.6 Workflow F: time-series tracking and trajectory analysis

**Applies to**: multi-frame images (one image per frame, with `meta_map` providing `timepoint_index` / `fov_id`) that need trajectories.

**Step 1: inference** (one `image_record` per frame; `meta_map` supplies the frame number/FOV)

```python
import tifffile

records = []
for name, t, fov in frames:   # frames comes from your experiment table
    records.append({
        "image_name": name,
        "image": tifffile.imread(f"/data/time_lapse/{name}.tif"),
        "meta_map": {
            "timepoint_index": {name: int(t)},
            "fov_id": {name: str(fov)},
        },
    })

exp.infer(
    anndata_path="/data/time_lapse/result.h5ad",
    infer_source="images",
    image_records=records,
    ...
)
```

**Step 2: extract the cell table**

```python
exp.extract_cell_table(
    source_kind="h5ad",
    h5ad_path="/data/time_lapse/result.h5ad",
    instance_source="pred",
    target_channel=[0],
    train_image_names=[r["image_name"] for r in records],
    test_image_names=(),
    attach_tracks=False,
    device="cuda",
    n_jobs=4,
)
```

**Step 3: track**

```python
exp.track_cells(
    h5ad_path="/data/time_lapse/result.h5ad",
    config_path="/opt/luma/scripts_formal/config/laptrack_config.json",
)
```

**Step 4: re-extract and analyze (attach tracks + stage F)**

Stages A and B/E need a real train/test split. The example takes the first half and the second half in time order; a real project should split by FOV or by time block so that no trajectory crosses train/test.

```python
all_names = [r["image_name"] for r in records]
cut = len(all_names) // 2
train_names, test_names = all_names[:cut], all_names[cut:]

exp.analyze_cell_features(
    source_kind="h5ad",
    data_name="time_lapse",
    h5ad_path="/data/time_lapse/result.h5ad",
    train_image_names=train_names,
    test_image_names=test_names,
    run_extract_flag=True,       # re-extract so that track_id is attached
    attach_tracks=True,
    instance_source="pred",
    target_channel=[0],
    meta_key="timepoint_index",
    target_kind="continuous",
    device="cuda",
    n_jobs=4,
)
```

Key points:

- `track_cells` reads the H5AD's `obs`, so a cell table must be extracted first;
- re-extracting after tracking (`run_extract_flag=True`) attaches `track_id` to the new table;
- stage F needs `track_id`, `fov_id`, and `timepoint_index` in `obs`;
- distances are in pixels; with `sqeuclidean` the configured cutoff is passed squared (writing 40 px actually sets a cutoff of 1600);
- there is no official entry point for "tracking visualization only, without B/E": if you only need trajectory figures, either supply a valid target and split, or call the visualizer classes under `utils` directly.

### 5.7 Workflow G: transferring an existing model to new data

**Scenario 1: inference only** - use workflow B/C directly and make sure the whole compatibility contract in Sec. 3.9 holds.

**Scenario 2: fine-tuning on the new data** - use `train_model`'s `load_model_path` / `load_components`:

```python
history, best_model = exp.train_model(
    dataset_path="/data/new_project/dataset",
    sam_checkpoint_path="/opt/luma/sam_checkpoints/sam_vit_l.pth",
    log_dir="/data/new_project/training_logs",
    title="finetune_new_project",
    load_model_path="/data/models/previous_best.pth",
    load_components=["vit", "upsampler", "predictors"],  # without volume_adapter
    train_encoder=False,          # keep the backbone frozen first
    epochs=50,
    learning_rate=1e-4,
    target_image_channel=[0],
)
```

Valid `load_components` names are `vit`, `upsampler`, `volume_adapter`, `predictors`, or `"all"`.

**Scenario 3: LoRA fine-tuning** - set `vit_adapt="lora"` and `lora_rank=4` in `trainer_params`, and keep `use_image_skip=True` so the decoder structure stays the same.

**Transfer checklist**:

- does the channel semantics of the new data match the old model (same channel index);
- are the pixel value range and normalization the same;
- does the cell size distribution fall inside a sensible range for `object_attri_limit` and `object_size` (and for the default major-axis filter);
- does the type set of the new data match `num_class` / `type_index_to_name`;
- for 3D data, do `input_z_size` and the volume-adapter weights exist.

---

## 6. Training and inference internals

### 6.1 Model architecture

`SamUNetR` = SAM ViT encoder (neck removed) + two UNETR-style upsampling decoders + three output heads:

- `raw_mask`: 1-channel cell-probability logits;
- `raw_hv`: 2-channel horizontal/vertical flow field;
- `cyto_type`: `num_class`-channel type logits;
- `embed_feat`: the ViT's last-layer token features.

The input is always 5D `BCZYX`:

- `input_z_size=1`: requires Z=1 and C=1 and squeezes directly;
- `input_z_size>1`: fused into a single plane by `TrainableVolumeToSamAdapter` (1x1x1 channel adaptation + 3D residual blocks + spatial-z attention).

The resulting plane is then `repeat_interleave`d to 3 channels before entering the ViT.

The forward pass also takes a `blur_skip` flag: training uses the raw encoder-1 skip, while the evaluation stitch path passes `blur_skip=True` whenever `use_image_skip` is enabled.

### 6.2 Losses

| Head | Default loss | Notes |
|---|---|---|
| `raw_mask` | `dicebce_logits` | Dice + BCE. With `uncertainty_weight=True` each part gets a learnable log-variance weight; with `False` the two parts are fixed at 0.5/0.5 |
| `raw_hv` | `cpdynamics_hv` | foreground/background-weighted L2, internally scaled by `hv_loss_weight` (default 2.0); `hovernet_hv` adds a Sobel gradient-consistency term |
| `cyto_type` | `ce_dice_loss_logits` | masked cross-entropy + class-weighted macro soft-Dice; this head keeps the fixed 0.5/0.5 split (its uncertainty weighting is disabled) |

The total loss is the **plain sum** of the three head losses (then divided by `accumulation_steps` before back-propagation), so the effective weighting comes from the internal factors above, not from three equal terms. The model with the lowest validation total loss is kept as best.

### 6.3 Training behaviour

- optimizer: AdamW (`optimizer_type="adam"`, no weight decay) or SGD (`momentum=0.9`, `weight_decay=1e-4`);
- AMP is on by default; `use_bf16=True` switches to bfloat16;
- gradient accumulation, gradient clipping (`max_norm=1.0`), and NaN/Inf batch skipping are supported;
- the validation set is split out of the training patches by `train_val_ratio`;
- a cosine annealing scheduler is created, but it is only stepped on the SGD path, so with Adam the learning rate stays constant;
- early stopping stops training after `patience` epochs without improvement. The **returned path** points at the checkpoint of the best epoch, but the model object kept in memory still holds the last epoch's weights, so reload the checkpoint if you continue using `exp.trainer`.

### 6.4 Inference and stitching

1. patch forward; for LoRA models the frozen representation is extracted with `use_lora=False`;
2. predictions are resized back to the native patch size and the half margin is cropped;
3. patches are added (`+=`) into the full image on the grid (valid regions are complementary, so there is no overlap);
4. the full image is resized back to the original size;
5. the background channel of the type map is clamped to `null_hypothesis_threshold` and then `argmax` is applied;
6. CP-dynamics recovers the instances.

### 6.5 CP-dynamics instance recovery

`[mask, h, v]` to instance mask:

1. flows are assembled as `dP=[v,h]` in CellPose axis order; the map is rescaled so its long edge equals `max_cpimage_size` (because the scale is computed as `max_size / max(H, W)`, maps smaller than the cap are **upscaled**);
2. optional Gaussian smoothing (on by default, sigma `1 * scale_ref`);
3. `mask > mask_thres` gives the foreground; small objects are removed and holes filled; the flow field is multiplied by the foreground mask;
4. `follow_flows` runs Euler integration for every foreground pixel and clusters the histogram into instances;
5. `flow_threshold` performs flow-consistency QC; `object_size` drops small instances;
6. nearest-neighbour resize back to the original size; an optional major-axis filter follows.

> The major-axis filter is **on by default**: `conduct_size_filter=True`, `min_major_axis_length='auto'`, which resolves to `sqrt(min_mask_size/pi)*2`, about 61.8 px with the default `min_mask_size=3000`. Instances below that are removed even if they pass `object_size`. Configure it through `CellExperiment(postprocessor_params={"conduct_size_filter": False})`, or pass an explicit `min_major_axis_length` / `min_mask_size`.

### 6.6 Evaluation metrics

| Task | Metrics |
|---|---|
| Segmentation | TP/FP/FN, Precision/Recall, AJI, DQ, SQ, PQ (single IoU threshold, not multi-threshold AP) |
| In-situ classification | per-class precision/recall/F1, macro/weighted AP, binary AUC, pixel-level and instance-level type mIoU |
| Single-cell classification | `macro_ap`, `weighted_ap`, `random_macro_ap`, per-class AP/AUC, optional `macro_auc` |
| Single-cell regression | `r2`, `rmse`, optional `mae` / `spearman`, FOV-time mean metrics |
| Tracking | assignment rate, track length/duration distributions, number of gap nodes |

Definitions and how to read the metrics are in chapter 7.

---

## 7. Cell representations and analysis

### 7.1 Representation definitions

Every cell has two representation matrices:

- **SAM representation** `X_sam`: mean pooling of the frozen SAM ViT patch features over the token region matching the extended bbox. Its width equals the ViT embedding width: **1024 for `vit_l`**, 768 for `vit_b`, 1280 for `vit_h`. The analysis code works on `n_mean=1024` and truncates wider embeddings to the first 1024 columns;
- **Morphological representation** `X_morph`: 136 columns split into 6 physical-quantity groups.

> Note: `X_sam` is always produced by the **frozen SAM ViT** (`use_lora=False`) and does not change with full fine-tuning or with LoRA weights. Representations from a fine-tuned encoder are not exposed by the current interface.

The two can be used separately or horizontally concatenated as `emb*||morph_ext`.

### 7.2 Stage A: cell-table extraction

- Dataset entry: re-encodes patches (frozen SAM); supports GT instances only;
- H5AD entry: concatenates the stored `patch_embeddings` and pools them with an integral image; supports GT / pred instances;
- only cells whose centroid lies inside the valid region are kept;
- the extended bbox is used for representation pooling and neighbourhood statistics;
- target columns and filters are **not written** into the shared `obs`; stage B resolves them in memory.

### 7.3 The 136 morphological columns

| Group | Columns | Content |
|---|---|---|
| `geometry_srel` | 24 | absolute-scale geometry: area, major/minor axis, Feret, perimeter, inertia moments, distance transform, radial statistics |
| `geometry_sinv` | 56 | dimensionless shape: eccentricity, Euler number, extent, solidity, circularity, Hu moments, Fourier, Zernike |
| `object_intensity` | 21 | in-object intensity location/dispersion + radial rings |
| `texture` | 22 | Haralick texture + HOG |
| `bbox_intensity` | 9 | intensity and gradient magnitude of the extended bbox |
| `neighbors` | 4 | contact and centroid distance inside the same patch |

Length normalization happens at extraction time: `objint_mass_displacement` is divided by the instance's equivalent radius, and `neighbor_dist_*` by the mean equivalent radius of all valid-region instances in that image.

### 7.4 Stage B: supervised prediction

- three input sets: `emb*`, `morph_ext`, `emb*||morph_ext`;
- head: `proj64_mlp` (`in_dim` -> 64 -> ReLU -> 256 -> ReLU -> `out_dim`);
- discrete: `K`-dimensional logits + cross-entropy with balanced class weights, early stopping on validation `macro_ap`;
- continuous: 1 dimension + sigmoid, early stopping on validation `r2`; training happens in scaled space and predictions are inverse-transformed back to the original unit before evaluation;
- 5 seeds by default (0 to 4); the summary reports median/mean/std/min/max;
- products: `type_prediction/{metrics, metrics_per_class, delta_metrics, metrics_summary, fov_time_mean_metrics, proj64_seed0/*}`.

### 7.5 Stage C: group-level ablation and CCA

- runs on `morph_ext` only, never concatenated;
- three kinds of variants: `full`, `drop_<group>`, `only_<group>`;
- in `ablation_delta`, `delta = metric_full - metric_drop`: a positive delta means removing the group lowered the metric;
- `ablation_only` holds the metric of that group used alone;
- read both numbers side by side; do not rank groups by delta alone;
- CCA: the embedding side is PCA-reduced to at most 64 dimensions, the morphology side is PCA-reduced within the group; the fit is on train and the report on test gives `rho_1/2/3` plus `morph_ss_explained_r1/all`.

### 7.6 Stage D: summary

- `summary/wide`: rows = input x head, columns = scope x default metric;
- `summary/input_comparison`: medians of the three main inputs plus their pairwise differences.

### 7.7 Stage E: feature projection

- UMAP, shrunk LDA, Proj64-PCA, PHATE (continuous targets only);
- everything is fitted on train and applied to test;
- figures: scatter, LDA KDE, condition-mean heatmap (columns in natural order, no column clustering);
- for continuous tasks the LDA axis is correlated (Spearman) with time to check monotonicity;
- products: `coords/{emb,morph}`, `visualization/figures/*`, `visualization/scatter_limits/*`, `visualization/lda_geometry`.

### 7.8 Stage F: trajectory projection

- reads only `track_id` and the stage-E coordinates;
- per test FOV it takes the 6 longest qualifying chains (`n_obs >= 10`);
- products: trajectory scatter, a 20x20 displacement field, and small-multiple figures.

---

## 8. Result storage and export

### 8.1 Result H5AD structure

```
/
  X, obs, var, obsm           cell table (after stage A)
  uns/
    luma/
      project                  schema_version, storage_policy
      meta_map                 image-level metadata mapping (last image written wins)
      images/<image_id>/       inference image records (see 3.4)
      evaluations/<image_id>/  evaluation results (optional)
      tracking/                laptrack results (optional)
      fovs                     fov_column + fov_ids (optional, written when tracking sees FOVs)
    runs/<run_name>/           analysis results (see 3.6)
    tracking_quality           per-track quality (optional)
    tracking_quality_summary   overall quality (optional)
    cell_table_config          extraction configuration
    morph_columns              morphological column names
    morph_feature_groups       morphological groups
    skipped_images             skip log
    run_index                  index of analysis runs
```

`uns/luma/project` is written when the file is created (`schema_version="0.1.0"`, `storage_policy="image_records_in_uns_luma"`), but nothing validates it when the file is read later.

### 8.2 In-place write rules

- for a LUMA result file a whole-file `write_h5ad` is **forbidden**; use `AnnDataResultStore.write_table_elements` or `FeatureWriter.write_uns`, otherwise the large arrays under `/uns/luma/images` are lost;
- the H5AD entry of `extract_cell_table` already takes the in-place path; the Dataset entry writes a separate file, so a whole-file write is fine there.

### 8.3 Figure export

- figures are stored as SVG bytes under `uns/runs/<run_name>/visualization/figures/` by default;
- `export_figures` only reads, never redraws;
- PNG export needs `cairosvg`;
- to redraw from already-stored coordinates, use `redraw_saved_visualization` in `utils/elements_visualizer.py` (it is not part of `CellExperiment`; it is an advanced interface).

### 8.4 Reading results

| Need | Recommended way |
|---|---|
| read the cell-table matrices | `anndata.read_h5ad(...)` or `FeatureReader.load_cell_table` |
| read a run's metrics | `FeatureReader.read_uns(h5ad, "type_prediction/metrics", run_name=...)` |
| read reduction coordinates | `FeatureReader.read_run_coords(h5ad, run_name, tag)` |
| read a figure | `FeatureReader.read_figure(h5ad, key, run_name)` |
| read the seed-0 bottleneck weights | `FeatureReader.read_seed0(h5ad, tag, run_name)` |

> `FeatureReader` / `FeatureWriter` belong to the `utils` layer and are not public methods of `LUMAExperiment.py`; if you need a strict "`CellExperiment` only" boundary, treat result reading as data access rather than an algorithm interface.

---

## 9. Deployment, performance, and reproducibility

### 9.1 Hardware

| Task | Recommendation |
|---|---|
| Training / inference | GPU; memory must cover the forward pass of `stitch_batch_size x patch` |
| Morphology extraction | CPU, multi-threaded (`n_morph_workers`) |
| UMAP / PHATE / LDA / CCA | CPU |
| Tracking | CPU |

### 9.2 Time and space

- training time depends on the number of patches, the number of epochs, whether the ViT is unfrozen, and whether LoRA is used; do not quote numbers that have not been measured;
- a result H5AD stores the full-image predictions and the patch embeddings and can be very large; budget disk as images x patches;
- for large-image inference, `stitch_batch_size` is the main memory/throughput knob.

### 9.3 Reproducibility

- dataset-build parameters (`loading_options`) are stored inside `metadata.pkl` and mirrored into the H5 attributes (`grid_size`, `input_z_size`, `image_channel_count`, `channel_*`);
- training hyper-parameters and the per-epoch losses are written to `log.txt` / `loss.png` in the session directory;
- inference thresholds are stored per image in the record's `postprocess_params`;
- analysis parameters are stored in `run_config` under `uns/runs/<run_name>/`;
- a run's `mlp_seeds`, `viz_random_state`, and `sample_seed` should be fixed;
- the tracking `config_path` and `tracking_overrides` are written into `/uns/luma/tracking`;
- archive result files together with their `run_config`.

### 9.4 Version management

- on the project side, record: the LUMA code version, the SAM weight version, the model checkpoint, `metadata.csv`, `loading_options`, and the `laptrack` configuration;
- there is no enforced schema-version check; after upgrading, run one regression pass on a small dataset first.

---

## 10. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `build_dataset` raises `KeyError` (meta_map) | some image hits no dictionary key | use the full `image_name` as the key, or fill in the substring mapping |
| `build_dataset` raises `AssertionError` on `image_name` | an image name is used twice | make `image_name` unique |
| `build_dataset` raises `Target size mismatch` / `Type num mismatch` | images were built with different `grid_target_size` or `generate_classification` | make all images share the same geometry and type rules |
| `train_model` raises a channel-count error | `target_image_channel` does not match the data's channel count | pass the channel index explicitly; 2D accepts exactly one channel |
| `infer` raises a `C=...` / `Z=...` error | `input_z_size` does not match the data dimensionality | use `input_z_size=1` for 2D; use the training Z for 3D |
| inference result empty / very few instances | `mask_thres` too high, `object_size` too large, inconsistent normalization, or the default major-axis filter (`min_mask_size=3000`) removing small cells | first check `normalize` against training, then tune the thresholds, then try `postprocessor_params={"conduct_size_filter": False}` |
| every pixel typed as background | `null_hypothesis_threshold` too high | lower it; check the type-head class count |
| seams in the stitched image | `grid_size` / `margin_ratio` of the standalone-image entry differ from training | use the training-time geometry |
| `extract_cell_table` raises `missing gt_instance_map` | the H5AD has no GT but `instance_source="gt"` was used | switch to `"pred"`, or use the Dataset entry |
| `extract_cell_table` raises a split error | the H5AD entry got no Dataset path and the train/test lists are incomplete | pass `dataset_path`, or pass both `train_image_names` and `test_image_names` |
| `analyze_cell_features` raises a missing-class error | a class of the K types is absent from train or from all-test | check `exclude_obs_in` / the split; discrete tasks need every class |
| `max_cells_per_image` raises a multi-patch error | the dataset has several patches per image | disable sampling, or use single-patch data |
| tracking raises "sampled cell table cannot be tracked" | `cell_table_config.max_cells_per_image` is non-empty | re-extract from an unsampled table |
| trajectories are broken | distance threshold too small, or wrong frame/FOV columns | check `time_column` / `fov_column`; increase the cutoff |
| `export_figures` PNG raises | `cairosvg` missing | install `cairosvg`, or export SVG first |
| image records are lost after writing the H5AD | a whole-file `write_h5ad` was used | switch to the in-place write interface |
| `ValueError: Metadata file not found` when loading a dataset | `dataset_path` points at a directory without `metadata.pkl` | point at the directory produced by `build_dataset` |
| `Image record already exists` during inference | a record with the same `image_id` exists and `overwrite_image` / `overwrite_anndata` is off | set `overwrite_anndata=True`, or use a new target file |

---

## 11. Known limitations

1. **Defaults**: `LUMA4_ROOT` is derived from the environment variable or the install location, but `analyze_cell_features` still carries project-specific defaults for `target_channel`, `sam_checkpoint_path`, `type_index_to_name`, `type_name_to_index`, `data_name`, `dataset_path`, and `output_root` (see [Sec. 4.7](#47-analyze_cell_features)). A migrated deployment must override them.
2. **Standalone-image input adaptation**: `infer` does not expose `image_axis_order`, `rescale`, or `blur_skip`; multi-file channels must be stacked into `(C,Z,H,W)` by the user first.
3. **Split for pure inference**: the H5AD entry needs a Dataset path or at least the train list; there is no official "test-only" entry.
4. **Unlabeled representation analysis**: `analyze_cell_features` requires `meta_key` / `target_kind`; it cannot produce UMAP/PHATE/heatmaps directly for unlabeled data.
5. **Representation export**: there is no `export_cell_representation`; read `cell_table.h5ad` instead.
6. **External instance-type labels**: the type map can only be generated from the intensity thresholds in `generate_classification`; a manual type table cannot be loaded directly.
7. **Tracking configuration**: `CellExperiment.track_cells` requires `config_path`. The packaged `scripts_formal/config/laptrack_config.json` has `anndata_path: null`, so the input file must be supplied through `h5ad_path`; `LUMAExperiment.main()` defaults its tracking switch to `config/mdck2_timeseries_laptrack_config.json`.
8. **Image-level visualization**: `CellTimeSeriesVisualizer` / `CellSegmentationVisualizer` are not exposed on `CellExperiment`.
9. **Evaluation of standalone images with GT**: `image_records` does not accept GT, so segmentation/classification cannot be evaluated directly on standalone images; use a single-image Dataset instead.
10. **Batch directory discovery**: `build_dataset` has no `image_dir` / glob parameter; you must enumerate the records yourself (although `image_paths` does accept a directory path, which is expanded by file extension).

---

## Appendix A: Parameter quick reference

### A.1 Dataset building

| Parameter | Typical value | Effect |
|---|---|---|
| `grid_size` | `(2,2)` / `"auto"` | grid |
| `grid_target_size` | `(1024,1024)` | patch input size |
| `margin_ratio` | `0.1` | overlap |
| `input_z_size` | `1` (2D) / `10` (3D) | Z depth |
| `normalize` | `False` | whole-image z-score |
| `generate_classification` | `{"channel":[0],"threshold":["auto"]}` | type map |
| `type_mode` | `"inst_wise"` | type granularity |
| `extension_rate` | `1.0` | single-cell bbox expansion |
| `object_attri_limit` | `{"axis_major_length": (85, None)}` | filter small instances |
| `delete_edges` | `False` | erase mask edges |
| `image_axis_order` | `"auto"` / `"cyx"` / `"zyx"` / `"czyx"` / `"yxc"` | axis interpretation |

### A.2 Training

| Parameter | Typical value | Effect |
|---|---|---|
| `epochs` | 50-250 | training epochs |
| `batch_size` | 1-4 | patch batch size |
| `accumulation_steps` | 1-8 | gradient accumulation |
| `learning_rate` | 1e-4 to 5e-5 | main learning rate |
| `train_encoder` | `False` / `True` / `[block]` | backbone unfreezing policy |
| `encoder_learning_rate` | 1e-6 to 1e-5 | backbone learning rate |
| `loss_type_hv` | `"cpdynamics_hv"` / `"hovernet_hv"` | HV loss |
| `hv_loss_weight` | 1.0-2.0 | HV weight |
| `patience` | 20-25 | early stopping |
| `use_amp` / `use_bf16` | `True` / hardware-dependent | mixed precision |

### A.3 Inference

| Parameter | Typical value | Effect |
|---|---|---|
| `target_stitch_channel` | same as training | inference channel |
| `stitch_batch_size` | 1-8 | patch batch size |
| `input_z_size` | same as training | Z depth |
| `grid_size` | same as training | standalone-image grid |
| `margin_ratio` | same as training | standalone-image overlap |
| `normalize` | same as training | standalone-image preprocessing |
| `mask_thres` | 0.5 | cell-probability threshold |
| `flow_threshold` | 0.5 | flow-field QC |
| `object_size` | 10 | minimum instance size |
| `null_hypothesis_threshold` | 0.15 | lower clamp of the type background |

### A.4 Analysis

| Parameter | Typical value | Effect |
|---|---|---|
| `meta_key` | `"gt_cell_type"` / a metadata column | target |
| `target_kind` | `"discrete"` / `"continuous"` | task type |
| `obs_equals` / `exclude_obs_in` | `{}` | filters |
| `target_scale_max` | magnitude of the continuous target | regression scaling |
| `stratify_eval_key` | a metadata column | stratified evaluation |
| `mlp_seeds` | `(0,1,2,3,4)` | random seeds |
| `max_epochs` / `patience` | 200 / 20 | MLP training |
| `column_zscore` | `True` | column standardization |
| `feature_transforms` | `[False]*4` | per-image z-score / row L2 |
| `extra_metrics` | `("macro_auc",)` / `("mae","spearman")` | additional metrics |

---

## Appendix B: Glossary

| Term | Meaning |
|---|---|
| AJI | Aggregated Jaccard Index, an aggregate instance-segmentation metric |
| AP | Average Precision |
| CCA | Canonical Correlation Analysis |
| CP-dynamics | CellPose-style flow-dynamics instance recovery |
| DQ / SQ / PQ | Detection / Segmentation / Panoptic Quality |
| HV | horizontal/vertical instance flow field |
| LoRA | low-rank adaptation; only the Q/V branches are trained |
| valid region | the effective region left after trimming the half margin of a patch |
| run_name | analysis directory name `{meta_key}\|{target_kind}\|{filter}` |
| representation | in this document, `X_sam` and `X_morph` |

---

## Appendix C: Configuration templates

### C.1 laptrack configuration

This is the content of the shipped `scripts_formal/config/laptrack_config.json`:

```json
{
  "anndata_path": null,
  "time_column": "timepoint_index",
  "x_column": "centroid_x",
  "y_column": "centroid_y",
  "distance_metric": "sqeuclidean",
  "max_linking_distance_px": 40.0,
  "gap_closing_distance_px": 60.0,
  "gap_closing_max_frame_count": 2,
  "allow_splitting": false,
  "splitting_distance_px": 40.0,
  "allow_merging": false,
  "merging_distance_px": 40.0,
  "visual_feature_columns": [],
  "visual_feature_normalization": "zscore",
  "visual_feature_clip_percentiles": [1.0, 99.0],
  "visual_feature_weight": 1.0,
  "laptrack_costs": {
    "track_start_cost": null,
    "track_end_cost": null,
    "segment_start_cost": null,
    "segment_end_cost": null,
    "no_splitting_cost": null,
    "no_merging_cost": null,
    "alternative_cost_factor": null
  }
}
```

- `anndata_path` is `null` in the file; `track_cells(h5ad_path=...)` supplies it and overrides any value written there;
- `fov_column` is not part of the shipped file; it defaults to `fov_id` and is read from `obs` (or from `/uns/luma/fovs` when present);
- without a `fov_id` column, tracking runs once over the whole table;
- with `sqeuclidean` the cutoffs are passed squared (40 px becomes 1600);
- `laptrack_costs` entries set to `null` are simply not forwarded.

### C.2 Minimal training configuration

```python
trainer_params = {
    "backbone_type": "vit_l",
    "num_class": 1,
    "use_prediction_head": True,
    "input_z_size": 1,
    "use_image_skip": True,
    "vit_adapt": "none",
}

train_kwargs = {
    "epochs": 100,
    "batch_size": 1,
    "accumulation_steps": 8,
    "learning_rate": 1e-4,
    "encoder_learning_rate": 1e-6,
    "train_encoder": False,
    "patience": 20,
}
```

---

## Appendix D: Corrections relative to the Chinese edition

The English edition was checked against `LUMAExperiment.py`, `LUMACore.py`, and `utils/*.py` on 2026-11-08. The Chinese original `LUMA_User_Guide.md` is unchanged; the following points were corrected here.

| # | Location | Chinese original | Verified behaviour / correction |
|---|---|---|---|
| 1 | Front matter, Sec. 1.4, 4.7, 5.1, 5.6, 7.1, 11 | Points to a companion file `LUMA_User_Guide_接口需求.md` ("notes G2/G3/G6/G7/G15/G16") | That file does not exist anywhere in the project; every pointer was replaced by an inline explanation |
| 2 | Sec. 1.3 | Embeds `LUMA4_architecture_overview.png` | The asset is not present in `scripts_formal/`; the embed was replaced by a note plus the textual pipeline |
| 3 | Sec. 2.2 item 1 | "`LUMA4_ROOT` must point at the actual install directory" | It is resolved as the environment variable `LUMA4_ROOT` when set, otherwise the parent of the `scripts_formal` package |
| 4 | Sec. 3.3 defaults table | 13 loading options | The real dictionary has 16 keys; `defined_attri_func` (`None`) and `delete_edges` (`False`) were added |
| 5 | Sec. 3.2 | Formula only | `patch_h` / `margin_h` are truncated to integers and the image is resized to the exact stitched size when it differs, which was added |
| 6 | Sec. 3.4 | "`/uns/luma/meta_map` is written wholesale while stitching" | Each image overwrites it, so a batch run keeps only the last image's map; per-image metadata is in `attrs.meta_info` |
| 7 | Sec. 3.5 | `gt_cell_type` described only as a majority vote | It is stored as an integer class index; name-based filters decode it through `type_index_to_name` |
| 8 | Sec. 4.3 (and 4.1 table) | `backbone_2d.pth` = `vit` + `upsampler` | It also contains the `predictors` component |
| 9 | Sec. 4.3 | Three training modes | The volume adapter only exists when `input_z_size>1`; with `vit_adapt="lora"` the `train_encoder` argument is ignored and the ViT stays frozen |
| 10 | Sec. 4.4, 2.2 | Threshold table had no post-processor size filter | `conduct_size_filter=True` by default, with `min_major_axis_length='auto'` at about 61.8 px (`min_mask_size=3000`), removing small instances after CP-dynamics |
| 11 | Sec. 4.6 | `track_cells(h5ad_path, config_path=None, ...)` | `config_path` is required on `CellExperiment.track_cells`; the packaged default exists only on `CellTracker.track_anndata` / `elements_tracker.main()` |
| 12 | Sec. 4.6 | Tracking writes `obs.track_id` | It also writes `obs.track_order`, `/uns/luma/fovs` when FOVs are present, and the two quality tables |
| 13 | Sec. 4.7 | "the signature writes project-specific defaults as neutral values or `None`" | The real defaults are project-specific: `data_name="a4bad_cpdynamics_hv"`, `target_channel` to `[2]`, `{0:"rpe",1:"hesc"}`, `{"rpe":0,"GFP-":0,"hesc":1,"GFP+":1}`, `<root>/sam_checkpoints/sam_vit_l_0b3195.pth`, `<root>/datasets/<data_name>`, `<root>/embedding_results/...`. Replaced by the real signature plus a resolution table, including the `lda_label_key` fallback chain |
| 14 | Sec. 4.5 | Attach-tracks described as a simple column copy | On the H5AD entry it also refreshes `uns/tracking_quality` / `uns/tracking_quality_summary` when tracks exist |
| 15 | Sec. 6.2 | "the total loss is an equally weighted sum of the three terms" | It is a plain sum of the three head losses, but the HV term is internally scaled by `hv_loss_weight` (default 2.0), and the mask term uses learned uncertainty weights (or fixed 0.5/0.5); the type head keeps 0.5/0.5 |
| 16 | Sec. 6.3 | (nothing stated) | Added: the cosine scheduler is only stepped on the SGD path, so Adam keeps a constant learning rate |
| 17 | Sec. 6.3 | "early stopping rolls back to the epoch with the lowest validation total loss" | No rollback happens: training stops after `patience` stale epochs, the returned path points at the best-epoch checkpoint, and the in-memory model still holds the last epoch's weights |
| 18 | Sec. 6.5 | "the long edge is downscaled when it exceeds `max_cpimage_size`"; "optional major-axis filter" | The maps are always rescaled so the long edge equals `max_cpimage_size` (small maps are upscaled); the major-axis filter is on by default at about 61.8 px |
| 19 | Sec. 7.1 | `X_sam` is "1024-dimensional" | The width equals the ViT embedding width: 1024 (`vit_l`), 768 (`vit_b`), 1280 (`vit_h`); the analysis truncates to `n_mean=1024` |
| 20 | Sec. 9.3 | "dataset-build parameters, training hyper-parameters, inference thresholds, and analysis parameters are all written to `metadata.pkl` / H5AD / `run_config`" | Training hyper-parameters are logged to `log.txt` (plus `loss.png`) in the session directory, not stored in `metadata.pkl`; the other three claims hold and were made specific |
| 21 | Sec. 11 item 7 | "`config_path` already defaults to the packaged `scripts_formal/config/laptrack_config.json`" | Not on `CellExperiment.track_cells`; `LUMAExperiment.main()` defaults to `config/mdck2_timeseries_laptrack_config.json` |
| 22 | Sec. 11 item 10, 4.2 | "there is no `image_dir` / glob entry" | True for `build_dataset`, but `image_paths` accepts a directory path that `GridImage` expands by extension, noted in both places |
| 23 | Sec. 8.1 | Structure listing | Added `/uns/luma/fovs`, plus the note that `uns/luma/project.schema_version` is written but never validated |
| 24 | Appendix C.1 | Template with `"anndata_path": "/data/results.h5ad"`, a `fov_column` key, and only 2 cost keys | The shipped file has `anndata_path: null`, no `fov_column`, and 7 cost keys; the template was replaced by the real file |
| 25 | Sec. 10 | Troubleshooting table | Added rows for `Target size mismatch` / `Type num mismatch`, `Metadata file not found`, `Image record already exists`, and the size-filter hint for empty results |

---

**Document maintenance**: whenever a public signature, a default value, or the H5AD schema of `CellExperiment` changes, update chapters 4 and 8 and the appendices in the same commit.

