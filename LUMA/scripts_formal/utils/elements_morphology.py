"""Extended morphology features: canonical columns and grouping by physical quantity.

Group by two criteria: whether a grayscale image is required and whether geometry explicitly depends on absolute pixel scale.
Merge `intensity_distribution` into `object_intensity`, and Haralick d1 and HOG into `texture`.
Write only canonical columns; do not extract theoretically redundant columns or deduplicate columns on the training set.
Divide `objint_mass_displacement` by the instance's equivalent radius; divide `neighbor_dist_*` by the mean
equivalent radius of all valid-region instances in that image.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from mahotas.features import haralick, zernike_moments
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import kurtosis, median_abs_deviation, skew
from skimage.measure import find_contours, regionprops
from skimage.morphology import binary_erosion, disk

# Build structuring elements for boundary erosion and neighborhood dilation only once.
_EROSION_FOOTPRINT = disk(1)
_NEIGHBOR_STRUCT = ndimage.generate_binary_structure(2, 2)
# Pad each instance's axis-aligned bounding box by 2 pixels for erosion / contours / contact rings.
_INSTANCE_CROP_PAD = 2

ZERNIKE_DEGREE = 9
ZERNIKE_NM = [
    (n, m) for n in range(ZERNIKE_DEGREE + 1) for m in range(n + 1) if (n - m) % 2 == 0
]
ZERNIKE_DROP = {(0, 0), (1, 1)}
ZERNIKE_NM_KEEP = [(n, m) for n, m in ZERNIKE_NM if (n, m) not in ZERNIKE_DROP]
HARALICK_DISTANCES = (1,)
HARALICK_SHORT = (
    "asm", "contrast", "correlation", "variance", "idm",
    "sum_avg", "sum_var", "sum_entropy", "entropy",
    "diff_var", "diff_entropy", "infomeas1", "infomeas2",
)
HARALICK_LEVELS = 32
N_INTDIST_BINS = 4
N_FOURIER_COEFFS = 8
N_HOG_BINS = 9
OBJINT_NAMES = (
    "objint_median", "objint_p25", "objint_p75", "objint_mad",
    "objint_mass_displacement",
    "objint_edge_mean", "objint_edge_std",
    "objint_edge_min", "objint_edge_max",
)
INTDIST_NAMES = tuple(
    f"intdist_{stat}_b{b:02d}"
    for stat in ("frac", "meanfrac", "radialcv")
    for b in range(N_INTDIST_BINS)
)
BBOX_INTENSITY_STATS = ("skew", "kurt", "p10", "p25", "p75", "p90", "range", "grad_mean", "grad_std")
DISTTRANS_STATS = ("mean", "std", "max", "median")
RADIAL_STATS = ("mean", "std", "max", "min", "median")

# Mask-only regionprops scalars; exclude grayscale and columns analytically derivable from area / convexity / perimeter.
REGIONPROPS_SCALAR_NAMES = (
    "morph_area",
    "morph_axis_major_length",
    "morph_axis_minor_length",
    "morph_eccentricity",
    "morph_euler_number",
    "morph_extent",
    "morph_feret_diameter_max",
    "morph_orientation",
    "morph_perimeter",
    "morph_solidity",
)

GEOMETRY_SREL_NAMES = (
    "morph_area",
    "morph_axis_major_length",
    "morph_axis_minor_length",
    "morph_feret_diameter_max",
    "morph_feret_diameter_min",
    "morph_perimeter",
    "inertia_xx",
    "inertia_xy",
    "inertia_yy",
    "inertia_eig_0",
    "inertia_eig_1",
    "moment_central_30",
    "moment_central_21",
    "moment_central_12",
    "moment_central_03",
    "disttrans_mean",
    "disttrans_std",
    "disttrans_median",
    "disttrans_max",
    "radial_mean",
    "radial_std",
    "radial_median",
    "radial_max",
    "radial_min",
)
GEOMETRY_SINV_NAMES = (
    "morph_eccentricity",
    "morph_euler_number",
    "morph_extent",
    "morph_solidity",
    "morph_circularity",
    "morph_aspect_ratio",
    "morph_orientation",
    "moment_norm_20",
    "moment_norm_11",
    "moment_norm_02",
    "moment_norm_30",
    "moment_norm_21",
    "moment_norm_12",
    "moment_norm_03",
    "moment_hu_1",
    "moment_hu_2",
    "moment_hu_3",
    "moment_hu_4",
    "moment_hu_5",
    "moment_hu_6",
    "moment_hu_7",
    *tuple(f"fourier_{i:02d}" for i in range(1, N_FOURIER_COEFFS)),
    *tuple(f"zernike_{n}_{m}" for n, m in ZERNIKE_NM_KEEP),
)
OBJECT_INTENSITY_NAMES = OBJINT_NAMES + INTDIST_NAMES
TEXTURE_NAMES = tuple(f"haralick_d1_{name}" for name in HARALICK_SHORT) + tuple(
    f"hog_{i:02d}" for i in range(N_HOG_BINS)
)
BBOX_INTENSITY_NAMES = tuple(f"bbox_intensity_{s}" for s in BBOX_INTENSITY_STATS)
NEIGHBOR_NAMES = (
    "neighbor_n",
    "neighbor_percent_touching",
    "neighbor_dist_closest",
    "neighbor_dist_second",
)
FEATURE_GROUP_ORDER = (
    "geometry_srel",
    "geometry_sinv",
    "object_intensity",
    "texture",
    "bbox_intensity",
    "neighbors",
)
FEATURE_GROUP_COLUMNS = {
    "geometry_srel": GEOMETRY_SREL_NAMES,
    "geometry_sinv": GEOMETRY_SINV_NAMES,
    "object_intensity": OBJECT_INTENSITY_NAMES,
    "texture": TEXTURE_NAMES,
    "bbox_intensity": BBOX_INTENSITY_NAMES,
    "neighbors": NEIGHBOR_NAMES,
}


def compute_cell_features(
    image: np.ndarray,
    mask: np.ndarray,
    inst_mask: np.ndarray,
    inst_id: int,
    cy: float,
    cx: float,
    ey1: int, ex1: int, ey2: int, ex2: int,
    patch_centroids: dict[int, tuple[float, float]],
    image_mean_equiv_radius: float,
) -> dict[str, float]:
    """Compute all canonical extended morphology features for one cell."""

    # Compute geometry / within-object texture only in the padded instance bbox; extended-bbox grayscale statistics use original image coordinates.
    crop = crop_instance_arrays(image, mask, inst_mask)
    if crop is None:
        return all_feature_names(return_empty=True)
    inst_c = crop["inst"]
    img_c = crop["image"]
    label_c = crop["labels"]
    cy_c = float(cy) - crop["y0"]
    cx_c = float(cx) - crop["x0"]
    dt = ndimage.distance_transform_edt(inst_c)
    boundary = inst_c ^ binary_erosion(inst_c, _EROSION_FOOTPRINT)

    feats = {}
    feats.update(extract_regionprops_scalar(inst_c))
    feats.update(extract_distance_transform_features(inst_c, dt=dt))
    feats.update(extract_boundary_radial_features(inst_c, cy_c, cx_c, boundary=boundary))
    feats.update(extract_fourier_descriptor_features(inst_c))
    feats.update(extract_zernike_features(inst_c))
    # Intensity / texture quantities require image; HOG and bbox statistics are defined on the extended bbox.
    feats.update(extract_bbox_intensity_extended(image, ey1, ex1, ey2, ex2))
    feats.update(extract_gradient_hog_features(image, inst_mask, ey1, ex1, ey2, ex2))
    feats.update(extract_object_intensity_features(
        img_c, inst_c, cy_c, cx_c, boundary=boundary,
    ))
    feats.update(extract_intensity_distribution_features(img_c, inst_c, dt=dt))
    feats.update(extract_haralick_features(img_c, inst_c))
    feats.update(extract_neighbor_features(
        label_c, inst_id, patch_centroids, image_mean_equiv_radius,
    ))
    return feats


def compute_morph_matrix(
    image_hw: np.ndarray,
    instance_map: np.ndarray,
    candidates: list[dict],
    morph_columns: list[str],
    image_mean_equiv_radius: float,
    n_workers: int = 7,
) -> np.ndarray:
    """Compute canonical extended morphology for selected cells; neighborhoods still use all instances in the current plane."""

    n_workers = max(1, int(n_workers))
    n_row = len(candidates)
    n_col = len(morph_columns)
    rows = np.empty((n_row, n_col), dtype=np.float32)
    if n_row == 0:
        return rows
    # Compute centroids once and reuse them for neighborhood distances.
    patch_centroids = {
        int(prop.label): (float(prop.centroid[0]), float(prop.centroid[1]))
        for prop in regionprops(instance_map)
    }

    def fill_row(row_idx: int) -> np.ndarray:
        cell = candidates[row_idx]
        inst_id = int(cell["instance_id"])
        inst_mask = instance_map == inst_id
        ey1, ex1, ey2, ex2 = [int(v) for v in cell["ext_bbox"]]
        if not inst_mask.any():
            feats = all_feature_names(return_empty=True)
        else:
            feats = compute_cell_features(
                image_hw, instance_map, inst_mask, inst_id,
                float(cell["centroid_y"]), float(cell["centroid_x"]),
                ey1, ex1, ey2, ex2, patch_centroids,
                float(image_mean_equiv_radius),
            )
        return np.array([feats[name] for name in morph_columns], dtype=np.float32)

    # Per-cell mahotas / ndimage operations release the GIL; the thread pool uses the job's available CPUs.
    if n_workers == 1 or n_row == 1:
        for row_idx in range(n_row):
            rows[row_idx] = fill_row(row_idx)
        return rows
    with ThreadPoolExecutor(max_workers=min(n_workers, n_row)) as pool:
        for row_idx, vec in enumerate(pool.map(fill_row, range(n_row), chunksize=1)):
            rows[row_idx] = vec
    return rows


def crop_instance_arrays(
    image: np.ndarray,
    mask: np.ndarray,
    inst_mask: np.ndarray,
    pad: int = _INSTANCE_CROP_PAD,
) -> dict[str, object] | None:
    """Find the padded axis-aligned bounding box of foreground pixels; return None if there is no foreground."""

    ys, xs = np.where(inst_mask)
    if ys.size == 0:
        return None
    height, width = inst_mask.shape
    y0 = max(0, int(ys.min()) - pad)
    x0 = max(0, int(xs.min()) - pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    x1 = min(width, int(xs.max()) + 1 + pad)
    return {
        "image": image[y0:y1, x0:x1],
        "labels": mask[y0:y1, x0:x1],
        "inst": inst_mask[y0:y1, x0:x1],
        "y0": y0,
        "x0": x0,
    }


def all_feature_names(return_empty: bool = False) -> list[str] | dict[str, float]:
    """Return canonical column names to align extraction and grouping; return an all-nan dictionary when return_empty is set."""

    names = []
    for group_name in FEATURE_GROUP_ORDER:
        names.extend(FEATURE_GROUP_COLUMNS[group_name])
    if return_empty:
        return {c: np.nan for c in names}
    return names


def extract_regionprops_scalar(inst_mask: np.ndarray) -> dict[str, float]:
    """Extract mask-only regionprops scalars, derived shape descriptors, and moments; exclude grayscale statistics."""

    props = regionprops(inst_mask.astype(np.int32))
    if not props:
        keys = [n for n in all_feature_names() if n.startswith(("morph_", "moment_", "inertia_"))]
        return {k: np.nan for k in keys}
    p = props[0]
    # Geometric scalars retained directly.
    feats = {
        "morph_area": float(p.area),
        "morph_axis_major_length": float(p.axis_major_length),
        "morph_axis_minor_length": float(p.axis_minor_length),
        "morph_eccentricity": float(p.eccentricity),
        "morph_euler_number": float(p.euler_number),
        "morph_extent": float(p.extent),
        "morph_feret_diameter_max": float(p.feret_diameter_max),
        "morph_orientation": float(p.orientation),
        "morph_perimeter": float(p.perimeter),
        "morph_solidity": float(p.solidity),
    }
    # Isoperimetric quotient and major/minor axis ratio; omit compactness as it is the reciprocal of circularity.
    area = float(p.area)
    perimeter = float(p.perimeter) if p.perimeter > 0 else np.nan
    major = float(p.axis_major_length)
    minor = float(p.axis_minor_length) if p.axis_minor_length > 0 else np.nan
    feats["morph_circularity"] = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter == perimeter else np.nan
    feats["morph_aspect_ratio"] = float(major / minor) if minor == minor and minor > 0 else np.nan
    feats["morph_feret_diameter_min"] = min_feret_diameter(inst_mask)
    feats.update(extract_moment_features(p))
    return feats


def extract_moment_features(prop) -> dict[str, float]:
    """Compute normalized moments, Hu moments, inertia tensor, and third-order central moments (absolute-scale supplements)."""

    Mu = np.asarray(prop.moments_central, dtype=np.float64)
    Nu = np.asarray(prop.moments_normalized, dtype=np.float64)
    hu = np.asarray(prop.moments_hu, dtype=np.float64)
    It = np.asarray(prop.inertia_tensor, dtype=np.float64)
    eig = np.asarray(prop.inertia_tensor_eigvals, dtype=np.float64)
    feats = {}
    # Second-order central moments duplicate inertia_*; retain only third-order central moments and omit all raw moments.
    for i in range(4):
        for j in range(4 - i):
            if i + j >= 3:
                feats[f"moment_central_{i}{j}"] = float(Mu[i, j])
            if i + j >= 2:
                feats[f"moment_norm_{i}{j}"] = float(Nu[i, j])
    for k in range(7):
        feats[f"moment_hu_{k + 1}"] = float(hu[k])
    feats["inertia_xx"] = float(It[0, 0])
    feats["inertia_xy"] = float(It[0, 1])
    feats["inertia_yy"] = float(It[1, 1])
    feats["inertia_eig_0"] = float(eig[0])
    feats["inertia_eig_1"] = float(eig[1])
    return feats


def min_feret_diameter(inst_mask: np.ndarray) -> float:
    """Use rotating calipers on the convex hull to find the minimum distance between parallel tangents (MinFeretDiameter)."""

    ys, xs = np.where(inst_mask)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    if len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return np.nan
    hull_pts = pts[hull.vertices]
    n = len(hull_pts)
    min_w = np.inf
    for i in range(n):
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        elen = float(np.hypot(edge[0], edge[1]))
        if elen < 1e-12:
            continue
        nx, ny = -edge[1] / elen, edge[0] / elen
        proj = hull_pts[:, 0] * nx + hull_pts[:, 1] * ny
        min_w = min(min_w, float(proj.max() - proj.min()))
    return float(min_w) if min_w < np.inf else np.nan


def extract_bbox_intensity_extended(image: np.ndarray, y1: int, x1: int, y2: int, x2: int) -> dict[str, float]:
    """Extended-bbox intensity statistics: skewness, kurtosis, additional percentiles, range, and gradient magnitude."""

    if image[y1:y2, x1:x2].size == 0:
        return {f"bbox_intensity_{s}": np.nan for s in BBOX_INTENSITY_STATS}
    crop = image[y1:y2, x1:x2]
    vals = crop.ravel()
    gy, gx = np.gradient(crop)
    grad = np.sqrt(gy ** 2 + gx ** 2).ravel()
    q01, q10, q25, q75, q90, q99 = np.percentile(vals, (1, 10, 25, 75, 90, 99))
    return {
        "bbox_intensity_skew": float(skew(vals)),
        "bbox_intensity_kurt": float(kurtosis(vals)),
        "bbox_intensity_p10": float(q10),
        "bbox_intensity_p25": float(q25),
        "bbox_intensity_p75": float(q75),
        "bbox_intensity_p90": float(q90),
        "bbox_intensity_range": float(q99 - q01),
        "bbox_intensity_grad_mean": float(grad.mean()),
        "bbox_intensity_grad_std": float(grad.std()),
    }


def extract_distance_transform_features(
    inst_mask: np.ndarray,
    dt: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute distance-transform statistics inside the cell mask; omit the minimum, which is always 0 at the boundary."""

    if not inst_mask.any():
        return {f"disttrans_{s}": np.nan for s in DISTTRANS_STATS}
    if dt is None:
        dt = ndimage.distance_transform_edt(inst_mask)
    vals = dt[inst_mask]
    return {
        "disttrans_mean": float(vals.mean()),
        "disttrans_std": float(vals.std()),
        "disttrans_max": float(vals.max()),
        "disttrans_median": float(np.median(vals)),
    }


def extract_boundary_radial_features(
    inst_mask: np.ndarray,
    cy: float,
    cx: float,
    boundary: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute radial-distance statistics from cell boundary points to the centroid."""

    if not inst_mask.any():
        return {f"radial_{s}": np.nan for s in RADIAL_STATS}
    if boundary is None:
        boundary = inst_mask ^ binary_erosion(inst_mask, _EROSION_FOOTPRINT)
    ys, xs = np.where(boundary)
    if ys.size == 0:
        return {f"radial_{s}": np.nan for s in RADIAL_STATS}
    dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return {
        "radial_mean": float(dists.mean()),
        "radial_std": float(dists.std()),
        "radial_max": float(dists.max()),
        "radial_min": float(dists.min()),
        "radial_median": float(np.median(dists)),
    }


def extract_fourier_descriptor_features(inst_mask: np.ndarray, n_coeffs: int = N_FOURIER_COEFFS) -> dict[str, float]:
    """Compute relative amplitudes of contour Fourier descriptors; omit the DC term, which is always 1 after normalization."""

    feats = {f"fourier_{i:02d}": np.nan for i in range(1, n_coeffs)}
    contours = find_contours(inst_mask, 0.5)
    if not contours:
        return feats
    contour = max(contours, key=len)
    if len(contour) < 4:
        return feats
    pts = contour[:, 0] + 1j * contour[:, 1]
    fft = np.fft.fft(pts)
    amplitudes = np.abs(fft)
    amplitudes = amplitudes / (amplitudes[0] + 1e-12)
    n_keep = min(n_coeffs, len(amplitudes))
    for i in range(1, n_keep):
        feats[f"fourier_{i:02d}"] = float(amplitudes[i])
    return feats


def extract_gradient_hog_features(
    image: np.ndarray, inst_mask: np.ndarray, y1: int, x1: int, y2: int, x2: int, n_bins: int = N_HOG_BINS,
) -> dict[str, float]:
    """Compute a gradient-orientation histogram (HOG style) for cell pixels in the bbox; return normalized bin values."""

    feats = {f"hog_{i:02d}": np.nan for i in range(n_bins)}
    crop = image[y1:y2, x1:x2]
    m_crop = inst_mask[y1:y2, x1:x2]
    if crop.size == 0 or not m_crop.any():
        return feats
    gy, gx = np.gradient(crop)
    magnitude = np.sqrt(gy ** 2 + gx ** 2)
    orientation = np.arctan2(gy, gx) * 180.0 / np.pi
    orientation = (orientation + 180.0) % 180.0
    mag_fg = magnitude[m_crop]
    ori_fg = orientation[m_crop]
    bin_idx = np.floor(ori_fg * n_bins / 180.0).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    hist = np.bincount(bin_idx, weights=mag_fg, minlength=n_bins).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    for i in range(n_bins):
        feats[f"hog_{i:02d}"] = float(hist[i])
    return feats


def extract_zernike_features(inst_mask: np.ndarray) -> dict[str, float]:
    """Compute amplitude coefficients for Zernike orders 0-9; omit (0,0) and (1,1), which are nearly constant after normalization."""

    names = [f"zernike_{n}_{m}" for n, m in ZERNIKE_NM_KEEP]
    ys, xs = np.where(inst_mask)
    if ys.size < 4:
        return {c: np.nan for c in names}
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = inst_mask[y1:y2, x1:x2].astype(np.uint8)
    cy, cx = float(ys.mean() - y1), float(xs.mean() - x1)
    radius = float(np.sqrt((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).max())
    radius = max(1, int(np.ceil(radius)))
    try:
        zvals = zernike_moments(crop, radius=radius, degree=ZERNIKE_DEGREE, cm=(cy, cx))
    except ValueError:
        return {c: np.nan for c in names}
    if len(zvals) != len(ZERNIKE_NM):
        return {c: np.nan for c in names}
    return {
        f"zernike_{n}_{m}": float(zvals[i])
        for i, (n, m) in enumerate(ZERNIKE_NM)
        if (n, m) not in ZERNIKE_DROP
    }


def extract_object_intensity_features(
    image: np.ndarray,
    inst_mask: np.ndarray,
    cy: float,
    cx: float,
    boundary: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute intensity location / dispersion statistics for interior and edge pixels; normalize mass displacement by the instance's equivalent radius."""

    nan_out = {c: np.nan for c in OBJINT_NAMES}
    vals = image[inst_mask]
    if vals.size == 0:
        return nan_out
    if boundary is None:
        boundary = inst_mask ^ binary_erosion(inst_mask, _EROSION_FOOTPRINT)
    edge_vals = image[boundary] if boundary.any() else np.array([], dtype=np.float64)
    ys, xs = np.where(inst_mask)
    wsum = float(vals.sum())
    area = float(vals.size)
    r_eq = float(np.sqrt(area / np.pi)) if area > 0 else np.nan
    if wsum > 0 and r_eq > 0:
        cy_I = float(np.dot(ys, vals) / wsum)
        cx_I = float(np.dot(xs, vals) / wsum)
        mass_disp = float(np.hypot(cy_I - cy, cx_I - cx) / r_eq)
    else:
        mass_disp = np.nan
    feats = {
        "objint_median": float(np.median(vals)),
        "objint_p25": float(np.percentile(vals, 25)),
        "objint_p75": float(np.percentile(vals, 75)),
        "objint_mad": float(median_abs_deviation(vals, scale=1.0)),
        "objint_mass_displacement": mass_disp,
    }
    if edge_vals.size == 0:
        feats.update({
            "objint_edge_mean": np.nan,
            "objint_edge_std": np.nan,
            "objint_edge_min": np.nan,
            "objint_edge_max": np.nan,
        })
    else:
        feats.update({
            "objint_edge_mean": float(edge_vals.mean()),
            "objint_edge_std": float(edge_vals.std()),
            "objint_edge_min": float(edge_vals.min()),
            "objint_edge_max": float(edge_vals.max()),
        })
    return feats


def extract_intensity_distribution_features(
    image: np.ndarray,
    inst_mask: np.ndarray,
    n_bins: int = N_INTDIST_BINS,
    dt: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute intensity distributions in outward rings from the distance-transform maximum; sample only inst_mask foreground."""

    names = list(INTDIST_NAMES) if n_bins == N_INTDIST_BINS else [
        f"intdist_{stat}_b{b:02d}" for stat in ("frac", "meanfrac", "radialcv") for b in range(n_bins)
    ]
    nan_out = {c: np.nan for c in names}
    if not inst_mask.any():
        return nan_out
    if dt is None:
        dt = ndimage.distance_transform_edt(inst_mask)
    cy_c, cx_c = np.unravel_index(np.argmax(dt), dt.shape)
    ys, xs = np.where(inst_mask)
    dist = np.sqrt((ys - float(cy_c)) ** 2 + (xs - float(cx_c)) ** 2)
    rmax = float(dist.max())
    if rmax < 1e-12:
        return nan_out
    bin_id = np.minimum((dist / rmax * n_bins).astype(np.int64), n_bins - 1)
    intens = image[ys, xs].astype(np.float64)
    i_tot = float(intens.sum())
    n_tot = float(len(intens))
    mean_all = i_tot / n_tot if n_tot > 0 else np.nan
    feats = {}
    for b in range(n_bins):
        sel = intens[bin_id == b]
        if sel.size == 0 or i_tot == 0 or mean_all == 0:
            feats[f"intdist_frac_b{b:02d}"] = np.nan
            feats[f"intdist_meanfrac_b{b:02d}"] = np.nan
            feats[f"intdist_radialcv_b{b:02d}"] = np.nan
            continue
        feats[f"intdist_frac_b{b:02d}"] = float(sel.sum() / i_tot)
        feats[f"intdist_meanfrac_b{b:02d}"] = float((sel.mean()) / mean_all)
        mu = float(sel.mean())
        feats[f"intdist_radialcv_b{b:02d}"] = float(sel.std() / mu) if abs(mu) > 1e-12 else np.nan
    return feats


def extract_haralick_features(image: np.ndarray, inst_mask: np.ndarray) -> dict[str, float]:
    """Compute 13 within-object Haralick features at distance 1, averaged over four directions; omit d3 as a repeat measurement of d1."""

    names = [f"haralick_d{d}_{name}" for d in HARALICK_DISTANCES for name in HARALICK_SHORT]
    nan_out = {c: np.nan for c in names}
    ys, xs = np.where(inst_mask)
    if ys.size < 4:
        return nan_out
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = image[y1:y2, x1:x2]
    m = inst_mask[y1:y2, x1:x2]
    if min(crop.shape) < 4:
        return nan_out
    vals = crop[m]
    vmin, vmax = float(vals.min()), float(vals.max())
    q = np.zeros(crop.shape, dtype=np.uint8)
    if vmax > vmin:
        scaled = np.floor((vals - vmin) / (vmax - vmin) * (HARALICK_LEVELS - 1)).astype(np.uint8) + 1
        q[m] = np.clip(scaled, 1, HARALICK_LEVELS)
    else:
        q[m] = 1
    feats = {}
    for d in HARALICK_DISTANCES:
        if min(q.shape) <= d:
            for name in HARALICK_SHORT:
                feats[f"haralick_d{d}_{name}"] = np.nan
            continue
        try:
            vec = haralick(q, ignore_zeros=True, return_mean=True, distance=int(d))
        except ValueError:
            for name in HARALICK_SHORT:
                feats[f"haralick_d{d}_{name}"] = np.nan
            continue
        for i, name in enumerate(HARALICK_SHORT):
            feats[f"haralick_d{d}_{name}"] = float(vec[i])
    return feats


def extract_neighbor_features(
    mask: np.ndarray,
    inst_id: int,
    patch_centroids: dict[int, tuple[float, float]],
    image_mean_equiv_radius: float,
) -> dict[str, float]:
    """Compute neighborhood contacts and nearest centroid distances within the same patch (MeasureObjectNeighbors).

    Divide centroid distances by the mean equivalent radius of all valid-region instances in that image, yielding multiples of a typical cell radius.
    """

    inst_mask = (mask == inst_id)
    dil = ndimage.binary_dilation(inst_mask, structure=_NEIGHBOR_STRUCT)
    ring = dil & ~inst_mask
    n_ring = int(ring.sum())
    ring_labels = mask[ring]
    other = ring_labels[ring_labels > 0]
    neighbor_ids = set(int(v) for v in np.unique(other) if int(v) != inst_id)
    percent = float(other.size / n_ring) if n_ring > 0 else 0.0
    cy, cx = patch_centroids.get(inst_id, (np.nan, np.nan))
    dists = []
    for oid, (oy, ox) in patch_centroids.items():
        if oid == inst_id:
            continue
        dists.append(float(np.hypot(oy - cy, ox - cx)))
    dists.sort()
    d1 = dists[0] if len(dists) >= 1 else np.nan
    d2 = dists[1] if len(dists) >= 2 else np.nan
    r_bar = float(image_mean_equiv_radius)
    if r_bar > 0:
        d1 = float(d1 / r_bar) if d1 == d1 else np.nan
        d2 = float(d2 / r_bar) if d2 == d2 else np.nan
    else:
        d1 = np.nan
        d2 = np.nan
    return {
        "neighbor_n": float(len(neighbor_ids)),
        "neighbor_percent_touching": percent,
        "neighbor_dist_closest": d1,
        "neighbor_dist_second": d2,
    }


def build_feature_groups(ext_df: pd.DataFrame) -> dict[str, list[str]]:
    """Assign extended features to six physical-quantity groups using canonical column names."""

    groups: dict[str, list[str]] = {name: [] for name in FEATURE_GROUP_ORDER}
    name_to_group = {
        col: group_name
        for group_name, cols in FEATURE_GROUP_COLUMNS.items()
        for col in cols
    }
    for c in ext_df.columns:
        if c == "obs_name":
            continue
        groups[name_to_group[c]].append(c)
    return groups
