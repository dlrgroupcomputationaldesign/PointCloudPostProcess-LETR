"""Dense length-z histogram + opening detection for one wall.

This version keeps the original wall-length/z histogram workflow and adds a
stronger opening detector built from the log-count image:

  1. local background subtraction on log(1 + count),
  2. adaptive low-density candidate extraction,
  3. connected-component rectangular proposals,
  4. sliding-window rectangular proposals for messy/merged regions,
  5. paired edge-support scoring from the log image,
  6. optional snap-to-frame-edge box refinement,
  7. non-maximum suppression,
  8. window-row consistency filtering/boosting.

The sign convention is:

    residual = blurred_log_count - log_count

so high residual means "this cell has fewer points than expected locally".
"""

import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
WALL_ID = 17
FT_TO_M = 0.3048
LENGTH_BIN_M = 0.05
Z_BIN_M = 0.05
MARGIN_M = 0.25
COUNT_THRESHOLD = 100
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000
OUTPUT_PNG = "wall_17_dense_lengthz_opening_detection_advanced.png"
OUTPUT_CSV = "wall_17_opening_candidates_advanced.csv"

# Set to (zmin_m, zmax_m) to clip explicitly; None = auto-clip to populated rows.
Z_CROP_M = None

# -----------------------------------------------------------------------------
# Opening-detection tuning parameters
# -----------------------------------------------------------------------------
# Local background blur. Larger values ignore small frame/detail texture and keep
# the broad wall-density trend. Order is z, length. These are intentionally
# larger than the first version because your residual map still had a lot of
# local texture/noise.
LOCAL_BG_SIGMA_Z_M = 0.55
LOCAL_BG_SIGMA_X_M = 1.20

# Candidate threshold. A cell is an opening candidate if its residual is high.
# The final threshold is max(percentile threshold, robust MAD threshold).
RESIDUAL_PERCENTILE = 90.0
RESIDUAL_MAD_K = 1.5
MIN_LOCAL_LOG_BACKGROUND = 3.8  # avoids selecting globally empty/unscanned zones

# Morphology on the candidate mask, in meters.
MORPH_CLOSE_Z_M = 0.12
MORPH_CLOSE_X_M = 0.20
MORPH_OPEN_Z_M = 0.05
MORPH_OPEN_X_M = 0.05

# Window geometry filters, in meters. These defaults are tuned for the repeated
# upper-window pattern in your wall_17 result. Loosen these for other buildings.
ENABLE_WINDOW_DETECTION = True
WINDOW_MIN_WIDTH_M = 0.45
WINDOW_MAX_WIDTH_M = 1.45
WINDOW_MIN_HEIGHT_M = 0.55
WINDOW_MAX_HEIGHT_M = 1.55
WINDOW_MIN_AREA_M2 = 0.25
WINDOW_MAX_AREA_M2 = 2.50
WINDOW_MIN_ASPECT_RATIO = 0.35  # width / height
WINDOW_MAX_ASPECT_RATIO = 2.20

# Window vertical band relative to the cropped z_min. Set any value to None to
# disable that particular constraint. These are intentionally broad.
WINDOW_MIN_BOTTOM_ABOVE_ZMIN_M = 1.10
WINDOW_MAX_BOTTOM_ABOVE_ZMIN_M = 2.45
WINDOW_MIN_TOP_ABOVE_ZMIN_M = 1.95
WINDOW_MAX_TOP_BELOW_ZMAX_M = None

# Door geometry filters, in meters. Door detection is kept separate from window
# detection so the window priors do not incorrectly reject doors.
ENABLE_DOOR_DETECTION = True
DOOR_MIN_WIDTH_M = 0.55
DOOR_MAX_WIDTH_M = 1.80
DOOR_MIN_HEIGHT_M = 1.65
DOOR_MAX_HEIGHT_M = 2.90
DOOR_MIN_AREA_M2 = 0.90
DOOR_MAX_AREA_M2 = 5.00
DOOR_MIN_ASPECT_RATIO = 0.20
DOOR_MAX_ASPECT_RATIO = 1.20
DOOR_MAX_BOTTOM_ABOVE_ZMIN_M = 0.45

# Connected-component filters.
MIN_COMPONENT_FILL_RATIO = 0.08  # component pixels / bbox pixels
MAX_COMPONENT_FILL_RATIO = 1.00

# Edge scoring. Edges are measured in narrow bands around the bbox. The paired
# vertical score is the important part: real windows usually have both a left and
# a right edge, while wall seams often only have one.
EDGE_BAND_M = 0.10
INTERIOR_EDGE_MARGIN_M = 0.12
MIN_VERTICAL_PAIR_SCORE = 0.18
MIN_HORIZONTAL_PAIR_SCORE_WINDOW = 0.06
MIN_EDGE_SCORE = 0.25
MIN_FINAL_SCORE = 0.75

# Snap each rectangular proposal outward/inward to the nearest strong frame edge.
SNAP_TO_EDGES = True
SNAP_SEARCH_MARGIN_M = 0.22

# Sliding-window proposals recover windows when the candidate mask is merged into
# a large noisy region, which happened on the left side of your example.
ENABLE_SLIDING_RECTANGLE_PROPOSALS = True
SLIDE_WINDOW_WIDTHS_M = (0.55, 0.70, 0.85, 1.00, 1.20)
SLIDE_WINDOW_HEIGHTS_M = (0.75, 0.90, 1.10, 1.30)
SLIDE_STEP_X_M = 0.10
SLIDE_STEP_Z_M = 0.10
SLIDE_QUICK_DENSITY_MIN = 0.22
SLIDE_QUICK_CANDIDATE_MIN = 0.03
SLIDE_MAX_RAW_PROPOSALS = 2500

# Do not trust detections touching the crop/image boundary.
BOUNDARY_MARGIN_M = 0.15
BOUNDARY_PENALTY = 0.40

# Row consistency for repeated windows. This boosts boxes that align with the
# dominant z row and rejects weak boxes outside that row.
APPLY_WINDOW_ROW_CONSISTENCY = True
ROW_CONSISTENCY_MIN_WINDOWS = 4
ROW_REFERENCE_TOP_K = 12
ROW_REFERENCE_MIN_SCORE = 0.68
WINDOW_ROW_TOLERANCE_Z0_M = 0.35
WINDOW_ROW_TOLERANCE_Z1_M = 0.35
ROW_ALIGNED_BONUS = 0.12
ROW_MISALIGNED_PENALTY = 0.22
ROW_MISALIGNED_KEEP_SCORE = 1.05

# Non-maximum suppression across component and sliding proposals.
NMS_IOU_THRESHOLD = 0.35
MAX_DETECTIONS = None  # set to an int to keep only the top detections


def _compute_xyz_min_from_annotation(path, chunksize=ANNOTATION_CHUNK_SIZE):
    """Mirror np.amin(data_label, axis=0)[0:3] inside create_labels()."""
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(
            xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0)
        )
    return xyz_min


def _transform_wall_to_e57_meters(wall_ft, xyz_min_ft, ft_to_m=FT_TO_M):
    """Inverse of create_labels shift: e57_m = (csv_ft + xyz_min) * 0.3048."""
    bbox = [
        {
            "x": (p["x"] + xyz_min_ft[0]) * ft_to_m,
            "y": (p["y"] + xyz_min_ft[1]) * ft_to_m,
            "z": (p["z"] + xyz_min_ft[2]) * ft_to_m,
        }
        for p in wall_ft["bbox"]
    ]
    footprint = [
        {
            "x": (p["x"] + xyz_min_ft[0]) * ft_to_m,
            "y": (p["y"] + xyz_min_ft[1]) * ft_to_m,
        }
        for p in wall_ft.get("footprint", [])
    ]
    zr = wall_ft["zRange"]
    return {
        "id": wall_ft["id"],
        "bbox": bbox,
        "footprint": footprint,
        "zRange": {
            "min": (zr["min"] + xyz_min_ft[2]) * ft_to_m,
            "max": (zr["max"] + xyz_min_ft[2]) * ft_to_m,
        },
    }


def _longest_footprint_edge_axis(wall):
    fp = wall.get("footprint")
    if fp and len(fp) >= 3:
        pts = np.array([[p["x"], p["y"]] for p in fp], dtype=np.float64)
        edges = []
        for i in range(len(pts)):
            edges.append(pts[(i + 1) % len(pts)] - pts[i])
    else:
        pts = np.array([[p["x"], p["y"]] for p in wall["bbox"]], dtype=np.float64)
        edges = [pts[j] - pts[i] for i in range(len(pts)) for j in range(i + 1, len(pts))]

    norms = np.linalg.norm(edges, axis=1)
    best = int(np.argmax(norms))
    axis = edges[best] / norms[best]
    projected = pts @ axis
    return axis, float(projected.min()), float(projected.max())


def _iter_e57_raw_xyz_chunks(e57_path, chunk_size=CHUNK_SIZE):
    import pye57

    e57 = pye57.E57(e57_path)
    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                count = reader.read()
                if count <= 0:
                    break
                yield np.column_stack(
                    (
                        data["cartesianX"][:count],
                        data["cartesianY"][:count],
                        data["cartesianZ"][:count],
                    )
                ).astype(np.float64, copy=False)
    finally:
        e57.close()


def _meters_to_pixels(distance_m, bin_m, minimum=1):
    return max(minimum, int(round(float(distance_m) / float(bin_m))))


def _clip_slice(start, stop, limit):
    start = max(0, int(start))
    stop = min(int(limit), int(stop))
    if stop <= start:
        return slice(0, 0)
    return slice(start, stop)


def _safe_mean(arr):
    if arr.size == 0:
        return 0.0
    return float(np.nanmean(arr))


def _integral_image(arr):
    """Integral image with a zero-padded first row/column."""
    arr = np.asarray(arr, dtype=np.float64)
    return np.pad(arr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")


def _box_sum(ii, r0, r1, c0, c1):
    return float(ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0])


def _box_mean(ii, r0, r1, c0, c1):
    area = max(1, (r1 - r0) * (c1 - c0))
    return _box_sum(ii, r0, r1, c0, c1) / area


def _prepare_edge_maps(log_img):
    from scipy import ndimage as ndi

    smooth = ndi.gaussian_filter(log_img.astype(np.float32), sigma=1.0, mode="nearest")
    grad_x = np.abs(ndi.sobel(smooth, axis=1))
    grad_z = np.abs(ndi.sobel(smooth, axis=0))
    gx_scale = float(np.percentile(grad_x, 95)) + 1e-6
    gz_scale = float(np.percentile(grad_z, 95)) + 1e-6
    return {
        "grad_x": grad_x,
        "grad_z": grad_z,
        "gx_scale": gx_scale,
        "gz_scale": gz_scale,
    }


def _edge_band_mean(grad, r0, r1, c0, c1):
    h, w = grad.shape
    return _safe_mean(grad[_clip_slice(r0, r1, h), _clip_slice(c0, c1, w)])


def _edge_support_score(edge_maps, log_img, r0, r1, c0, c1, z_bin_m, x_bin_m, opening_type="window"):
    """Measure whether a candidate bbox is supported by paired frame-like edges."""
    grad_x = edge_maps["grad_x"]
    grad_z = edge_maps["grad_z"]
    gx_scale = edge_maps["gx_scale"]
    gz_scale = edge_maps["gz_scale"]
    h, w = log_img.shape

    c_band = _meters_to_pixels(EDGE_BAND_M, x_bin_m)
    r_band = _meters_to_pixels(EDGE_BAND_M, z_bin_m)

    left = _edge_band_mean(grad_x, r0, r1, c0 - c_band, c0 + c_band + 1) / gx_scale
    right = _edge_band_mean(grad_x, r0, r1, c1 - c_band - 1, c1 + c_band) / gx_scale
    bottom = _edge_band_mean(grad_z, r0 - r_band, r0 + r_band + 1, c0, c1) / gz_scale
    top = _edge_band_mean(grad_z, r1 - r_band - 1, r1 + r_band, c0, c1) / gz_scale

    vertical_pair = min(left, right)
    horizontal_pair = min(top, bottom)
    horizontal_best = max(top, bottom)

    # Penalize strong clutter inside the opening, but only gently; real windows
    # can contain mullions/blinds, so this should not dominate the score.
    margin_c = _meters_to_pixels(INTERIOR_EDGE_MARGIN_M, x_bin_m)
    margin_r = _meters_to_pixels(INTERIOR_EDGE_MARGIN_M, z_bin_m)
    inner_r0 = min(max(0, r0 + margin_r), h)
    inner_r1 = max(min(h, r1 - margin_r), inner_r0)
    inner_c0 = min(max(0, c0 + margin_c), w)
    inner_c1 = max(min(w, c1 - margin_c), inner_c0)
    interior_x = _safe_mean(grad_x[inner_r0:inner_r1, inner_c0:inner_c1]) / gx_scale
    interior_z = _safe_mean(grad_z[inner_r0:inner_r1, inner_c0:inner_c1]) / gz_scale
    interior_clutter = 0.5 * (interior_x + interior_z)

    # Cap very strong edges so a single extreme seam cannot dominate.
    vertical_pair = min(vertical_pair, 3.0)
    horizontal_pair = min(horizontal_pair, 3.0)
    horizontal_best = min(horizontal_best, 3.0)
    interior_clutter = min(interior_clutter, 3.0)

    if opening_type == "door":
        edge_score = 0.62 * vertical_pair + 0.28 * top + 0.10 * horizontal_best
    else:
        edge_score = 0.55 * vertical_pair + 0.35 * horizontal_pair + 0.10 * horizontal_best
    edge_score = max(0.0, edge_score - 0.10 * interior_clutter)

    return {
        "edge_score": float(edge_score),
        "left_edge": float(left),
        "right_edge": float(right),
        "top_edge": float(top),
        "bottom_edge": float(bottom),
        "vertical_pair": float(vertical_pair),
        "horizontal_pair": float(horizontal_pair),
        "interior_edge_clutter": float(interior_clutter),
    }


def _snap_bbox_to_edges(edge_maps, r0, r1, c0, c1, z_bin_m, x_bin_m, h, w):
    """Move bbox sides to nearby strong x/z gradients.

    The proposal can be the low-density interior; snapping moves its sides toward
    the frame edges. Search is local so boxes do not jump to unrelated wall seams.
    """
    if not SNAP_TO_EDGES:
        return r0, r1, c0, c1

    grad_x = edge_maps["grad_x"]
    grad_z = edge_maps["grad_z"]
    search_c = _meters_to_pixels(SNAP_SEARCH_MARGIN_M, x_bin_m)
    search_r = _meters_to_pixels(SNAP_SEARCH_MARGIN_M, z_bin_m)
    c_band = _meters_to_pixels(EDGE_BAND_M, x_bin_m)
    r_band = _meters_to_pixels(EDGE_BAND_M, z_bin_m)

    def col_score(c):
        return _edge_band_mean(grad_x, r0, r1, c - c_band, c + c_band + 1)

    def row_score(r):
        return _edge_band_mean(grad_z, r - r_band, r + r_band + 1, c0, c1)

    c0_candidates = range(max(0, c0 - search_c), min(w - 1, c0 + search_c) + 1)
    c1_candidates = range(max(1, c1 - search_c), min(w, c1 + search_c) + 1)
    r0_candidates = range(max(0, r0 - search_r), min(h - 1, r0 + search_r) + 1)
    r1_candidates = range(max(1, r1 - search_r), min(h, r1 + search_r) + 1)

    new_c0 = max(c0_candidates, key=col_score, default=c0)
    # c1 is exclusive, so evaluate the edge at c1 - 1.
    new_c1 = max(c1_candidates, key=lambda c: col_score(c - 1), default=c1)
    new_r0 = max(r0_candidates, key=row_score, default=r0)
    new_r1 = max(r1_candidates, key=lambda r: row_score(r - 1), default=r1)

    # Do not allow snapping to invert or collapse the rectangle.
    min_w_px = _meters_to_pixels(min(WINDOW_MIN_WIDTH_M, DOOR_MIN_WIDTH_M), x_bin_m)
    min_h_px = _meters_to_pixels(min(WINDOW_MIN_HEIGHT_M, DOOR_MIN_HEIGHT_M), z_bin_m)
    if new_c1 - new_c0 < min_w_px:
        new_c0, new_c1 = c0, c1
    if new_r1 - new_r0 < min_h_px:
        new_r0, new_r1 = r0, r1

    return int(new_r0), int(new_r1), int(new_c0), int(new_c1)


def _classify_opening(width_m, height_m, z0_m, z1_m, z_min_m, z_max_m):
    area_m2 = width_m * height_m
    aspect = width_m / max(height_m, 1e-6)
    bottom_above = z0_m - z_min_m
    top_above = z1_m - z_min_m
    top_below = z_max_m - z1_m

    if ENABLE_WINDOW_DETECTION:
        ok = (
            WINDOW_MIN_WIDTH_M <= width_m <= WINDOW_MAX_WIDTH_M
            and WINDOW_MIN_HEIGHT_M <= height_m <= WINDOW_MAX_HEIGHT_M
            and WINDOW_MIN_AREA_M2 <= area_m2 <= WINDOW_MAX_AREA_M2
            and WINDOW_MIN_ASPECT_RATIO <= aspect <= WINDOW_MAX_ASPECT_RATIO
        )
        if WINDOW_MIN_BOTTOM_ABOVE_ZMIN_M is not None:
            ok = ok and bottom_above >= WINDOW_MIN_BOTTOM_ABOVE_ZMIN_M
        if WINDOW_MAX_BOTTOM_ABOVE_ZMIN_M is not None:
            ok = ok and bottom_above <= WINDOW_MAX_BOTTOM_ABOVE_ZMIN_M
        if WINDOW_MIN_TOP_ABOVE_ZMIN_M is not None:
            ok = ok and top_above >= WINDOW_MIN_TOP_ABOVE_ZMIN_M
        if WINDOW_MAX_TOP_BELOW_ZMAX_M is not None:
            ok = ok and top_below >= WINDOW_MAX_TOP_BELOW_ZMAX_M
        if ok:
            return "window"

    if ENABLE_DOOR_DETECTION:
        ok = (
            DOOR_MIN_WIDTH_M <= width_m <= DOOR_MAX_WIDTH_M
            and DOOR_MIN_HEIGHT_M <= height_m <= DOOR_MAX_HEIGHT_M
            and DOOR_MIN_AREA_M2 <= area_m2 <= DOOR_MAX_AREA_M2
            and DOOR_MIN_ASPECT_RATIO <= aspect <= DOOR_MAX_ASPECT_RATIO
            and bottom_above <= DOOR_MAX_BOTTOM_ABOVE_ZMIN_M
        )
        if ok:
            return "door"

    return None


def _score_bbox(
    *,
    log_img,
    background,
    residual,
    candidate,
    edge_maps,
    threshold,
    r0,
    r1,
    c0,
    c1,
    x_bin_m,
    z_bin_m,
    wall_length_m,
    z_min_m,
    z_max_m,
    source,
    label_id=None,
    component_fill_ratio=None,
    allow_snap=True,
):
    h, w = log_img.shape
    r0 = int(np.clip(r0, 0, h - 1))
    r1 = int(np.clip(r1, r0 + 1, h))
    c0 = int(np.clip(c0, 0, w - 1))
    c1 = int(np.clip(c1, c0 + 1, w))

    if allow_snap:
        r0, r1, c0, c1 = _snap_bbox_to_edges(edge_maps, r0, r1, c0, c1, z_bin_m, x_bin_m, h, w)

    width_m = (c1 - c0) * x_bin_m
    height_m = (r1 - r0) * z_bin_m
    x0_m = c0 * x_bin_m
    x1_m = min(wall_length_m, c1 * x_bin_m)
    z0_m = z_min_m + r0 * z_bin_m
    z1_m = z_min_m + r1 * z_bin_m
    opening_type = _classify_opening(width_m, height_m, z0_m, z1_m, z_min_m, z_max_m)
    if opening_type is None:
        return None

    if component_fill_ratio is not None:
        fill_ratio = float(component_fill_ratio)
        if not (MIN_COMPONENT_FILL_RATIO <= fill_ratio <= MAX_COMPONENT_FILL_RATIO):
            return None
    else:
        fill_ratio = float(np.mean(candidate[r0:r1, c0:c1])) if candidate[r0:r1, c0:c1].size else 0.0

    edge = _edge_support_score(edge_maps, log_img, r0, r1, c0, c1, z_bin_m, x_bin_m, opening_type)
    if edge["vertical_pair"] < MIN_VERTICAL_PAIR_SCORE:
        return None
    if opening_type == "window" and edge["horizontal_pair"] < MIN_HORIZONTAL_PAIR_SCORE_WINDOW:
        return None
    if edge["edge_score"] < MIN_EDGE_SCORE:
        return None

    bbox_residual = residual[r0:r1, c0:c1]
    bbox_background = background[r0:r1, c0:c1]
    bbox_log = log_img[r0:r1, c0:c1]
    bbox_candidate = candidate[r0:r1, c0:c1]

    mean_residual_bbox = _safe_mean(bbox_residual)
    mean_positive_residual = _safe_mean(np.maximum(bbox_residual, 0.0))
    mean_background = _safe_mean(bbox_background)
    mean_log = _safe_mean(bbox_log)
    candidate_support = float(np.mean(bbox_candidate)) if bbox_candidate.size else 0.0

    density_score = mean_positive_residual / (threshold + 1e-6)
    contrast_score = max(0.0, mean_background - mean_log) / (threshold + 1e-6)
    if component_fill_ratio is not None:
        rect_score = min(fill_ratio / max(MIN_COMPONENT_FILL_RATIO, 1e-6), 1.0)
    else:
        rect_score = min(candidate_support / 0.25, 1.0)

    boundary_margin_c = _meters_to_pixels(BOUNDARY_MARGIN_M, x_bin_m)
    boundary_margin_r = _meters_to_pixels(BOUNDARY_MARGIN_M, z_bin_m)
    touches_boundary = (
        c0 <= boundary_margin_c
        or c1 >= w - boundary_margin_c
        or r0 <= boundary_margin_r
        or r1 >= h - boundary_margin_r
    )
    boundary_penalty = BOUNDARY_PENALTY if touches_boundary else 0.0

    # Weighting: density finds holes, paired edges reject random low-density blobs,
    # rect/candidate support keeps component/sliding proposals anchored to the mask.
    final_score = (
        0.32 * density_score
        + 0.13 * contrast_score
        + 0.38 * edge["edge_score"]
        + 0.17 * rect_score
        - boundary_penalty
    )

    if final_score < MIN_FINAL_SCORE:
        return None

    area_m2 = width_m * height_m
    aspect = width_m / max(height_m, 1e-6)
    det = {
        "label_id": label_id,
        "source": source,
        "opening_type": opening_type,
        "score": float(final_score),
        "density_score": float(density_score),
        "contrast_score": float(contrast_score),
        "edge_score": float(edge["edge_score"]),
        "rect_score": float(rect_score),
        "candidate_support": float(candidate_support),
        "x0_m": float(x0_m),
        "x1_m": float(x1_m),
        "z0_m": float(z0_m),
        "z1_m": float(z1_m),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "area_m2": float(area_m2),
        "aspect": float(aspect),
        "fill_ratio": float(fill_ratio),
        "mean_residual_bbox": float(mean_residual_bbox),
        "mean_positive_residual": float(mean_positive_residual),
        "mean_background": float(mean_background),
        "mean_log": float(mean_log),
        "threshold": float(threshold),
        "touches_boundary": bool(touches_boundary),
        "left_edge": edge["left_edge"],
        "right_edge": edge["right_edge"],
        "top_edge": edge["top_edge"],
        "bottom_edge": edge["bottom_edge"],
        "vertical_pair": edge["vertical_pair"],
        "horizontal_pair": edge["horizontal_pair"],
        "interior_edge_clutter": edge["interior_edge_clutter"],
        "row_aligned": None,
        "r0": int(r0),
        "r1": int(r1),
        "c0": int(c0),
        "c1": int(c1),
    }
    return det


def _component_proposals(log_img, background, residual, candidate, edge_maps, threshold, x_bin_m, z_bin_m, wall_length_m, z_min_m, z_max_m):
    from scipy import ndimage as ndi

    labeled, _ = ndi.label(candidate)
    objects = ndi.find_objects(labeled)
    detections = []

    for label_id, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        rs, cs = obj
        r0, r1 = int(rs.start), int(rs.stop)
        c0, c1 = int(cs.start), int(cs.stop)
        bbox_px = max(1, (r1 - r0) * (c1 - c0))
        component_area_px = int(np.sum(labeled[rs, cs] == label_id))
        fill_ratio = component_area_px / bbox_px

        det = _score_bbox(
            log_img=log_img,
            background=background,
            residual=residual,
            candidate=candidate,
            edge_maps=edge_maps,
            threshold=threshold,
            r0=r0,
            r1=r1,
            c0=c0,
            c1=c1,
            x_bin_m=x_bin_m,
            z_bin_m=z_bin_m,
            wall_length_m=wall_length_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            source="component",
            label_id=label_id,
            component_fill_ratio=fill_ratio,
            allow_snap=True,
        )
        if det is not None:
            detections.append(det)

    return detections


def _sliding_rectangle_proposals(log_img, background, residual, candidate, edge_maps, threshold, x_bin_m, z_bin_m, wall_length_m, z_min_m, z_max_m):
    if not ENABLE_SLIDING_RECTANGLE_PROPOSALS or not ENABLE_WINDOW_DETECTION:
        return []

    h, w = log_img.shape
    residual_pos_ii = _integral_image(np.maximum(residual, 0.0))
    candidate_ii = _integral_image(candidate.astype(np.float64))
    background_ii = _integral_image(background)

    step_c = _meters_to_pixels(SLIDE_STEP_X_M, x_bin_m)
    step_r = _meters_to_pixels(SLIDE_STEP_Z_M, z_bin_m)

    # Restrict the vertical search to the broad window band when possible.
    r0_min = 0
    r0_max = h - 1
    if WINDOW_MIN_BOTTOM_ABOVE_ZMIN_M is not None:
        r0_min = max(r0_min, int(np.floor(WINDOW_MIN_BOTTOM_ABOVE_ZMIN_M / z_bin_m)) - 2)
    if WINDOW_MAX_BOTTOM_ABOVE_ZMIN_M is not None:
        r0_max = min(r0_max, int(np.ceil(WINDOW_MAX_BOTTOM_ABOVE_ZMIN_M / z_bin_m)) + 2)

    raw = []
    for width_m in SLIDE_WINDOW_WIDTHS_M:
        win_w = _meters_to_pixels(width_m, x_bin_m)
        if win_w >= w:
            continue
        for height_m in SLIDE_WINDOW_HEIGHTS_M:
            win_h = _meters_to_pixels(height_m, z_bin_m)
            if win_h >= h:
                continue
            max_r0 = min(h - win_h, r0_max)
            for r0 in range(max(0, r0_min), max_r0 + 1, step_r):
                r1 = r0 + win_h
                # Top-band constraint, if enabled.
                z1_m = z_min_m + r1 * z_bin_m
                if WINDOW_MIN_TOP_ABOVE_ZMIN_M is not None and (z1_m - z_min_m) < WINDOW_MIN_TOP_ABOVE_ZMIN_M:
                    continue
                if WINDOW_MAX_TOP_BELOW_ZMAX_M is not None and (z_max_m - z1_m) < WINDOW_MAX_TOP_BELOW_ZMAX_M:
                    continue

                for c0 in range(0, w - win_w + 1, step_c):
                    c1 = c0 + win_w
                    mean_background = _box_mean(background_ii, r0, r1, c0, c1)
                    if mean_background < MIN_LOCAL_LOG_BACKGROUND:
                        continue
                    quick_density = _box_mean(residual_pos_ii, r0, r1, c0, c1) / (threshold + 1e-6)
                    quick_candidate = _box_mean(candidate_ii, r0, r1, c0, c1)
                    if quick_density < SLIDE_QUICK_DENSITY_MIN and quick_candidate < SLIDE_QUICK_CANDIDATE_MIN:
                        continue

                    det = _score_bbox(
                        log_img=log_img,
                        background=background,
                        residual=residual,
                        candidate=candidate,
                        edge_maps=edge_maps,
                        threshold=threshold,
                        r0=r0,
                        r1=r1,
                        c0=c0,
                        c1=c1,
                        x_bin_m=x_bin_m,
                        z_bin_m=z_bin_m,
                        wall_length_m=wall_length_m,
                        z_min_m=z_min_m,
                        z_max_m=z_max_m,
                        source="sliding",
                        label_id=None,
                        component_fill_ratio=None,
                        allow_snap=True,
                    )
                    if det is not None:
                        raw.append(det)

    raw.sort(key=lambda d: d["score"], reverse=True)
    return raw[:SLIDE_MAX_RAW_PROPOSALS]


def _bbox_iou(a, b):
    x0 = max(a["x0_m"], b["x0_m"])
    y0 = max(a["z0_m"], b["z0_m"])
    x1 = min(a["x1_m"], b["x1_m"])
    y1 = min(a["z1_m"], b["z1_m"])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a["x1_m"] - a["x0_m"]) * max(0.0, a["z1_m"] - a["z0_m"])
    area_b = max(0.0, b["x1_m"] - b["x0_m"]) * max(0.0, b["z1_m"] - b["z0_m"])
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def _non_max_suppression(detections, iou_threshold=NMS_IOU_THRESHOLD):
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for det in detections:
        duplicate = False
        for prev in kept:
            # Suppress overlapping boxes even if one came from component and the
            # other from sliding proposals. This keeps the best-aligned version.
            if _bbox_iou(det, prev) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _apply_window_row_consistency(detections):
    if not APPLY_WINDOW_ROW_CONSISTENCY:
        return detections

    windows = [d for d in detections if d["opening_type"] == "window"]
    if len(windows) < ROW_CONSISTENCY_MIN_WINDOWS:
        return detections

    reference = [d for d in windows if d["score"] >= ROW_REFERENCE_MIN_SCORE and not d["touches_boundary"]]
    reference = sorted(reference, key=lambda d: d["score"], reverse=True)[:ROW_REFERENCE_TOP_K]
    if len(reference) < ROW_CONSISTENCY_MIN_WINDOWS:
        return detections

    median_z0 = float(np.median([d["z0_m"] for d in reference]))
    median_z1 = float(np.median([d["z1_m"] for d in reference]))

    adjusted = []
    for det in detections:
        if det["opening_type"] != "window":
            adjusted.append(det)
            continue

        z0_ok = abs(det["z0_m"] - median_z0) <= WINDOW_ROW_TOLERANCE_Z0_M
        z1_ok = abs(det["z1_m"] - median_z1) <= WINDOW_ROW_TOLERANCE_Z1_M
        aligned = bool(z0_ok and z1_ok)
        det = dict(det)
        det["row_median_z0_m"] = median_z0
        det["row_median_z1_m"] = median_z1
        det["row_aligned"] = aligned
        if aligned:
            det["score"] = float(det["score"] + ROW_ALIGNED_BONUS)
            adjusted.append(det)
        else:
            det["score"] = float(det["score"] - ROW_MISALIGNED_PENALTY)
            if det["score"] >= ROW_MISALIGNED_KEEP_SCORE:
                adjusted.append(det)

    adjusted.sort(key=lambda d: d["score"], reverse=True)
    return adjusted


def detect_openings_from_log_image(log_img, x_bin_m, z_bin_m, wall_length_m, z_min_m):
    """Return background, residual, candidate mask, and ranked detections."""
    from scipy import ndimage as ndi

    log_img = log_img.astype(np.float32, copy=False)
    h, w = log_img.shape
    z_max_m = z_min_m + h * z_bin_m

    sigma_z_px = LOCAL_BG_SIGMA_Z_M / z_bin_m
    sigma_x_px = LOCAL_BG_SIGMA_X_M / x_bin_m
    background = ndi.gaussian_filter(log_img, sigma=(sigma_z_px, sigma_x_px), mode="nearest")
    residual = background - log_img

    finite = np.isfinite(residual)
    if not np.any(finite):
        return background, residual, np.zeros_like(log_img, dtype=bool), []

    vals = residual[finite]
    percentile_thr = float(np.percentile(vals, RESIDUAL_PERCENTILE))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    robust_thr = med + RESIDUAL_MAD_K * 1.4826 * mad
    threshold = max(percentile_thr, robust_thr)

    candidate = (residual >= threshold) & (background >= MIN_LOCAL_LOG_BACKGROUND)

    close_structure = np.ones(
        (
            _meters_to_pixels(MORPH_CLOSE_Z_M, z_bin_m),
            _meters_to_pixels(MORPH_CLOSE_X_M, x_bin_m),
        ),
        dtype=bool,
    )
    open_structure = np.ones(
        (
            _meters_to_pixels(MORPH_OPEN_Z_M, z_bin_m),
            _meters_to_pixels(MORPH_OPEN_X_M, x_bin_m),
        ),
        dtype=bool,
    )
    candidate = ndi.binary_closing(candidate, structure=close_structure)
    candidate = ndi.binary_opening(candidate, structure=open_structure)

    edge_maps = _prepare_edge_maps(log_img)

    component_detections = _component_proposals(
        log_img,
        background,
        residual,
        candidate,
        edge_maps,
        threshold,
        x_bin_m,
        z_bin_m,
        wall_length_m,
        z_min_m,
        z_max_m,
    )
    sliding_detections = _sliding_rectangle_proposals(
        log_img,
        background,
        residual,
        candidate,
        edge_maps,
        threshold,
        x_bin_m,
        z_bin_m,
        wall_length_m,
        z_min_m,
        z_max_m,
    )

    detections = component_detections + sliding_detections
    detections = _non_max_suppression(detections)
    detections = _apply_window_row_consistency(detections)
    detections = _non_max_suppression(detections)
    detections.sort(key=lambda d: d["score"], reverse=True)

    if MAX_DETECTIONS is not None:
        detections = detections[: int(MAX_DETECTIONS)]

    print(
        "  opening detector: "
        f"threshold={threshold:.3f}, "
        f"component={len(component_detections)}, "
        f"sliding={len(sliding_detections)}, "
        f"final={len(detections)}"
    )

    return background, residual, candidate, detections


def _draw_detection_boxes(ax, detections, show_labels=True):
    for i, det in enumerate(detections, start=1):
        edgecolor = "red" if det["opening_type"] == "window" else "blue"
        rect = patches.Rectangle(
            (det["x0_m"], det["z0_m"]),
            det["width_m"],
            det["height_m"],
            fill=False,
            edgecolor=edgecolor,
            linewidth=1.2,
        )
        ax.add_patch(rect)
        if show_labels:
            source_abbrev = "C" if det["source"] == "component" else "S"
            ax.text(
                det["x0_m"],
                det["z1_m"],
                f"{i}:{source_abbrev}:{det['score']:.2f}",
                color=edgecolor,
                fontsize=7,
                va="bottom",
                ha="left",
            )


def main() -> None:
    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    wall_ft = next(w for w in wall_output["walls"] if int(w["id"]) == WALL_ID)

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft) from annotation: {xyz_min_ft}")
    print(f"  equiv offset (m): {xyz_min_ft * FT_TO_M}")

    wall_m = _transform_wall_to_e57_meters(wall_ft, xyz_min_ft)
    length_axis, s_min, s_max = _longest_footprint_edge_axis(wall_m)
    wall_length = s_max - s_min
    print(f"wall_{WALL_ID} length axis={length_axis}, length={wall_length:.2f} m")

    z_min_full = float(wall_m["zRange"]["min"])
    z_max_full = float(wall_m["zRange"]["max"])
    z_min, z_max = (
        (float(Z_CROP_M[0]), float(Z_CROP_M[1])) if Z_CROP_M else (z_min_full, z_max_full)
    )
    print(f"  z range: full=({z_min_full:.2f},{z_max_full:.2f}) m, used=({z_min:.2f},{z_max:.2f}) m")

    n_length = max(5, int(np.ceil(wall_length / LENGTH_BIN_M)))
    n_z = max(5, int(np.ceil((z_max - z_min) / Z_BIN_M)))
    count_img = np.zeros((n_z, n_length), dtype=np.uint32)

    crop_geom = _wall_crop_geometry(wall_m, MARGIN_M)

    pts_in_crop = 0
    pts_used = 0
    for chunk in _iter_e57_raw_xyz_chunks(E57_PATH, CHUNK_SIZE):
        cropped = _filter_chunk_to_wall_crop(chunk, crop_geom)
        if not len(cropped):
            continue
        pts_in_crop += len(cropped)

        s = cropped[:, :2] @ length_axis - s_min
        z = cropped[:, 2]
        keep = (s >= 0) & (s <= wall_length) & (z >= z_min) & (z <= z_max)
        if not np.any(keep):
            continue

        s = s[keep]
        z = z[keep]
        x_idx = np.clip(np.floor(s / wall_length * n_length).astype(np.int64), 0, n_length - 1)
        y_idx = np.clip(np.floor((z - z_min) / (z_max - z_min) * n_z).astype(np.int64), 0, n_z - 1)
        np.add.at(count_img, (y_idx, x_idx), 1)
        pts_used += int(keep.sum())

    print(f"  pts_in_crop={pts_in_crop}, pts_used={pts_used}")

    if Z_CROP_M is None:
        nonempty = np.flatnonzero(count_img.sum(axis=1) > 0)
        if len(nonempty):
            r0, r1 = int(nonempty.min()), int(nonempty.max()) + 1
            if (r1 - r0) < n_z:
                count_img = count_img[r0:r1]
                z_min = z_min + r0 * Z_BIN_M
                z_max = z_min + (r1 - r0) * Z_BIN_M
                n_z = r1 - r0
                print(f"  auto-clipped z to ({z_min:.2f},{z_max:.2f}) m ({n_z} rows)")

    log_img = np.log1p(count_img)
    binary_img = (count_img >= COUNT_THRESHOLD).astype(float)
    background, residual, opening_candidate, detections = detect_openings_from_log_image(
        log_img=log_img,
        x_bin_m=LENGTH_BIN_M,
        z_bin_m=Z_BIN_M,
        wall_length_m=wall_length,
        z_min_m=z_min,
    )

    print(f"  detected {len(detections)} final opening candidates")
    if detections:
        df = pd.DataFrame(detections)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  wrote {OUTPUT_CSV}")
        cols = [
            "score",
            "opening_type",
            "source",
            "x0_m",
            "x1_m",
            "z0_m",
            "z1_m",
            "width_m",
            "height_m",
            "edge_score",
            "vertical_pair",
            "horizontal_pair",
            "row_aligned",
        ]
        print(df[cols].head(30))
    else:
        pd.DataFrame(detections).to_csv(OUTPUT_CSV, index=False)
        print(f"  wrote empty {OUTPUT_CSV}")

    extent = (0.0, wall_length, z_min, z_max)

    # True e57 meter scale: 1 m along wall length == 1 m along z.
    in_per_m = 0.30
    height_per_panel = (z_max - z_min) * in_per_m
    n_panels = 6
    fig_width = wall_length * in_per_m
    fig_height = height_per_panel * n_panels + 2.3

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
    )

    im0 = axes[0].imshow(count_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[0].set_title("Raw count per cell")
    plt.colorbar(im0, ax=axes[0], label="point count", fraction=0.025, pad=0.01)

    im1 = axes[1].imshow(log_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[1].set_title("log(1 + count)")
    plt.colorbar(im1, ax=axes[1], label="log(1 + count)", fraction=0.025, pad=0.01)

    im2 = axes[2].imshow(residual, origin="lower", extent=extent, cmap="gray", aspect="equal")
    axes[2].set_title("Local low-density residual: blurred log - log")
    plt.colorbar(im2, ax=axes[2], label="residual", fraction=0.025, pad=0.01)

    axes[3].imshow(opening_candidate.astype(float), origin="lower", extent=extent, cmap="gray", aspect="equal", vmin=0, vmax=1)
    axes[3].set_title("Adaptive candidate mask after morphology: white = low-density candidate")
    _draw_detection_boxes(axes[3], detections, show_labels=False)

    axes[4].imshow(binary_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal", vmin=0, vmax=1)
    axes[4].set_title(f"Original global binary count >= {COUNT_THRESHOLD} for comparison")

    axes[5].imshow(log_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[5].set_title("Final candidates over log image; red=window, blue=door; C=component, S=sliding")
    _draw_detection_boxes(axes[5], detections, show_labels=True)

    for ax in axes:
        ax.set_ylabel("z (m)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("wall length (m)")

    fig.suptitle(
        f"wall_{WALL_ID}: dense length-z + advanced opening detection, "
        f"{pts_used} pts used / {pts_in_crop} in crop, "
        f"{wall_length:.2f} m x {z_max - z_min:.2f} m (true scale)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"  wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
