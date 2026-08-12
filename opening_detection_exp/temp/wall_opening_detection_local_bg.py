"""Dense length-z histogram + opening detection for one wall.

This version keeps your existing wall-length/z histogram workflow and adds:
  1. local background subtraction on log(1 + count),
  2. adaptive low-density candidate extraction,
  3. connected-component rectangular filtering,
  4. edge-support scoring from the log image,
  5. an annotated figure and CSV of candidate openings.

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
WALL_ID = 34
FT_TO_M = 0.3048
LENGTH_BIN_M = 0.05
Z_BIN_M = 0.05
MARGIN_M = 0.22
COUNT_THRESHOLD = 100
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000
OUTPUT_PNG = "wall_17_opening_tune3.png"
OUTPUT_CSV = "wall_17_opening_candidates.csv"

# Set to (zmin_m, zmax_m) to clip explicitly; None = auto-clip to populated rows.
Z_CROP_M = None

# -----------------------------------------------------------------------------
# Opening-detection tuning parameters
# -----------------------------------------------------------------------------
# Local background blur. Larger values ignore small frame/details and keep the
# broad wall-density trend. Order is z, length.
LOCAL_BG_SIGMA_Z_M = 0.42
LOCAL_BG_SIGMA_X_M = 1.00

# Candidate threshold. A cell is an opening candidate if its residual is high.
# The final threshold is max(percentile threshold, robust MAD threshold).
RESIDUAL_PERCENTILE = 90.0
RESIDUAL_MAD_K = 1.5
MIN_LOCAL_LOG_BACKGROUND = 2.0 # avoids selecting globally empty/unscanned zones

# Morphology on the candidate mask, in meters.
MORPH_CLOSE_Z_M = 0.12
MORPH_CLOSE_X_M = 0.20
MORPH_OPEN_Z_M = 0.05
MORPH_OPEN_X_M = 0.05

# Rectangular component filters, in meters.  for WINDOW!
MIN_OPENING_WIDTH_M = 0.40
MAX_OPENING_WIDTH_M = 1.90
MIN_OPENING_HEIGHT_M = 0.45
MAX_OPENING_HEIGHT_M = 1.70
MIN_OPENING_AREA_M2 = 0.22
MAX_OPENING_AREA_M2 = 3.00

MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 2.80
MIN_FILL_RATIO = 0.12
MAX_FILL_RATIO = 0.95

# Edge scoring. Edges are measured in narrow bands around the bbox.
EDGE_BAND_M = 0.12
MIN_EDGE_SCORE = 0.18
MIN_FINAL_SCORE = 0.68

# Do not trust detections touching the crop/image boundary.
BOUNDARY_MARGIN_M = 0.2
BOUNDARY_PENALTY = 0.40

# Keep only this many top candidates in the output CSV/plot. Set None to keep all.
MAX_DETECTIONS = None


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


def _edge_support_score(log_img, r0, r1, c0, c1, z_bin_m, x_bin_m):
    """Measure whether a candidate bbox is supported by frame-like edges.

    Vertical edge support uses x-gradient bands at left and right sides.
    Horizontal edge support uses z-gradient bands at bottom and top sides.
    The score is normalized by robust global gradient magnitudes.
    """
    from scipy import ndimage as ndi

    h, w = log_img.shape
    smooth = ndi.gaussian_filter(log_img.astype(np.float32), sigma=1.0, mode="nearest")
    grad_x = np.abs(ndi.sobel(smooth, axis=1))
    grad_z = np.abs(ndi.sobel(smooth, axis=0))

    # Robust global normalization. Add epsilon to avoid divide-by-zero.
    gx_scale = float(np.percentile(grad_x, 95)) + 1e-6
    gz_scale = float(np.percentile(grad_z, 95)) + 1e-6

    c_band = _meters_to_pixels(EDGE_BAND_M, x_bin_m)
    r_band = _meters_to_pixels(EDGE_BAND_M, z_bin_m)

    left = _safe_mean(grad_x[_clip_slice(r0, r1, h), _clip_slice(c0 - c_band, c0 + c_band + 1, w)])
    right = _safe_mean(grad_x[_clip_slice(r0, r1, h), _clip_slice(c1 - c_band - 1, c1 + c_band, w)])
    bottom = _safe_mean(grad_z[_clip_slice(r0 - r_band, r0 + r_band + 1, h), _clip_slice(c0, c1, w)])
    top = _safe_mean(grad_z[_clip_slice(r1 - r_band - 1, r1 + r_band, h), _clip_slice(c0, c1, w)])

    # Require both vertical sides where possible. For horizontal support, top is
    # usually more reliable than bottom for doors, so use the stronger one.
    vertical_pair = min(left, right) / gx_scale
    horizontal_support = max(top, bottom) / gz_scale

    # Cap to keep extremely strong edges from dominating the whole score.
    vertical_pair = min(vertical_pair, 3.0)
    horizontal_support = min(horizontal_support, 3.0)
    edge_score = 0.70 * vertical_pair + 0.30 * horizontal_support

    return {
        "edge_score": float(edge_score),
        "left_edge": float(left / gx_scale),
        "right_edge": float(right / gx_scale),
        "top_edge": float(top / gz_scale),
        "bottom_edge": float(bottom / gz_scale),
    }


def detect_openings_from_log_image(log_img, x_bin_m, z_bin_m, wall_length_m, z_min_m):
    """Return residual map, candidate mask, and ranked opening detections."""
    from scipy import ndimage as ndi

    log_img = log_img.astype(np.float32, copy=False)
    h, w = log_img.shape

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

    labeled, n_labels = ndi.label(candidate)
    objects = ndi.find_objects(labeled)
    detections = []

    boundary_margin_c = _meters_to_pixels(BOUNDARY_MARGIN_M, x_bin_m)
    boundary_margin_r = _meters_to_pixels(BOUNDARY_MARGIN_M, z_bin_m)

    for label_id, obj in enumerate(objects, start=1):
        if obj is None:
            continue
        rs, cs = obj
        r0, r1 = int(rs.start), int(rs.stop)
        c0, c1 = int(cs.start), int(cs.stop)

        bbox_px = max(1, (r1 - r0) * (c1 - c0))
        component_area_px = int(np.sum(labeled[rs, cs] == label_id))
        fill_ratio = component_area_px / bbox_px

        width_m = (c1 - c0) * x_bin_m
        height_m = (r1 - r0) * z_bin_m
        area_m2 = width_m * height_m
        aspect = width_m / max(height_m, 1e-6)

        if not (MIN_OPENING_WIDTH_M <= width_m <= MAX_OPENING_WIDTH_M):
            continue
        if not (MIN_OPENING_HEIGHT_M <= height_m <= MAX_OPENING_HEIGHT_M):
            continue
        if not (MIN_OPENING_AREA_M2 <= area_m2 <= MAX_OPENING_AREA_M2):
            continue
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
            continue
        if not (MIN_FILL_RATIO <= fill_ratio <= MAX_FILL_RATIO):
            continue

        edge = _edge_support_score(log_img, r0, r1, c0, c1, z_bin_m, x_bin_m)
        if edge["edge_score"] < MIN_EDGE_SCORE:
            continue

        component_mask = labeled[rs, cs] == label_id
        residual_component = residual[rs, cs][component_mask]
        residual_bbox = residual[rs, cs]
        mean_residual_component = float(np.mean(residual_component)) if residual_component.size else 0.0
        mean_residual_bbox = float(np.mean(residual_bbox)) if residual_bbox.size else 0.0

        density_score = mean_residual_component / (threshold + 1e-6)
        rect_score = min(fill_ratio / max(MIN_FILL_RATIO, 1e-6), 1.0)

        touches_boundary = (
            c0 <= boundary_margin_c
            or c1 >= w - boundary_margin_c
            or r0 <= boundary_margin_r
            or r1 >= h - boundary_margin_r
        )
        boundary_penalty = BOUNDARY_PENALTY if touches_boundary else 0.0

        final_score = (
            0.45 * density_score
            + 0.35 * edge["edge_score"]
            + 0.20 * rect_score
            - boundary_penalty
        )

        if final_score < MIN_FINAL_SCORE:
            continue

        x0_m = c0 * x_bin_m
        x1_m = min(wall_length_m, c1 * x_bin_m)
        z0_m = z_min_m + r0 * z_bin_m
        z1_m = z_min_m + r1 * z_bin_m

        detections.append(
            {
                "label_id": label_id,
                "score": float(final_score),
                "density_score": float(density_score),
                "edge_score": float(edge["edge_score"]),
                "rect_score": float(rect_score),
                "x0_m": float(x0_m),
                "x1_m": float(x1_m),
                "z0_m": float(z0_m),
                "z1_m": float(z1_m),
                "width_m": float(width_m),
                "height_m": float(height_m),
                "area_m2": float(area_m2),
                "aspect": float(aspect),
                "fill_ratio": float(fill_ratio),
                "mean_residual_component": float(mean_residual_component),
                "mean_residual_bbox": float(mean_residual_bbox),
                "threshold": float(threshold),
                "touches_boundary": bool(touches_boundary),
                "left_edge": edge["left_edge"],
                "right_edge": edge["right_edge"],
                "top_edge": edge["top_edge"],
                "bottom_edge": edge["bottom_edge"],
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    if MAX_DETECTIONS is not None:
        detections = detections[: int(MAX_DETECTIONS)]

    return background, residual, candidate, detections


def _draw_detection_boxes(ax, detections, show_labels=True):
    for i, det in enumerate(detections, start=1):
        rect = patches.Rectangle(
            (det["x0_m"], det["z0_m"]),
            det["width_m"],
            det["height_m"],
            fill=False,
            edgecolor="red",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        if show_labels:
            ax.text(
                det["x0_m"],
                det["z1_m"],
                f"{i}: {det['score']:.2f}",
                color="red",
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

    print(f"  detected {len(detections)} opening candidates")
    if detections:
        df = pd.DataFrame(detections)
        # Drop pixel indices from CSV if you only want metric coordinates.
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"  wrote {OUTPUT_CSV}")
        print(df[["score", "x0_m", "x1_m", "z0_m", "z1_m", "width_m", "height_m", "edge_score"]].head(20))
    else:
        # Still write an empty CSV with useful columns for downstream scripts.
        pd.DataFrame(detections).to_csv(OUTPUT_CSV, index=False)
        print(f"  wrote empty {OUTPUT_CSV}")

    extent = (0.0, wall_length, z_min, z_max)

    # True e57 meter scale: 1 m along wall length == 1 m along z.
    in_per_m = 0.30
    height_per_panel = (z_max - z_min) * in_per_m
    n_panels = 5
    fig_width = wall_length * in_per_m
    fig_height = height_per_panel * n_panels + 2.0

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

    axes[4].imshow(log_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[4].set_title("Final rectangular candidates over log image, label = rank:score")
    _draw_detection_boxes(axes[4], detections, show_labels=True)

    for ax in axes:
        ax.set_ylabel("z (m)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("wall length (m)")

    fig.suptitle(
        f"wall_{WALL_ID}: dense length-z + opening detection, {pts_used} pts used / {pts_in_crop} in crop, "
        f"{wall_length:.2f} m x {z_max - z_min:.2f} m (true scale)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"  wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
