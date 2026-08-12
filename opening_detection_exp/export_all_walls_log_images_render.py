"""Export per-wall log/intensity images using the SAME renderer as the
post-process package (post_process_src.post_process.utils.opening_image_util).

Differences vs. export_all_walls_log_images.py (the simple version):
  - Calls render_log_image / render_intensity_image, so output matches the
    pipeline's images byte-for-byte (vmax scaling, denoise/CLAHE/unsharp,
    flipud orientation, integer upscale by CELL_PX, RGB encoding).
  - Optional intensity rendering (reflectance mean per cell). Falls back to
    density per wall if intensity coverage < OPENING_IMAGE_INTENSITY_MIN_COVERAGE.
  - Saves via cv2.imwrite so no matplotlib decorations leak in.

Streams the e57 once and accumulates every wall's histogram (+ intensity_sum
when enabled) in parallel, then writes one PNG per wall.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)
from post_process_src.post_process.utils.opening_image_util import (
    render_intensity_image,
    render_log_image,
)

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
OUTPUT_DIR = "wall_log_images_render"
FT_TO_M = 0.3048
LENGTH_BIN_M = 0.05
Z_BIN_M = 0.05
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000

# ---- Package render parameters (match post_process_src.config.OpeningConfig defaults) ----
IMAGE_SOURCE = "density"          # "density" | "intensity"
IMAGE_TRANSFORM = "log1p"         # "raw" | "log1p" | "sqrt"
IMAGE_CLIP_PCT = 99.0             # percentile clip for vmax; None to disable
IMAGE_VMAX_SCALE = 1.8            # brightness lever (>1 brightens gray_r)
IMAGE_CMAP = "gray_r"             # "gray_r" (openings bright) | "gray"
IMAGE_GAMMA = None                # tone curve; None to disable
IMAGE_CELL_PX = 4                 # integer upscale per cell
IMAGE_DENOISE_SIGMA = 0.0         # Gaussian denoise on uint8 (0 = off)
IMAGE_CLAHE_CLIP = 0.0            # CLAHE clip limit (0 = off)
IMAGE_CLAHE_TILE = 8
IMAGE_UNSHARP_AMOUNT = 0.0        # unsharp mask amount (0 = off)
IMAGE_UNSHARP_SIGMA = 1.0
IMAGE_INTENSITY_CLIP_PCT = 99.0
IMAGE_INTENSITY_MIN_COVERAGE = 0.35


def _render_parameters():
    """Build the parameters dict the package render functions read from."""
    return {
        "OPENING_IMAGE_TRANSFORM": IMAGE_TRANSFORM,
        "OPENING_IMAGE_CLIP_PCT": IMAGE_CLIP_PCT,
        "OPENING_IMAGE_VMAX_SCALE": IMAGE_VMAX_SCALE,
        "OPENING_IMAGE_CMAP": IMAGE_CMAP,
        "OPENING_IMAGE_GAMMA": IMAGE_GAMMA,
        "OPENING_IMAGE_CELL_PX": IMAGE_CELL_PX,
        "OPENING_IMAGE_DENOISE_SIGMA": IMAGE_DENOISE_SIGMA,
        "OPENING_IMAGE_CLAHE_CLIP": IMAGE_CLAHE_CLIP,
        "OPENING_IMAGE_CLAHE_TILE": IMAGE_CLAHE_TILE,
        "OPENING_IMAGE_UNSHARP_AMOUNT": IMAGE_UNSHARP_AMOUNT,
        "OPENING_IMAGE_UNSHARP_SIGMA": IMAGE_UNSHARP_SIGMA,
        "OPENING_IMAGE_INTENSITY_CLIP_PCT": IMAGE_INTENSITY_CLIP_PCT,
        "OPENING_IMAGE_INTENSITY_MIN_COVERAGE": IMAGE_INTENSITY_MIN_COVERAGE,
    }


def _compute_xyz_min_from_annotation(path, chunksize=ANNOTATION_CHUNK_SIZE):
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0))
    return xyz_min


def _transform_wall_to_e57_meters(wall_ft, xyz_min_ft, ft_to_m=FT_TO_M):
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
        edges = [pts[(i + 1) % len(pts)] - pts[i] for i in range(len(pts))]
    else:
        pts = np.array([[p["x"], p["y"]] for p in wall["bbox"]], dtype=np.float64)
        edges = [pts[j] - pts[i] for i in range(len(pts)) for j in range(i + 1, len(pts))]
    norms = np.linalg.norm(edges, axis=1)
    best = int(np.argmax(norms))
    axis = edges[best] / norms[best]
    projected = pts @ axis
    return axis, float(projected.min()), float(projected.max())


def _iter_e57_chunks(e57_path, chunk_size=CHUNK_SIZE, with_intensity=False):
    import pye57

    e57 = pye57.E57(str(e57_path))
    base_fields = ["cartesianX", "cartesianY", "cartesianZ"]
    if with_intensity:
        try:
            field_names = list(e57.get_header(0).point_fields)
            has_intensity = "intensity" in field_names
        except Exception:
            has_intensity = False
    else:
        has_intensity = False

    fields = base_fields + (["intensity"] if has_intensity else [])
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                count = reader.read()
                if count <= 0:
                    break
                cols = [data[f][:count] for f in base_fields]
                if has_intensity:
                    cols.append(data["intensity"][:count])
                yield np.column_stack(cols).astype(np.float64, copy=False), has_intensity
    finally:
        e57.close()


def _intensity_render_or_fallback(intensity_sum, count_img, parameters):
    """Mirror _select_render's per-wall coverage check."""
    coverage = float((count_img > 0).mean()) if count_img.size else 0.0
    min_cov = float(parameters.get("OPENING_IMAGE_INTENSITY_MIN_COVERAGE", 0.0) or 0.0)
    if coverage >= min_cov:
        return render_intensity_image(intensity_sum, count_img, parameters), "intensity"
    return render_log_image(count_img, parameters), f"density (coverage {coverage:.2f} < {min_cov:.2f})"


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    print(f"loaded {len(walls_ft)} walls from {PICKLE_PATH}")

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft): {xyz_min_ft}")

    use_intensity_requested = IMAGE_SOURCE.lower() == "intensity"
    parameters = _render_parameters()

    walls_data = []
    for w in walls_ft:
        try:
            wall_m = _transform_wall_to_e57_meters(w, xyz_min_ft)
            length_axis, s_min, s_max = _longest_footprint_edge_axis(wall_m)
            wall_length = s_max - s_min
            z_min = float(wall_m["zRange"]["min"])
            z_max = float(wall_m["zRange"]["max"])
            if wall_length <= 0 or z_max <= z_min:
                print(f"  skip wall_{w['id']}: degenerate dims")
                continue
            n_length = max(5, int(np.ceil(wall_length / LENGTH_BIN_M)))
            n_z = max(5, int(np.ceil((z_max - z_min) / Z_BIN_M)))
            walls_data.append(
                {
                    "id": w["id"],
                    "length_axis": length_axis,
                    "s_min": s_min,
                    "wall_length": wall_length,
                    "z_min": z_min,
                    "z_max": z_max,
                    "crop_geom": _wall_crop_geometry(wall_m, MARGIN_M),
                    "n_length": n_length,
                    "n_z": n_z,
                    "count_img": np.zeros((n_z, n_length), dtype=np.uint32),
                    "intensity_sum": (
                        np.zeros((n_z, n_length), dtype=np.float64)
                        if use_intensity_requested else None
                    ),
                    "pts_used": 0,
                }
            )
        except Exception as e:
            print(f"  skip wall_{w['id']}: {e}")
    print(f"prepared {len(walls_data)} wall accumulators")

    has_intensity_field = False
    for ci, (chunk, has_intensity_field) in enumerate(
        _iter_e57_chunks(E57_PATH, CHUNK_SIZE, with_intensity=use_intensity_requested)
    ):
        xyz = chunk[:, :3]
        intensity = chunk[:, 3] if (chunk.shape[1] >= 4 and has_intensity_field) else None
        for d in walls_data:
            # cropping is by xyz only; reuse package helper
            mask_aabb = (
                (xyz[:, 0] >= d["crop_geom"]["minb"][0]) & (xyz[:, 0] <= d["crop_geom"]["maxb"][0])
                & (xyz[:, 1] >= d["crop_geom"]["minb"][1]) & (xyz[:, 1] <= d["crop_geom"]["maxb"][1])
                & (xyz[:, 2] >= d["crop_geom"]["minb"][2]) & (xyz[:, 2] <= d["crop_geom"]["maxb"][2])
            )
            if not np.any(mask_aabb):
                continue
            xyz_a = xyz[mask_aabb]
            int_a = intensity[mask_aabb] if intensity is not None else None
            polygon = d["crop_geom"].get("polygon")
            if polygon is not None:
                try:
                    from shapely import contains_xy
                    inside = contains_xy(polygon, xyz_a[:, 0], xyz_a[:, 1])
                except ImportError:
                    from shapely.geometry import Point
                    inside = np.array(
                        [polygon.contains(Point(x, y)) for x, y in xyz_a[:, :2]], dtype=bool
                    )
                xyz_a = xyz_a[inside]
                if int_a is not None:
                    int_a = int_a[inside]
            if not len(xyz_a):
                continue

            s = xyz_a[:, :2] @ d["length_axis"] - d["s_min"]
            z = xyz_a[:, 2]
            keep = (
                (s >= 0) & (s <= d["wall_length"])
                & (z >= d["z_min"]) & (z <= d["z_max"])
            )
            if not np.any(keep):
                continue
            s = s[keep]
            z = z[keep]
            x_idx = np.clip(
                np.floor(s / d["wall_length"] * d["n_length"]).astype(np.int64),
                0, d["n_length"] - 1,
            )
            y_idx = np.clip(
                np.floor((z - d["z_min"]) / (d["z_max"] - d["z_min"]) * d["n_z"]).astype(np.int64),
                0, d["n_z"] - 1,
            )
            np.add.at(d["count_img"], (y_idx, x_idx), 1)
            if d["intensity_sum"] is not None and int_a is not None:
                np.add.at(d["intensity_sum"], (y_idx, x_idx), int_a[keep])
            d["pts_used"] += int(keep.sum())
        if (ci + 1) % 5 == 0:
            print(f"  processed {ci + 1} chunks")

    use_intensity = use_intensity_requested and has_intensity_field
    if use_intensity_requested and not has_intensity_field:
        print(f"  WARNING: IMAGE_SOURCE=intensity requested but e57 has no intensity field; falling back to density")

    for d in walls_data:
        if use_intensity and d["intensity_sum"] is not None:
            img_rgb, mode = _intensity_render_or_fallback(d["intensity_sum"], d["count_img"], parameters)
        else:
            img_rgb = render_log_image(d["count_img"], parameters)
            mode = "density"

        out_path = out_dir / f"wall_{d['id']}_{mode.split()[0]}.png"
        # render_*_image returns RGB; cv2.imwrite needs BGR.
        cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(
            f"  wrote {out_path}  pts={d['pts_used']}  "
            f"shape={d['count_img'].shape}  mode={mode}"
        )


if __name__ == "__main__":
    main()
