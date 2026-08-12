"""Render every wall's image for WyomingStateFair_Laramie in several modalities,
so the contrast levers AND the choice of signal (density / reflectance / color)
can be eyeballed before a detection run.

Per wall, one PNG is written per variant:
  density baseline  -- committed log-density render (geometry only)
  density enhanced  -- log-density + clip/gamma/denoise/CLAHE/unsharp
  intensity         -- mean per-cell reflectance (a real material signal)
  truecolor         -- mean per-cell RGB orthophoto (most photo-like)

All four come from a SINGLE pass over the E57: each wall accumulates, per fine
(z, s) cell, a point count, an intensity sum and an RGB sum; means are taken at
render time. Cells with no points (openings, occlusions) render black.

Data (all next to this script):
  e57/WyomingStateFair_Laramie.e57              dense cloud (meters), with
                                                intensity + colorRGB per point
  wall_output_pickle/WyomingStateFair_Laramie.pickle   wall geometry (feet)
  csv_label/WyomingStateFair_Laramie.csv        annotation, for xyz_min

Because the cloud is streamed in meters, the renderer runs with
POINT_CLOUD_TO_POST_PROCESSING_SCALE = 1.0 (bins/sizes are already metres).
"""

import pickle
import sys
from pathlib import Path

# Make the repo root importable so `post_process_src.*` resolves from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)
from post_process_src.post_process.utils.opening_image_util import (
    downsample_sum,
    render_log_image,
)

SCRIPT_DIR = Path(__file__).resolve().parent

NAME = "WyomingStateFair_Laramie"
FT_TO_M = 0.3048
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000
FINE_BIN_M = 0.025  # accumulate fine, block-sum to the 0.05 render bin
RENDER_BIN_M = 0.05
CELL_PX = 4
# Reflectance normalization: map this percentile of nonempty cell means to white.
INTENSITY_CLIP_PCT = 99.0


# ---------------------------------------------------------------------------
# Density render parameters (for render_log_image). Cloud is in meters here, so
# the scale is 1.0 and all *_M bins are metres.
# ---------------------------------------------------------------------------
def _base(**overrides):
    p = {
        "POINT_CLOUD_TO_POST_PROCESSING_SCALE": 1.0,
        "OPENING_IMAGE_BIN_M": RENDER_BIN_M,
        "OPENING_IMAGE_FINE_BIN_M": FINE_BIN_M,
        "OPENING_IMAGE_CELL_PX": CELL_PX,
        "OPENING_IMAGE_TRANSFORM": "log1p",
        "OPENING_IMAGE_CLIP_PCT": None,
        "OPENING_IMAGE_GAMMA": None,
        "OPENING_IMAGE_VMAX_SCALE": 1.8,
        "OPENING_IMAGE_CMAP": "gray_r",
        "OPENING_IMAGE_DENOISE_SIGMA": 0.0,
        "OPENING_IMAGE_CLAHE_CLIP": 0.0,
        "OPENING_IMAGE_CLAHE_TILE": 8,
        "OPENING_IMAGE_UNSHARP_AMOUNT": 0.0,
        "OPENING_IMAGE_UNSHARP_SIGMA": 1.0,
    }
    p.update(overrides)
    return p


DENSITY_VARIANTS = {
    "density_baseline": _base(),
    "density_enhanced": _base(
        OPENING_IMAGE_CLIP_PCT=99.0,
        OPENING_IMAGE_VMAX_SCALE=1.2,
        OPENING_IMAGE_GAMMA=0.8,
        OPENING_IMAGE_DENOISE_SIGMA=0.8,
        OPENING_IMAGE_CLAHE_CLIP=2.0,
        OPENING_IMAGE_CLAHE_TILE=8,
        OPENING_IMAGE_UNSHARP_AMOUNT=0.7,
        OPENING_IMAGE_UNSHARP_SIGMA=1.0,
    ),
}
# Non-density modalities rendered from the same accumulators.
EXTRA_VARIANTS = ["intensity", "truecolor"]
ALL_VARIANTS = list(DENSITY_VARIANTS) + EXTRA_VARIANTS

E57_FIELDS = [
    "cartesianX", "cartesianY", "cartesianZ",
    "intensity", "colorRed", "colorGreen", "colorBlue",
]


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


def _iter_e57_chunks(e57_path, chunk_size=CHUNK_SIZE):
    """Yield (N, 7) chunks: x, y, z, intensity, R, G, B (all float64).

    Column order matches _filter_chunk_to_wall_crop's use of cols 0-2 for the
    spatial mask, so the extra fields ride along with their points.
    """
    import pye57

    e57 = pye57.E57(str(e57_path))
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(E57_FIELDS, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                count = reader.read()
                if count <= 0:
                    break
                yield np.column_stack(
                    [data[f][:count] for f in E57_FIELDS]
                ).astype(np.float64, copy=False)
    finally:
        e57.close()


def _render_mean_gray(value_sum, count, factor, clip_pct=INTENSITY_CLIP_PCT):
    """Per-cell mean of a scalar field -> uint8 grayscale, oriented like render_log_image.

    Block-sums both the value sum and the count to the render bin, divides to a
    mean, normalizes nonempty cells by a percentile (robust to specular spikes),
    and leaves empty cells black. Then flips vertically and upscales by CELL_PX.
    """
    vs = downsample_sum(value_sum, factor)
    cs = downsample_sum(count.astype(np.float64), factor)
    mean = np.zeros_like(vs)
    nonempty = cs > 0
    mean[nonempty] = vs[nonempty] / cs[nonempty]

    pos = mean[nonempty]
    if pos.size:
        vmax = float(np.percentile(pos, clip_pct))
    else:
        vmax = 1.0
    if vmax <= 0.0:
        vmax = 1.0

    gray = np.clip(mean / vmax, 0.0, 1.0)
    img8 = (gray * 255.0).astype(np.uint8)
    img8[~nonempty] = 0  # openings / occlusions stay black
    img8 = np.flipud(img8)
    if CELL_PX > 1:
        img8 = np.repeat(np.repeat(img8, CELL_PX, axis=0), CELL_PX, axis=1)
    return np.stack([img8, img8, img8], axis=-1)


def _render_truecolor(color_sum, count, factor, color_scale):
    """Per-cell mean RGB -> uint8 BGR image (cv2 order), oriented like render_log_image.

    color_scale maps the source color range onto 0-255 (1.0 for 8-bit color).
    Empty cells render black.
    """
    cs = downsample_sum(count.astype(np.float64), factor)
    nonempty = cs > 0
    rgb = np.zeros((*cs.shape, 3), dtype=np.float64)
    for c in range(3):
        ch = downsample_sum(color_sum[:, :, c], factor)
        rgb[nonempty, c] = ch[nonempty] / cs[nonempty]
    rgb = np.clip(rgb * color_scale, 0.0, 255.0).astype(np.uint8)
    rgb = np.flipud(rgb)
    if CELL_PX > 1:
        rgb = np.repeat(np.repeat(rgb, CELL_PX, axis=0), CELL_PX, axis=1)
    return rgb[:, :, ::-1]  # RGB -> BGR for cv2.imwrite


def main() -> None:
    pickle_path = SCRIPT_DIR / "wall_output_pickle" / f"{NAME}.pickle"
    e57_path = SCRIPT_DIR / "e57" / f"{NAME}.e57"
    annotation_csv_path = SCRIPT_DIR / "csv_label" / f"{NAME}.csv"
    output_root = SCRIPT_DIR / "output_log_img_test" / NAME

    for vname in ALL_VARIANTS:
        (output_root / vname).mkdir(parents=True, exist_ok=True)

    print(f"=== {NAME} ===")
    with open(pickle_path, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    print(f"loaded {len(walls_ft)} walls from {pickle_path.name}")

    xyz_min_ft = _compute_xyz_min_from_annotation(annotation_csv_path)
    print(f"xyz_min (ft): {xyz_min_ft}")

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
            n_length = max(5, int(np.ceil(wall_length / FINE_BIN_M)))
            n_z = max(5, int(np.ceil((z_max - z_min) / FINE_BIN_M)))
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
                    "intensity_sum": np.zeros((n_z, n_length), dtype=np.float64),
                    "color_sum": np.zeros((n_z, n_length, 3), dtype=np.float64),
                    "pts_used": 0,
                }
            )
        except Exception as e:
            print(f"  skip wall_{w['id']}: {e}")
    print(f"prepared {len(walls_data)} wall accumulators")

    color_max_seen = 0.0
    for ci, chunk in enumerate(_iter_e57_chunks(e57_path, CHUNK_SIZE)):
        for d in walls_data:
            cropped = _filter_chunk_to_wall_crop(chunk, d["crop_geom"])
            if not len(cropped):
                continue
            xy = cropped[:, :2]
            z = cropped[:, 2]
            intensity = cropped[:, 3]
            rgb = cropped[:, 4:7]
            s = xy @ d["length_axis"] - d["s_min"]
            keep = (
                (s >= 0)
                & (s <= d["wall_length"])
                & (z >= d["z_min"])
                & (z <= d["z_max"])
            )
            if not np.any(keep):
                continue
            s = s[keep]
            zc = z[keep]
            ik = intensity[keep]
            rk = rgb[keep]
            color_max_seen = max(color_max_seen, float(rk.max()) if rk.size else 0.0)
            x_idx = np.clip(
                np.floor(s / d["wall_length"] * d["n_length"]).astype(np.int64),
                0, d["n_length"] - 1,
            )
            y_idx = np.clip(
                np.floor((zc - d["z_min"]) / (d["z_max"] - d["z_min"]) * d["n_z"]).astype(np.int64),
                0, d["n_z"] - 1,
            )
            np.add.at(d["count_img"], (y_idx, x_idx), 1)
            np.add.at(d["intensity_sum"], (y_idx, x_idx), ik)
            np.add.at(d["color_sum"], (y_idx, x_idx), rk)
            d["pts_used"] += int(keep.sum())
        if (ci + 1) % 5 == 0:
            print(f"  processed {ci + 1} chunks")

    # 8-bit color needs no scaling; if the E57 stores 16-bit color, map to 0-255.
    color_scale = 255.0 / color_max_seen if color_max_seen > 255.0 else 1.0
    print(f"color max seen: {color_max_seen:.1f} -> color_scale {color_scale:.4f}")

    factor = max(1, int(round(RENDER_BIN_M / FINE_BIN_M)))  # fine -> render bin
    for d in walls_data:
        render_counts = downsample_sum(d["count_img"].astype(np.float64), factor)
        for vname, params in DENSITY_VARIANTS.items():
            img_rgb = render_log_image(render_counts, params)
            cv2.imwrite(str(output_root / vname / f"wall_{d['id']}.png"), img_rgb)

        intensity_img = _render_mean_gray(d["intensity_sum"], d["count_img"], factor)
        cv2.imwrite(str(output_root / "intensity" / f"wall_{d['id']}.png"), intensity_img)

        truecolor_img = _render_truecolor(d["color_sum"], d["count_img"], factor, color_scale)
        cv2.imwrite(str(output_root / "truecolor" / f"wall_{d['id']}.png"), truecolor_img)

        print(f"  wall_{d['id']}: pts={d['pts_used']}  shape={d['count_img'].shape}"
              f"  -> {len(ALL_VARIANTS)} variants")

    print(f"\ndone -> {output_root}")


if __name__ == "__main__":
    main()
