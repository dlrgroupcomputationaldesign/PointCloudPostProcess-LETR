"""Export the log(1+count) length-z image for every wall in wall_output.pickle.

One PNG per wall, no axes / colorbars / titles — just the rasterized grayscale
density image. Streams LaramieCM.e57 once and accumulates every wall's
histogram in parallel (so we don't re-read 95M points per wall).
"""

import pickle
import sys
from pathlib import Path

# Make the repo root importable so `post_process_src.*` resolves regardless
# of the cwd or which subdirectory the script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)

# Anchor all paths to this script's folder so it runs from any cwd. The data
# dirs (wall_output_pickle/, e57/, csv_label/) sit next to this file.
SCRIPT_DIR = Path(__file__).resolve().parent

# Datasets to process. Each is streamed (its own .e57) once per run.
NAMES = [
    "00-10231-20_CortevaYorkTest",
    "WyomingStateFair_Laramie",
]
FT_TO_M = 0.3048
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000

# Accumulate the histogram ONCE at this fine bin size, then derive every coarser
# variant by block-summing. Keep it a clean divisor of the variant bin sizes
# below (0.025 -> 0.05 = factor 2, 0.10 = factor 4).
FINE_BIN_M = 0.025

# Output pixel scale per histogram cell. 1 = saved image has exactly n_z × n_length pixels.
CELL_PX = 4

# ---------------------------------------------------------------------------
# Parameter sweep. Each entry renders one image per wall into its own subfolder.
# All of these are cheap post-processing on the single accumulated count grid,
# so add/remove freely and re-run to compare for Grounding DINO.
#
#   bin_m     : effective histogram cell size in meters (block-summed from FINE_BIN_M)
#   transform : "log1p" | "sqrt" | "raw"  -- how counts are compressed
#   clip_pct  : percentile of nonzero values mapped to black (None = use max)
#   gamma     : tone curve applied after normalization (None = linear; <1 brightens)
#   cmap      : "gray_r" (openings bright) | "gray" (openings dark)
# ---------------------------------------------------------------------------
VARIANTS = [
    # current baseline: log(1+count), 5cm bins, full-range, openings bright
    {"name": "log_0.05",            "bin_m": 0.05, "transform": "log1p", "clip_pct": None, "gamma": None, "cmap": "gray_r"},
    # contrast-stretched: clip the densest 1% so mid-tones (frames/openings) pop
    {"name": "log_clip99_0.05",     "bin_m": 0.05, "transform": "log1p", "clip_pct": 99.0, "gamma": None, "cmap": "gray_r"},
    # sqrt is a gentler compression than log -- keeps more density gradient
    {"name": "sqrt_clip99_0.05",    "bin_m": 0.05, "transform": "sqrt",  "clip_pct": 99.0, "gamma": None, "cmap": "gray_r"},
    # gamma<1 brightens mid/low tones, exaggerating opening boundaries
    {"name": "log_clip99_g0.6_0.05","bin_m": 0.05, "transform": "log1p", "clip_pct": 99.0, "gamma": 0.6,  "cmap": "gray_r"},
    # coarser bins -> smoother, less speckle; sometimes reads more like a photo
    {"name": "log_clip99_0.10",     "bin_m": 0.10, "transform": "log1p", "clip_pct": 99.0, "gamma": None, "cmap": "gray_r"},
    # inverted polarity: openings dark on a light wall (door-as-dark-rectangle)
    {"name": "log_clip99_inv_0.05", "bin_m": 0.05, "transform": "log1p", "clip_pct": 99.0, "gamma": None, "cmap": "gray"},
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


def _iter_e57_raw_xyz_chunks(e57_path, chunk_size=CHUNK_SIZE):
    import pye57

    e57 = pye57.E57(str(e57_path))
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


def _save_clean_image(out_path, img, cmap="gray_r", vmin=None, vmax=None, cell_px=CELL_PX):
    """Save just the rasterized image, no axes/colorbar/title.

    With cell_px=1 we get a 1-cell-per-pixel image (small). Larger values
    upscale; nearest-neighbor keeps pixels crisp. vmin/vmax fix the contrast
    mapping (default None = matplotlib auto-scales to data min/max).
    """
    h, w = img.shape
    dpi = 100
    fig_w = max(0.5, w * cell_px / dpi)
    fig_h = max(0.5, h * cell_px / dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, origin="lower", aspect="equal",
              interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _downsample_sum(img, factor):
    """Block-sum an image by an integer factor (coarsen the histogram).

    Trims any remainder rows/cols that don't fill a full block.
    """
    if factor <= 1:
        return img
    h, w = img.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    if h2 == 0 or w2 == 0:
        return img
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).sum(axis=(1, 3))


def _render_variant(count_img, variant):
    """Turn the raw count grid into a display array + contrast range for one variant."""
    factor = max(1, int(round(variant["bin_m"] / FINE_BIN_M)))
    img = _downsample_sum(count_img.astype(np.float64), factor)

    t = variant["transform"]
    if t == "raw":
        arr = img
    elif t == "log1p":
        arr = np.log1p(img)
    elif t == "sqrt":
        arr = np.sqrt(img)
    else:
        raise ValueError(f"unknown transform: {t}")

    pos = arr[arr > 0]
    clip = variant.get("clip_pct")
    if clip is not None and pos.size:
        vmax = float(np.percentile(pos, clip))
    else:
        vmax = float(arr.max()) if arr.size else 1.0
    vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    gamma = variant.get("gamma")
    if gamma:
        # normalize to [0,1], apply the tone curve, then let imshow span [0,1]
        norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0) ** gamma
        return norm, 0.0, 1.0
    return arr, vmin, vmax


def process_dataset(name) -> None:
    """Stream one dataset's E57 once and write every sweep variant per wall."""
    pickle_path = SCRIPT_DIR / "wall_output_pickle" / f"{name}.pickle"
    e57_path = SCRIPT_DIR / "e57" / f"{name}.e57"
    annotation_csv_path = SCRIPT_DIR / "csv_label" / f"{name}.csv"
    output_root = SCRIPT_DIR / "output_sweep" / name

    print(f"\n=== {name} ===")
    variant_dirs = {}
    for v in VARIANTS:
        d = output_root / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        variant_dirs[v["name"]] = d

    with open(pickle_path, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    print(f"loaded {len(walls_ft)} walls from {pickle_path}")

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
                    "pts_used": 0,
                }
            )
        except Exception as e:
            print(f"  skip wall_{w['id']}: {e}")
    print(f"prepared {len(walls_data)} wall accumulators")

    for ci, chunk in enumerate(_iter_e57_raw_xyz_chunks(e57_path, CHUNK_SIZE)):
        for d in walls_data:
            cropped = _filter_chunk_to_wall_crop(chunk, d["crop_geom"])
            if not len(cropped):
                continue
            s = cropped[:, :2] @ d["length_axis"] - d["s_min"]
            z = cropped[:, 2]
            keep = (
                (s >= 0)
                & (s <= d["wall_length"])
                & (z >= d["z_min"])
                & (z <= d["z_max"])
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
            d["pts_used"] += int(keep.sum())
        if (ci + 1) % 5 == 0:
            print(f"  processed {ci + 1} chunks")

    for d in walls_data:
        for v in VARIANTS:
            img, vmin, vmax = _render_variant(d["count_img"], v)
            out_path = variant_dirs[v["name"]] / f"wall_{d['id']}.png"
            _save_clean_image(str(out_path), img, cmap=v["cmap"], vmin=vmin, vmax=vmax)
        print(f"  wall_{d['id']}: pts={d['pts_used']}  fine_shape={d['count_img'].shape}"
              f"  -> {len(VARIANTS)} variants")


def main() -> None:
    for name in NAMES:
        process_dataset(name)


if __name__ == "__main__":
    main()
