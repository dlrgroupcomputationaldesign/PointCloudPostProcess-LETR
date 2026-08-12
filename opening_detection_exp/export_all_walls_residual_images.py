"""Export the local-background residual (background - log) image for every
wall in wall_output.pickle.

One PNG per wall, no axes / colorbars / titles — bright cells = below local
density average (opening candidates). Streams LaramieCM.e57 once and
accumulates every wall's histogram in parallel.

residual = gaussian_blur(log(1+count)) - log(1+count)
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
OUTPUT_DIR = "wall_residual_images"
FT_TO_M = 0.3048
LENGTH_BIN_M = 0.05
Z_BIN_M = 0.05
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000

# Local-background blur (matches wall_opening_detection_*). Bigger σ ignores
# small frame details, captures broader density trends.
LOCAL_BG_SIGMA_Z_M = 0.35
LOCAL_BG_SIGMA_X_M = 0.90

# Output pixel scale per histogram cell.
CELL_PX = 4
# "gray": negative residuals = dark, positive (openings) = bright.
# Try "RdBu_r" if you want a diverging colormap centered at 0.
CMAP = "gray"


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


def _save_clean_image(out_path, img, cmap=CMAP, cell_px=CELL_PX):
    """Save just the rasterized image, no axes/colorbar/title."""
    h, w = img.shape
    dpi = 100
    fig_w = max(0.5, w * cell_px / dpi)
    fig_h = max(0.5, h * cell_px / dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, origin="lower", aspect="equal", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    print(f"loaded {len(walls_ft)} walls from {PICKLE_PATH}")

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
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
                    "pts_used": 0,
                }
            )
        except Exception as e:
            print(f"  skip wall_{w['id']}: {e}")
    print(f"prepared {len(walls_data)} wall accumulators")

    for ci, chunk in enumerate(_iter_e57_raw_xyz_chunks(E57_PATH, CHUNK_SIZE)):
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

    sigma_z_px = LOCAL_BG_SIGMA_Z_M / Z_BIN_M
    sigma_x_px = LOCAL_BG_SIGMA_X_M / LENGTH_BIN_M
    print(f"gaussian sigma (px): z={sigma_z_px:.2f}, x={sigma_x_px:.2f}")

    for d in walls_data:
        log_img = np.log1p(d["count_img"].astype(np.float32))
        background = ndi.gaussian_filter(log_img, sigma=(sigma_z_px, sigma_x_px), mode="nearest")
        residual = background - log_img
        out_path = out_dir / f"wall_{d['id']}_residual.png"
        _save_clean_image(str(out_path), residual)
        print(
            f"  wrote {out_path}  pts={d['pts_used']}  shape={residual.shape}  "
            f"residual=[{residual.min():.2f},{residual.max():.2f}]"
        )


if __name__ == "__main__":
    main()
