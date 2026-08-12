"""Dense length-z histogram for wall_17, following the meter-based method that
produced wall_17_e57_rgb_log_binary_2d.png.

Pipeline (matches json_output_visualization.create_e57_wall_length_z_image):
  1. Load wall from pickle (feet, npy-shifted coord system — same as inference CSV).
  2. Map to e57-native meters: e57_xyz_m = (wall_xyz_ft + xyz_min) * 0.3048,
     where xyz_min is the per-axis min that create_labels subtracted from the
     annotation CSV. Read directly from the annotation CSV at startup.
  3. Length axis = longest footprint-edge direction (NOT PCA — that smeared stripes).
  4. Stream raw e57 chunks, crop to wall footprint+margin, project to (s, z).
  5. Histogram in meters, auto-clip z to rows that actually contain points
     (a tall bbox padded with empty space washes out log/binary contrast).
  6. 3-panel figure: raw count / log(1+count) / binary, all gray_r so openings
     appear as bright rectangles.
"""

import pickle

import matplotlib.pyplot as plt
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
OUTPUT_PNG = "wall_17_dense_lengthz_panels.png"

# Set to (zmin_m, zmax_m) to clip explicitly; None = auto-clip to populated rows.
Z_CROP_M = None


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


def main() -> None:
    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    wall_ft = next(w for w in wall_output["walls"] if int(w["id"]) == WALL_ID)

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft) from annotation: {xyz_min_ft}")
    print(f"  ≡ offset (m): {xyz_min_ft * FT_TO_M}")

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
    extent = (0.0, wall_length, z_min, z_max)

    # True e57 meter scale: 1 m along wall length == 1 m along z.
    # Vertical stack of 3 panels because wall is ~15:1 (length:height).
    in_per_m = 0.30   # 0.30 inches per meter — tune for paper/screen size
    height_per_panel = (z_max - z_min) * in_per_m
    fig_width = wall_length * in_per_m
    fig_height = height_per_panel * 3 + 1.5  # extra inches for titles & xlabel

    fig, axes = plt.subplots(
        3, 1, figsize=(fig_width, fig_height),
        sharex=True, sharey=True,
    )

    im0 = axes[0].imshow(count_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[0].set_title("Raw count per cell")
    plt.colorbar(im0, ax=axes[0], label="point count", fraction=0.025, pad=0.01)

    im1 = axes[1].imshow(log_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal")
    axes[1].set_title("log(1 + count)")
    plt.colorbar(im1, ax=axes[1], label="log(1 + count)", fraction=0.025, pad=0.01)

    axes[2].imshow(binary_img, origin="lower", extent=extent, cmap="gray_r", aspect="equal", vmin=0, vmax=1)
    axes[2].set_title(f"Binary (count >= {COUNT_THRESHOLD}) — white = opening candidate")

    for ax in axes:
        ax.set_ylabel("z (m)")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("wall length (m)")

    fig.suptitle(
        f"wall_{WALL_ID}: dense length-z, {pts_used} pts used / {pts_in_crop} in crop, "
        f"{wall_length:.2f} m × {z_max - z_min:.2f} m (true scale)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    print(f"  wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
