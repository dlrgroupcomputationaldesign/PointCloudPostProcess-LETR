"""Sanity-check that the e57 points cropped to wall_17's footprint actually
overlap the pickle's wall_17 points.

Both are mapped to e57-native meters using:
    e57_xyz_m = (pickle_xyz_ft + xyz_min) * 0.3048
where xyz_min is the per-axis min that create_labels subtracted from the
annotation CSV (the only file with the exact pre-shift origin).

If the coord mapping is correct, the two clouds should occupy the same XYZ
region. Outputs:
  - console: bbox and centroid of each set
  - plotly HTML: red = pickle wall_17, gray = e57 crop, overlaid in meters
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from post_process_src.post_process.stages.openings import (
    _filter_chunk_to_wall_crop,
    _wall_crop_geometry,
)

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
WALL_ID = 17
FT_TO_M = 0.3048
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000
MAX_E57_POINTS_FOR_PLOT = 300_000
OUTPUT_HTML = "verify_wall_17_alignment.html"


def _compute_xyz_min_from_annotation(path, chunksize=ANNOTATION_CHUNK_SIZE):
    """Mirror np.amin(data_label, axis=0)[0:3] inside create_labels()."""
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(
            xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0)
        )
    return xyz_min


def _iter_e57_chunks(path, chunk_size):
    import pye57

    e57 = pye57.E57(path)
    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                n = reader.read()
                if n <= 0:
                    break
                yield np.column_stack(
                    (data["cartesianX"][:n], data["cartesianY"][:n], data["cartesianZ"][:n])
                ).astype(np.float64, copy=False)
    finally:
        e57.close()


def _transform_wall_to_meters(wall_ft, xyz_min_ft, ft_to_m=FT_TO_M):
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


def _summary(name, pts):
    if len(pts) == 0:
        print(f"  {name}: 0 points")
        return
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = pts.mean(axis=0)
    print(
        f"  {name}: n={len(pts):>8d}  "
        f"x=({lo[0]:.2f},{hi[0]:.2f}) y=({lo[1]:.2f},{hi[1]:.2f}) z=({lo[2]:.2f},{hi[2]:.2f})  "
        f"centroid=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})"
    )


def main() -> None:
    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)

    wall_ft = next(w for w in wall_output["walls"] if int(w["id"]) == WALL_ID)
    pickle_pts_ft = np.array(
        [
            [p["location"]["x"], p["location"]["y"], p["location"]["z"]]
            for p in wall_output["points"]
            if int(p["id"]) == WALL_ID
        ],
        dtype=np.float64,
    )

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft) from annotation: {xyz_min_ft}")
    print(f"  ≡ offset (m): {xyz_min_ft * FT_TO_M}")

    pickle_pts_m = (pickle_pts_ft + xyz_min_ft) * FT_TO_M
    wall_m = _transform_wall_to_meters(wall_ft, xyz_min_ft)

    print("\nBBox comparison (all in e57 meters):")
    _summary("pickle wall_17  ", pickle_pts_m)

    crop_geom = _wall_crop_geometry(wall_m, MARGIN_M)
    e57_pts_list = []
    total_seen = 0
    for chunk in _iter_e57_chunks(E57_PATH, CHUNK_SIZE):
        total_seen += len(chunk)
        cropped = _filter_chunk_to_wall_crop(chunk, crop_geom)
        if len(cropped):
            e57_pts_list.append(cropped)
    e57_pts_m = np.vstack(e57_pts_list) if e57_pts_list else np.empty((0, 3))
    print(f"  (scanned {total_seen} raw e57 points)")
    _summary("e57 crop wall_17", e57_pts_m)

    if len(pickle_pts_m) and len(e57_pts_m):
        lo = np.maximum(pickle_pts_m.min(axis=0), e57_pts_m.min(axis=0))
        hi = np.minimum(pickle_pts_m.max(axis=0), e57_pts_m.max(axis=0))
        overlap = np.maximum(hi - lo, 0)
        pickle_extent = pickle_pts_m.max(axis=0) - pickle_pts_m.min(axis=0)
        ratio = overlap / np.where(pickle_extent > 0, pickle_extent, 1)
        print(f"  XY-Z overlap / pickle extent: {ratio}")

    if len(e57_pts_m) > MAX_E57_POINTS_FOR_PLOT:
        idx = np.linspace(0, len(e57_pts_m) - 1, MAX_E57_POINTS_FOR_PLOT, dtype=np.int64)
        e57_plot = e57_pts_m[idx]
    else:
        e57_plot = e57_pts_m

    fig = go.Figure()
    if len(e57_plot):
        fig.add_trace(
            go.Scatter3d(
                x=e57_plot[:, 0], y=e57_plot[:, 1], z=e57_plot[:, 2],
                mode="markers",
                marker=dict(size=1, color="lightgray", opacity=0.5),
                name=f"e57 crop ({len(e57_pts_m)} pts, showing {len(e57_plot)})",
            )
        )
    if len(pickle_pts_m):
        fig.add_trace(
            go.Scatter3d(
                x=pickle_pts_m[:, 0], y=pickle_pts_m[:, 1], z=pickle_pts_m[:, 2],
                mode="markers",
                marker=dict(size=3, color="red", opacity=1.0),
                name=f"pickle wall_17 ({len(pickle_pts_m)} pts)",
            )
        )
    bbox_xyz = np.array([[c["x"], c["y"], c["z"]] for c in wall_m["bbox"]])
    fig.add_trace(
        go.Scatter3d(
            x=bbox_xyz[:, 0], y=bbox_xyz[:, 1], z=bbox_xyz[:, 2],
            mode="markers",
            marker=dict(size=4, color="green", symbol="diamond"),
            name="wall_17 bbox corners (mapped)",
        )
    )
    fig.update_layout(
        title=f"wall_{WALL_ID} alignment: pickle (red) vs e57 crop (gray) in meters",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
    )
    fig.write_html(OUTPUT_HTML)
    print(f"\nwrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
