"""Render the LaramieCM.e57 point cloud (heavily downsampled) with wall_{WALL_ID}
overlaid so we can see whether the wall is in the right place.

Things drawn in e57-native meters:
  - dense e57 sample (RGB-colored, ~MAX_E57_POINTS points, stride-sampled)
  - pickle wall_{WALL_ID} sparse points mapped to meters (red dots)
  - wall_{WALL_ID} bbox wireframe (red lines)
  - all other walls' bboxes (semi-transparent green) for context

Pickle-to-e57 mapping:
    e57_xyz_m = (pickle_xyz_ft + xyz_min) * 0.3048
where xyz_min is read from the annotation CSV (the per-axis min that
create_labels subtracted before training).
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
WALL_ID = 2
FT_TO_M = 0.3048
CHUNK_SIZE = 1_000_000
ANNOTATION_CHUNK_SIZE = 500_000
MAX_E57_POINTS = 500_000
OUTPUT_HTML = f"e57_with_wall_{WALL_ID}.html"

BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _compute_xyz_min_from_annotation(path, chunksize=ANNOTATION_CHUNK_SIZE):
    """Mirror np.amin(data_label, axis=0)[0:3] inside create_labels()."""
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(
            xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0)
        )
    return xyz_min


def _iter_e57_xyzrgb_chunks(path, chunk_size):
    import pye57

    e57 = pye57.E57(path)
    color_fields = ["cartesianX", "cartesianY", "cartesianZ", "colorRed", "colorGreen", "colorBlue"]
    plain_fields = ["cartesianX", "cartesianY", "cartesianZ"]
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            try:
                data, buffers = e57.make_buffers(color_fields, chunk_size)
                has_color = True
            except Exception:
                data, buffers = e57.make_buffers(plain_fields, chunk_size)
                has_color = False
            reader = header.points.reader(buffers)
            while True:
                n = reader.read()
                if n <= 0:
                    break
                xyz = np.column_stack(
                    (data["cartesianX"][:n], data["cartesianY"][:n], data["cartesianZ"][:n])
                ).astype(np.float64, copy=False)
                if has_color:
                    rgb = np.column_stack(
                        (data["colorRed"][:n], data["colorGreen"][:n], data["colorBlue"][:n])
                    ).astype(np.uint8, copy=False)
                else:
                    rgb = np.full((n, 3), 180, dtype=np.uint8)
                yield xyz, rgb
    finally:
        e57.close()


def _bbox_trace(bbox_pts, color, width, name, showlegend=True, legendgroup=None):
    xs, ys, zs = [], [], []
    for a, b in BBOX_EDGES:
        xs.extend([bbox_pts[a, 0], bbox_pts[b, 0], None])
        ys.extend([bbox_pts[a, 1], bbox_pts[b, 1], None])
        zs.extend([bbox_pts[a, 2], bbox_pts[b, 2], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name, showlegend=showlegend, legendgroup=legendgroup,
        hoverinfo="name",
    )


def _stride_sample(xyz, rgb, kept_so_far, target_total, approx_total):
    """Pick a stride-based subsample of this chunk so we end up near target_total."""
    if approx_total <= target_total:
        return xyz, rgb
    keep_ratio = target_total / approx_total
    n_keep = max(1, int(len(xyz) * keep_ratio))
    idx = np.linspace(0, len(xyz) - 1, n_keep, dtype=np.int64)
    return xyz[idx], rgb[idx]


def main() -> None:
    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft) from annotation: {xyz_min_ft}")
    print(f"  ≡ offset (m): {xyz_min_ft * FT_TO_M}")

    target = next(w for w in wall_output["walls"] if int(w["id"]) == WALL_ID)
    target_bbox_m = (
        np.array([[p["x"], p["y"], p["z"]] for p in target["bbox"]]) + xyz_min_ft
    ) * FT_TO_M
    print(
        f"wall_{WALL_ID} bbox (m) "
        f"x=({target_bbox_m[:,0].min():.2f},{target_bbox_m[:,0].max():.2f}) "
        f"y=({target_bbox_m[:,1].min():.2f},{target_bbox_m[:,1].max():.2f}) "
        f"z=({target_bbox_m[:,2].min():.2f},{target_bbox_m[:,2].max():.2f})"
    )

    pickle_pts_ft = np.array(
        [
            [p["location"]["x"], p["location"]["y"], p["location"]["z"]]
            for p in wall_output["points"]
            if int(p["id"]) == WALL_ID
        ],
        dtype=np.float64,
    )
    pickle_pts_m = (pickle_pts_ft + xyz_min_ft) * FT_TO_M

    import pye57
    e57 = pye57.E57(E57_PATH)
    approx_total = sum(e57.get_header(i).point_count for i in range(e57.scan_count))
    e57.close()
    print(f"e57 approx total points: {approx_total}")

    xyz_kept, rgb_kept = [], []
    seen = 0
    for xyz, rgb in _iter_e57_xyzrgb_chunks(E57_PATH, CHUNK_SIZE):
        seen += len(xyz)
        xyz_s, rgb_s = _stride_sample(xyz, rgb, len(xyz_kept), MAX_E57_POINTS, approx_total)
        xyz_kept.append(xyz_s)
        rgb_kept.append(rgb_s)
    xyz_plot = np.vstack(xyz_kept)
    rgb_plot = np.vstack(rgb_kept)
    print(f"e57 scanned: {seen}, sampled for plot: {len(xyz_plot)}")
    print(
        f"e57 bbox (m) "
        f"x=({xyz_plot[:,0].min():.2f},{xyz_plot[:,0].max():.2f}) "
        f"y=({xyz_plot[:,1].min():.2f},{xyz_plot[:,1].max():.2f}) "
        f"z=({xyz_plot[:,2].min():.2f},{xyz_plot[:,2].max():.2f})"
    )

    color_strs = [f"rgb({r},{g},{b})" for r, g, b in rgb_plot]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xyz_plot[:, 0], y=xyz_plot[:, 1], z=xyz_plot[:, 2],
            mode="markers",
            marker=dict(size=1, color=color_strs, opacity=0.6),
            name=f"e57 sample ({len(xyz_plot)} pts)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=pickle_pts_m[:, 0], y=pickle_pts_m[:, 1], z=pickle_pts_m[:, 2],
            mode="markers",
            marker=dict(size=3, color="red", opacity=1.0),
            name=f"pickle wall_{WALL_ID} ({len(pickle_pts_m)} pts)",
        )
    )
    fig.add_trace(_bbox_trace(target_bbox_m, "red", 5, f"wall_{WALL_ID} bbox"))

    first_other = True
    for w in wall_output["walls"]:
        if int(w["id"]) == WALL_ID:
            continue
        bbox = w.get("bbox", [])
        if len(bbox) != 8:
            continue
        bbox_m = (np.array([[p["x"], p["y"], p["z"]] for p in bbox]) + xyz_min_ft) * FT_TO_M
        fig.add_trace(
            _bbox_trace(
                bbox_m, "rgba(0,180,0,0.4)", 2,
                name="other walls" if first_other else f"wall_{w['id']}",
                showlegend=first_other,
                legendgroup="other_walls",
            )
        )
        first_other = False

    fig.update_layout(
        title=f"e57 + wall_{WALL_ID} overlay (meters)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )
    fig.write_html(OUTPUT_HTML)
    print(f"wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
