"""For every wall in wall_output.pickle, write a 3D HTML showing the dense
e57 points + an attached length-z log(1+count) surface, mirroring
plot_wall_17_log_attached.py but bulk-processed.

One pass through LaramieCM.e57 accumulates every wall's count_img AND a
stride-sampled subset of points for visualization. Per-chunk stride sampling
keeps memory bounded at ~MAX_WALL_POINTS per wall (not per chunk).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PICKLE_PATH = "wall_output.pickle"
E57_PATH = r"e57\Corteva.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
OUTPUT_DIR = "wall_log_html"
FT_TO_M = 0.3048
LENGTH_BIN_M = 0.05
Z_BIN_M = 0.05
MARGIN_M = 0.25
CHUNK_SIZE = 2_000_000
ANNOTATION_CHUNK_SIZE = 500_000
MAX_WALL_POINTS = 200_000
LOG_NORMAL_OFFSET = 0.10
LOG_OPACITY = 0.95
POINT_SIZE = 1.3
POINT_OPACITY = 0.25

# Restrict to specific wall IDs (list of ints) or None for every wall.
ONLY_WALL_IDS = None

BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _compute_xyz_min_from_annotation(path, chunksize=ANNOTATION_CHUNK_SIZE):
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0))
    return xyz_min


def _transform_wall_to_meters(wall_ft, xyz_min_ft, ft_to_m=FT_TO_M):
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


def _wall_frame(wall):
    fp = wall.get("footprint")
    if fp and len(fp) >= 3:
        pts = np.array([[p["x"], p["y"]] for p in fp], dtype=np.float64)
        edges = [pts[(i + 1) % len(pts)] - pts[i] for i in range(len(pts))]
    else:
        pts = np.array([[p["x"], p["y"]] for p in wall["bbox"]], dtype=np.float64)
        edges = [pts[j] - pts[i] for i in range(len(pts)) for j in range(i + 1, len(pts))]
    norms = np.linalg.norm(edges, axis=1)
    best = int(np.argmax(norms))
    length_axis = edges[best] / norms[best]
    normal_axis = np.array([-length_axis[1], length_axis[0]])
    origin_xy = pts.mean(axis=0)
    centered = pts - origin_xy
    s_proj = centered @ length_axis
    return {
        "origin_xy": origin_xy,
        "length_axis": length_axis,
        "normal_axis": normal_axis,
        "s_min": float(s_proj.min()),
        "s_max": float(s_proj.max()),
        "z_min": float(wall["zRange"]["min"]),
        "z_max": float(wall["zRange"]["max"]),
    }


def _wall_crop_geometry(wall, margin):
    from shapely.geometry import Polygon

    fp = wall.get("footprint") or []
    pts = np.asarray([[p["x"], p["y"]] for p in fp], dtype=np.float64) if fp else None
    polygon = None
    if pts is not None and len(pts) >= 3:
        polygon = Polygon(pts)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and margin:
            polygon = polygon.buffer(float(margin))

    z_min = float(wall["zRange"]["min"]) - margin
    z_max = float(wall["zRange"]["max"]) + margin
    if polygon is not None and not polygon.is_empty:
        minx, miny, maxx, maxy = polygon.bounds
        minb = np.array([minx, miny, z_min])
        maxb = np.array([maxx, maxy, z_max])
    else:
        bbox = np.array([[p["x"], p["y"], p["z"]] for p in wall["bbox"]])
        minb = bbox.min(axis=0) - margin
        maxb = bbox.max(axis=0) + margin
    return {"minb": minb, "maxb": maxb, "polygon": polygon}


def _crop_xyz_rgb(xyz, rgb, crop_geom):
    minb, maxb = crop_geom["minb"], crop_geom["maxb"]
    aabb = (
        (xyz[:, 0] >= minb[0]) & (xyz[:, 0] <= maxb[0])
        & (xyz[:, 1] >= minb[1]) & (xyz[:, 1] <= maxb[1])
        & (xyz[:, 2] >= minb[2]) & (xyz[:, 2] <= maxb[2])
    )
    if not np.any(aabb):
        return xyz[:0], rgb[:0]
    xyz_a = xyz[aabb]
    rgb_a = rgb[aabb]
    polygon = crop_geom.get("polygon")
    if polygon is None:
        return xyz_a, rgb_a
    try:
        from shapely import contains_xy

        inside = contains_xy(polygon, xyz_a[:, 0], xyz_a[:, 1])
    except ImportError:
        from shapely.geometry import Point

        inside = np.array(
            [polygon.contains(Point(x, y)) for x, y in xyz_a[:, :2]], dtype=bool
        )
    return xyz_a[inside], rgb_a[inside]


def _iter_e57_xyzrgb(path, chunk_size=CHUNK_SIZE):
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


def _surface_grid_on_wall(frame, n_length, n_z, normal_offset):
    s_edges = np.linspace(frame["s_min"], frame["s_max"], n_length + 1)
    z_edges = np.linspace(frame["z_min"], frame["z_max"], n_z + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    s_grid, z_grid = np.meshgrid(s_centers, z_centers)
    origin = frame["origin_xy"] + normal_offset * frame["normal_axis"]
    x_grid = origin[0] + s_grid * frame["length_axis"][0]
    y_grid = origin[1] + s_grid * frame["length_axis"][1]
    return x_grid, y_grid, z_grid


def _e57_total_point_count(path):
    import pye57

    e57 = pye57.E57(path)
    try:
        return sum(e57.get_header(i).point_count for i in range(e57.scan_count))
    finally:
        e57.close()


def _build_html_for_wall(d, out_path):
    """Build the per-wall plotly HTML from accumulated data."""
    wall_id = d["id"]
    frame = d["frame"]
    wall_m = d["wall_m"]
    count_img = d["count_img"]
    pts_total = d["pts_used"]

    wall_xyz = np.vstack(d["xyz_kept"]) if d["xyz_kept"] else np.empty((0, 3))
    wall_rgb = np.vstack(d["rgb_kept"]) if d["rgb_kept"] else np.empty((0, 3), dtype=np.uint8)

    # Second sampling pass to enforce final MAX_WALL_POINTS cap.
    if len(wall_xyz) > MAX_WALL_POINTS:
        idx = np.linspace(0, len(wall_xyz) - 1, MAX_WALL_POINTS, dtype=np.int64)
        wall_xyz = wall_xyz[idx]
        wall_rgb = wall_rgb[idx]

    log_img = np.log1p(count_img.astype(np.float64))
    bbox_pts = np.array([[c["x"], c["y"], c["z"]] for c in wall_m["bbox"]])

    fig = go.Figure()
    if len(wall_xyz):
        colors = [f"rgb({r},{g},{b})" for r, g, b in wall_rgb]
        fig.add_trace(
            go.Scatter3d(
                x=wall_xyz[:, 0], y=wall_xyz[:, 1], z=wall_xyz[:, 2],
                mode="markers",
                marker=dict(size=POINT_SIZE, color=colors, opacity=POINT_OPACITY),
                name=f"E57 wall points ({len(wall_xyz)} shown of {d['cropped_total']})",
                hoverinfo="skip",
            )
        )

    n_z, n_length = count_img.shape
    for side_offset, side_label in ((LOG_NORMAL_OFFSET, "+normal"), (-LOG_NORMAL_OFFSET, "-normal")):
        xg, yg, zg = _surface_grid_on_wall(frame, n_length, n_z, side_offset)
        fig.add_trace(
            go.Surface(
                x=xg, y=yg, z=zg,
                surfacecolor=log_img,
                colorscale="Greys",
                opacity=LOG_OPACITY,
                name=f"log(1 + count) [{side_label}]",
                colorbar=dict(title="log(1+count)", x=1.02) if side_offset > 0 else None,
                showscale=side_offset > 0,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
                showlegend=True,
            )
        )

    xs, ys, zs = [], [], []
    for a, b in BBOX_EDGES:
        xs.extend([bbox_pts[a, 0], bbox_pts[b, 0], None])
        ys.extend([bbox_pts[a, 1], bbox_pts[b, 1], None])
        zs.extend([bbox_pts[a, 2], bbox_pts[b, 2], None])
    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color="rgb(220,40,40)", width=5),
            name=f"wall_{wall_id} bbox",
        )
    )

    fig.update_layout(
        title=f"wall_{wall_id}: e57 + log surface ({pts_total} pts in histogram, {d['cropped_total']} cropped)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    if ONLY_WALL_IDS is not None:
        wanted = {int(i) for i in ONLY_WALL_IDS}
        walls_ft = [w for w in walls_ft if int(w["id"]) in wanted]
    print(f"loaded {len(walls_ft)} walls")

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft): {xyz_min_ft}")

    total_pts = _e57_total_point_count(E57_PATH)
    expected_chunks = max(1, total_pts // CHUNK_SIZE)
    per_chunk_cap = max(1, MAX_WALL_POINTS // expected_chunks)
    print(
        f"e57 total points: {total_pts}, expected ~{expected_chunks} chunks, "
        f"per-chunk visualization cap per wall: {per_chunk_cap}"
    )

    walls_data = []
    for w in walls_ft:
        try:
            wall_m = _transform_wall_to_meters(w, xyz_min_ft)
            frame = _wall_frame(wall_m)
            wall_length = frame["s_max"] - frame["s_min"]
            wall_height = frame["z_max"] - frame["z_min"]
            if wall_length <= 0 or wall_height <= 0:
                print(f"  skip wall_{w['id']}: degenerate dims")
                continue
            n_length = max(5, int(np.ceil(wall_length / LENGTH_BIN_M)))
            n_z = max(5, int(np.ceil(wall_height / Z_BIN_M)))
            walls_data.append(
                {
                    "id": w["id"],
                    "wall_m": wall_m,
                    "frame": frame,
                    "crop_geom": _wall_crop_geometry(wall_m, MARGIN_M),
                    "count_img": np.zeros((n_z, n_length), dtype=np.uint32),
                    "n_length": n_length,
                    "n_z": n_z,
                    "xyz_kept": [],
                    "rgb_kept": [],
                    "pts_used": 0,
                    "cropped_total": 0,
                }
            )
        except Exception as e:
            print(f"  skip wall_{w['id']}: {e}")
    print(f"prepared {len(walls_data)} wall accumulators")

    for ci, (xyz, rgb) in enumerate(_iter_e57_xyzrgb(E57_PATH, CHUNK_SIZE)):
        for d in walls_data:
            c_xyz, c_rgb = _crop_xyz_rgb(xyz, rgb, d["crop_geom"])
            n_cropped = len(c_xyz)
            if not n_cropped:
                continue
            d["cropped_total"] += n_cropped

            # Histogram uses ALL cropped points.
            rel = c_xyz[:, :2] - d["frame"]["origin_xy"]
            s = rel @ d["frame"]["length_axis"]
            z = c_xyz[:, 2]
            keep = (
                (s >= d["frame"]["s_min"]) & (s <= d["frame"]["s_max"])
                & (z >= d["frame"]["z_min"]) & (z <= d["frame"]["z_max"])
            )
            if np.any(keep):
                s_k = s[keep] - d["frame"]["s_min"]
                z_k = z[keep] - d["frame"]["z_min"]
                x_idx = np.clip(
                    np.floor(s_k / LENGTH_BIN_M).astype(np.int64), 0, d["n_length"] - 1
                )
                y_idx = np.clip(
                    np.floor(z_k / Z_BIN_M).astype(np.int64), 0, d["n_z"] - 1
                )
                np.add.at(d["count_img"], (y_idx, x_idx), 1)
                d["pts_used"] += int(keep.sum())

            # Stride-sample what we store for visualization, capped per chunk.
            if n_cropped > per_chunk_cap:
                idx = np.linspace(0, n_cropped - 1, per_chunk_cap, dtype=np.int64)
                d["xyz_kept"].append(c_xyz[idx])
                d["rgb_kept"].append(c_rgb[idx])
            else:
                d["xyz_kept"].append(c_xyz)
                d["rgb_kept"].append(c_rgb)
        if (ci + 1) % 5 == 0:
            print(f"  processed {ci + 1} chunks")

    print("writing per-wall HTMLs...")
    for d in walls_data:
        out_path = out_dir / f"wall_{d['id']}_e57_log_attached.html"
        _build_html_for_wall(d, str(out_path))
        print(
            f"  wrote {out_path}  "
            f"hist={d['pts_used']}  cropped={d['cropped_total']}  shape={d['count_img'].shape}"
        )


if __name__ == "__main__":
    main()
