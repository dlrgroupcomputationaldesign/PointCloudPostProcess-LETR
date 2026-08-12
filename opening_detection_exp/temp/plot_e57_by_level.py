"""Split LaramieCM.e57 into per-level point clouds based on the levels
implied by wall_output.pickle, and write one Plotly HTML per level.

Levels are derived from each wall's `levelIndex`; each level's z band is
the union of all its walls' zRanges (mapped to e57 meters via the
annotation CSV's xyz_min). Walls' bboxes are drawn on the corresponding
level's HTML for context.

Output: per-level HTML files named e57_level_<idx>.html in the working
directory.
"""

import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PICKLE_PATH = "wall_output.pickle"
E57_PATH = "LaramieCM.e57"
ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
FT_TO_M = 0.3048
CHUNK_SIZE = 1_000_000
ANNOTATION_CHUNK_SIZE = 500_000
MAX_POINTS_PER_LEVEL = 400_000
Z_PADDING_M = 0.30          # extend each level's band slightly to catch floor/ceiling slabs

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


def _ft_to_m(value_ft, axis_offset_ft, ft_to_m=FT_TO_M):
    return (float(value_ft) + float(axis_offset_ft)) * ft_to_m


def _wall_bbox_to_meters(wall_ft, xyz_min_ft):
    pts = np.array([[p["x"], p["y"], p["z"]] for p in wall_ft["bbox"]], dtype=np.float64)
    return (pts + xyz_min_ft) * FT_TO_M


def _group_walls_by_level(walls_ft, xyz_min_ft):
    """Return {level_idx: {"z_min_m":..., "z_max_m":..., "walls":[bbox_m, ...]}}."""
    levels = {}
    for w in walls_ft:
        level = int(w.get("levelIndex", 0))
        zr = w["zRange"]
        z_min_m = _ft_to_m(zr["min"], xyz_min_ft[2])
        z_max_m = _ft_to_m(zr["max"], xyz_min_ft[2])
        bbox_m = _wall_bbox_to_meters(w, xyz_min_ft)
        entry = levels.setdefault(
            level,
            {"z_min_m": np.inf, "z_max_m": -np.inf, "walls": []},
        )
        entry["z_min_m"] = min(entry["z_min_m"], z_min_m)
        entry["z_max_m"] = max(entry["z_max_m"], z_max_m)
        entry["walls"].append({"id": w["id"], "bbox_m": bbox_m})
    return dict(sorted(levels.items()))


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


def _bbox_trace(bbox_m, color, name, width=3, legendgroup=None, showlegend=True):
    xs, ys, zs = [], [], []
    for a, b in BBOX_EDGES:
        xs.extend([bbox_m[a, 0], bbox_m[b, 0], None])
        ys.extend([bbox_m[a, 1], bbox_m[b, 1], None])
        zs.extend([bbox_m[a, 2], bbox_m[b, 2], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name, legendgroup=legendgroup, showlegend=showlegend,
        hoverinfo="name",
    )


def main() -> None:
    with open(PICKLE_PATH, "rb") as f:
        wall_output = pickle.load(f)
    walls_ft = wall_output["walls"]
    print(f"loaded {len(walls_ft)} walls")

    xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
    print(f"xyz_min (ft): {xyz_min_ft}")

    levels = _group_walls_by_level(walls_ft, xyz_min_ft)
    print(f"detected {len(levels)} level(s):")
    for idx, info in levels.items():
        z0 = info["z_min_m"] - Z_PADDING_M
        z1 = info["z_max_m"] + Z_PADDING_M
        info["filter_z_min"] = z0
        info["filter_z_max"] = z1
        info["xyz_kept"] = []
        info["rgb_kept"] = []
        info["pts_seen"] = 0
        print(
            f"  level {idx}: {len(info['walls'])} walls, "
            f"z=({info['z_min_m']:.2f},{info['z_max_m']:.2f}) m → "
            f"filter z=({z0:.2f},{z1:.2f}) m (with {Z_PADDING_M} m padding)"
        )

    for ci, (xyz, rgb) in enumerate(_iter_e57_xyzrgb(E57_PATH, CHUNK_SIZE)):
        for idx, info in levels.items():
            mask = (xyz[:, 2] >= info["filter_z_min"]) & (xyz[:, 2] <= info["filter_z_max"])
            if not np.any(mask):
                continue
            info["xyz_kept"].append(xyz[mask])
            info["rgb_kept"].append(rgb[mask])
            info["pts_seen"] += int(mask.sum())
        if (ci + 1) % 10 == 0:
            print(f"  processed {ci + 1} chunks")

    for idx, info in levels.items():
        if not info["xyz_kept"]:
            print(f"  level {idx}: no points found, skipping")
            continue
        xyz_all = np.vstack(info["xyz_kept"])
        rgb_all = np.vstack(info["rgb_kept"])
        n = len(xyz_all)

        if n > MAX_POINTS_PER_LEVEL:
            sel = np.linspace(0, n - 1, MAX_POINTS_PER_LEVEL, dtype=np.int64)
            xyz_plot = xyz_all[sel]
            rgb_plot = rgb_all[sel]
        else:
            xyz_plot = xyz_all
            rgb_plot = rgb_all
        colors = [f"rgb({r},{g},{b})" for r, g, b in rgb_plot]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=xyz_plot[:, 0], y=xyz_plot[:, 1], z=xyz_plot[:, 2],
                mode="markers",
                marker=dict(size=1, color=colors, opacity=0.6),
                name=f"level {idx} e57 ({len(xyz_plot)} of {n} pts)",
            )
        )

        first = True
        for w in info["walls"]:
            fig.add_trace(
                _bbox_trace(
                    w["bbox_m"],
                    "rgba(220,40,40,0.9)",
                    f"level {idx} walls" if first else f"wall_{w['id']}",
                    width=3,
                    legendgroup=f"level_{idx}_walls",
                    showlegend=first,
                )
            )
            first = False

        fig.update_layout(
            title=(
                f"e57 level {idx} — z=({info['z_min_m']:.2f},{info['z_max_m']:.2f}) m, "
                f"{len(info['walls'])} walls, {n} pts in band"
            ),
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                aspectmode="data",
            ),
        )
        out_path = f"e57_level_{idx}.html"
        fig.write_html(out_path)
        print(f"  wrote {out_path}  pts={n}")


if __name__ == "__main__":
    main()
