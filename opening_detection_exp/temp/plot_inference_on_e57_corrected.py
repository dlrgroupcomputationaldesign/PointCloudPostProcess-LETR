"""Overlay the inference CSV on the e57, using the exact xyz_min that was
subtracted by create_labels (instead of approximating with e57_header_min).

Transform chain (forward):
    e57_m --(*1/0.3048)--> e57_ft  ==  annotation CSV X,Y,Z (feet)
    annotation_ft --(-= xyz_min)--> npy_ft  --train/infer--> inference CSV x,y,z

Inverse (this script):
    inference_ft + xyz_min  -->  annotation_ft
    annotation_ft * 0.3048  -->  e57_m

So: e57_m = (inference_csv_ft + xyz_min) * 0.3048

xyz_min is computed from the ANNOTATION CSV (with X,Y,Z columns), matching the
np.amin in create_labels.

Set ANNOTATION_CSV_PATH to the intermediate CSV. If left as None, the script
falls back to estimating xyz_min from e57_header_min (less accurate).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ANNOTATION_CSV_PATH = "WyomingStateFair_Laramie.csv"
INFERENCE_CSV_PATH = "WyomingStateFair_Laramie_inference_output.csv"
E57_PATH = "LaramieCM.e57"
FT_TO_M = 0.3048
CHUNK_SIZE = 1_000_000
MAX_E57_POINTS = 300_000
MAX_INFERENCE_POINTS = 200_000
OUTPUT_HTML = "inference_on_e57_corrected.html"

LABEL_NAMES = {0: "Other", 1: "Floor", 2: "Ceiling", 3: "Wall"}
LABEL_COLORS = {0: "lightgray", 1: "tan", 2: "lightblue", 3: "salmon"}

BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _read_e57_header_min(path):
    import pye57

    e57 = pye57.E57(path)
    try:
        h = e57.get_header(0)
        return np.array([h.xMinimum, h.yMinimum, h.zMinimum], dtype=np.float64)
    finally:
        e57.close()


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


def _compute_xyz_min_from_annotation(path):
    """Mirror np.amin(data_label, axis=0)[0:3] in create_labels()."""
    cols = ["X", "Y", "Z"]
    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
        xyz_min = np.minimum(xyz_min, chunk[cols].to_numpy(dtype=np.float64).min(axis=0))
    return xyz_min


def _aabb_corners(lo, hi):
    return np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )


def _bbox_trace(corners, color, name, width=4):
    xs, ys, zs = [], [], []
    for a, b in BBOX_EDGES:
        xs.extend([corners[a, 0], corners[b, 0], None])
        ys.extend([corners[a, 1], corners[b, 1], None])
        zs.extend([corners[a, 2], corners[b, 2], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        name=name,
    )


def main() -> None:
    if ANNOTATION_CSV_PATH:
        print(f"Reading annotation CSV: {ANNOTATION_CSV_PATH}")
        xyz_min_ft = _compute_xyz_min_from_annotation(ANNOTATION_CSV_PATH)
        print(f"  xyz_min (ft) from annotation: {xyz_min_ft}")
        offset_m = xyz_min_ft * FT_TO_M
        print(f"  offset (m) = xyz_min * 0.3048: {offset_m}")
    else:
        header_min = _read_e57_header_min(E57_PATH)
        offset_m = header_min
        xyz_min_ft = header_min / FT_TO_M
        print(f"Annotation CSV not provided; falling back to e57_header_min as offset.")
        print(f"  e57_header_min (m): {header_min}")
        print(f"  implied xyz_min (ft): {xyz_min_ft}")
        print("  NOTE: this is approximate; provide ANNOTATION_CSV_PATH for exact xyz_min.")

    df = pd.read_csv(INFERENCE_CSV_PATH)
    inf_ft = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    inf_m_all = (inf_ft + xyz_min_ft) * FT_TO_M
    print(f"\ninference CSV n={len(inf_ft)}")
    print(f"  inference bbox (ft):  min={inf_ft.min(axis=0)}, max={inf_ft.max(axis=0)}")
    print(f"  mapped to e57 m:      min={inf_m_all.min(axis=0)}, max={inf_m_all.max(axis=0)}")

    e57_xyz_chunks, e57_rgb_chunks = [], []
    e57_lo = np.array([np.inf] * 3)
    e57_hi = np.array([-np.inf] * 3)
    e57_total = 0
    for xyz, rgb in _iter_e57_xyzrgb_chunks(E57_PATH, CHUNK_SIZE):
        e57_lo = np.minimum(e57_lo, xyz.min(axis=0))
        e57_hi = np.maximum(e57_hi, xyz.max(axis=0))
        e57_xyz_chunks.append(xyz)
        e57_rgb_chunks.append(rgb)
        e57_total += len(xyz)
    e57_xyz = np.vstack(e57_xyz_chunks)
    e57_rgb = np.vstack(e57_rgb_chunks)
    print(f"e57 n={e57_total}, bbox (m): min={e57_lo}, max={e57_hi}")
    print(f"\nshift after correction:")
    print(f"  min e57 - min inf_m = {e57_lo - inf_m_all.min(axis=0)}")
    print(f"  max e57 - max inf_m = {e57_hi - inf_m_all.max(axis=0)}")

    if len(e57_xyz) > MAX_E57_POINTS:
        idx = np.linspace(0, len(e57_xyz) - 1, MAX_E57_POINTS, dtype=np.int64)
        e57_xyz = e57_xyz[idx]
        e57_rgb = e57_rgb[idx]
    e57_colors = [f"rgb({r},{g},{b})" for r, g, b in e57_rgb]

    rng = np.random.default_rng(0)
    if len(inf_m_all) > MAX_INFERENCE_POINTS:
        sel = rng.choice(len(inf_m_all), MAX_INFERENCE_POINTS, replace=False)
        inf_m_plot = inf_m_all[sel]
        labels_plot = df["pred_label"].to_numpy(dtype=int)[sel]
    else:
        inf_m_plot = inf_m_all
        labels_plot = df["pred_label"].to_numpy(dtype=int)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=e57_xyz[:, 0], y=e57_xyz[:, 1], z=e57_xyz[:, 2],
            mode="markers",
            marker=dict(size=1, color=e57_colors, opacity=0.35),
            name=f"e57 sample ({len(e57_xyz)} of {e57_total})",
        )
    )
    for label_int, label_name in LABEL_NAMES.items():
        mask = labels_plot == label_int
        if not np.any(mask):
            continue
        pts = inf_m_plot[mask]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.8, color=LABEL_COLORS[label_int], opacity=0.85),
                name=f"inf {label_name} ({len(pts)})",
            )
        )

    fig.add_trace(_bbox_trace(_aabb_corners(e57_lo, e57_hi), "orange", "e57 bbox", width=5))
    fig.add_trace(
        _bbox_trace(
            _aabb_corners(inf_m_all.min(axis=0), inf_m_all.max(axis=0)),
            "cyan", "inference bbox (mapped)", width=5,
        )
    )

    title_offset = (
        f"xyz_min = {xyz_min_ft} ft → offset = {offset_m} m"
        if ANNOTATION_CSV_PATH
        else f"FALLBACK: offset = e57_header_min = {offset_m}"
    )
    fig.update_layout(
        title=f"Inference CSV mapped to e57 meters — {title_offset}",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )
    fig.write_html(OUTPUT_HTML)
    print(f"\nwrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
