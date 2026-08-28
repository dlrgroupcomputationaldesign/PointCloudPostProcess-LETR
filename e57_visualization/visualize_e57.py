"""Visualize one or more original e57 point clouds.

Three rendering modes, all keeping native e57 meters (no shift to origin):

  --mode voxel   (default) — voxel-downsample to ``--voxel`` m, write a Plotly
                  HTML. Best for sharing a static page; output is light (~80k–
                  500k points).
  --mode stride  — keep every Nth point (``--stride``), write a Plotly HTML.
                  Cheaper than voxel (O(1) per chunk) but spatially uneven.
  --mode open3d  — open an Open3D desktop window with EVERY point. No HTML.
                  Smooth GPU rendering of 95M+ points, but only viewable on
                  this machine.

Examples:
    python visualize_e57.py --input e57/LaramieCM.e57                        # default voxel HTML
    python visualize_e57.py --input e57/ --voxel 0.2 --output detailed.html  # finer voxel
    python visualize_e57.py --input e57/LaramieCM.e57 --mode stride --stride 200
    python visualize_e57.py --input e57/LaramieCM.e57 --mode open3d          # all points, desktop window
"""

import argparse
from pathlib import Path

import numpy as np


def iter_e57_files(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.e57")) + sorted(p.glob("*.E57"))
    raise SystemExit(f"input path not found: {input_path}")


def _open_e57_chunks(e57_path, chunk_size):
    """Yield (xyz, rgb, has_color) chunks from an e57 file."""
    import pye57

    e57 = pye57.E57(str(e57_path))
    XYZ = ["cartesianX", "cartesianY", "cartesianZ"]
    COLOR = ["colorRed", "colorGreen", "colorBlue"]
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            # Ask the scan what it actually stores rather than trying colour and
            # catching the failure. Plenty of e57s are geometry-only -- a laser
            # scan without a camera pass defines just the cartesian fields -- and
            # requesting an undefined one raises ErrorPathUndefined.
            #
            # It has to be a lookup, not a try/except: make_buffers() only
            # ALLOCATES, so it succeeds for fields the file has never heard of.
            # The path check happens in points.reader(), so wrapping
            # make_buffers alone catches nothing and the error escapes.
            #
            # Decided per scan, since a multi-scan file can mix the two.
            available = set(header.point_fields)
            has_color = all(f in available for f in COLOR)
            data, buffers = e57.make_buffers(XYZ + COLOR if has_color else XYZ,
                                             chunk_size)
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
                yield xyz, rgb, has_color
    finally:
        e57.close()


def _voxel_downsample_streamed(e57_path, voxel_size, chunk_size):
    """Stream the e57 in chunks and keep one representative point per voxel."""
    KEY_OFFSET = 1 << 20
    seen = set()
    kept_xyz, kept_rgb = [], []

    for xyz, rgb, _ in _open_e57_chunks(e57_path, chunk_size):
        keys_3d = np.floor(xyz / voxel_size).astype(np.int64)
        if np.abs(keys_3d).max(initial=0) >= KEY_OFFSET:
            raise ValueError(
                "voxel keys exceed pack range; scan spans >~400km or voxel too small."
            )
        kx = keys_3d[:, 0] + KEY_OFFSET
        ky = keys_3d[:, 1] + KEY_OFFSET
        kz = keys_3d[:, 2] + KEY_OFFSET
        packed = (kx << 42) | (ky << 21) | kz

        _, first_idx = np.unique(packed, return_index=True)
        packed_u = packed[first_idx]
        xyz_u = xyz[first_idx]
        rgb_u = rgb[first_idx]

        new_mask = np.fromiter(
            (int(k) not in seen for k in packed_u),
            dtype=bool,
            count=len(packed_u),
        )
        if np.any(new_mask):
            kept_xyz.append(xyz_u[new_mask])
            kept_rgb.append(rgb_u[new_mask])
            seen.update(int(k) for k in packed_u[new_mask])

    if not kept_xyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(kept_xyz), np.vstack(kept_rgb)


def _stride_sample_streamed(e57_path, stride, chunk_size):
    """Keep every Nth point globally across chunks."""
    stride = max(1, int(stride))
    total_seen = 0
    kept_xyz, kept_rgb = [], []
    for xyz, rgb, _ in _open_e57_chunks(e57_path, chunk_size):
        # First local index that lines up with the global Nth-point sequence.
        offset = total_seen % stride
        first_local = (stride - offset) % stride
        if first_local < len(xyz):
            kept_xyz.append(xyz[first_local::stride])
            kept_rgb.append(rgb[first_local::stride])
        total_seen += len(xyz)
    if not kept_xyz:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(kept_xyz), np.vstack(kept_rgb)


def _load_full_e57(e57_path, chunk_size):
    """Load every point from an e57 into one (xyz, rgb) pair. Memory-heavy."""
    xyz_chunks, rgb_chunks = [], []
    for xyz, rgb, _ in _open_e57_chunks(e57_path, chunk_size):
        xyz_chunks.append(xyz)
        rgb_chunks.append(rgb)
    if not xyz_chunks:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    return np.vstack(xyz_chunks), np.vstack(rgb_chunks)


def _show_open3d(files, no_color, point_size, chunk_size):
    """Open an interactive Open3D window with all points from all files."""
    import open3d as o3d

    geometries = []
    for f in files:
        print(f"reading {f}...")
        xyz, rgb = _load_full_e57(f, chunk_size)
        print(f"  {len(xyz)} points")
        if not len(xyz):
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if no_color:
            pcd.colors = o3d.utility.Vector3dVector(np.full((len(xyz), 3), 0.55))
        else:
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        geometries.append(pcd)

    if not geometries:
        raise SystemExit("no points loaded")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="e57 viewer", width=1280, height=800)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([0.05, 0.05, 0.07])
    print("Open3D window opened — close it to exit.")
    vis.run()
    vis.destroy_window()


def _write_html(files, sample_fn, args):
    """Sample each e57 and write one Plotly HTML with one trace per file."""
    import plotly.graph_objects as go

    fig = go.Figure()
    total_kept = 0
    for f in files:
        print(f"reading {f}...")
        xyz, rgb = sample_fn(f)
        print(f"  {len(xyz)} points kept")
        if not len(xyz):
            continue
        total_kept += len(xyz)
        if args.no_color:
            color = "rgb(140,140,140)"
        else:
            color = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=args.point_size, color=color, opacity=1.0),
                name=f"{f.stem} ({len(xyz)} pts)",
            )
        )

    if total_kept == 0:
        raise SystemExit("no points kept")

    if args.mode == "voxel":
        title = f"e57 visualization (voxel = {args.voxel} m, no shift)"
    else:
        title = f"e57 visualization (stride = {args.stride}, no shift)"

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            camera=dict(up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=title,
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
    print(f"wrote {args.output}  total {total_kept} pts across {len(files)} file(s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="single .e57 file or a directory containing .e57 files")
    ap.add_argument("--mode", choices=["voxel", "stride", "open3d"], default="voxel",
                    help="voxel: downsample to HTML; stride: thin to HTML; open3d: all points in desktop window")
    ap.add_argument("--output", default="e57_visualization.html",
                    help="(html modes only) output html path")
    ap.add_argument("--voxel", type=float, default=0.4, help="(voxel mode) cube size in meters")
    ap.add_argument("--stride", type=int, default=100,
                    help="(stride mode) keep every Nth point")
    ap.add_argument("--point-size", type=float, default=1.5,
                    help="marker size — Plotly units for html, pixels for open3d")
    ap.add_argument("--no-color", action="store_true",
                    help="ignore e57 RGB and draw all points in solid gray")
    ap.add_argument("--chunk-size", type=int, default=2_000_000,
                    help="e57 read chunk size (points)")
    args = ap.parse_args()

    files = iter_e57_files(args.input)
    if not files:
        raise SystemExit(f"no .e57 files found in {args.input}")
    print(f"found {len(files)} e57 file(s); mode={args.mode}")

    if args.mode == "open3d":
        _show_open3d(files, args.no_color, args.point_size, args.chunk_size)
        return

    if args.mode == "voxel":
        sample_fn = lambda f: _voxel_downsample_streamed(f, args.voxel, args.chunk_size)
    else:  # stride
        sample_fn = lambda f: _stride_sample_streamed(f, args.stride, args.chunk_size)

    _write_html(files, sample_fn, args)


if __name__ == "__main__":
    main()
