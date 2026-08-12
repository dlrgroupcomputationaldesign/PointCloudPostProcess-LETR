"""Visualize post-process output JSON as an interactive 3-D HTML.

Points are colored by class (floor / ceiling / wall / door / window / opening),
wall and opening bounding boxes are drawn as wireframes, and floor/ceiling
outlines as closed loops. Reads a local JSON (elements.json / all_output.json /
any stage file) or downloads one from blob storage.

When ``--show-openings`` is set (and either an `opening/` folder exists locally
next to --input, or you ran with blob args), every per-wall log image +
annotated detection image found under ``{folder}/opening/`` is embedded into
the same HTML as a gallery section below the 3D scene -- useful for sharing a
single self-contained demo file.

Examples:
    python visualize_outputs.py --input all_output.json --output viz.html
    python visualize_outputs.py --input walls.json --show-openings
    # from blob (key via --account-key or AZURE_STORAGE_KEY env):
    python visualize_outputs.py --account-name pointcloudapistorage \
        --container jobs --folder <job_id> --file elements.json --show-openings
"""

import argparse
import base64
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# class -> color
CLASS_COLORS = {
    "floor": "#1f77b4",
    "ceiling": "#2ca02c",
    "wall": "#9e9e9e",
    "door": "#d62728",
    "window": "#ff7f0e",
    "opening": "#e377c2",
}

# 8-corner box edge connectivity, keyed by how each source orders its corners.
# Wall bbox (convert_to_edge_points): bottom face 0-1-2-3, top face 4-5-6-7.
WALL_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
]
# Opening bbox (_candidate_bbox): corner index = z*4 + t*2 + s.
OPENING_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # vary s
    (0, 2), (1, 3), (4, 6), (5, 7),   # vary t
    (0, 4), (1, 5), (2, 6), (3, 7),   # vary z
]


_WALL_ID_RE = re.compile(r"wall_(?P<id>[^_]+)_(?P<kind>log|detected)\.png$", re.IGNORECASE)


def _parse_opening_blob_name(name):
    """Return (wall_id, kind) for an 'opening/wall_X_log.png' style name, else None."""
    m = _WALL_ID_RE.search(name)
    if not m:
        return None
    return m.group("id"), m.group("kind").lower()


def _load_opening_images_local(folder):
    """Walk a local opening/ folder and return {wall_id: {kind: bytes}}."""
    folder = Path(folder)
    grouped = defaultdict(dict)
    if not folder.is_dir():
        return grouped
    for png in folder.glob("wall_*_*.png"):
        parsed = _parse_opening_blob_name(png.name)
        if parsed is None:
            continue
        wall_id, kind = parsed
        grouped[wall_id][kind] = png.read_bytes()
    return grouped


def _download_opening_folder_to_local(args, dest_dir):
    """Download every blob under {folder}/opening/ into dest_dir. Returns the count."""
    key = args.account_key or os.environ.get("AZURE_STORAGE_KEY")
    if not (args.account_name and args.container and args.folder and key):
        raise SystemExit(
            "downloading opening images requires --account-name, --container, "
            "--folder, and AZURE_STORAGE_KEY (or --account-key)"
        )

    from azure.storage.blob import BlobServiceClient

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    service = BlobServiceClient(
        account_url=f"https://{args.account_name}.blob.core.windows.net",
        credential=key,
    )
    container = service.get_container_client(args.container)
    prefix = f"{args.folder.strip('/')}/opening/"
    n = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        # Strip the folder prefix so files land flat in dest_dir.
        rel = blob.name[len(prefix):] if blob.name.startswith(prefix) else Path(blob.name).name
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(container.download_blob(blob.name).readall())
        n += 1
    print(f"downloaded {n} opening blob(s) into {dest}")
    return n


def _load_opening_images_blob(args):
    """List blobs under {folder}/opening/ and return {wall_id: {kind: bytes}}."""
    grouped = defaultdict(dict)
    key = args.account_key or os.environ.get("AZURE_STORAGE_KEY")
    if not (args.account_name and args.container and args.folder and key):
        return grouped

    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(
        account_url=f"https://{args.account_name}.blob.core.windows.net",
        credential=key,
    )
    container = service.get_container_client(args.container)
    prefix = f"{args.folder.strip('/')}/opening/"
    for blob in container.list_blobs(name_starts_with=prefix):
        parsed = _parse_opening_blob_name(blob.name)
        if parsed is None:
            continue
        wall_id, kind = parsed
        grouped[wall_id][kind] = container.download_blob(blob.name).readall()
    return grouped


def _load_opening_images(args):
    """Resolve opening images in this order:
       1. --opening-dir (downloads from blob first if --download-openings).
       2. <input-dir>/opening/ next to --input.
       3. blob {folder}/opening/ (streams without caching).
    """
    if args.opening_dir:
        dest = Path(args.opening_dir)
        if args.download_openings:
            _download_opening_folder_to_local(args, dest)
        return _load_opening_images_local(dest)
    if args.input:
        local = Path(args.input).resolve().parent / "opening"
        grouped = _load_opening_images_local(local)
        if grouped:
            return grouped
    return _load_opening_images_blob(args)


def _natural_wall_key(wall_id):
    """Sort wall ids numerically when possible ('2' before '10')."""
    try:
        return (0, int(wall_id))
    except (TypeError, ValueError):
        return (1, str(wall_id))


def _opening_gallery_html(grouped):
    """Build a self-contained gallery section from {wall_id: {kind: bytes}}."""
    if not grouped:
        return ""

    cards = []
    for wall_id in sorted(grouped.keys(), key=_natural_wall_key):
        entries = grouped[wall_id]
        imgs_html = []
        for kind in ("log", "detected"):
            data = entries.get(kind)
            if not data:
                continue
            b64 = base64.b64encode(data).decode("ascii")
            imgs_html.append(
                f'<figure><figcaption>{kind}</figcaption>'
                f'<img src="data:image/png;base64,{b64}" alt="wall {wall_id} {kind}"></figure>'
            )
        if not imgs_html:
            continue
        cards.append(
            f'<section class="wall-card"><h3>Wall {wall_id}</h3>'
            f'<div class="wall-imgs">{"".join(imgs_html)}</div></section>'
        )

    if not cards:
        return ""

    css = """
    <style>
      .opening-gallery { font-family: -apple-system, Segoe UI, sans-serif; padding: 1.5rem;
                         background: #fafafa; }
      .opening-gallery h2 { margin-top: 0; }
      .opening-gallery .wall-card { background: white; border: 1px solid #e0e0e0;
                                    border-radius: 6px; margin: 0 0 1rem 0; padding: 0.8rem 1rem; }
      .opening-gallery .wall-card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #555; }
      .opening-gallery .wall-imgs { display: flex; gap: 1rem; flex-wrap: wrap; }
      .opening-gallery figure { margin: 0; }
      .opening-gallery figcaption { font-size: 0.8rem; color: #888; margin-bottom: 4px; }
      .opening-gallery img { max-width: 100%; height: auto; border: 1px solid #ddd;
                             image-rendering: pixelated; }
    </style>
    """
    return (
        css
        + '<div class="opening-gallery">'
        + f'<h2>Opening images ({len(cards)} walls)</h2>'
        + "".join(cards)
        + "</div>"
    )


def load_json(args):
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            return json.load(f)
    if not (args.account_name and args.container and args.folder and args.file):
        raise SystemExit("provide --input, or --account-name/--container/--folder/--file")
    key = args.account_key or os.environ.get("AZURE_STORAGE_KEY")
    if not key:
        raise SystemExit("provide --account-key or set AZURE_STORAGE_KEY")
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(
        account_url=f"https://{args.account_name}.blob.core.windows.net",
        credential=key,
    )
    client = service.get_container_client(args.container)
    blob = client.get_blob_client(f"{args.folder.strip('/')}/{args.file}")
    return json.loads(blob.download_blob().readall())


def box_edges(corners):
    """12 edges of an (oriented) box from its 8 corners, ordering-independent.

    Classifies each corner by the sign of its projection onto the box's 3 PCA
    axes; two corners share an edge when their sign patterns differ in exactly
    one axis. Works for thin/long boxes and any corner order.
    """
    corners = np.asarray(corners, dtype=float)
    rel = corners - corners.mean(axis=0)
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    signs = np.sign(rel @ vt.T)
    signs[signs == 0] = 1
    edges = []
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if int(np.sum(signs[i] != signs[j])) == 1:
                edges.append((i, j))
    return edges


def _line_trace(go, segments, color, name, showlegend, legendgroup=None):
    xs, ys, zs = [], [], []
    for a, b in segments:
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(width=4, color=color), name=name,
        showlegend=showlegend, legendgroup=legendgroup,
    )


def box_trace(go, corners, color, name, edges=None, showlegend=False, legendgroup=None):
    pts = np.asarray([[c["x"], c["y"], c["z"]] for c in corners], dtype=float)
    if len(pts) < 4:
        return None
    # Use the source's known corner ordering for 8-corner boxes; otherwise fall
    # back to the orientation-derived edges (robust but can zigzag on near-cubes).
    use_edges = edges if (edges and len(pts) == 8) else box_edges(pts)
    segs = [(pts[i], pts[j]) for i, j in use_edges]
    return _line_trace(go, segs, color, name, showlegend, legendgroup=legendgroup)


def _ring_segments(ring):
    segs = [(ring[k], ring[k + 1]) for k in range(len(ring) - 1)]
    if not np.allclose(ring[0], ring[-1]):
        segs.append((ring[-1], ring[0]))  # close
    return segs


def prism_trace(go, edge_points, color, name, showlegend=False, legendgroup=None):
    """Floor/ceiling edgePoints are an extruded boundary interleaved as
    [c0_bottom, c0_top, c1_bottom, c1_top, ...]. Draw it as a prism: the bottom
    loop, the top loop, and the vertical edges joining them."""
    pts = np.asarray([[p["x"], p["y"], p["z"]] for p in edge_points], dtype=float)
    if len(pts) < 2:
        return None
    if len(pts) % 2 == 0:
        bottom, top = pts[0::2], pts[1::2]   # even = bottom level, odd = top level
        segs = _ring_segments(bottom) + _ring_segments(top)
        segs += [(b, t) for b, t in zip(bottom, top)]   # verticals
    else:
        segs = _ring_segments(pts)           # fallback: single closed loop
    return _line_trace(go, segs, color, name, showlegend, legendgroup=legendgroup)


def point_traces(go, data, max_per_class):
    """One scatter trace per class, subsampled for rendering."""
    by_class = {}
    for point in data.get("points", []):
        loc = point.get("location")
        if not loc:
            continue
        by_class.setdefault(point.get("category", "opening"), []).append(
            (loc["x"], loc["y"], loc["z"])
        )
    traces = []
    rng = np.random.default_rng(0)
    for category, pts in by_class.items():
        arr = np.asarray(pts, dtype=float)
        if max_per_class and len(arr) > max_per_class:
            arr = arr[rng.choice(len(arr), max_per_class, replace=False)]
        traces.append(
            go.Scatter3d(
                x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="markers",
                marker=dict(size=1.5, color=CLASS_COLORS.get(category, "#777777")),
                name=f"{category} pts ({len(pts)})",
            )
        )
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="local output JSON (elements/all_output/stage file)")
    ap.add_argument("--account-name")
    ap.add_argument("--container")
    ap.add_argument("--folder")
    ap.add_argument("--file", default="elements.json")
    ap.add_argument("--account-key")
    ap.add_argument("--output", default="output_visualization.html")
    ap.add_argument("--max-points-per-class", type=int, default=50_000)
    ap.add_argument("--no-points", action="store_true", help="boxes/outlines only")
    ap.add_argument("--show-openings", action="store_true",
                    help="append a gallery of per-wall opening images from "
                         "<input-dir>/opening/ (local) or <blob>/{folder}/opening/")
    ap.add_argument("--opening-dir",
                    help="local folder of opening images to use for the gallery "
                         "(takes precedence over auto-detected paths)")
    ap.add_argument("--download-openings", action="store_true",
                    help="download blob {folder}/opening/ into --opening-dir first")
    args = ap.parse_args()

    if args.download_openings and not args.opening_dir:
        raise SystemExit("--download-openings requires --opening-dir")

    import plotly.graph_objects as go

    data = load_json(args)
    traces = []

    if not args.no_points:
        traces += point_traces(go, data, args.max_points_per_class)

    # Floor / ceiling outlines (extruded prisms). Each class is its own
    # legendgroup so clicking the single visible legend entry toggles every
    # floor/ceiling outline at once (requires legend.groupclick="togglegroup").
    for key in ("floors", "ceilings"):
        color = CLASS_COLORS["floor" if key == "floors" else "ceiling"]
        group_name = f"{key} outlines"
        first = True
        for element in data.get(key, []):
            t = prism_trace(
                go, element.get("edgePoints", []), color,
                group_name if first else f"{key[:-1]} {element.get('id','')}",
                showlegend=first, legendgroup=key,
            )
            if t:
                traces.append(t); first = False

    # Bounding boxes, with each source's known corner ordering. Same
    # legendgroup trick so all doors / windows / openings toggle as a unit.
    box_groups = [("walls", "wall", WALL_BOX_EDGES)]
    box_groups += [(k, k[:-1], OPENING_BOX_EDGES) for k in ("doors", "windows", "openings")]
    for key, cls, edges in box_groups:
        color = CLASS_COLORS.get(cls, "#777777")
        group_name = f"{key} bboxes"
        first = True
        for element in data.get(key, []):
            t = box_trace(
                go, element.get("bbox", []), color,
                group_name if first else f"{cls} {element.get('id','')}",
                edges=edges, showlegend=first, legendgroup=key,
            )
            if t:
                traces.append(t); first = False

    if not traces:
        raise SystemExit("no drawable geometry found in the JSON")

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data", camera=dict(up=dict(x=0, y=0, z=1))),
        margin=dict(l=0, r=0, t=30, b=0),
        title=os.path.basename(args.input or args.file),
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
    )
    gallery_html = ""
    if args.show_openings:
        grouped = _load_opening_images(args)
        gallery_html = _opening_gallery_html(grouped)
        if not gallery_html:
            print("--show-openings set but no opening images were found")

    if gallery_html:
        # Inject the gallery right before </body> in the Plotly HTML.
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        anchor = "</body>"
        if anchor in html:
            html = html.replace(anchor, gallery_html + anchor, 1)
        else:
            html += gallery_html
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        n_walls = html.count('class="wall-card"')
        print(f"wrote {args.output}  ({len(traces)} traces, {n_walls} opening cards)")
    else:
        fig.write_html(args.output, include_plotlyjs="cdn", full_html=True)
        print(f"wrote {args.output}  ({len(traces)} traces)")


if __name__ == "__main__":
    main()
