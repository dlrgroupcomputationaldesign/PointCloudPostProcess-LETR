import json
import numpy as np
import copy

def load_points_and_objects(json_path, object_type):
    with open(json_path, "r") as f:
        data = json.load(f)

    pts = data["points"]
    object_data = data[object_type]

    # Points: Nx3 + Nx3 color
    xyz = np.array([[p["location"]["x"], p["location"]["y"], p["location"]["z"]] for p in pts], dtype=np.float64)

    rgb = np.array([[p["color"]["r"], p["color"]["g"], p["color"]["b"]] for p in pts], dtype=np.float64)
    rgb = np.clip(rgb / 255.0, 0.0, 1.0)

    return xyz, rgb, object_data

def make_point_cloud(xyz, rgb):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def build_lineset(points, color):
    import open3d as o3d

    lines = [[i, i + 1] for i in range(len(points) - 1)]

    # close loop if polygon
    if len(points) >= 3:
        lines.append([len(points) - 1, 0])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color), (len(lines), 1))
    )
    return ls
    
def make_edgepoints_linesets(object_data,
                                  lower_color=(1.0, 0.2, 0.2),
                                  upper_color=(0.2, 0.8, 0.2)):
    """
    Each floor.edgePoints is:
      [(x1,y1,z1),(x1,y1,z2),(x2,y2,z1),(x2,y2,z2),...]

    Returns two LineSets per floor:
      - lower polyline
      - upper polyline
    """
    geoms = []

    for obj in object_data:
        ep = obj.get("edgePoints", [])
        if len(ep) < 4 or len(ep) % 2 != 0:
            continue

        pts = np.array([[p["x"], p["y"], p["z"]] for p in ep], dtype=np.float64)

        # split into lower / upper
        lower = pts[0::2]   # even indices
        upper = pts[1::2]   # odd indices

        geoms.append(build_lineset(lower, lower_color))
        geoms.append(build_lineset(upper, upper_color))

    return geoms

def make_bbox_linesets(object_data):
    import open3d as o3d

    geoms = []

    for obj in object_data:
        ep = obj.get("bbox", [])
        if len(ep) != 8:
            continue

        pts = np.array([[p["x"], p["y"], p["z"]] for p in ep], dtype=np.float64)

        def build_bbox_lineset(bbox_points, color=(0, 1, 0)):
            """
            bbox_points: (8,3) array in the exact order you listed.
            """
            bbox_points = np.asarray(bbox_points, dtype=np.float64)

            lines = np.array([
                # bottom rectangle (minz)
                [0, 1], [1, 2], [2, 3], [3, 0],
                # top rectangle (maxz)
                [4, 5], [5, 6], [6, 7], [7, 4],
                # vertical edges
                [0, 4], [1, 5], [2, 6], [3, 7],
            ], dtype=np.int32)

            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(bbox_points)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
            )
            return ls

        geoms.append(build_bbox_lineset(pts, (1.0, 0.0, 0.0)))  # Red for walls

    return geoms

def build_level_indicator_square(center_xy, z, size, color=(0.2, 0.2, 0.2)):
    """
    Draw a thin square (LineSet) at height z, centered at (cx, cy).
    size is half-width in XY units.
    """
    cx, cy = center_xy
    pts = np.array([
        [cx - size, cy - size, z],
        [cx + size, cy - size, z],
        [cx + size, cy + size, z],
        [cx - size, cy + size, z],
    ], dtype=np.float64)

    return build_lineset(pts, color)

def visualize(json_path, object_type):
    import open3d as o3d

    xyz, rgb, object_data = load_points_and_objects(json_path, object_type)
    pcd = make_point_cloud(xyz, rgb)

    if object_type == "floors" or object_type =="ceilings":
        object_lines = make_edgepoints_linesets(object_data)
    elif object_type == "walls":
        object_lines = make_bbox_linesets(object_data)

    # Optional coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries(
        [pcd, frame, *object_lines],
        window_name=f"{object_type} Visualization",
        point_show_normal=False
    )

    pcd_grey = copy.deepcopy(pcd)
    pcd_grey.paint_uniform_color([0.97, 0.97, 0.97])
    o3d.visualization.draw_geometries([pcd_grey, *object_lines])

def visualize_all(json_path,
                  level_square_size=50):
    import open3d as o3d

    with open(json_path, "r") as f:
        data = json.load(f)

    points = data.get("points", [])
    levels = data.get("levels", [])
    floors = data.get("floors", [])
    ceilings = data.get("ceilings", [])
    walls = data.get("walls", [])

    # --- build point cloud colored by category ---
    xyz = np.array([[p["location"]["x"], p["location"]["y"], p["location"]["z"]] for p in points], dtype=np.float64)
    cats = [p.get("category", "unknown") for p in points]

    # choose colors per category (RGB 0..1)
    cat_color = {
        "floor":   (0.98, 0.95, 0.75),  # red-ish
        "ceiling": (0.70, 0.95, 0.70),  # green-ish
        "wall":    (0.75, 0.85, 0.98),  # blue-ish
        "unknown": (0.85, 0.85, 0.85),  # grey
    }

    rgb = np.array([cat_color.get(c, cat_color["unknown"]) for c in cats], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # --- compute overall XY center for level indicators ---
    if len(xyz) > 0:
        cx, cy = float(np.mean(xyz[:, 0])), float(np.mean(xyz[:, 1]))
    else:
        cx, cy = 0.0, 0.0

    # --- levels: draw zMode indicators ---
    level_geoms = []
    for lv in levels:
        z = lv.get("zMode", None)
        if z is None:
            continue
        ls = build_level_indicator_square((cx, cy), float(z), size=level_square_size, color=(0.1, 0.1, 0.1))
        if ls is not None:
            level_geoms.append(ls)

    # floors: lower line red, upper line orange
    floor_geoms = make_edgepoints_linesets(floors,
                                    lower_color=(0.0, 0.0, 0.0),
                                    upper_color=(0.0, 0.0, 0.0))
    # ceilings: lower line green, upper line teal
    ceiling_geoms = make_edgepoints_linesets(ceilings,
                                    lower_color=(0.0, 0.0, 0.0),
                                    upper_color=(0.0, 0.0, 0.0))

    wall_geoms = make_bbox_linesets(walls)

    o3d.visualization.draw_geometries([pcd, *floor_geoms],
                                      window_name="floor Visualization")
    o3d.visualization.draw_geometries([pcd, *ceiling_geoms],
                                      window_name="ceiling Visualization")
    o3d.visualization.draw_geometries([pcd, *wall_geoms],
                                      window_name="wall Visualization")
    o3d.visualization.draw_geometries([pcd, *level_geoms],
                                      window_name="level Visualization")


WALL_BBOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)

OPENING_BBOX_EDGES = np.array([
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)


def _bbox_points(obj):
    return np.asarray([[p["x"], p["y"], p["z"]] for p in obj.get("bbox", [])], dtype=np.float64)


def _aabb_object(minb, maxb, object_id="crop"):
    minb = np.asarray(minb, dtype=np.float64)
    maxb = np.asarray(maxb, dtype=np.float64)
    x0, y0, z0 = minb
    x1, y1, z1 = maxb
    return {
        "id": str(object_id),
        "bbox": [
            {"x": x0, "y": y0, "z": z0},
            {"x": x1, "y": y0, "z": z0},
            {"x": x1, "y": y1, "z": z0},
            {"x": x0, "y": y1, "z": z0},
            {"x": x0, "y": y0, "z": z1},
            {"x": x1, "y": y0, "z": z1},
            {"x": x1, "y": y1, "z": z1},
            {"x": x0, "y": y1, "z": z1},
        ],
    }


def _bbox_trace(objects, edges, name, color, line_width=4):
    import plotly.graph_objects as go

    xs, ys, zs = [], [], []
    for obj in objects:
        pts = _bbox_points(obj)
        if pts.shape != (8, 3):
            continue
        for a, b in edges:
            xs.extend([pts[a, 0], pts[b, 0], None])
            ys.extend([pts[a, 1], pts[b, 1], None])
            zs.extend([pts[a, 2], pts[b, 2], None])

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=line_width),
        name=f"{name} ({len(objects)})",
    )


def _opening_center_trace(objects, name, color):
    import plotly.graph_objects as go

    centers = []
    hover = []
    for obj in objects:
        pts = _bbox_points(obj)
        if pts.shape != (8, 3):
            continue
        center = pts.mean(axis=0)
        centers.append(center)
        hover.append(
            f"{name[:-1]} {obj.get('id')}<br>"
            f"wall {obj.get('wallId')}<br>"
            f"w {obj.get('width', 0):.2f}, h {obj.get('height', 0):.2f}<br>"
            f"z {obj.get('bottomZ', 0):.2f} - {obj.get('topZ', 0):.2f}<br>"
            f"conf {obj.get('confidence', 0):.2f}"
        )

    if not centers:
        centers = np.empty((0, 3))
    else:
        centers = np.asarray(centers, dtype=np.float64)

    return go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="markers",
        marker=dict(size=4, color=color, symbol="diamond"),
        name=f"{name} centers",
        text=hover,
        hoverinfo="text",
    )


def _sample_points(points, max_points):
    if max_points is None or len(points) <= max_points:
        return points

    idx = np.linspace(0, len(points) - 1, int(max_points), dtype=np.int64)
    return [points[i] for i in idx]


def export_walls_openings_html(
    json_path,
    html_path="walls_windows_visualization.html",
    max_points=120000,
):
    import plotly.graph_objects as go

    with open(json_path, "r") as f:
        data = json.load(f)

    points = _sample_points(data.get("points", []), max_points)
    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])
    openings = data.get("openings", [])

    xyz = np.asarray(
        [[p["location"]["x"], p["location"]["y"], p["location"]["z"]] for p in points],
        dtype=np.float64,
    )
    cats = [p.get("category", "unknown") for p in points]
    cat_color = {
        "floor": "rgb(190,190,190)",
        "ceiling": "rgb(170,210,170)",
        "wall": "rgb(135,165,215)",
        "unknown": "rgb(180,180,180)",
    }
    point_colors = [cat_color.get(cat, cat_color["unknown"]) for cat in cats]

    fig = go.Figure()
    if xyz.size:
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=1.2, color=point_colors, opacity=0.35),
                name=f"points ({len(points)})",
                hoverinfo="skip",
            )
        )

    fig.add_trace(_bbox_trace(walls, WALL_BBOX_EDGES, "walls", "rgb(220,40,40)", 3))
    fig.add_trace(_bbox_trace(windows, OPENING_BBOX_EDGES, "windows", "rgb(255,165,0)", 6))
    fig.add_trace(_opening_center_trace(windows, "windows", "rgb(255,165,0)"))

    if doors:
        fig.add_trace(_bbox_trace(doors, OPENING_BBOX_EDGES, "doors", "rgb(0,180,80)", 6))
        fig.add_trace(_opening_center_trace(doors, "doors", "rgb(0,180,80)"))

    if openings:
        fig.add_trace(_bbox_trace(openings, OPENING_BBOX_EDGES, "openings", "rgb(160,80,220)", 6))
        fig.add_trace(_opening_center_trace(openings, "openings", "rgb(160,80,220)"))

    fig.update_layout(
        title=(
            f"Walls and Openings: {len(walls)} walls, "
            f"{len(windows)} windows, {len(doors)} doors, {len(openings)} openings"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def _sample_dataframe(df, max_points):
    if max_points is None or len(df) <= max_points:
        return df

    idx = np.linspace(0, len(df) - 1, int(max_points), dtype=np.int64)
    return df.iloc[idx].copy()


def export_walls_openings_with_original_csv_html(
    csv_path,
    json_path,
    html_path="walls_windows_original_cloud.html",
    max_points=160000,
):
    import pandas as pd
    import plotly.graph_objects as go

    df = pd.read_csv(csv_path)
    df = _sample_dataframe(df, max_points)
    with open(json_path, "r") as f:
        data = json.load(f)

    walls = data.get("walls", [])
    doors = data.get("doors", [])
    windows = data.get("windows", [])
    openings = data.get("openings", [])

    rgb = df[["r", "g", "b"]].fillna(160).clip(0, 255).astype(int).to_numpy()
    rgb_colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]

    label_colors = {
        0: "rgb(170,170,170)",  # Other
        1: "rgb(210,190,80)",   # Floor
        2: "rgb(80,180,100)",   # Ceiling
        3: "rgb(80,130,220)",   # Wall
    }
    pred_labels = df["pred_label"].fillna(-1).astype(int).to_numpy()
    semantic_colors = [label_colors.get(label, "rgb(120,120,120)") for label in pred_labels]

    hover = [
        f"label {label}<br>rgb {r},{g},{b}"
        for label, (r, g, b) in zip(pred_labels, rgb)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(size=1.2, color=rgb_colors, opacity=0.45),
            name=f"original RGB points ({len(df)})",
            text=hover,
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(size=1.2, color=semantic_colors, opacity=0.4),
            name="prediction labels",
            visible="legendonly",
            hoverinfo="skip",
        )
    )

    fig.add_trace(_bbox_trace(walls, WALL_BBOX_EDGES, "walls", "rgb(220,40,40)", 3))
    fig.add_trace(_bbox_trace(windows, OPENING_BBOX_EDGES, "windows", "rgb(255,165,0)", 6))
    fig.add_trace(_opening_center_trace(windows, "windows", "rgb(255,165,0)"))

    if doors:
        fig.add_trace(_bbox_trace(doors, OPENING_BBOX_EDGES, "doors", "rgb(0,180,80)", 6))
        fig.add_trace(_opening_center_trace(doors, "doors", "rgb(0,180,80)"))

    if openings:
        fig.add_trace(_bbox_trace(openings, OPENING_BBOX_EDGES, "openings", "rgb(160,80,220)", 6))
        fig.add_trace(_opening_center_trace(openings, "openings", "rgb(160,80,220)"))

    fig.update_layout(
        title=(
            f"Original Point Cloud + Walls/Openings: {len(walls)} walls, "
            f"{len(windows)} windows, {len(doors)} doors, {len(openings)} openings"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def _objects_for_wall(objects, wall_id):
    wall_id = str(wall_id)
    return [obj for obj in objects if str(obj.get("wallId")) == wall_id]


def _bbox_extent(objects):
    arrays = [_bbox_points(obj) for obj in objects if _bbox_points(obj).shape == (8, 3)]
    if not arrays:
        return None
    pts = np.vstack(arrays)
    return pts.min(axis=0), pts.max(axis=0)


def export_wall_detail_with_original_csv_html(
    csv_path,
    json_path,
    wall_id,
    html_path=None,
    margin=2.0,
):
    import pandas as pd
    import plotly.graph_objects as go

    wall_id = str(wall_id)
    if html_path is None:
        html_path = f"wall_{wall_id}_original_cloud_windows.html"

    with open(json_path, "r") as f:
        data = json.load(f)

    walls = [wall for wall in data.get("walls", []) if str(wall.get("id")) == wall_id]
    if not walls:
        raise ValueError(f"wall id {wall_id!r} was not found")

    windows = _objects_for_wall(data.get("windows", []), wall_id)
    doors = _objects_for_wall(data.get("doors", []), wall_id)
    openings = _objects_for_wall(data.get("openings", []), wall_id)

    extent = _bbox_extent(walls + windows + doors + openings)
    if extent is None:
        raise ValueError(f"wall id {wall_id!r} has no valid bbox geometry")
    minb, maxb = extent
    minb = minb - float(margin)
    maxb = maxb + float(margin)

    df = pd.read_csv(csv_path)
    mask = (
        (df["x"] >= minb[0])
        & (df["x"] <= maxb[0])
        & (df["y"] >= minb[1])
        & (df["y"] <= maxb[1])
        & (df["z"] >= minb[2])
        & (df["z"] <= maxb[2])
    )
    df = df[mask].copy()

    rgb = df[["r", "g", "b"]].fillna(160).clip(0, 255).astype(int).to_numpy()
    rgb_colors = [f"rgb({r},{g},{b})" for r, g, b in rgb]
    pred_labels = df["pred_label"].fillna(-1).astype(int).to_numpy()
    hover = [
        f"label {label}<br>rgb {r},{g},{b}"
        for label, (r, g, b) in zip(pred_labels, rgb)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(size=1.8, color=rgb_colors, opacity=0.65),
            name=f"original points near wall {wall_id} ({len(df)})",
            text=hover,
            hoverinfo="text",
        )
    )
    fig.add_trace(_bbox_trace(walls, WALL_BBOX_EDGES, "wall", "rgb(220,40,40)", 5))
    fig.add_trace(_bbox_trace(windows, OPENING_BBOX_EDGES, "windows", "rgb(255,165,0)", 8))
    fig.add_trace(_opening_center_trace(windows, "windows", "rgb(255,165,0)"))

    if doors:
        fig.add_trace(_bbox_trace(doors, OPENING_BBOX_EDGES, "doors", "rgb(0,180,80)", 8))
        fig.add_trace(_opening_center_trace(doors, "doors", "rgb(0,180,80)"))

    if openings:
        fig.add_trace(_bbox_trace(openings, OPENING_BBOX_EDGES, "openings", "rgb(160,80,220)", 8))
        fig.add_trace(_opening_center_trace(openings, "openings", "rgb(160,80,220)"))

    fig.update_layout(
        title=(
            f"Wall {wall_id}: {len(windows)} windows, "
            f"{len(doors)} doors, {len(openings)} openings"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def export_wall_detail_with_e57_dense_html(
    e57_path,
    json_path,
    wall_id,
    html_path=None,
    margin=0.5,
    chunk_size=2_000_000,
    include_crop_points=True,
    max_crop_points=250000,
    crop_points_visible="legendonly",
    e57_to_csv_offset=None,
    e57_to_csv_scale=3.280839895013123,
    dense_openings_path=None,
):
    import plotly.graph_objects as go

    from post_process_src.post_process import OpeningConfig, PostProcessConfig
    from post_process_src.post_process.stages.openings import (
        _filter_chunk_to_wall_crop,
        _iter_e57_csv_chunks,
        _project_points,
        _wall_crop_geometry,
        _wall_frame,
    )

    wall_id = str(wall_id)
    if html_path is None:
        html_path = f"wall_{wall_id}_e57_dense_windows.html"

    with open(json_path, "r") as f:
        data = json.load(f)

    wall = next((wall for wall in data.get("walls", []) if str(wall.get("id")) == wall_id), None)
    if wall is None:
        raise ValueError(f"wall id {wall_id!r} was not found")

    sparse_windows = _objects_for_wall(data.get("windows", []), wall_id)
    dense_json = dense_openings_path or html_path.replace(".html", "_openings.json")
    dense_windows = []
    try:
        with open(dense_json, "r") as f:
            dense_data = json.load(f)
            dense_windows = dense_data.get("windows", [])
    except FileNotFoundError:
        pass

    params = PostProcessConfig(
        openings=OpeningConfig(
            dense_source_path=e57_path,
            dense_wall_ids=[wall_id],
            dense_wall_margin=margin,
            e57_chunk_size=chunk_size,
            e57_to_csv_offset=e57_to_csv_offset,
            e57_to_csv_scale=e57_to_csv_scale,
        )
    ).to_parameters()
    frame = _wall_frame(wall)
    crop_geometry = _wall_crop_geometry(wall, margin)
    minb = crop_geometry["minb"]
    maxb = crop_geometry["maxb"]
    points = []
    crop_points = []
    crop_count = 0

    for chunk in _iter_e57_csv_chunks(e57_path, params):
        crop = _filter_chunk_to_wall_crop(chunk, crop_geometry)
        if not len(crop):
            continue

        crop_count += len(crop)
        if include_crop_points and max_crop_points != 0:
            remaining = None if max_crop_points is None else max_crop_points - sum(len(p) for p in crop_points)
            if remaining is None or remaining > 0:
                if remaining is None or len(crop) <= remaining:
                    crop_points.append(crop)
                else:
                    idx = np.linspace(0, len(crop) - 1, int(remaining), dtype=np.int64)
                    crop_points.append(crop[idx])

        s, t, z = _project_points(crop, frame)
        tol = float(params["OPENING_WALL_DISTANCE_TOLERANCE"])
        wall_mask = (
            (s >= frame["s_min"])
            & (s <= frame["s_max"])
            & (t >= frame["t_min"] - tol)
            & (t <= frame["t_max"] + tol)
            & (z >= frame["z_min"])
            & (z <= frame["z_max"])
        )
        if wall_mask.any():
            points.append(crop[wall_mask])

    pts = np.vstack(points) if points else np.empty((0, 3))
    crop_pts = np.vstack(crop_points) if crop_points else np.empty((0, 3))

    fig = go.Figure()
    if include_crop_points and len(crop_pts):
        fig.add_trace(
            go.Scatter3d(
                x=crop_pts[:, 0],
                y=crop_pts[:, 1],
                z=crop_pts[:, 2],
                mode="markers",
                marker=dict(size=1.2, color="rgb(170,170,170)", opacity=0.18),
                name=(
                    f"E57 {crop_geometry['source']} crop, sampled "
                    f"({len(crop_pts)} of {crop_count})"
                ),
                visible=crop_points_visible,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=2.0, color="rgb(60,140,230)", opacity=0.75),
            name=f"E57 wall-plane points ({len(pts)})",
            hoverinfo="skip",
        )
    )
    fig.add_trace(_bbox_trace([_aabb_object(minb, maxb, "e57_crop_aabb")], WALL_BBOX_EDGES, f"E57 {crop_geometry['source']} crop bounds", "rgb(0,180,220)", 4))
    fig.add_trace(_bbox_trace([wall], WALL_BBOX_EDGES, f"wall {wall_id} bbox", "rgb(220,40,40)", 5))
    fig.add_trace(_bbox_trace(sparse_windows, OPENING_BBOX_EDGES, "sparse CSV windows", "rgb(255,165,0)", 7))
    fig.add_trace(_opening_center_trace(sparse_windows, "sparse CSV windows", "rgb(255,165,0)"))
    fig.add_trace(_bbox_trace(dense_windows, OPENING_BBOX_EDGES, "E57 windows", "rgb(0,220,120)", 8))

    fig.update_layout(
        title=(
            f"Wall {wall_id} E57 dense wall-plane points: {len(pts)} pts, "
            f"crop {crop_count} pts, sparse windows {len(sparse_windows)}, "
            f"E57 windows {len(dense_windows)}"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return html_path


def _transform_wall_to_e57_meters(wall, offset_m, ft_to_m=0.3048):
    offset_m = np.asarray(offset_m, dtype=np.float64)

    def transform_point(point):
        xyz_ft = np.array([point["x"], point["y"], point["z"]], dtype=np.float64)
        xyz_m = xyz_ft * ft_to_m + offset_m
        return {"x": float(xyz_m[0]), "y": float(xyz_m[1]), "z": float(xyz_m[2])}

    out = dict(wall)
    out["bbox"] = [transform_point(point) for point in wall.get("bbox", [])]

    if "footprint" in wall:
        out["footprint"] = []
        for point in wall["footprint"]:
            xy_ft = np.array([point["x"], point["y"]], dtype=np.float64)
            xy_m = xy_ft * ft_to_m + offset_m[:2]
            out["footprint"].append({"x": float(xy_m[0]), "y": float(xy_m[1])})

    if "zRange" in wall:
        out["zRange"] = {
            "min": float(wall["zRange"]["min"] * ft_to_m + offset_m[2]),
            "max": float(wall["zRange"]["max"] * ft_to_m + offset_m[2]),
        }

    return out


def _read_e57_header_bounds(e57_path):
    import pye57

    e57 = pye57.E57(e57_path)
    try:
        header = e57.get_header(0)
        return {
            "min": np.array([header.xMinimum, header.yMinimum, header.zMinimum], dtype=np.float64),
            "max": np.array([header.xMaximum, header.yMaximum, header.zMaximum], dtype=np.float64),
        }
    finally:
        e57.close()


def _iter_e57_raw_xyz_chunks(e57_path, chunk_size=2_000_000):
    import pye57

    e57 = pye57.E57(e57_path)
    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    try:
        for scan_index in range(e57.scan_count):
            header = e57.get_header(scan_index)
            data, buffers = e57.make_buffers(fields, int(chunk_size))
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


def _iter_e57_raw_xyzrgb_chunks(e57_path, chunk_size=2_000_000):
    import pye57

    e57 = pye57.E57(e57_path)
    xyz_fields = ["cartesianX", "cartesianY", "cartesianZ"]
    rgb_fields = ["colorRed", "colorGreen", "colorBlue"]
    try:
        for scan_index in range(e57.scan_count):
            header = e57.get_header(scan_index)
            available_fields = set(header.point_fields)
            has_rgb = all(field in available_fields for field in rgb_fields)
            fields = xyz_fields + (rgb_fields if has_rgb else [])
            data, buffers = e57.make_buffers(fields, int(chunk_size))
            reader = header.points.reader(buffers)
            while True:
                count = reader.read()
                if count <= 0:
                    break

                xyz = np.column_stack(
                    (
                        data["cartesianX"][:count],
                        data["cartesianY"][:count],
                        data["cartesianZ"][:count],
                    )
                ).astype(np.float64, copy=False)

                if has_rgb:
                    rgb = np.column_stack(
                        (
                            data["colorRed"][:count],
                            data["colorGreen"][:count],
                            data["colorBlue"][:count],
                        )
                    ).astype(np.float64, copy=False)
                else:
                    rgb = np.full((count, 3), 128.0, dtype=np.float64)

                yield xyz, rgb
    finally:
        e57.close()


def _filter_xyzrgb_chunk_to_wall_crop(xyz, rgb, crop_geometry):
    from post_process_src.post_process.stages.openings import Point, contains_xy

    minb = crop_geometry["minb"]
    maxb = crop_geometry["maxb"]
    mask = (
        (xyz[:, 0] >= minb[0])
        & (xyz[:, 0] <= maxb[0])
        & (xyz[:, 1] >= minb[1])
        & (xyz[:, 1] <= maxb[1])
        & (xyz[:, 2] >= minb[2])
        & (xyz[:, 2] <= maxb[2])
    )
    if not np.any(mask):
        return xyz[:0], rgb[:0]

    cropped_xyz = xyz[mask]
    cropped_rgb = rgb[mask]
    polygon = crop_geometry.get("polygon")
    if polygon is None:
        return cropped_xyz, cropped_rgb

    if contains_xy is not None:
        inside = contains_xy(polygon, cropped_xyz[:, 0], cropped_xyz[:, 1])
    else:
        inside = np.array(
            [polygon.contains(Point(x, y)) for x, y in cropped_xyz[:, :2]],
            dtype=bool,
        )

    return cropped_xyz[inside], cropped_rgb[inside]


def export_wall_from_pickle_mapped_to_e57_html(
    wall_output_pickle,
    e57_path,
    wall_id,
    output_html="wall_e57_points.html",
    ft_to_m=0.3048,
    offset_m=None,
    margin_m=0.5,
    chunk_size=2_000_000,
    max_crop_points=300000,
    show_source_wall_points=True,
):
    import pickle
    import plotly.graph_objects as go

    from post_process_src.post_process.stages.openings import (
        _filter_chunk_to_wall_crop,
        _wall_crop_geometry,
    )

    with open(wall_output_pickle, "rb") as f:
        wall_output_dict = pickle.load(f)

    wall_id = str(wall_id)
    wall = next((w for w in wall_output_dict["walls"] if str(w.get("id")) == wall_id), None)
    if wall is None:
        raise ValueError(f"wall id {wall_id!r} was not found")

    e57_bounds = _read_e57_header_bounds(e57_path)
    if offset_m is None:
        offset_m = e57_bounds["min"]
    else:
        offset_m = np.asarray(offset_m, dtype=np.float64)
    
    wall_e57 = _transform_wall_to_e57_meters(wall, offset_m, ft_to_m=ft_to_m)
    crop_geometry = _wall_crop_geometry(wall_e57, margin_m)

    crop_points = []
    total_crop_points = 0
    for chunk in _iter_e57_raw_xyz_chunks(e57_path, chunk_size=chunk_size):
        cropped = _filter_chunk_to_wall_crop(chunk, crop_geometry)
        if not len(cropped):
            continue

        total_crop_points += len(cropped)
        remaining = None if max_crop_points is None else max_crop_points - sum(len(p) for p in crop_points)
        if remaining is None or remaining > 0:
            if remaining is None or len(cropped) <= remaining:
                crop_points.append(cropped)
            else:
                idx = np.linspace(0, len(cropped) - 1, int(remaining), dtype=np.int64)
                crop_points.append(cropped[idx])

    e57_points = np.vstack(crop_points) if crop_points else np.empty((0, 3), dtype=np.float64)

    fig = go.Figure()
    if len(e57_points):
        fig.add_trace(
            go.Scatter3d(
                x=e57_points[:, 0],
                y=e57_points[:, 1],
                z=e57_points[:, 2],
                mode="markers",
                marker=dict(size=1.6, color="rgb(70,140,230)", opacity=0.75),
                name=f"E57 points in wall footprint ({len(e57_points)} of {total_crop_points})",
                hoverinfo="skip",
            )
        )

    if show_source_wall_points:
        source_points = []
        source_colors = []
        for point in wall_output_dict.get("points", []):
            if point.get("category") != "wall" or str(point.get("id")) != wall_id:
                continue
            loc = point["location"]
            color = point.get("color", {"r": 255, "g": 165, "b": 0})
            xyz_m = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64) * ft_to_m + offset_m
            source_points.append(xyz_m)
            source_colors.append(f"rgb({int(color['r'])},{int(color['g'])},{int(color['b'])})")

        if source_points:
            source_points = np.vstack(source_points)
            fig.add_trace(
                go.Scatter3d(
                    x=source_points[:, 0],
                    y=source_points[:, 1],
                    z=source_points[:, 2],
                    mode="markers",
                    marker=dict(size=3.0, color=source_colors, opacity=0.9),
                    name=f"wall {wall_id} CSV points mapped to E57",
                    hoverinfo="skip",
                )
            )

    fig.add_trace(_bbox_trace([wall_e57], WALL_BBOX_EDGES, f"wall {wall_id} bbox mapped to E57", "rgb(220,40,40)", 5))

    if wall_e57.get("footprint"):
        z_min = wall_e57["zRange"]["min"]
        z_max = wall_e57["zRange"]["max"]
        footprint_xy = np.array([[p["x"], p["y"]] for p in wall_e57["footprint"]], dtype=np.float64)
        for z, name, color in (
            (z_min, "footprint bottom mapped to E57", "rgb(255,165,0)"),
            (z_max, "footprint top mapped to E57", "rgb(255,220,80)"),
        ):
            footprint = np.column_stack([footprint_xy[:, 0], footprint_xy[:, 1], np.full(len(footprint_xy), z)])
            footprint = np.vstack([footprint, footprint[0]])
            fig.add_trace(
                go.Scatter3d(
                    x=footprint[:, 0],
                    y=footprint[:, 1],
                    z=footprint[:, 2],
                    mode="lines",
                    line=dict(color=color, width=5),
                    name=name,
                )
            )

    fig.update_layout(
        title=(
            f"Wall {wall_id} mapped to E57 meters: "
            f"{total_crop_points} E57 points in footprint crop"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    return fig


def _load_wall_output_pickle(wall_output_pickle):
    import pickle

    with open(wall_output_pickle, "rb") as f:
        return pickle.load(f)


def _get_wall_by_id_from_output(wall_output_dict, wall_id):
    wall_id = str(wall_id)
    wall = next((w for w in wall_output_dict["walls"] if str(w.get("id")) == wall_id), None)
    if wall is None:
        raise ValueError(f"wall id {wall_id!r} was not found")
    return wall


def _wall_length_projection_range(wall):
    if wall.get("footprint"):
        footprint_xy = np.array([[p["x"], p["y"]] for p in wall["footprint"]], dtype=np.float64)
    else:
        footprint_xy = np.array([[p["x"], p["y"]] for p in wall.get("bbox", [])], dtype=np.float64)

    if len(footprint_xy) < 2:
        raise ValueError("wall needs footprint or bbox points to estimate length axis")

    max_len = -1.0
    best_axis = None

    if wall.get("footprint") and len(footprint_xy) >= 3:
        for i in range(len(footprint_xy)):
            edge = footprint_xy[(i + 1) % len(footprint_xy)] - footprint_xy[i]
            length = np.linalg.norm(edge)
            if length > 0 and length > max_len:
                max_len = length
                best_axis = edge / length
    else:
        for i in range(len(footprint_xy)):
            for j in range(i + 1, len(footprint_xy)):
                edge = footprint_xy[j] - footprint_xy[i]
                length = np.linalg.norm(edge)
                if length > 0 and length > max_len:
                    max_len = length
                    best_axis = edge / length

    if best_axis is None or max_len <= 0:
        raise ValueError("could not estimate wall length axis")

    projected = footprint_xy @ best_axis
    return best_axis, float(projected.min()), float(projected.max())

import pickle
import copy
import numpy as np
import pye57
import plotly.graph_objects as go


M_TO_FT = 3.280839895


def _load_wall_output_pickle(wall_output_pickle):
    with open(wall_output_pickle, "rb") as f:
        return pickle.load(f)


def _get_wall_by_id_from_output(wall_output_dict, wall_id):
    wall_id = str(wall_id)

    for wall in wall_output_dict["walls"]:
        if str(wall["id"]) == wall_id:
            return wall

    raise ValueError(f"Wall id {wall_id} not found.")


def _read_e57_xyz_rgb(e57_path, scan_index=0):
    import numpy as np
    import pye57

    e57 = pye57.E57(e57_path)

    # Important: request colors=True
    scan_data = e57.read_scan(
        scan_index,
        colors=True,
        intensity=True,
        ignore_missing_fields=True,
    )

    print("Available E57 fields:")
    print(scan_data.keys())

    xyz = np.vstack(
        [
            scan_data["cartesianX"],
            scan_data["cartesianY"],
            scan_data["cartesianZ"],
        ]
    ).T.astype(np.float64)

    valid = np.isfinite(xyz).all(axis=1)

    has_rgb = (
        "colorRed" in scan_data
        and "colorGreen" in scan_data
        and "colorBlue" in scan_data
    )

    if has_rgb:
        rgb_raw = np.vstack(
            [
                scan_data["colorRed"],
                scan_data["colorGreen"],
                scan_data["colorBlue"],
            ]
        ).T.astype(np.float64)

        rgb_raw = rgb_raw[valid]

        print("RGB raw min:", rgb_raw.min(axis=0))
        print("RGB raw max:", rgb_raw.max(axis=0))
        print("RGB raw dtype:", rgb_raw.dtype)

        # Normalize color robustly
        rgb_max = float(np.nanmax(rgb_raw))

        if rgb_max <= 1.0:
            # RGB stored as 0-1 float
            rgb = rgb_raw * 255.0
        elif rgb_max <= 255.0:
            # RGB stored as 0-255
            rgb = rgb_raw
        else:
            # RGB stored as 16-bit, often 0-65535
            rgb = rgb_raw / 65535.0 * 255.0

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    else:
        print("No RGB fields found in E57 scan.")

        # Fallback: use intensity if available
        if "intensity" in scan_data:
            intensity = np.asarray(scan_data["intensity"], dtype=np.float64)
            intensity = intensity[valid]

            i_min = np.nanpercentile(intensity, 1)
            i_max = np.nanpercentile(intensity, 99)

            intensity_norm = (intensity - i_min) / (i_max - i_min + 1e-8)
            intensity_norm = np.clip(intensity_norm, 0, 1)

            gray = (intensity_norm * 255).astype(np.uint8)
            rgb = np.column_stack([gray, gray, gray])

            print("Using intensity as grayscale color.")
        else:
            xyz_valid_len = int(valid.sum())
            rgb = np.full((xyz_valid_len, 3), 180, dtype=np.uint8)

            print("No intensity field either. Using constant gray.")

    xyz = xyz[valid]

    return xyz, rgb


def _transform_wall_to_e57_meters(wall_ft, offset_m, ft_to_m=0.3048):
    """
    wall_output coordinates are assumed to be in feet.
    E57 coordinates are in meters.

    e57_xyz_m = wall_xyz_ft * ft_to_m + offset_m
    """

    offset_m = np.asarray(offset_m, dtype=np.float64)

    wall_m = copy.deepcopy(wall_ft)

    wall_m["footprint"] = [
        {
            "x": p["x"] * ft_to_m + offset_m[0],
            "y": p["y"] * ft_to_m + offset_m[1],
        }
        for p in wall_ft["footprint"]
    ]

    wall_m["bbox"] = [
        {
            "x": p["x"] * ft_to_m + offset_m[0],
            "y": p["y"] * ft_to_m + offset_m[1],
            "z": p["z"] * ft_to_m + offset_m[2],
        }
        for p in wall_ft["bbox"]
    ]

    wall_m["zRange"] = {
        "min": wall_ft["zRange"]["min"] * ft_to_m + offset_m[2],
        "max": wall_ft["zRange"]["max"] * ft_to_m + offset_m[2],
    }

    return wall_m


def _wall_crop_basis_from_footprint(wall_m):
    """
    Get wall-local length and normal axes from the transformed wall footprint.
    """

    footprint_xy = np.array(
        [[p["x"], p["y"]] for p in wall_m["footprint"]],
        dtype=np.float64,
    )

    center = footprint_xy.mean(axis=0)
    centered = footprint_xy - center

    # PCA direction = wall length direction
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)

    length_axis = eigvecs[:, np.argmax(eigvals)]
    length_axis = length_axis / np.linalg.norm(length_axis)

    normal_axis = np.array([-length_axis[1], length_axis[0]], dtype=np.float64)

    s_vals = footprint_xy @ length_axis
    n_vals = footprint_xy @ normal_axis

    return {
        "length_axis": length_axis,
        "normal_axis": normal_axis,
        "s_min": float(s_vals.min()),
        "s_max": float(s_vals.max()),
        "n_min": float(n_vals.min()),
        "n_max": float(n_vals.max()),
        "z_min": float(wall_m["zRange"]["min"]),
        "z_max": float(wall_m["zRange"]["max"]),
    }


def _crop_xyz_rgb_near_wall(xyz, rgb, wall_m, margin_m=0.25, z_margin_m=None):
    """
    Crop original E57 points near the transformed wall footprint.

    This crops in wall-local coordinates:
      s = along wall length
      n = wall thickness / normal direction
      z = height
    """

    if z_margin_m is None:
        z_margin_m = margin_m

    geom = _wall_crop_basis_from_footprint(wall_m)

    xy = xyz[:, :2]
    z = xyz[:, 2]

    s = xy @ geom["length_axis"]
    n = xy @ geom["normal_axis"]

    keep = (
        (s >= geom["s_min"] - margin_m)
        & (s <= geom["s_max"] + margin_m)
        & (n >= geom["n_min"] - margin_m)
        & (n <= geom["n_max"] + margin_m)
        & (z >= geom["z_min"] - z_margin_m)
        & (z <= geom["z_max"] + z_margin_m)
    )

    return xyz[keep], rgb[keep], geom, keep


def _rgb_to_plotly_strings(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    return [
        f"rgb({int(r)},{int(g)},{int(b)})"
        for r, g, b in rgb
    ]


def _downsample_points(points, rgb, max_points, seed=0):
    if max_points is None or max_points <= 0:
        return points, rgb

    if len(points) <= max_points:
        return points, rgb

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)

    return points[idx], rgb[idx]


def _wall_outline_traces(wall_m, display_unit="m"):
    scale = M_TO_FT if display_unit == "ft" else 1.0

    footprint_xy = np.array(
        [[p["x"], p["y"]] for p in wall_m["footprint"]],
        dtype=np.float64,
    )

    z_min = float(wall_m["zRange"]["min"])
    z_max = float(wall_m["zRange"]["max"])

    bottom = np.column_stack(
        [
            footprint_xy[:, 0],
            footprint_xy[:, 1],
            np.full(len(footprint_xy), z_min),
        ]
    )

    top = np.column_stack(
        [
            footprint_xy[:, 0],
            footprint_xy[:, 1],
            np.full(len(footprint_xy), z_max),
        ]
    )

    bottom = bottom * scale
    top = top * scale

    bottom_closed = np.vstack([bottom, bottom[0]])
    top_closed = np.vstack([top, top[0]])

    traces = []

    traces.append(
        go.Scatter3d(
            x=bottom_closed[:, 0],
            y=bottom_closed[:, 1],
            z=bottom_closed[:, 2],
            mode="lines+markers",
            line=dict(width=6, color="red"),
            marker=dict(size=4, color="red"),
            name="wall footprint z_min",
        )
    )

    traces.append(
        go.Scatter3d(
            x=top_closed[:, 0],
            y=top_closed[:, 1],
            z=top_closed[:, 2],
            mode="lines+markers",
            line=dict(width=6, color="orange"),
            marker=dict(size=4, color="orange"),
            name="wall footprint z_max",
        )
    )

    for i in range(len(bottom)):
        traces.append(
            go.Scatter3d(
                x=[bottom[i, 0], top[i, 0]],
                y=[bottom[i, 1], top[i, 1]],
                z=[bottom[i, 2], top[i, 2]],
                mode="lines",
                line=dict(width=4, color="red"),
                showlegend=False,
            )
        )

    return traces


def export_e57_wall_crop_rgb_html(
    wall_output_pickle,
    e57_path,
    output_html="wall_17_e57_crop_rgb.html",
    wall_id=17,
    scan_index=0,
    ft_to_m=0.3048,
    offset_m=None,
    margin_m=0.25,
    z_margin_m=None,
    max_html_points=500_000,
    marker_size=1.5,
    display_unit="m",
    show_wall_outline=True,
):
    """
    Visualize original E57 RGB points cropped near one wall footprint.

    Pipeline:
        wall_output.pickle wall 17
            -> map feet to E57 meters
        original LaramieCM.e57 points
            -> crop near that mapped wall footprint
        cropped RGB points
            -> Plotly HTML

    display_unit:
        "m"  = show original E57 meter coordinates
        "ft" = show converted display coordinates in feet
    """

    # -------------------------
    # Load wall output
    # -------------------------
    wall_output_dict = _load_wall_output_pickle(wall_output_pickle)
    wall_ft = _get_wall_by_id_from_output(wall_output_dict, wall_id)

    # -------------------------
    # Load original E57 points + RGB
    # -------------------------
    xyz_m, rgb = _read_e57_xyz_rgb(e57_path, scan_index=scan_index)

    print(f"Loaded E57 points: {len(xyz_m):,}")

    # -------------------------
    # Determine feet -> E57 meter offset
    # -------------------------
    if offset_m is None:
        # Same idea as e57_header_min fallback.
        # If you already have _read_e57_header_bounds(), you can use that instead.
        offset_m = xyz_m.min(axis=0)
    else:
        offset_m = np.asarray(offset_m, dtype=np.float64)

    print(f"Using offset_m: {offset_m}")

    # -------------------------
    # Transform wall from package feet to E57 meters
    # -------------------------
    wall_m = _transform_wall_to_e57_meters(
        wall_ft,
        offset_m=offset_m,
        ft_to_m=ft_to_m,
    )

    # -------------------------
    # Crop original E57 points near wall footprint
    # -------------------------
    cropped_xyz_m, cropped_rgb, crop_geom, keep_mask = _crop_xyz_rgb_near_wall(
        xyz_m,
        rgb,
        wall_m,
        margin_m=margin_m,
        z_margin_m=z_margin_m,
    )

    print(f"Cropped points near wall {wall_id}: {len(cropped_xyz_m):,}")

    if len(cropped_xyz_m) == 0:
        raise ValueError(
            "No E57 points found near this wall. "
            "Try increasing margin_m, checking offset_m, or checking ft_to_m."
        )

    # -------------------------
    # Downsample for browser performance
    # -------------------------
    html_xyz_m, html_rgb = _downsample_points(
        cropped_xyz_m,
        cropped_rgb,
        max_points=max_html_points,
        seed=0,
    )

    print(f"Points written to HTML: {len(html_xyz_m):,}")

    # -------------------------
    # Display coordinates
    # -------------------------
    if display_unit == "ft":
        html_xyz = html_xyz_m * M_TO_FT
        unit_name = "feet"
    else:
        html_xyz = html_xyz_m
        unit_name = "meters"

    rgb_strings = _rgb_to_plotly_strings(html_rgb)

    # -------------------------
    # Plotly HTML
    # -------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=html_xyz[:, 0],
            y=html_xyz[:, 1],
            z=html_xyz[:, 2],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=rgb_strings,
                opacity=1.0,
            ),
            name=f"E57 cropped RGB points near wall {wall_id}",
        )
    )

    if show_wall_outline:
        for tr in _wall_outline_traces(wall_m, display_unit=display_unit):
            fig.add_trace(tr)

    fig.update_layout(
        title=f"Original E57 RGB crop near wall {wall_id}",
        scene=dict(
            xaxis_title=f"X ({unit_name})",
            yaxis_title=f"Y ({unit_name})",
            zaxis_title=f"Z ({unit_name})",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")

    print(f"Saved HTML to: {output_html}")

    return {
        "fig": fig,
        "cropped_xyz_m": cropped_xyz_m,
        "cropped_rgb": cropped_rgb,
        "wall_m": wall_m,
        "crop_geom": crop_geom,
        "offset_m": offset_m,
        "html_path": output_html,
    }

def create_e57_wall_length_z_image(
    wall_output_pickle,
    e57_path,
    wall_id=17,
    ft_to_m=0.3048,
    offset_m=None,
    margin_m=0.25,
    length_bin_size=0.05,
    z_bin_size=0.05,
    count_threshold=1,
    z_crop=None,
    chunk_size=2_000_000,
    max_sample_points=0,
):
    """
    Stream raw E57 points for one mapped wall and rasterize them on a 2D wall plane.

    The image coordinate system matches the notebook prototype:
      - x axis = distance along wall length
      - y axis = E57 z height
      - pixel value = point count

    If offset_m is None, the wall geometry is mapped from package feet to raw E57
    meters with: e57_xyz = wall_xyz_ft * ft_to_m + e57_header_min.
    """
    from post_process_src.post_process.stages.openings import (
        _filter_chunk_to_wall_crop,
        _wall_crop_geometry,
    )

    wall_output_dict = _load_wall_output_pickle(wall_output_pickle)
    wall = _get_wall_by_id_from_output(wall_output_dict, wall_id)

    e57_bounds = _read_e57_header_bounds(e57_path)
    if offset_m is None:
        offset_m = e57_bounds["min"]
    else:
        offset_m = np.asarray(offset_m, dtype=np.float64)

    wall_e57 = _transform_wall_to_e57_meters(wall, offset_m, ft_to_m=ft_to_m)
    crop_geometry = _wall_crop_geometry(wall_e57, margin_m)

    length_axis, s_min, s_max = _wall_length_projection_range(wall_e57)
    wall_length = s_max - s_min
    print(f"Wall {wall_id} length axis: {length_axis}, s range: {s_min:.2f} to {s_max:.2f}, length: {wall_length:.2f} m")
    if wall_length <= 0:
        raise ValueError("wall length is zero after projection")

    if z_crop is None:
        z_min = float(wall_e57["zRange"]["min"])
        z_max = float(wall_e57["zRange"]["max"])
    else:
        z_min = float(z_crop[0])
        z_max = float(z_crop[1])

    if z_max <= z_min:
        raise ValueError("z range must be positive")

    n_length = max(5, int(np.ceil(wall_length / length_bin_size)))
    n_z = max(5, int(np.ceil((z_max - z_min) / z_bin_size)))

    count_img = np.zeros((n_z, n_length), dtype=np.uint32)
    points_in_crop = 0
    points_used = 0
    sample_points = []
    sampled_points = 0

    for chunk in _iter_e57_raw_xyz_chunks(e57_path, chunk_size=chunk_size):
        cropped = _filter_chunk_to_wall_crop(chunk, crop_geometry)
        if not len(cropped):
            continue

        points_in_crop += len(cropped)
        s = cropped[:, :2] @ length_axis - s_min
        z = cropped[:, 2]

        keep = (s >= 0.0) & (s <= wall_length) & (z >= z_min) & (z <= z_max)
        if not np.any(keep):
            continue

        cropped = cropped[keep]
        s = s[keep]
        z = z[keep]
        points_used += len(s)

        if max_sample_points is not None and max_sample_points > 0:
            remaining = int(max_sample_points) - sampled_points
            if remaining > 0:
                if len(cropped) <= remaining:
                    selected = cropped
                else:
                    idx = np.linspace(0, len(cropped) - 1, remaining, dtype=np.int64)
                    selected = cropped[idx]
                sample_points.append(selected)
                sampled_points += len(selected)

        x_idx = np.floor(s / wall_length * n_length).astype(np.int64)
        y_idx = np.floor((z - z_min) / (z_max - z_min) * n_z).astype(np.int64)
        x_idx = np.clip(x_idx, 0, n_length - 1)
        y_idx = np.clip(y_idx, 0, n_z - 1)

        np.add.at(count_img, (y_idx, x_idx), 1)

    binary_img = count_img >= count_threshold
    s_edges = np.linspace(0.0, wall_length, n_length + 1)
    z_edges = np.linspace(z_min, z_max, n_z + 1)

    return {
        "count_img": count_img,
        "binary_img": binary_img,
        "s_edges": s_edges,
        "z_edges": z_edges,
        "wall_length": wall_length,
        "z_min": z_min,
        "z_max": z_max,
        "length_axis": length_axis,
        "wall_e57": wall_e57,
        "offset_m": offset_m,
        "points_in_crop": points_in_crop,
        "points_used": points_used,
        "count_threshold": count_threshold,
        "length_bin_size": length_bin_size,
        "z_bin_size": z_bin_size,
        "sample_points": (
            np.vstack(sample_points)
            if sample_points
            else np.empty((0, 3), dtype=np.float64)
        ),
    }


def create_e57_wall_length_z_rgb_image(
    wall_output_pickle,
    e57_path,
    wall_id=17,
    ft_to_m=0.3048,
    offset_m=None,
    margin_m=0.25,
    length_bin_size=0.05,
    z_bin_size=0.05,
    count_threshold=10,
    z_crop=None,
    chunk_size=2_000_000,
    rgb_color_max=255.0,
    empty_rgb=(1.0, 1.0, 1.0),
    max_sample_points=0,
):
    """
    Stream raw E57 XYZ+RGB for one mapped wall and rasterize it in length-z space.

    Returns the same count/binary images as create_e57_wall_length_z_image, plus
    avg_rgb_img: an RGB image where each cell is the average E57 color in that cell.
    """
    from post_process_src.post_process.stages.openings import _wall_crop_geometry

    wall_output_dict = _load_wall_output_pickle(wall_output_pickle)
    wall = _get_wall_by_id_from_output(wall_output_dict, wall_id)

    e57_bounds = _read_e57_header_bounds(e57_path)
    if offset_m is None:
        offset_m = e57_bounds["min"]
    else:
        offset_m = np.asarray(offset_m, dtype=np.float64)

    wall_e57 = _transform_wall_to_e57_meters(wall, offset_m, ft_to_m=ft_to_m)
    crop_geometry = _wall_crop_geometry(wall_e57, margin_m)

    length_axis, s_min, s_max = _wall_length_projection_range(wall_e57)
    wall_length = s_max - s_min
    if wall_length <= 0:
        raise ValueError("wall length is zero after projection")

    if z_crop is None:
        z_min = float(wall_e57["zRange"]["min"])
        z_max = float(wall_e57["zRange"]["max"])
    else:
        z_min = float(z_crop[0])
        z_max = float(z_crop[1])

    if z_max <= z_min:
        raise ValueError("z range must be positive")

    n_length = max(5, int(np.ceil(wall_length / length_bin_size)))
    n_z = max(5, int(np.ceil((z_max - z_min) / z_bin_size)))

    count_img = np.zeros((n_z, n_length), dtype=np.uint32)
    rgb_sum_img = np.zeros((n_z, n_length, 3), dtype=np.float64)
    points_in_crop = 0
    points_used = 0
    sample_s = []
    sample_z = []
    sample_rgb = []
    sampled_points = 0

    for xyz, rgb in _iter_e57_raw_xyzrgb_chunks(e57_path, chunk_size=chunk_size):
        cropped_xyz, cropped_rgb = _filter_xyzrgb_chunk_to_wall_crop(xyz, rgb, crop_geometry)
        if not len(cropped_xyz):
            continue

        points_in_crop += len(cropped_xyz)
        s = cropped_xyz[:, :2] @ length_axis - s_min
        z = cropped_xyz[:, 2]

        keep = (s >= 0.0) & (s <= wall_length) & (z >= z_min) & (z <= z_max)
        if not np.any(keep):
            continue

        s = s[keep]
        z = z[keep]
        rgb = cropped_rgb[keep]
        points_used += len(s)

        x_idx = np.floor(s / wall_length * n_length).astype(np.int64)
        y_idx = np.floor((z - z_min) / (z_max - z_min) * n_z).astype(np.int64)
        x_idx = np.clip(x_idx, 0, n_length - 1)
        y_idx = np.clip(y_idx, 0, n_z - 1)

        np.add.at(count_img, (y_idx, x_idx), 1)
        for channel in range(3):
            np.add.at(rgb_sum_img[:, :, channel], (y_idx, x_idx), rgb[:, channel])

        if max_sample_points is not None and max_sample_points > 0:
            remaining = int(max_sample_points) - sampled_points
            if remaining > 0:
                if len(s) <= remaining:
                    idx = slice(None)
                else:
                    idx = np.linspace(0, len(s) - 1, remaining, dtype=np.int64)
                sample_s.append(s[idx])
                sample_z.append(z[idx])
                sample_rgb.append(np.clip(rgb[idx] / rgb_color_max, 0.0, 1.0))
                sampled_points += len(s[idx])

    valid = count_img > 0
    avg_rgb_img = np.empty((n_z, n_length, 3), dtype=np.float64)
    avg_rgb_img[:, :, :] = np.asarray(empty_rgb, dtype=np.float64)
    avg_rgb_img[valid] = np.clip(
        (rgb_sum_img[valid] / count_img[valid, None]) / rgb_color_max,
        0.0,
        1.0,
    )

    binary_img = count_img >= count_threshold
    s_edges = np.linspace(0.0, wall_length, n_length + 1)
    z_edges = np.linspace(z_min, z_max, n_z + 1)

    return {
        "count_img": count_img,
        "binary_img": binary_img,
        "avg_rgb_img": avg_rgb_img,
        "s_edges": s_edges,
        "z_edges": z_edges,
        "wall_length": wall_length,
        "z_min": z_min,
        "z_max": z_max,
        "length_axis": length_axis,
        "wall_e57": wall_e57,
        "offset_m": offset_m,
        "points_in_crop": points_in_crop,
        "points_used": points_used,
        "count_threshold": count_threshold,
        "length_bin_size": length_bin_size,
        "z_bin_size": z_bin_size,
        "sample_s": np.concatenate(sample_s) if sample_s else np.empty(0, dtype=np.float64),
        "sample_z": np.concatenate(sample_z) if sample_z else np.empty(0, dtype=np.float64),
        "sample_rgb": (
            np.vstack(sample_rgb)
            if sample_rgb
            else np.empty((0, 3), dtype=np.float64)
        ),
    }


def plot_e57_wall_length_z_image(
    result,
    wall_id=17,
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    count_img = result["count_img"]
    binary_img = result["binary_img"]
    s_edges = result["s_edges"]
    z_edges = result["z_edges"]
    wall_length = s_edges[-1]
    z_min = z_edges[0]
    z_max = z_edges[-1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        count_img,
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
        cmap="gray_r",
    )
    axes[0].set_title("Point count per cell")
    plt.colorbar(im0, ax=axes[0], label="point count")

    im1 = axes[1].imshow(
        np.log1p(count_img),
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
        cmap="gray_r",
    )
    axes[1].set_title("Log point count")
    plt.colorbar(im1, ax=axes[1], label="log(1 + count)")

    axes[2].imshow(
        binary_img.astype(float),
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Binary image, threshold >= {result['count_threshold']}"
    )

    for ax in axes:
        ax.set_xlabel("Wall length direction (m)")
        ax.set_ylabel("Z height (m)")

    fig.suptitle(
        f"Wall {wall_id}: E57 length-z image, "
        f"{result['points_used']} points used from {result['points_in_crop']} crop points"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_e57_wall_length_z_rgb_image(
    result,
    wall_id=17,
    save_path=None,
    show=True,
    show_sample_points=False,
    sample_point_size=0.15,
):
    import matplotlib.pyplot as plt

    count_img = result["count_img"]
    binary_img = result["binary_img"]
    avg_rgb_img = result["avg_rgb_img"]
    s_edges = result["s_edges"]
    z_edges = result["z_edges"]
    wall_length = s_edges[-1]
    z_min = z_edges[0]
    z_max = z_edges[-1]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)

    axes[0].imshow(
        avg_rgb_img,
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
    )
    if show_sample_points and len(result.get("sample_s", [])):
        axes[0].scatter(
            result["sample_s"],
            result["sample_z"],
            c=result["sample_rgb"],
            s=sample_point_size,
            linewidths=0,
            alpha=0.8,
        )
    axes[0].set_title("Average E57 RGB per cell")

    im1 = axes[1].imshow(
        np.log1p(count_img),
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
        cmap="gray_r",
    )
    axes[1].set_title("Log point count")
    plt.colorbar(im1, ax=axes[1], label="log(1 + count)")

    axes[2].imshow(
        binary_img.astype(float),
        origin="lower",
        aspect="auto",
        extent=[0, wall_length, z_min, z_max],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Binary image, threshold >= {result['count_threshold']}"
    )

    for ax in axes:
        ax.set_xlabel("Wall length direction (m)")
        ax.set_ylabel("Z height (m)")

    fig.suptitle(
        f"Wall {wall_id}: E57 RGB length-z image, "
        f"{result['points_used']} points used from {result['points_in_crop']} crop points"
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _length_z_surface_grid(result, normal_offset=0.0):
    wall = result["wall_e57"]
    axis = np.asarray(result["length_axis"], dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        raise ValueError("result length_axis must be non-zero")
    axis = axis / axis_norm
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)

    if wall.get("footprint"):
        reference_xy = np.array([[p["x"], p["y"]] for p in wall["footprint"]], dtype=np.float64)
    else:
        reference_xy = np.array([[p["x"], p["y"]] for p in wall["bbox"]], dtype=np.float64)

    s_min = float((reference_xy @ axis).min())
    normal_center = float(np.median(reference_xy @ normal))

    s_edges = np.asarray(result["s_edges"], dtype=np.float64)
    z_edges = np.asarray(result["z_edges"], dtype=np.float64)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    local_s, z_grid = np.meshgrid(s_centers, z_centers)
    absolute_s = local_s + s_min
    normal_value = normal_center + normal_offset

    x_grid = absolute_s * axis[0] + normal_value * normal[0]
    y_grid = absolute_s * axis[1] + normal_value * normal[1]
    return x_grid, y_grid, z_grid


def _sample_result_or_e57_wall_points(
    result,
    e57_path=None,
    max_points=200000,
    margin_m=0.25,
    chunk_size=2_000_000,
):
    sample_points = np.asarray(result.get("sample_points", np.empty((0, 3))), dtype=np.float64)
    if len(sample_points) or not e57_path or max_points == 0:
        if max_points and len(sample_points) > max_points:
            idx = np.linspace(0, len(sample_points) - 1, max_points, dtype=np.int64)
            return sample_points[idx]
        return sample_points

    from post_process_src.post_process.stages.openings import (
        _filter_chunk_to_wall_crop,
        _wall_crop_geometry,
    )

    crop_geometry = _wall_crop_geometry(result["wall_e57"], margin_m)
    points = []
    sampled = 0
    for chunk in _iter_e57_raw_xyz_chunks(e57_path, chunk_size=chunk_size):
        cropped = _filter_chunk_to_wall_crop(chunk, crop_geometry)
        if not len(cropped):
            continue

        remaining = None if max_points is None else max_points - sampled
        if remaining is None:
            points.append(cropped)
            continue
        if remaining <= 0:
            break
        if len(cropped) <= remaining:
            selected = cropped
        else:
            idx = np.linspace(0, len(cropped) - 1, remaining, dtype=np.int64)
            selected = cropped[idx]
        points.append(selected)
        sampled += len(selected)

    return np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)


def _sample_e57_wall_xyzrgb_points(
    result,
    e57_path,
    max_points=300000,
    margin_m=0.25,
    chunk_size=2_000_000,
):
    if not e57_path:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )

    from post_process_src.post_process.stages.openings import _wall_crop_geometry

    crop_geometry = _wall_crop_geometry(result["wall_e57"], margin_m)
    length_axis = np.asarray(result["length_axis"], dtype=np.float64)
    s_edges = np.asarray(result["s_edges"], dtype=np.float64)
    z_edges = np.asarray(result["z_edges"], dtype=np.float64)
    wall_length = float(s_edges[-1])
    z_min = float(z_edges[0])
    z_max = float(z_edges[-1])
    n_length = len(s_edges) - 1

    axis = length_axis / np.linalg.norm(length_axis)
    if result["wall_e57"].get("footprint"):
        reference_xy = np.array(
            [[p["x"], p["y"]] for p in result["wall_e57"]["footprint"]],
            dtype=np.float64,
        )
    else:
        reference_xy = np.array(
            [[p["x"], p["y"]] for p in result["wall_e57"]["bbox"]],
            dtype=np.float64,
        )
    s_min = float((reference_xy @ axis).min())

    counts_by_col = np.asarray(result["count_img"], dtype=np.int64).sum(axis=0)
    total_available = int(counts_by_col.sum())
    if max_points is None or max_points >= total_available:
        quotas = counts_by_col.copy()
    else:
        quotas = np.zeros(n_length, dtype=np.int64)
        remaining = int(max_points)
        active = counts_by_col > 0
        while remaining > 0 and np.any(active):
            active_cols = np.flatnonzero(active)
            share = max(1, remaining // len(active_cols))
            progressed = 0
            for col in active_cols:
                room = counts_by_col[col] - quotas[col]
                add = min(room, share)
                if add > 0:
                    quotas[col] += add
                    remaining -= int(add)
                    progressed += int(add)
                if remaining <= 0:
                    break
            active = quotas < counts_by_col
            if progressed == 0:
                break

    targets_by_col = [
        np.linspace(0, counts_by_col[col] - 1, quotas[col], dtype=np.int64)
        if quotas[col] > 0
        else np.empty(0, dtype=np.int64)
        for col in range(n_length)
    ]
    target_cursor = np.zeros(n_length, dtype=np.int64)
    seen_by_col = np.zeros(n_length, dtype=np.int64)

    points = []
    colors = []
    for xyz, rgb in _iter_e57_raw_xyzrgb_chunks(e57_path, chunk_size=chunk_size):
        cropped_xyz, cropped_rgb = _filter_xyzrgb_chunk_to_wall_crop(
            xyz,
            rgb,
            crop_geometry,
        )
        if not len(cropped_xyz):
            continue

        s = cropped_xyz[:, :2] @ axis - s_min
        z = cropped_xyz[:, 2]
        keep = (s >= 0.0) & (s <= wall_length) & (z >= z_min) & (z <= z_max)
        if not np.any(keep):
            continue

        cropped_xyz = cropped_xyz[keep]
        cropped_rgb = cropped_rgb[keep]
        s = s[keep]
        x_idx = np.floor(s / wall_length * n_length).astype(np.int64)
        x_idx = np.clip(x_idx, 0, n_length - 1)

        selected_indices = []
        for col in np.unique(x_idx):
            targets = targets_by_col[col]
            cursor = target_cursor[col]
            if cursor >= len(targets):
                seen_by_col[col] += int(np.count_nonzero(x_idx == col))
                continue

            col_indices = np.flatnonzero(x_idx == col)
            start = seen_by_col[col]
            end = start + len(col_indices)
            stop = np.searchsorted(targets, end, side="left")
            if stop > cursor:
                local_indices = targets[cursor:stop] - start
                selected_indices.append(col_indices[local_indices])
                target_cursor[col] = stop
            seen_by_col[col] = end

        if selected_indices:
            selected_indices = np.concatenate(selected_indices)
            points.append(cropped_xyz[selected_indices])
            colors.append(np.clip(cropped_rgb[selected_indices], 0, 255).astype(np.uint8))

    if not points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )

    return np.vstack(points), np.vstack(colors)


def _rgb_array_to_plotly_colors(rgb):
    rgb = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def export_e57_wall_length_z_overlay_html(
    result,
    e57_path=None,
    wall_id=17,
    output_html="wall_17_e57_length_z_overlay.html",
    max_points=200000,
    margin_m=0.25,
    log_normal_offset=0.03,
    binary_normal_offset=0.12,
    log_opacity=0.75,
    binary_opacity=0.45,
    show_points=True,
):
    import plotly.graph_objects as go

    count_img = np.asarray(result["count_img"], dtype=np.float64)
    binary_img = np.asarray(result["binary_img"], dtype=np.float64)
    log_img = np.log1p(count_img)

    fig = go.Figure()

    if show_points:
        points = _sample_result_or_e57_wall_points(
            result,
            e57_path=e57_path,
            max_points=max_points,
            margin_m=margin_m,
        )
        if len(points):
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    marker=dict(size=1.4, color="rgb(150,150,150)", opacity=0.35),
                    name=f"E57 wall points sampled ({len(points)})",
                    hoverinfo="skip",
                )
            )

    x_log, y_log, z_log = _length_z_surface_grid(result, normal_offset=log_normal_offset)
    fig.add_trace(
        go.Surface(
            x=x_log,
            y=y_log,
            z=z_log,
            surfacecolor=log_img,
            colorscale="Viridis",
            opacity=log_opacity,
            name="log(1 + point count)",
            colorbar=dict(title="log count", x=1.02),
            showscale=True,
        )
    )

    x_bin, y_bin, z_bin = _length_z_surface_grid(result, normal_offset=binary_normal_offset)
    fig.add_trace(
        go.Surface(
            x=x_bin,
            y=y_bin,
            z=z_bin,
            surfacecolor=binary_img,
            colorscale=[
                [0.0, "rgb(255,255,255)"],
                [0.499, "rgb(255,255,255)"],
                [0.5, "rgb(20,20,20)"],
                [1.0, "rgb(20,20,20)"],
            ],
            cmin=0,
            cmax=1,
            opacity=binary_opacity,
            name=f"binary occupied mask >= {result['count_threshold']}",
            colorbar=dict(title="binary", x=1.11),
            showscale=True,
        )
    )

    fig.add_trace(
        _bbox_trace(
            [result["wall_e57"]],
            WALL_BBOX_EDGES,
            f"wall {wall_id} bbox mapped to E57",
            "rgb(220,40,40)",
            5,
        )
    )

    fig.update_layout(
        title=(
            f"Wall {wall_id}: E57 points with log-count and binary length-z images "
            f"({result['points_used']} points used)"
        ),
        scene=dict(
            aspectmode="data",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    return fig


def export_e57_wall_points_with_log_image_html(
    result,
    e57_path=None,
    wall_id=17,
    output_html="wall_17_e57_points_full_log_attached.html",
    max_points=None,
    margin_m=0.25,
    log_normal_offset=0.03,
    log_opacity=0.78,
    point_size=1.3,
    point_opacity=0.45,
    point_color_mode="solid",
    scene_background_color=None,
):
    """Plot E57 wall points and attach the length-z log-count image to the wall plane."""
    import plotly.graph_objects as go

    if point_color_mode == "rgb":
        points, rgb = _sample_e57_wall_xyzrgb_points(
            result,
            e57_path,
            max_points=max_points,
            margin_m=margin_m,
        )
        point_colors = _rgb_array_to_plotly_colors(rgb)
        point_trace_name = f"E57 RGB wall points ({len(points)} of {result['points_in_crop']})"
    else:
        points = _sample_result_or_e57_wall_points(
            result,
            e57_path=e57_path,
            max_points=max_points,
            margin_m=margin_m,
        )
        point_colors = "rgb(70,140,230)"
        point_trace_name = f"E57 wall points ({len(points)} of {result['points_in_crop']})"

    fig = go.Figure()
    if len(points):
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=point_colors,
                    opacity=point_opacity,
                ),
                name=point_trace_name,
                hoverinfo="skip",
            )
        )

    x_log, y_log, z_log = _length_z_surface_grid(
        result,
        normal_offset=log_normal_offset,
    )
    fig.add_trace(
        go.Surface(
            x=x_log,
            y=y_log,
            z=z_log,
            surfacecolor=np.log1p(np.asarray(result["count_img"], dtype=np.float64)),
            colorscale="Viridis",
            opacity=log_opacity,
            name="attached log(1 + point count)",
            colorbar=dict(title="log count", x=1.02),
            showscale=True,
        )
    )

    fig.add_trace(
        _bbox_trace(
            [result["wall_e57"]],
            WALL_BBOX_EDGES,
            f"wall {wall_id} bbox mapped to E57",
            "rgb(220,40,40)",
            5,
        )
    )

    scene_layout = dict(
        aspectmode="data",
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0), up=dict(x=0, y=0, z=1)),
    )
    if scene_background_color is not None:
        scene_layout.update(
            xaxis=dict(
                backgroundcolor=scene_background_color,
                gridcolor="rgb(80,80,80)",
                zerolinecolor="rgb(120,120,120)",
            ),
            yaxis=dict(
                backgroundcolor=scene_background_color,
                gridcolor="rgb(80,80,80)",
                zerolinecolor="rgb(120,120,120)",
            ),
            zaxis=dict(
                backgroundcolor=scene_background_color,
                gridcolor="rgb(80,80,80)",
                zerolinecolor="rgb(120,120,120)",
            ),
        )

    fig.update_layout(
        title=(
            f"Wall {wall_id}: full E57 wall points with attached length-z log image "
            f"({result['points_used']} points used for histogram)"
        ),
        scene=scene_layout,
        paper_bgcolor=scene_background_color if scene_background_color is not None else None,
        plot_bgcolor=scene_background_color if scene_background_color is not None else None,
        margin=dict(l=0, r=0, t=45, b=0),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default="WyomingStateFair_Laramie_postprocess_openings.json")
    parser.add_argument("--html", default="WyomingStateFair_Laramie_walls_windows.html")
    parser.add_argument("--csv")
    parser.add_argument("--wall-id")
    parser.add_argument("--e57")
    parser.add_argument("--e57-offset", nargs=3, type=float)
    parser.add_argument("--e57-scale", type=float, default=3.280839895013123)
    parser.add_argument("--dense-openings")
    parser.add_argument("--show-crop-points", action="store_true")
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--open3d", action="store_true")
    args = parser.parse_args()

    if args.open3d:
        visualize_all(args.json_path)
    elif args.e57 and args.wall_id:
        path = export_wall_detail_with_e57_dense_html(
            args.e57,
            args.json_path,
            args.wall_id,
            args.html,
            args.margin,
            e57_to_csv_offset=args.e57_offset,
            e57_to_csv_scale=args.e57_scale,
            dense_openings_path=args.dense_openings,
            crop_points_visible=True if args.show_crop_points else "legendonly",
        )
        print(path)
    elif args.csv and args.wall_id:
        path = export_wall_detail_with_original_csv_html(
            args.csv,
            args.json_path,
            args.wall_id,
            args.html,
            args.margin,
        )
        print(path)
    elif args.csv:
        path = export_walls_openings_with_original_csv_html(
            args.csv,
            args.json_path,
            args.html,
            args.max_points,
        )
        print(path)
    else:
        path = export_walls_openings_html(args.json_path, args.html, args.max_points)
        print(path)
