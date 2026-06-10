import numpy as np
import json
from pydantic.dataclasses import dataclass
import io
import logging

@dataclass
class blob_config:
    account_name: str
    account_key: str 
    container_name: str 
    folder: str 

class InMemoryLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.stream = io.StringIO()

    def emit(self, record):
        self.stream.write(self.format(record) + "\n")

    def get_text(self) -> str:
        return self.stream.getvalue()

def setup_blob_clients(logging_blob_location):
    from azure.storage.blob import BlobServiceClient

    blob_cfg = blob_config(**logging_blob_location)

    blob_service_client = BlobServiceClient(
        account_url=f"https://{blob_cfg.account_name}.blob.core.windows.net",
        credential=blob_cfg.account_key
    )
    container_client = blob_service_client.get_container_client(blob_cfg.container_name)
    try:
        container_client.create_container()
    except Exception:
        pass
    
    base = blob_cfg.folder.strip("/")

    def blob(name: str):
        # name like "floor/floor_plane_fit_1.png" or "floor_plane_fit_1.png"
        name = name.lstrip("/")
        return container_client.get_blob_client(f"{base}/{name}")

    return blob

def setup_logger_in_memory():
    logger = logging.getLogger("Infer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers if setup is called multiple times
    if not any(isinstance(h, InMemoryLogHandler) for h in logger.handlers):
        mh = InMemoryLogHandler()
        mh.setLevel(logging.INFO)
        mh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(mh)

    return logger

def upload_logger_to_blob(logger, log_blob_client):
    from azure.storage.blob import ContentSettings

    mh = next(h for h in logger.handlers if isinstance(h, InMemoryLogHandler))
    log_blob_client.upload_blob(
        mh.get_text().encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type="text/plain; charset=utf-8"),
    )

def upload_dict_to_blob_json(
    data: dict,
    json_blob_client,
    indent=2,
    encoding="utf-8"
    ):
    from azure.storage.blob import ContentSettings

    buf = io.StringIO()
    json.dump(data, buf, indent=indent)
    json_text = buf.getvalue()

    json_blob_client.upload_blob(
        json_text.encode(encoding),
        overwrite=True,
        content_settings=ContentSettings(
            content_type="application/json; charset=utf-8"
        ),
    )

def snapshot_plotly_html_bytes(points_xyz, colors_rgb=None, point_size=2, camera=None):
    import plotly.graph_objects as go

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    marker = {"size": point_size}
    if colors_rgb is not None:
        # colors_rgb expected in [0,255] uint8 or [0,1] float
        if colors_rgb.max() <= 1.0:
            c = (colors_rgb * 255).astype("uint8")
        else:
            c = colors_rgb.astype("uint8")
        marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in c]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=marker
        )
    ])

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera or dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    return html.encode("utf-8")

def plot_plane_inliers_outliers_html_bytes(
        inlier_pts,
        outlier_pts,
        point_size=2,
        camera=None,
    ):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=inlier_pts[:, 0],
        y=inlier_pts[:, 1],
        z=inlier_pts[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color="rgb(255,0,0)",  # red
        ),
        name="Plane (inliers)",
    ))

    fig.add_trace(go.Scatter3d(
        x=outlier_pts[:, 0],
        y=outlier_pts[:, 1],
        z=outlier_pts[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            color="rgb(0,0,255)",  # blue
        ),
        name="Non-plane (outliers)",
    ))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera or dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(itemsizing="constant"),
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

def obb_corners_from_o3d(obb):
    # Half sizes
    ex, ey, ez = obb.extent / 2.0

    # 8 corners in local frame
    local = np.array([
        [-ex, -ey, -ez],
        [ ex, -ey, -ez],
        [-ex,  ey, -ez],
        [ ex,  ey, -ez],
        [-ex, -ey,  ez],
        [ ex, -ey,  ez],
        [-ex,  ey,  ez],
        [ ex,  ey,  ez],
    ])

    # Rotate + translate to world frame
    R = obb.R          # (3,3)
    c = obb.center     # (3,)
    world = (local @ R.T) + c

    return world

def plot_inliers_with_obb_html_bytes(
        inlier_pts,
        bbox,
        point_size=2,
        camera=None,
    ):
    import plotly.graph_objects as go

    # OBB edges by corner index (12 edges total)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]
    obb_corners = obb_corners_from_o3d(bbox)

    # Make one line trace with None separators between segments
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [obb_corners[i, 0], obb_corners[j, 0], None]
        ys += [obb_corners[i, 1], obb_corners[j, 1], None]
        zs += [obb_corners[i, 2], obb_corners[j, 2], None]

    fig = go.Figure()

    # Inlier points
    fig.add_trace(go.Scatter3d(
        x=inlier_pts[:, 0], y=inlier_pts[:, 1], z=inlier_pts[:, 2],
        mode="markers",
        marker=dict(size=point_size, color="rgb(255,0,0)"),
        name="Inliers (plane)"
    ))

    # Bounding box lines
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=6, color="rgb(0,255,0)"),
        name="OBB"
    ))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera or dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(itemsizing="constant"),
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

def aabb_edges_from_minmax(minb, maxb):
    # 8 corners of AABB
    x0, y0, z0 = minb
    x1, y1, z1 = maxb
    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x0, y1, z0],  # 2
        [x1, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x0, y1, z1],  # 6
        [x1, y1, z1],  # 7
    ])

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]
    return xs, ys, zs

def plot_clusters_with_aabbs_html_bytes(clusters, point_size=2, camera=None, box_color_rgb=(255, 0, 0)):
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add clusters
    for idx, c in enumerate(clusters):
        pts = c["points"]
        col = c["color"]

        # color is [0,1] -> rgb string
        r, g, b = (np.clip(col, 0, 1) * 255).astype(int)
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=f"rgb({r},{g},{b})"),
            name=f"cluster_{idx}",
        ))

        # Add AABB
        xs, ys, zs = aabb_edges_from_minmax(c["min"], c["max"])
        br, bg, bb = box_color_rgb
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(width=6, color=f"rgb({br},{bg},{bb})"),
            name=f"aabb_{idx}",
            showlegend=False,  # keep legend clean; set True if you want each box listed
        ))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=camera or dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")

def upload_html_bytes_to_blob(html_bytes, blob_client, overwrite=True):
    from azure.storage.blob import ContentSettings

    blob_client.upload_blob(
        html_bytes,
        overwrite=overwrite,
        content_settings=ContentSettings(content_type="text/html; charset=utf-8"),
    )

def upload_image_array_to_blob(image_rgb, blob_client, overwrite=True):
    """PNG-encode an HxWx3 RGB uint8 array and upload it."""
    import cv2
    from azure.storage.blob import ContentSettings

    bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image for blob upload")

    blob_client.upload_blob(
        buf.tobytes(),
        overwrite=overwrite,
        content_settings=ContentSettings(content_type="image/png"),
    )


def upload_matplotlib_fig_to_blob(fig, blob_client, dpi=200, overwrite=True):
    import matplotlib.pyplot as plt
    from azure.storage.blob import ContentSettings

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # prevent memory leak in loops
    buf.seek(0)

    blob_client.upload_blob(
        buf.getvalue(),
        overwrite=overwrite,
        content_settings=ContentSettings(content_type="image/png"),
    )

