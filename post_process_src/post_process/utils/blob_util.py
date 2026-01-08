import pandas as pd
import numpy as np
from scipy.stats import mode
import json
from pydantic.dataclasses import dataclass
from azure.storage.blob import BlobServiceClient, ContentSettings
import io
import logging
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

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

def snapshot_png_bytes_visualizer(
        geometries,
        width=1280,
        height=720,
        visible=False,
        view="bbox"  # "cluster", "bbox"
    ):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=visible)

    try:
        for g in geometries:
            vis.add_geometry(g)

        ctr = vis.get_view_control()

        # Auto-center on geometry
        bbox = geometries[0].get_axis_aligned_bounding_box()
        center = bbox.get_center()
        ctr.set_lookat(center)

        if view == "cluster":
            ctr.set_front([0.1, 0.5, 0.4])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(1.5)
        elif view == "bbox":
            ctr.set_front([0.2, 0.5, 0.6])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(1.5)
        else:  # top-down
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.7)

        # Required to avoid black frame
        vis.poll_events()
        vis.update_renderer()

        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img_u8

    finally:
        vis.destroy_window()

def upload_snapshot_to_blob_from_u8(img_u8, snapshot_blob_client):
    # PNG encode using Pillow (recommended on Windows)
    buf = io.BytesIO()
    Image.fromarray(img_u8).save(buf, format="PNG")
    buf.seek(0)

    snapshot_blob_client.upload_blob(
        buf.getvalue(),
        overwrite=True,
        content_settings=ContentSettings(content_type="image/png"),
    )

def upload_matplotlib_fig_to_blob(fig, blob_client, dpi=200, overwrite=True):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # prevent memory leak in loops
    buf.seek(0)

    blob_client.upload_blob(
        buf.getvalue(),
        overwrite=overwrite,
        content_settings=ContentSettings(content_type="image/png"),
    )

