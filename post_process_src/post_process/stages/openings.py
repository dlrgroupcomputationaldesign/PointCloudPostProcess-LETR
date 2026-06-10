"""Detect doors, windows, and openings from per-wall log-density images.

Each wall's points are histogrammed in its own (length, height) frame and
rendered as a grayscale image (see ``utils.opening_image_util``). A pluggable
detector -- Grounding DINO by default, selected via ``OPENING_DETECTOR`` -- finds
boxes on that image, which are mapped back to 3-D bounding boxes. Both the input
log image and the annotated detection image are uploaded to blob alongside the
floor/ceiling/wall snapshots when ``logging_blob_location`` is set.
"""

import numpy as np
from shapely.geometry import Point, Polygon

try:
    from shapely import contains_xy
except ImportError:  # pragma: no cover - shapely 1.x fallback
    contains_xy = None

from ..config import coerce_parameters
from ..detectors import build_detector
from ..labels import label_mask
from ..runtime import logger
from ..utils.blob_util import (
    upload_dict_to_blob_json,
    upload_image_array_to_blob,
)
from ..utils.opening_image_util import (
    annotate_detections,
    box_to_grid_span,
    count_grid_dims,
    downsample_sum,
    render_log_image,
)
from .common import make_blob_factory


def _point_dict_to_array(points):
    return np.asarray(
        [[point["x"], point["y"], point["z"]] for point in points],
        dtype=float,
    )


def _length_axis(wall):
    """Unit vector along the wall's longest footprint edge (bbox fallback).

    Matches the experiment's ``_longest_footprint_edge_axis`` orientation.
    """
    footprint = wall.get("footprint")
    if footprint and len(footprint) >= 3:
        pts = np.asarray([[p["x"], p["y"]] for p in footprint], dtype=float)
        edges = np.asarray([pts[(i + 1) % len(pts)] - pts[i] for i in range(len(pts))])
    else:
        pts = _point_dict_to_array(wall["bbox"])[:, :2]
        edges = np.asarray(
            [pts[j] - pts[i] for i in range(len(pts)) for j in range(i + 1, len(pts))]
        )
    norms = np.linalg.norm(edges, axis=1)
    axis = edges[int(np.argmax(norms))]
    return axis / np.linalg.norm(axis)


def _wall_frame(wall):
    corners = _point_dict_to_array(wall["bbox"])
    xy = corners[:, :2]
    center_xy = xy.mean(axis=0)
    centered_xy = xy - center_xy

    length_axis = _length_axis(wall)
    normal_axis = np.array([-length_axis[1], length_axis[0]])

    s = centered_xy @ length_axis
    t = centered_xy @ normal_axis

    return {
        "origin_xy": center_xy,
        "length_axis": length_axis,
        "normal_axis": normal_axis,
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
        "z_min": float(corners[:, 2].min()),
        "z_max": float(corners[:, 2].max()),
    }


def _project_points(points_xyz, frame):
    xy = points_xyz[:, :2] - frame["origin_xy"]
    s = xy @ frame["length_axis"]
    t = xy @ frame["normal_axis"]
    z = points_xyz[:, 2]
    return s, t, z


def _candidate_bbox(frame, s_min, s_max, z_min, z_max):
    t_min = frame["t_min"]
    t_max = frame["t_max"]
    corners = []
    for z in (z_min, z_max):
        for t in (t_min, t_max):
            for s in (s_min, s_max):
                xy = (
                    frame["origin_xy"]
                    + s * frame["length_axis"]
                    + t * frame["normal_axis"]
                )
                corners.append(
                    {
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                        "z": float(z),
                    }
                )
    return corners


# ---------------------------------------------------------------------------
# Unit handling
#
# wall_output geometry is in the CSV's native units (feet for these datasets);
# OPENING_E57_TO_CSV_SCALE is the native-units-per-meter factor (3.2808 for feet,
# 1.0 for meter data). Physical parameters -- the image bin size and the
# width/height filters -- are expressed in METERS and converted to native units
# here, so the rendered grid matches the meter-binned images the detector expects.
# ---------------------------------------------------------------------------
def _units_per_meter(parameters):
    return float(parameters["OPENING_E57_TO_CSV_SCALE"])


def _fine_bin_size(parameters):
    """Accumulation bin in native units (fine resolution, block-summed later)."""
    return float(parameters["OPENING_IMAGE_FINE_BIN_M"]) * _units_per_meter(parameters)


def _render_bin_factor(parameters):
    """How many fine cells block-sum into one render cell (e.g. 0.05/0.025 = 2)."""
    fine = float(parameters["OPENING_IMAGE_FINE_BIN_M"])
    render = float(parameters["OPENING_IMAGE_BIN_M"])
    return max(1, int(round(render / fine)))


# ---------------------------------------------------------------------------
# Count grid (the FINE histogram; block-summed to the render bin before display)
# ---------------------------------------------------------------------------
def _empty_count_grid(frame, parameters):
    n_z, n_s = count_grid_dims(frame, _fine_bin_size(parameters))
    return np.zeros((n_z, n_s), dtype=np.uint32)


def _accumulate_counts(counts, frame, points_xyz, parameters):
    """Bin points that fall on this wall into its fine (z, s) count grid."""
    if points_xyz.size == 0:
        return 0

    s, t, z = _project_points(points_xyz, frame)
    distance_tolerance = float(parameters["OPENING_WALL_DISTANCE_TOLERANCE"])
    wall_mask = (
        (s >= frame["s_min"])
        & (s <= frame["s_max"])
        & (t >= frame["t_min"] - distance_tolerance)
        & (t <= frame["t_max"] + distance_tolerance)
        & (z >= frame["z_min"])
        & (z <= frame["z_max"])
    )
    if not np.any(wall_mask):
        return 0

    s = s[wall_mask]
    z = z[wall_mask]
    bin_native = _fine_bin_size(parameters)
    n_z, n_s = counts.shape
    s_idx = np.clip(np.floor((s - frame["s_min"]) / bin_native).astype(int), 0, n_s - 1)
    z_idx = np.clip(np.floor((z - frame["z_min"]) / bin_native).astype(int), 0, n_z - 1)
    np.add.at(counts, (z_idx, s_idx), 1)
    return int(s.size)


# ---------------------------------------------------------------------------
# Detection -> candidate mapping
# ---------------------------------------------------------------------------
def _classify_label(label):
    text = str(label).lower()
    if "door" in text:
        return "door"
    if "window" in text:
        return "window"
    return "opening"


def _detection_to_candidate(detection, wall, frame, n_z, n_s, parameters):
    span = box_to_grid_span(
        detection.box_xyxy,
        frame,
        n_z,
        n_s,
        parameters["OPENING_IMAGE_CELL_PX"],
    )
    if span is None:
        return None

    s_min, s_max, z_min, z_max = span
    width = s_max - s_min  # native units (feet); matches the bbox geometry below
    height = z_max - z_min
    # Size filters are physical (metres); convert the native extents to compare.
    units_per_meter = _units_per_meter(parameters)
    width_m = width / units_per_meter
    height_m = height / units_per_meter
    if width_m < parameters["OPENING_MIN_WIDTH"] or width_m > parameters["OPENING_MAX_WIDTH"]:
        return None
    if height_m < parameters["OPENING_MIN_HEIGHT"] or height_m > parameters["OPENING_MAX_HEIGHT"]:
        return None

    category = _classify_label(detection.label)
    return {
        "wallId": str(wall["id"]),
        "class": category,
        "category": category,  # consumed (popped) by _candidates_to_output
        "label": str(detection.label),
        "confidence": float(detection.score),
        "bbox": _candidate_bbox(frame, s_min, s_max, z_min, z_max),
        "width": float(width),
        "height": float(height),
        "bottomZ": float(z_min),
        "topZ": float(z_max),
    }


# ---------------------------------------------------------------------------
# Output sinks: upload to blob, or fall back to a local folder for inspection.
# A "name" is a forward-slash relative path like "opening/wall_3_log.png".
# ---------------------------------------------------------------------------
def _local_path(parameters, name):
    # Local output is flat: drop the blob-style "opening/" prefix so the log and
    # detection images land directly under OPENING_LOCAL_OUTPUT_DIR.
    import os

    base = os.path.abspath(str(parameters["OPENING_LOCAL_OUTPUT_DIR"]))
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, os.path.basename(name))


def _make_image_sink(blobs, parameters):
    """Return fn(name, image_rgb) writing to blob, else to OPENING_LOCAL_OUTPUT_DIR.

    Returns None when neither sink is configured (detection still runs; just no
    images are written).
    """
    if blobs:
        def _to_blob(name, image_rgb):
            upload_image_array_to_blob(image_rgb, blobs(name))

        return _to_blob

    if parameters.get("OPENING_LOCAL_OUTPUT_DIR"):
        import cv2

        def _to_local(name, image_rgb):
            cv2.imwrite(_local_path(parameters, name), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        return _to_local

    return None


def _write_output_json(data, name, blobs, parameters):
    """Upload the opening JSON to blob, or write it under the local output dir."""
    if blobs:
        upload_dict_to_blob_json(data, blobs(name))
    elif parameters.get("OPENING_LOCAL_OUTPUT_DIR"):
        import json

        with open(_local_path(parameters, name), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)


def _detect_wall(wall, frame, counts, point_count, detector, parameters, image_sink):
    """Render -> detect -> map for a single wall, writing both images via image_sink.

    ``counts`` is the FINE histogram; it is block-summed to the render bin first,
    so detection and pixel->3-D mapping use the render-resolution grid.
    """
    if point_count < int(parameters["OPENING_MIN_WALL_POINTS"]):
        return []
    if frame["s_max"] <= frame["s_min"] or frame["z_max"] <= frame["z_min"]:
        return []

    render_counts = downsample_sum(counts, _render_bin_factor(parameters))
    image_rgb = render_log_image(render_counts, parameters)
    n_z, n_s = render_counts.shape
    wall_id = str(wall["id"])

    if image_sink:
        image_sink("opening/wall_{}_log.png".format(wall_id), image_rgb)

    detections = detector.detect(image_rgb)

    candidates = []
    annotations = []
    for detection in detections:
        candidate = _detection_to_candidate(detection, wall, frame, n_z, n_s, parameters)
        if candidate is None:
            continue
        candidates.append(candidate)
        annotations.append(
            {
                "box_xyxy": detection.box_xyxy,
                "category": candidate["category"],
                "score": candidate["confidence"],
            }
        )

    if image_sink:
        annotated = annotate_detections(image_rgb, annotations)
        image_sink("opening/wall_{}_detected.png".format(wall_id), annotated)

    logger.info(
        "Wall {}: {} points, {} detections, {} kept".format(
            wall_id, point_count, len(detections), len(candidates)
        )
    )
    return candidates


# ---------------------------------------------------------------------------
# Wall selection and dense-E57 cropping
# ---------------------------------------------------------------------------
def _selected_walls(wall_output, parameters):
    walls = wall_output.get("walls", [])
    wall_ids = parameters.get("OPENING_DENSE_WALL_IDS")
    if not wall_ids:
        return walls

    selected = {str(wall_id) for wall_id in wall_ids}
    return [wall for wall in walls if str(wall.get("id")) in selected]


def _wall_aabb(wall, margin):
    points = _point_dict_to_array(wall["bbox"])
    return points.min(axis=0) - margin, points.max(axis=0) + margin


def _wall_z_range(wall, margin=0.0):
    if "zRange" in wall:
        z_range = wall["zRange"]
        return float(z_range["min"]) - margin, float(z_range["max"]) + margin

    points = _point_dict_to_array(wall["bbox"])
    return float(points[:, 2].min()) - margin, float(points[:, 2].max()) + margin


def _wall_footprint_polygon(wall, margin=0.0):
    footprint = wall.get("footprint")
    if not footprint:
        return None

    points = np.asarray([[pt["x"], pt["y"]] for pt in footprint], dtype=float)
    if len(points) < 3:
        return None

    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return None
    if margin:
        polygon = polygon.buffer(float(margin))
    if polygon.is_empty:
        return None
    return polygon


def _wall_crop_geometry(wall, margin):
    polygon = _wall_footprint_polygon(wall, margin)
    if polygon is None:
        minb, maxb = _wall_aabb(wall, margin)
        return {"minb": minb, "maxb": maxb, "polygon": None, "source": "bbox"}

    minx, miny, maxx, maxy = polygon.bounds
    z_min, z_max = _wall_z_range(wall, margin)
    return {
        "minb": np.array([minx, miny, z_min], dtype=float),
        "maxb": np.array([maxx, maxy, z_max], dtype=float),
        "polygon": polygon,
        "source": "footprint",
    }


def _filter_chunk_to_wall_crop(chunk, crop_geometry):
    minb = crop_geometry["minb"]
    maxb = crop_geometry["maxb"]
    mask = (
        (chunk[:, 0] >= minb[0])
        & (chunk[:, 0] <= maxb[0])
        & (chunk[:, 1] >= minb[1])
        & (chunk[:, 1] <= maxb[1])
        & (chunk[:, 2] >= minb[2])
        & (chunk[:, 2] <= maxb[2])
    )
    if not np.any(mask):
        return chunk[:0]

    cropped = chunk[mask]
    polygon = crop_geometry.get("polygon")
    if polygon is None:
        return cropped

    if contains_xy is not None:
        inside = contains_xy(polygon, cropped[:, 0], cropped[:, 1])
    else:
        inside = np.array(
            [polygon.contains(Point(x, y)) for x, y in cropped[:, :2]],
            dtype=bool,
        )

    return cropped[inside]


def _compute_xyz_min_from_annotation(parameters):
    """Min XYZ of the original (unshifted) annotation cloud, in feet.

    The post-process input CSV is shifted to the origin, so the original min --
    needed to align the E57 -- comes from a separate annotation CSV (the
    experiment's csv_label/<name>.csv). Reads X,Y,Z (falls back to x,y,z).
    """
    import pandas as pd

    path = parameters["OPENING_ANNOTATION_CSV_PATH"]
    chunksize = int(parameters["OPENING_ANNOTATION_CHUNK_SIZE"])
    header = pd.read_csv(path, nrows=0)
    cols = ["X", "Y", "Z"] if {"X", "Y", "Z"}.issubset(header.columns) else ["x", "y", "z"]

    xyz_min = np.array([np.inf, np.inf, np.inf])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        xyz_min = np.minimum(xyz_min, chunk[cols].to_numpy(dtype=float).min(axis=0))
    return xyz_min


def _resolve_e57_offset(parameters):
    """Derive the wall-frame offset from an annotation CSV when not given explicitly.

    With the ``original * scale - offset`` transform, the offset is the min of the
    already-scaled (feet) coordinates -- i.e. the annotation cloud's xyz min. Used
    only when neither xyz_offset nor OPENING_E57_TO_CSV_OFFSET was supplied.
    """
    if parameters.get("OPENING_E57_TO_CSV_OFFSET") is not None:
        return parameters
    if not parameters.get("OPENING_ANNOTATION_CSV_PATH"):
        return parameters

    xyz_min_ft = _compute_xyz_min_from_annotation(parameters)
    offset = xyz_min_ft.tolist()
    logger.info(
        "Computed wall-frame offset from annotation {}: {}".format(
            parameters["OPENING_ANNOTATION_CSV_PATH"], np.round(offset, 4).tolist()
        )
    )
    parameters = dict(parameters)
    parameters["OPENING_E57_TO_CSV_OFFSET"] = offset
    return parameters


def _to_wall_frame(points_xyz, parameters):
    """Map original-cloud points into the (scaled, shifted) wall coordinate frame.

    Reproduces the preprocessing transform that produced the walls: the cloud is
    scaled (e.g. metres -> feet) then shifted to the origin, i.e.
    ``wall_coords = original * scale - xyz_offset``. ``scale`` comes from
    OPENING_E57_TO_CSV_SCALE and ``xyz_offset`` from OPENING_E57_TO_CSV_OFFSET
    (set from the caller's xyz_offset, or the annotation min, or 0).
    """
    scale = float(parameters["OPENING_E57_TO_CSV_SCALE"])
    offset = parameters.get("OPENING_E57_TO_CSV_OFFSET")
    offset = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
    return points_xyz.astype(float, copy=False) * scale - offset


def _iter_e57_raw_chunks(path, chunk_size):
    import pye57

    e57 = pye57.E57(str(path))
    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    try:
        for scan_index in range(e57.scan_count):
            header = e57.get_header(scan_index)
            data, buffers = e57.make_buffers(fields, chunk_size)
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
                ).astype(float, copy=False)
    finally:
        e57.close()


def _iter_las_raw_chunks(path, chunk_size):
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading .las/.laz point clouds requires laspy: pip install laspy"
        ) from exc

    with laspy.open(str(path)) as reader:
        for points in reader.chunk_iterator(chunk_size):
            yield np.column_stack(
                (np.asarray(points.x), np.asarray(points.y), np.asarray(points.z))
            ).astype(float, copy=False)


def _iter_array_raw_chunks(points_xyz, chunk_size):
    for start in range(0, len(points_xyz), chunk_size):
        yield np.asarray(points_xyz[start : start + chunk_size], dtype=float)


def _iter_dense_chunks(path, parameters):
    """Yield raw (unconverted) XYZ chunks from a point cloud, dispatching on extension.

    Supports .e57 (pye57), .las/.laz (laspy), .npy (numpy), and anything Open3D
    can read (.ply/.pcd/.pts/.xyz). Large formats stream; others load once.
    """
    import os

    ext = os.path.splitext(str(path))[1].lower()
    chunk_size = int(parameters["OPENING_E57_CHUNK_SIZE"])

    if ext == ".e57":
        yield from _iter_e57_raw_chunks(path, chunk_size)
    elif ext in (".las", ".laz"):
        yield from _iter_las_raw_chunks(path, chunk_size)
    elif ext == ".npy":
        arr = np.load(str(path), mmap_mode="r")
        yield from _iter_array_raw_chunks(np.asarray(arr[:, :3]), chunk_size)
    else:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        yield from _iter_array_raw_chunks(np.asarray(pcd.points), chunk_size)


# ---------------------------------------------------------------------------
# Accumulation strategies
# ---------------------------------------------------------------------------
def _build_accumulators(walls, parameters, margin=None):
    accumulators = []
    for wall in walls:
        frame = _wall_frame(wall)
        acc = {
            "wall": wall,
            "frame": frame,
            "counts": _empty_count_grid(frame, parameters),
            "point_count": 0,
        }
        if margin is not None:
            acc["crop_geometry"] = _wall_crop_geometry(wall, margin)
        accumulators.append(acc)
    return accumulators


def _accumulate_from_dense(accumulators, point_cloud_path, parameters):
    logger.info("Streaming dense opening points from {}".format(point_cloud_path))
    for raw_chunk in _iter_dense_chunks(point_cloud_path, parameters):
        chunk = _to_wall_frame(raw_chunk, parameters)
        for acc in accumulators:
            cropped = _filter_chunk_to_wall_crop(chunk, acc["crop_geometry"])
            if len(cropped):
                acc["point_count"] += _accumulate_counts(
                    acc["counts"], acc["frame"], cropped, parameters
                )


def _accumulate_from_df(accumulators, wall_points_xyz, parameters):
    for acc in accumulators:
        acc["point_count"] += _accumulate_counts(
            acc["counts"], acc["frame"], wall_points_xyz, parameters
        )


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------
def _candidates_to_output(candidates):
    doors = []
    windows = []
    openings = []
    counters = {"door": 1, "window": 1, "opening": 1}

    for candidate in candidates:
        category = candidate.pop("category")
        candidate["id"] = str(counters[category])
        counters[category] += 1

        if category == "door":
            doors.append(candidate)
        elif category == "window":
            windows.append(candidate)
        else:
            openings.append(candidate)

    return {
        "points": [],
        "doors": doors,
        "windows": windows,
        "openings": openings,
    }


def run_openings(
    df,
    parameters,
    wall_output,
    point_cloud_path=None,
    xyz_offset=None,
    opening_detection_model=None,
    logging_blob_location=None,
):
    """Detect doors, windows, and openings on per-wall log-density images.

    Args:
        df: labelled point DataFrame (used only when no dense cloud is given).
        parameters: post-process parameters.
        wall_output: output of ``run_walls`` (the walls to scan).
        point_cloud_path: original dense cloud (.e57/.las/.npy/.ply...) to stream.
            Falls back to ``OPENING_DENSE_SOURCE_PATH``; if neither, uses ``df``.
        xyz_offset: the shift applied during preprocessing, so the dense cloud is
            mapped to the wall frame as ``original * scale - xyz_offset``
            (scale = OPENING_E57_TO_CSV_SCALE). If None, falls back to an explicit
            OPENING_E57_TO_CSV_OFFSET or an annotation CSV.
        opening_detection_model: path/id of the detector weights/model; injected
            so ``build_detector`` loads it.
        logging_blob_location: blob target for images/JSON; when None, images go
            to OPENING_LOCAL_OUTPUT_DIR if set.
    """
    parameters = coerce_parameters(parameters)

    # Fold caller-supplied runtime values into the parameters the helpers read.
    overrides = {}
    if opening_detection_model:
        overrides["OPENING_GD_WEIGHTS_PATH"] = opening_detection_model
        overrides["OPENING_GD_MODEL_ID"] = opening_detection_model
    if xyz_offset is not None:
        overrides["OPENING_E57_TO_CSV_OFFSET"] = list(xyz_offset)
    if overrides:
        parameters = {**parameters, **overrides}

    blobs = make_blob_factory(logging_blob_location)

    logger.info("Running opening post-processing...")

    image_sink = _make_image_sink(blobs, parameters)

    empty_output = {"points": [], "doors": [], "windows": [], "openings": []}
    walls = _selected_walls(wall_output, parameters)
    if not walls:
        _write_output_json(empty_output, "opening_output.json", blobs, parameters)
        return empty_output

    detector = build_detector(parameters)

    dense_path = point_cloud_path or parameters.get("OPENING_DENSE_SOURCE_PATH")
    if dense_path:
        # Align the dense cloud to the walls (xyz_offset, explicit offset, or annotation).
        if parameters.get("OPENING_E57_TO_CSV_OFFSET") is None:
            parameters = _resolve_e57_offset(parameters)
        if parameters.get("OPENING_E57_TO_CSV_OFFSET") is None:
            logger.warning(
                "No xyz_offset / OPENING_E57_TO_CSV_OFFSET / annotation given; the "
                "dense cloud may be misaligned with the walls."
            )
        # Crop margin is physical (metres); the crop geometry is in native units.
        margin = float(parameters["OPENING_DENSE_WALL_MARGIN"]) * _units_per_meter(parameters)
        accumulators = _build_accumulators(walls, parameters, margin=margin)
        _accumulate_from_dense(accumulators, dense_path, parameters)
    else:
        accumulators = _build_accumulators(walls, parameters)
        wall_df = df[label_mask(df, "Wall", parameters)]
        wall_points_xyz = wall_df[["x", "y", "z"]].to_numpy(dtype=float)
        _accumulate_from_df(accumulators, wall_points_xyz, parameters)

    candidates = []
    for acc in accumulators:
        candidates.extend(
            _detect_wall(
                acc["wall"],
                acc["frame"],
                acc["counts"],
                acc["point_count"],
                detector,
                parameters,
                image_sink,
            )
        )

    opening_output_dict = _candidates_to_output(candidates)
    _write_output_json(opening_output_dict, "opening_output.json", blobs, parameters)

    return opening_output_dict
