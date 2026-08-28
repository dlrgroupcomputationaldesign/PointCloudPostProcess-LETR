import numpy as np
from shapely.geometry import Point, Polygon

from ..config import coerce_parameters
from ..runtime import get_device, logger
from ..utils.blob_util import write_image_output, write_json_output
from ..utils.floor_ceiling_util import (
    point_axis_align,
    points_between_level,
    project_points_to_floor,
    sorted_merged_floor_ceiling_plane,
)
from .common import make_blob_factory, resolve_survey_basis


def run_walls(df, parameters, floor_bboxz, line_seg_model, logging_blob_location=None):
    import torch

    from ..utils.wall_seg_util import (
        buffer_from_wall_thickness,
        convert_to_edge_points,
        extract_bbox_minmax,
        find_zrgb,
        img_process_model_input,
        line_segmentation_inf,
        load_line_segmentation_model,
        pixel_to_xy,
        slope_thresholds,
        wall_overlay_image,
    )

    parameters = coerce_parameters(parameters)
    # Resolve the basis ONCE and write it back, so the rotate-back-out calls
    # below use the same matrix the alignment used. An estimated basis that
    # differed between align and un-align would leave output in neither frame.
    parameters["SURVEY_BASIS"] = resolve_survey_basis(df, parameters)
    blobs = make_blob_factory(logging_blob_location)
    # Same sink fallback as the floor and ceiling stages: blob when configured,
    # otherwise LOCAL_OUTPUT_DIR. Without this the stage produced no snapshots
    # at all unless a blob location was set.
    local_dir = parameters.get("LOCAL_OUTPUT_DIR")

    logger.info("Running wall post-processing...")
    wall_lst, point_lst = [], []
    num_level = len(floor_bboxz)
    plane_arr = sorted_merged_floor_ceiling_plane(floor_bboxz)

    # Preserve current behavior: wall reconstruction projects all points between levels.
    align_axis_df = point_axis_align(df, np.array(parameters["SURVEY_BASIS"]))
    xyzrgb = align_axis_df[["x", "y", "z", "r", "g", "b"]].values

    checkpoint = torch.load(line_seg_model, map_location=get_device())
    model = load_line_segmentation_model(checkpoint)
    wall_id = 1

    # WALL_ANGLE_TOL_DEG replaces the VERT_THR / HORI_THR pair, which were
    # reciprocals encoding one angle (the shipped 10 and 0.1 are both
    # tan(5.71 deg)). The old keys still win if either is set explicitly.
    # Explicit legacy values win; otherwise derive both from the one angle.
    # WALL_ANGLE_TOL_DEG now always has a default, so checking it first would
    # make VERT_THR/HORI_THR unreachable for callers still setting them.
    vert_thr = parameters.get("VERT_THR")
    hori_thr = parameters.get("HORI_THR")
    if vert_thr is None or hori_thr is None:
        vert_thr, hori_thr = slope_thresholds(
            parameters.get("WALL_ANGLE_TOL_DEG", 5.71))

    resize_width = int(parameters["RESIZE_WIDTH"])
    units_per_meter = float(parameters.get("POINT_CLOUD_TO_POST_PROCESSING_SCALE", 1.0))
    wall_thickness = float(parameters.get("WALL_THICKNESS_M", 0.279)) * units_per_meter
    morph_kernel = int(parameters.get("WALL_MORPH_KERNEL", 3))

    for level in range(num_level):
        if level == num_level - 1:
            z_min_floor = plane_arr[-1][1]
            filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor)]
        else:
            filtered_points = points_between_level(
                plane_arr[level],
                plane_arr[level + 1],
                xyzrgb,
            )

        xy_projected, x_edges, y_edges, projected_img_arr = project_points_to_floor(
            filtered_points,
            parameters["PROJECTED_BINS"],
        )
        inputs, orig_size, resize_ratio = img_process_model_input(
            projected_img_arr,
            resize_width,
            parameters["INT_THR"],
            morph_kernel=morph_kernel,
        )

        # "auto" converts the physical wall thickness into resized-image pixels,
        # which is the space the line buffers live in. A fixed number means a
        # different physical width on every building.
        buffer_thr = parameters["BUFFER_THR"]
        if isinstance(buffer_thr, str) and str(buffer_thr).lower() == "auto":
            buffer_thr = buffer_from_wall_thickness(x_edges, resize_width, wall_thickness)
            logger.info(
                "wall buffer: thickness %.2f units over %.0f-unit extent at "
                "%d px -> BUFFER_THR %.2f px",
                wall_thickness, float(x_edges[-1] - x_edges[0]), resize_width,
                buffer_thr,
            )

        polyhv_arr = line_segmentation_inf(
            model,
            inputs,
            orig_size,
            projected_img_arr,
            resize_ratio,
            parameters["SCORE_THR"],
            vert_thr,
            hori_thr,
            buffer_thr,
        )
        if blobs is not None or local_dir:
            write_image_output(
                wall_overlay_image(projected_img_arr, polyhv_arr),
                f"wall_projection_{level + 1}.png",
                blobs,
                local_dir,
            )

        img_width, img_height = projected_img_arr.shape[1], projected_img_arr.shape[0]
        ori_poly = [
            [
                pixel_to_xy(
                    x,
                    img_height - y,
                    x_edges,
                    y_edges,
                    img_width,
                    img_height,
                    parameters["PROJECTED_BINS"],
                )
                for y, x in poly
            ]
            for poly in polyhv_arr
        ]
        polygons = [Polygon(row) for row in ori_poly]

        wall_segments = []
        for poly in polygons:
            inside_points = [
                [point[0], point[1]]
                for point in xy_projected
                if poly.contains(Point(point))
            ]
            if inside_points:
                wall_segments.append(
                    {
                        "polygon": poly,
                        "inside_points": inside_points,
                    }
                )

        lookup = {
            (float(x), float(y)): filtered_points[i, 2:6]
            for i, (x, y) in enumerate(filtered_points[:, :2])
        }
        pts_in_poly = [segment["inside_points"] for segment in wall_segments]
        points_zrgb = [find_zrgb(np.array(poly), lookup) for poly in pts_in_poly]

        points_xyz = [arr[:, :3] for arr in points_zrgb]
        points_rgb = [arr[:, 3:] for arr in points_zrgb]
        bbox_minmax = extract_bbox_minmax(
            points_xyz,
            blobs=blobs,
            snapshot_idx=level + 1,
            local_dir=local_dir,
        )
        edge_points = convert_to_edge_points(bbox_minmax)

        rotated_edge = [pts @ np.array(parameters["SURVEY_BASIS"]).T for pts in edge_points]
        rotated_footprints = []
        survey_basis_t = np.array(parameters["SURVEY_BASIS"]).T
        for segment in wall_segments:
            footprint_xy = np.asarray(segment["polygon"].exterior.coords[:-1], dtype=float)
            footprint_xyz = np.column_stack(
                (
                    footprint_xy[:, 0],
                    footprint_xy[:, 1],
                    np.zeros(len(footprint_xy)),
                )
            )
            rotated_footprint = footprint_xyz @ survey_basis_t
            rotated_footprints.append(rotated_footprint[:, :2])

        rotated_wall_xyz = [
            pts @ np.array(parameters["SURVEY_BASIS"]).T for pts in points_xyz
        ]
        rotated_with_rgb = [
            np.hstack((xyz, rgb)) for xyz, rgb in zip(rotated_wall_xyz, points_rgb)
        ]

        for i, bbox in enumerate(rotated_with_rgb):
            for pt in bbox:
                point_lst.append(
                    {
                        "category": "wall",
                        "id": str(wall_id),
                        "location": {
                            "x": float(pt[0]),
                            "y": float(pt[1]),
                            "z": float(pt[2]),
                        },
                        "color": {
                            "r": int(pt[3]),
                            "g": int(pt[4]),
                            "b": int(pt[5]),
                        },
                    }
                )

            wall_lst.append(
                {
                    "id": str(wall_id),
                    "levelIndex": int(level),
                    "zRange": {
                        "min": float(np.min(rotated_edge[i][:, 2])),
                        "max": float(np.max(rotated_edge[i][:, 2])),
                    },
                    "footprint": [
                        {"x": float(x), "y": float(y)}
                        for x, y in rotated_footprints[i]
                    ],
                    "bbox": [
                        {"x": float(x), "y": float(y), "z": float(z)}
                        for x, y, z in rotated_edge[i]
                    ],
                }
            )

            wall_id += 1

    wall_output_dict = {
        "points": point_lst,
        "walls": wall_lst,
    }

    written = write_json_output(wall_output_dict, "wall_output.json", blobs, local_dir)
    if written:
        logger.info("wall output -> %s", written)

    return wall_output_dict
