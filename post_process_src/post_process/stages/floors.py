import numpy as np
import pandas as pd

from ..config import coerce_parameters
from ..labels import label_mask
from ..runtime import logger
from ..utils.blob_util import upload_dict_to_blob_json
from ..utils.floor_ceiling_util import cluster_floor_ceiling, fit_ceiling_floor, point_axis_align
from .common import get_parameter, make_blob_factory, scalar_mode


def run_floors(df, parameters, logging_blob_location=None):
    parameters = coerce_parameters(parameters)
    blobs = make_blob_factory(logging_blob_location)

    logger.info("Running floor post-processing...")
    floor_lst, floor_level, point_lst = [], [], []
    segment_floor_df = df[label_mask(df, "Floor", parameters)].reset_index(drop=True)

    align_axis_floor_df = point_axis_align(
        segment_floor_df,
        np.array(parameters["SURVEY_BASIS"]).T,
    )

    cluster_dict_floor = cluster_floor_ceiling(
        align_axis_floor_df,
        get_parameter(parameters, "EPS_F", "EPS"),
        get_parameter(parameters, "MIN_SAMPLES_F", "MIN_SAMPLES"),
        type="floor",
        blobs=blobs,
    )

    floor_bboxz = []
    floor_id = 1

    logger.info("Number of floor clusters: {}".format(len(cluster_dict_floor)))
    for i, num in enumerate(cluster_dict_floor):
        df_points_colors = pd.DataFrame(
            cluster_dict_floor[num],
            columns=["x", "y", "z", "r", "g", "b"],
        )

        bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(
            df_points_colors,
            parameters["DIS_THR_F"],
            parameters["RANSAC_N_F"],
            parameters["NUM_ITER_F"],
            parameters["ALPHA_F"],
            logger,
            type="floor",
            blobs=blobs,
            snapshot_idx=i + 1,
        )

        floor_bboxz.append([bbox_zmin, bbox_zmax])

        rotated_corner = corner_xyz @ np.array(parameters["SURVEY_BASIS"]).T

        cluster_xyz_arr = cluster_dict_floor[num][:, :3]
        cluster_rgb_arr = cluster_dict_floor[num][:, 3:]
        rotated_xyz = cluster_xyz_arr @ np.array(parameters["SURVEY_BASIS"]).T
        rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))

        z_values = [pt[2] for pt in rotated_with_rgb]
        mode_z = scalar_mode(z_values)

        for pt in rotated_with_rgb:
            point_lst.append(
                {
                    "category": "floor",
                    "id": str(floor_id),
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

        floor_level.append(mode_z)

        floor_lst.append(
            {
                "id": str(floor_id),
                "edgePoints": [
                    {"x": float(x), "y": float(y), "z": float(z)}
                    for x, y, z in rotated_corner
                ],
            }
        )
        floor_id += 1

    floor_output_dict = {
        "points": point_lst,
        "floors": floor_lst,
    }

    if blobs:
        upload_dict_to_blob_json(floor_output_dict, blobs("floor_output.json"))

    return floor_output_dict, floor_bboxz, floor_level
