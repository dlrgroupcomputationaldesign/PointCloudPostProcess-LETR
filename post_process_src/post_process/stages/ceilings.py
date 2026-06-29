import numpy as np
import pandas as pd

from ..config import coerce_parameters
from ..labels import label_mask
from ..runtime import logger
from ..utils.blob_util import upload_dict_to_blob_json
from ..utils.floor_ceiling_util import cluster_floor_ceiling, fit_ceiling_floor, point_axis_align
from .common import get_parameter, make_blob_factory, scalar_mode


def run_ceilings(df, parameters, logging_blob_location=None):
    parameters = coerce_parameters(parameters)
    blobs = make_blob_factory(logging_blob_location)

    logger.info("Running ceiling post-processing...")
    ceiling_lst, ceiling_level, point_lst = [], [], []
    segment_ceiling_df = df[label_mask(df, "Ceiling", parameters)].reset_index(drop=True)
    align_axis_ceiling_df = point_axis_align(
        segment_ceiling_df,
        np.array(parameters["SURVEY_BASIS"]).T,
    )
    cluster_dict_ceiling = cluster_floor_ceiling(
        align_axis_ceiling_df,
        get_parameter(parameters, "EPS_C", "EPS"),
        get_parameter(parameters, "MIN_SAMPLES_C", "MIN_SAMPLES"),
        type="ceiling",
        blobs=blobs,
    )
    ceiling_id = 1

    logger.info("Number of ceiling clusters: {}".format(len(cluster_dict_ceiling)))
    for i, num in enumerate(cluster_dict_ceiling):
        df_points_colors = pd.DataFrame(
            cluster_dict_ceiling[num],
            columns=["x", "y", "z", "r", "g", "b"],
        )
        bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(
            df_points_colors,
            parameters["DIS_THR_C"],
            parameters["RANSAC_N_C"],
            parameters["NUM_ITER_C"],
            parameters["ALPHA_C"],
            logger,
            type="ceiling",
            blobs=blobs,
            snapshot_idx=i + 1,
            boundary_opts={
                "method": parameters.get("BOUNDARY_METHOD_C", "alphashape"),
                "cell": parameters.get("BOUNDARY_CELL_C", 0.25),
                "fill_gap": parameters.get("BOUNDARY_FILL_GAP_C", 0.8),
                "simplify_eps_frac": parameters.get("BOUNDARY_SIMPLIFY_EPS_FRAC_C", 0.02),
            },
        )

        rotated_corner = corner_xyz @ np.array(parameters["SURVEY_BASIS"]).T

        cluster_xyz_arr = cluster_dict_ceiling[num][:, :3]
        cluster_rgb_arr = cluster_dict_ceiling[num][:, 3:]
        rotated_xyz = cluster_xyz_arr @ np.array(parameters["SURVEY_BASIS"]).T
        rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))
        z_values = [pt[2] for pt in rotated_with_rgb]
        mode_z = scalar_mode(z_values)

        for pt in rotated_with_rgb:
            point_lst.append(
                {
                    "category": "ceiling",
                    "id": str(ceiling_id),
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

        ceiling_level.append(mode_z)

        ceiling_lst.append(
            {
                "id": str(ceiling_id),
                "edgePoints": [
                    {"x": float(x), "y": float(y), "z": float(z)}
                    for x, y, z in rotated_corner
                ],
            }
        )
        ceiling_id += 1

    ceiling_output_dict = {
        "points": point_lst,
        "ceilings": ceiling_lst,
    }

    if blobs:
        upload_dict_to_blob_json(ceiling_output_dict, blobs("ceiling_output.json"))

    return ceiling_output_dict, ceiling_level
