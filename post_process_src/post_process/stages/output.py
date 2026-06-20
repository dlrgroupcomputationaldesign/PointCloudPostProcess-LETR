from ..runtime import logger
from ..utils.blob_util import (
    upload_dict_to_blob_json,
    upload_logger_to_blob,
)
from ..utils.coordinate_util import to_original_coordinates
from .common import make_blob_factory


def final_output(
    floor_output,
    ceiling_output,
    wall_output,
    floor_level,
    ceiling_level,
    opening_output=None,
    logging_blob_location=None,
    point_cloud_offset=None,
    point_cloud_scale=None,
):
    """Compile floors/ceilings/walls/openings into one output dict.

    If ``point_cloud_offset`` is given, all geometry is converted from the
    post-process frame back to the original cloud's coordinates via
    ``original = (post + offset) / scale`` (``point_cloud_scale`` defaults to the
    feet-per-meter constant). Leave it None to keep post-process coordinates.
    """
    blobs = make_blob_factory(logging_blob_location)

    logger.info("Compiling final output...")
    level_lst = []
    level_id = 1
    level = floor_level + ceiling_level
    for mode_z in level:
        level_lst.append(
            {
                "id": str(level_id),
                "zMode": mode_z,
            }
        )
        level_id += 1

    opening_points = []
    if opening_output:
        opening_points = opening_output.get("points", [])

    final_output_dict = {
        "points": (
            floor_output["points"]
            + ceiling_output["points"]
            + wall_output["points"]
            + opening_points
        ),
        "levels": level_lst,
        "floors": floor_output["floors"],
        "ceilings": ceiling_output["ceilings"],
        "walls": wall_output["walls"],
    }

    if opening_output:
        final_output_dict["doors"] = opening_output.get("doors", [])
        final_output_dict["windows"] = opening_output.get("windows", [])
        final_output_dict["openings"] = opening_output.get("openings", [])

    if point_cloud_offset is not None:
        scale = 3.280839895013123 if point_cloud_scale is None else point_cloud_scale
        logger.info(
            "Converting output to original coordinates (scale={}, offset={})".format(
                scale, list(point_cloud_offset)
            )
        )
        final_output_dict = to_original_coordinates(
            final_output_dict, scale, point_cloud_offset
        )

    if blobs:
        upload_logger_to_blob(logger, blobs("postprocess_log.txt"))
        upload_dict_to_blob_json(final_output_dict, blobs("all_output.json"))

    return final_output_dict
