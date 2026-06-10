import pickle

from .config import coerce_parameters
from .stages.ceilings import run_ceilings
from .stages.floors import run_floors
from .stages.openings import run_openings
from .stages.output import final_output
from .stages.walls import run_walls


def run_post_process(
    df,
    parameters,
    line_seg_model,
    logging_blob_location=None,
    include_openings=False,
):
    parameters = coerce_parameters(parameters)

    floor_output, floor_bboxz, floor_level = run_floors(
        df,
        parameters,
        logging_blob_location=logging_blob_location,
    )
    ceiling_output, ceiling_level = run_ceilings(
        df,
        parameters,
        logging_blob_location=logging_blob_location,
    )
    wall_output = run_walls(
        df,
        parameters,
        floor_bboxz,
        line_seg_model,
        logging_blob_location=logging_blob_location,
    )

    # with open('wall_output.pickle', 'wb') as f:
    #     pickle.dump(wall_output, f)
                
    opening_output = None
    if include_openings or parameters.get("OPENINGS_ENABLED", False):
        opening_output = run_openings(
            df,
            parameters,
            wall_output,
            logging_blob_location=logging_blob_location,
        )

    return final_output(
        floor_output,
        ceiling_output,
        wall_output,
        floor_level,
        ceiling_level,
        opening_output=opening_output,
        logging_blob_location=logging_blob_location,
    )
