import json

import pandas as pd

# Canonical import: this is the name pip-installers get (after `pip install -e .`).
from post_process import run_post_process
from survey_basis import survey_basis_from_df, survey_basis_from_e57

e57 = "Ty_50mm.e57"
csv_path = r"src\ty_inference_prediction.csv"
line_seg_model = r"checkpoints\checkpoint0024.pth"

# Where to estimate SURVEY_BASIS from:
#   "csv" -- uses pred_label == Wall. Instant (the frame is already in memory).
#   "e57" -- uses a mid-height slab of the raw scan. Slower (streams the whole
#            file, ~1-2 min for 2.7 GB) but more accurate: on Laramie it landed
#            0.028 deg from the registered basis vs the CSV's 0.064 deg, because
#            95M raw points beat 692k segmented ones for normal estimation.
# Both must describe the SAME building as csv_path.
SURVEY_BASIS_SOURCE = "e57"

# Read the cloud first: SURVEY_BASIS is derived from it rather than supplied.
df = pd.read_csv(csv_path)

# Recover the building's rotation from the cloud itself, so no externally
# registered survey basis is needed. See survey_basis.py.
print(f"estimating SURVEY_BASIS from the {SURVEY_BASIS_SOURCE}...")
if SURVEY_BASIS_SOURCE == "e57":
    survey_basis = survey_basis_from_e57(e57)
else:
    survey_basis = survey_basis_from_df(df)
print("SURVEY_BASIS (row-major):", survey_basis)
# --- Opening detection (Grounding DINO) ---------------------------------------
# Using the HuggingFace transformers port (no CUDA build needed). The default
# checkpoint IDEA-Research/grounding-dino-tiny is the same Swin-T OGC model as
# groundingdino_swint_ogc.pth and downloads automatically on first run.
# Requires: pip install "transformers>=4.40" pillow
parameters = {
    # DBSCAN
    "EPS": 0.5,
    "MIN_SAMPLES": 10,

    # Floor RANSAC
    "DIS_THR_F": 1,
    "RANSAC_N_F": 10, 
    "NUM_ITER_F": 1000, 
    "ALPHA_F": 1, #2
    
    # Ceiling RANSAC
    "DIS_THR_C": 4,  #1
    "RANSAC_N_C": 10, 
    "NUM_ITER_C": 1000, 
    "ALPHA_C": 1,

    "PROJECTED_BINS": 300,
    "RESIZE_WIDTH": 400,
    "INT_THR": 35,     #20
    "SCORE_THR": 0.55,  #0.6
    "VERT_THR": 10,
    "HORI_THR": 0.1,
    "BUFFER_THR": 2,

    # 3x3 rotation matrix, estimated above from the cloud's wall normals.
    "SURVEY_BASIS": survey_basis,

    # --- Opening detection ---
    "OPENINGS_ENABLED": True,
    "OPENING_DETECTOR": "grounding_dino_hf",  # HF port (no CUDA build)
    # No blob here, so write log + annotated detection images locally for inspection.
    "OPENING_LOCAL_OUTPUT_DIR": "opening_output",
    # Dense E57 path (matches the experiment's log images): stream the E57 and
    # align it to the walls using xyz_min from the original annotation CSV.
    # Must match csv_path's building, or openings get aligned to the wrong cloud.
    "OPENING_DENSE_SOURCE_PATH": e57,
    "OPENING_ANNOTATION_CSV_PATH": r"opening_detection_exp\csv_label\WyomingStateFair_Laramie.csv",
    "OPENING_DENSE_WALL_MARGIN": 0.25,  # m margin around the wall to include points
    }

from datetime import datetime
import uuid
from post_process.utils.coordinate_util import to_original_coordinates_all

def generate_job_name(prefix="postporcess"):
    date_str = datetime.utcnow().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:6]
    return f"{prefix}_{date_str}_{short_id}"

logging_blob_location = {
    "account_name": "pointcloudapistorage",
    "account_key": "yUMEWqPMmETNPCiffgPAdPkPMcCWWDfvV3UOY9ZgUKKRCXzt54kqpuqyb1W9cDwT88NOSmlls6oB+AStXFrmSg==",
    "container_name": "jobs",
    "folder": generate_job_name(),
}

print(f"\nrunning post-process -> blob folder {logging_blob_location['folder']}")
result = run_post_process(
    df,
    parameters,
    line_seg_model,
    logging_blob_location=logging_blob_location,
    include_openings=True,
)

print(
    "openings -> doors={} windows={} openings={}".format(
        len(result.get("doors", [])),
        len(result.get("windows", [])),
        len(result.get("openings", [])),
    )
)
# with open("all_output.json", "w") as f:
#     json.dump(result, f, indent=2)
# print("wrote all_output.json")

# from json_output_visualization import (
#     create_e57_wall_length_z_image,
#     export_e57_wall_points_with_log_image_html,
#     export_wall_from_pickle_mapped_to_e57_html,
#     plot_e57_wall_length_z_image,
#     export_e57_wall_crop_rgb_html
# )

# e57_result = create_e57_wall_length_z_image(
#     "wall_output.pickle",
#     "LaramieCM.e57",
#     wall_id=17,
#     length_bin_size=0.05,
#     z_bin_size=0.05,
#     count_threshold=10,  # dense E57 needs higher threshold than sparse CSV
#     margin_m=0.25,
# )

# fig = plot_e57_wall_length_z_image(
#     e57_result,
#     wall_id=17,
#     save_path="wall_17_e57_length_z_image_bbox.png",
# )


# fig = export_e57_wall_points_with_log_image_html(
#     e57_result,
#     e57_path="LaramieCM.e57",
#     wall_id=17,
#     output_html="wall_17_e57_points_rgb_log_attached_clear.html",
#     max_points=800000,
#     log_normal_offset=-0.2,
#     log_opacity=0.35,
#     point_size=2.0,
#     point_opacity=1.0,
#     point_color_mode="rgb",
#     scene_background_color="rgb(20,20,20)",
# )
# result = export_e57_wall_crop_rgb_html(
#     wall_output_pickle="wall_output.pickle",
#     e57_path="LaramieCM.e57",
#     output_html="wall_17_original_e57_crop_rgb.html",
#     wall_id=17,
#     margin_m=0.35,
#     max_html_points=500_000,
#     marker_size=1.5,
#     display_unit="m",
#     show_wall_outline=True,
# )


# fig = export_wall_from_pickle_mapped_to_e57_html(
#     "wall_output.pickle",
#     "LaramieCM.e57",
#     wall_id=17,
#     output_html="wall_17_e57_points.html",
#     ft_to_m=0.3048,
# )

# fig.show()

# floor_bboxz: [[zmin,zmax],[zmin,zmax],...]
# print('Starting floor post-processing...')
# floor_output, floor_bboxz, floor_level = post_process_src.post_process.run_floors(df, parameters)
# print('Starting ceiling post-processing...')
# ceiling_output, ceiling_level = post_process_src.post_process.run_ceilings(df, parameters)
# print('Starting wall post-processing...')
# wall_output = post_process_src.post_process.run_walls(df, parameters, floor_bboxz, line_seg_model)
# print('Starting all post-processing...')
# all_output = post_process_src.post_process.final_output(floor_output, ceiling_output, wall_output, floor_level, ceiling_level)
# print('all output generated.', all_output)

# floor_output, floor_bboxz, floor_level = post_process_src.post_process.run_floors(df, parameters, logging_blob_location=logging_blob_location)
# ceiling_output, ceiling_level = post_process_src.post_process.run_ceilings(df, parameters, logging_blob_location=logging_blob_location)
# wall_output = post_process_src.post_process.run_walls(df, parameters, floor_bboxz, line_seg_model, logging_blob_location=logging_blob_location)
# all_output = post_process_src.post_process.final_output(floor_output, ceiling_output, wall_output, floor_level, ceiling_level, logging_blob_location=logging_blob_location)
