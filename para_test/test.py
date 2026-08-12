from post_process import post_process
import pandas as pd

internal_params = {
    "EPS_F": 0.5,
    "MIN_SAMPLES_F": 10,
    "DIS_THR_F": 0.1,
    "RANSAC_N_F": 3,
    "NUM_ITER_F": 1000,
    "ALPHA_F": 0.05,
    "BOUNDARY_METHOD_F": "alphashape",
    "BOUNDARY_CELL_F": 0.25,
    "BOUNDARY_FILL_GAP_F": 0.8,
    "BOUNDARY_SIMPLIFY_EPS_FRAC_F": 0.02,
}
df_pred = pd.read_csv("src\\laram_inference_prediction.csv")  # from csv
floor_output, floor_bboxz, floor_level = post_process.run_floors(
                df_pred, 
                internal_params, 
            )

print(floor_output)