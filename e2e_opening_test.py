"""End-to-end smoke test for the opening-detection stage on real data.

Runs the real floors->ceilings->walls pipeline on the CortevaYork CSV, then
runs run_openings with a STUB detector (groundingdino weights aren't available
in this env) so every piece of new non-ML code is exercised against real walls.
Also saves the rendered log images locally for visual inspection.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "post_process_src")

import cv2
import pandas as pd

from post_process import run_ceilings, run_floors, run_walls
from post_process.config import coerce_parameters
from post_process.detectors.base import Detection
from post_process.stages import openings as op
from post_process.utils.opening_image_util import render_log_image

CSV_PATH = r"src/00-10231-20_CortevaYorkTest_Output.csv"
LINE_SEG_MODEL = r"checkpoints\checkpoint0024.pth"

# Mirror run.py's parameters.
PARAMETERS = {
    "EPS": 0.5,
    "MIN_SAMPLES": 10,
    "DIS_THR_F": 1,
    "RANSAC_N_F": 10,
    "NUM_ITER_F": 1000,
    "ALPHA_F": 1,
    "DIS_THR_C": 4,
    "RANSAC_N_C": 10,
    "NUM_ITER_C": 1000,
    "ALPHA_C": 1,
    "PROJECTED_BINS": 300,
    "RESIZE_WIDTH": 400,
    "INT_THR": 35,
    "SCORE_THR": 0.55,
    "VERT_THR": 10,
    "HORI_THR": 0.1,
    "BUFFER_THR": 2,
    "SURVEY_BASIS": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
}


class StubDetector:
    """Returns one plausible lower-center 'door' box per image to exercise mapping."""

    def detect(self, image_rgb):
        h, w = image_rgb.shape[:2]
        return [
            Detection(
                label="door",
                score=0.8,
                box_xyxy=(w * 0.40, h * 0.45, w * 0.60, h * 0.99),
            )
        ]


def main():
    df = pd.read_csv(CSV_PATH)
    print("rows: {}  cols: {}".format(len(df), list(df.columns)), flush=True)

    print("running floors...", flush=True)
    floor_output, floor_bboxz, floor_level = run_floors(df, PARAMETERS)
    print("running ceilings...", flush=True)
    ceiling_output, ceiling_level = run_ceilings(df, PARAMETERS)
    print("running walls...", flush=True)
    wall_output = run_walls(df, PARAMETERS, floor_bboxz, LINE_SEG_MODEL)
    print("walls detected: {}".format(len(wall_output["walls"])), flush=True)
    with open("wall_output.pickle", "wb") as f:
        pickle.dump(wall_output, f)

    # Swap in the stub so run_openings runs without groundingdino/weights.
    op.build_detector = lambda params: StubDetector()

    opening_output = op.run_openings(
        df,
        PARAMETERS,
        wall_output,
        floor_level,
        ceiling_level,
        logging_blob_location=None,
    )
    print(
        "openings -> doors={} windows={} openings={}".format(
            len(opening_output["doors"]),
            len(opening_output["windows"]),
            len(opening_output["openings"]),
        ),
        flush=True,
    )
    if opening_output["doors"]:
        d = opening_output["doors"][0]
        print(
            "sample door: wall={} class={} conf={:.2f} w={:.2f} h={:.2f} "
            "bottomZ={:.2f} topZ={:.2f} bbox_corners={}".format(
                d["wallId"], d["class"], d["confidence"], d["width"], d["height"],
                d["bottomZ"], d["topZ"], len(d["bbox"]),
            ),
            flush=True,
        )

    # Save the real rendered log images locally (what the detector would consume).
    cp = coerce_parameters(PARAMETERS)
    outdir = Path("opening_test_output")
    outdir.mkdir(exist_ok=True)
    accumulators = op._build_accumulators(wall_output["walls"], cp)
    wall_points = df[op.label_mask(df, "Wall", cp)][["x", "y", "z"]].to_numpy(float)
    op._accumulate_from_df(accumulators, wall_points, cp)
    saved = 0
    for acc in accumulators:
        if acc["point_count"] < cp["OPENING_MIN_WALL_POINTS"]:
            continue
        image_rgb = render_log_image(acc["counts"], cp)
        out_path = outdir / "wall_{}_log.png".format(acc["wall"]["id"])
        cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        saved += 1
    print("saved {} log images to {}".format(saved, outdir), flush=True)
    print("E2E_OK", flush=True)


if __name__ == "__main__":
    main()
