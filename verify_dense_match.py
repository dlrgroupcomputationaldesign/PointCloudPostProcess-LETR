"""Verify the revised dense openings path matches the experiment's log images.

Runs run_openings (dense E57 + annotation offset + fine/block-sum + longest-edge
axis) on the EXPERIMENT's own wall pickle, with a stub detector, and compares the
rendered wall_17 image to opening_detection_exp/output_log_img.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "post_process_src")

import cv2
import numpy as np

from post_process.config import coerce_parameters
from post_process.detectors.base import Detection
from post_process.stages import openings as op

EXP = Path("opening_detection_exp")
NAME = "00-10231-20_CortevaYorkTest"


class StubDetector:
    def detect(self, image_rgb):
        return []  # we only care about the rendered log image here


def main():
    wall_output = pickle.load(open(EXP / "wall_output_pickle" / f"{NAME}.pickle", "rb"))
    params = {
        "OPENING_DENSE_SOURCE_PATH": str(EXP / "e57" / f"{NAME}.e57"),
        "OPENING_ANNOTATION_CSV_PATH": str(EXP / "csv_label" / f"{NAME}.csv"),
        "OPENING_LOCAL_OUTPUT_DIR": "opening_verify_dense",
    }
    op.build_detector = lambda p: StubDetector()
    op.run_openings(None, params, wall_output, logging_blob_location=None)

    mine = cv2.imread("opening_verify_dense/opening/wall_17_log.png", cv2.IMREAD_GRAYSCALE)
    exp = cv2.imread(str(EXP / "output_log_img" / NAME / "wall_17_log.png"), cv2.IMREAD_GRAYSCALE)
    print("mine:", None if mine is None else mine.shape, " experiment:", exp.shape, flush=True)
    if mine is not None:
        a = cv2.resize(mine, (exp.shape[1], exp.shape[0]), interpolation=cv2.INTER_AREA)
        corr = np.corrcoef(a.ravel(), exp.ravel())[0, 1]
        print("wall_17 pixel correlation vs experiment: %.3f" % corr, flush=True)
    print("VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
