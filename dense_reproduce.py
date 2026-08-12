"""Reproduce the experiment's dense per-wall log images through the package.

Streams the CortevaYork E57 (meters) into the openings stage with the correct
feet<->meter alignment, renders each wall at 0.05 m bins, and saves the images
so they can be compared against opening_detection_exp/output_log_img.
Uses the already-produced wall_output.pickle and a stub detector.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "post_process_src")

import cv2
import numpy as np
import pandas as pd

from post_process.config import coerce_parameters
from post_process.detectors.base import Detection
from post_process.stages import openings as op
from post_process.utils.opening_image_util import render_log_image

E57_PATH = r"opening_detection_exp/e57/00-10231-20_CortevaYorkTest.e57"
LABEL_CSV = r"opening_detection_exp/csv_label/00-10231-20_CortevaYorkTest.csv"
FT_TO_M = 0.3048


def compute_xyz_min_ft(path, chunk=500_000):
    cols = ["X", "Y", "Z"]
    mn = np.array([np.inf, np.inf, np.inf])
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunk):
        mn = np.minimum(mn, ch[cols].to_numpy(float).min(axis=0))
    return mn


class StubDetector:
    def detect(self, image_rgb):
        h, w = image_rgb.shape[:2]
        return [Detection("door", 0.8, (w * 0.40, h * 0.45, w * 0.60, h * 0.99))]


def main():
    xyz_min_ft = compute_xyz_min_ft(LABEL_CSV)
    offset_m = (xyz_min_ft * FT_TO_M).tolist()
    print("xyz_min_ft:", np.round(xyz_min_ft, 3), "-> E57_TO_CSV_OFFSET (m):",
          np.round(offset_m, 3), flush=True)

    params = {
        "SURVEY_BASIS": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "OPENING_DENSE_SOURCE_PATH": E57_PATH,
        "OPENING_E57_TO_CSV_OFFSET": offset_m,
        # E57_TO_CSV_SCALE stays default 3.2808 (also the units-per-meter factor)
        "OPENING_IMAGE_BIN_M": 0.05,
        "OPENING_IMAGE_CELL_PX": 4,
        "OPENING_DENSE_WALL_MARGIN": 0.25,
        "OPENING_E57_CHUNK_SIZE": 2_000_000,
    }
    cp = coerce_parameters(params)

    walls = pickle.load(open("wall_output.pickle", "rb"))["walls"]
    print("walls:", len(walls), flush=True)

    margin = float(cp["OPENING_DENSE_WALL_MARGIN"]) * op._units_per_meter(cp)
    accumulators = op._build_accumulators(walls, cp, margin=margin)
    print("streaming E57 (this takes a few minutes)...", flush=True)
    op._accumulate_from_e57(accumulators, E57_PATH, cp)

    op.build_detector = lambda p: StubDetector()
    detector = StubDetector()

    outdir = Path("opening_test_output_dense")
    outdir.mkdir(exist_ok=True)
    saved = 0
    candidates = []
    for acc in accumulators:
        if acc["point_count"] < cp["OPENING_MIN_WALL_POINTS"]:
            continue
        image_rgb = render_log_image(acc["counts"], cp)
        cv2.imwrite(
            str(outdir / "wall_{}_log.png".format(acc["wall"]["id"])),
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        )
        saved += 1
        candidates.extend(
            op._detect_wall(
                acc["wall"], acc["frame"], acc["counts"], acc["point_count"],
                detector, cp, None,
            )
        )

    # report a few grid shapes to confirm meter binning (expect ~88 x ~294 for wall 17)
    for acc in accumulators:
        if str(acc["wall"]["id"]) == "17":
            print("wall 17 grid (n_z, n_s):", acc["counts"].shape,
                  "points:", acc["point_count"], flush=True)
    print("saved {} dense images to {}".format(saved, outdir), flush=True)
    print("candidates:", len(candidates), flush=True)
    print("DENSE_OK", flush=True)


if __name__ == "__main__":
    main()
