"""Find the OPENING_IMAGE_VMAX_SCALE that makes the package's wall_17 render
match opening_detection_exp/output_log_img/WyomingStateFair_Laramie/wall_17.

Streams the Laramie E57 once (accumulates only wall 17), then renders at several
vmax_scale values (clip_pct=None baseline) and compares brightness + correlation
to the reference image.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, "post_process_src")

import cv2
import numpy as np

from post_process.config import coerce_parameters
from post_process.stages import openings as op
from post_process.utils.opening_image_util import downsample_sum, render_log_image

EXP = Path("opening_detection_exp")
NAME = "WyomingStateFair_Laramie"
WALL_ID = "17"
REF = EXP / "output_log_img" / NAME / f"wall_{WALL_ID}_log.png"
SCALES = [1.0, 1.5, 1.8, 2.0, 2.5, 3.0]


def compare(pkg_gray, ref_gray):
    a = cv2.resize(pkg_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    corr = max(
        float(np.corrcoef(a.ravel(), ref_gray.ravel())[0, 1]),
        float(np.corrcoef(np.fliplr(a).ravel(), ref_gray.ravel())[0, 1]),
    )
    return a.mean(), corr


def main():
    params = {
        "OPENING_DENSE_SOURCE_PATH": str(EXP / "e57" / f"{NAME}.e57"),
        "OPENING_ANNOTATION_CSV_PATH": str(EXP / "csv_label" / f"{NAME}.csv"),
        "OPENING_IMAGE_CLIP_PCT": None,   # match the log_0.05 full-range baseline
    }
    cp = coerce_parameters(params)
    cp = op._resolve_e57_offset(cp)

    walls = pickle.load(open(EXP / "wall_output_pickle" / f"{NAME}.pickle", "rb"))["walls"]
    wall = next(w for w in walls if str(w["id"]) == WALL_ID)

    margin = float(cp["OPENING_DENSE_WALL_MARGIN"]) * op._units_per_meter(cp)
    accs = op._build_accumulators([wall], cp, margin=margin)
    print("streaming Laramie E57 for wall 17 ...", flush=True)
    op._accumulate_from_dense(accs, cp["OPENING_DENSE_SOURCE_PATH"], cp)

    counts = accs[0]["counts"]
    render = downsample_sum(counts, op._render_bin_factor(cp))
    print("wall 17 render grid:", render.shape, " points:", accs[0]["point_count"], flush=True)

    ref = cv2.imread(str(REF), cv2.IMREAD_GRAYSCALE).astype(float)
    print(f"reference mean={ref.mean():.1f}\n", flush=True)

    outdir = Path("vmax_calib"); outdir.mkdir(exist_ok=True)
    print(f"{'scale':>6} {'pkg_mean':>9} {'ref_mean':>9} {'corr':>6}", flush=True)
    for scale in SCALES:
        p = dict(cp); p["OPENING_IMAGE_VMAX_SCALE"] = scale
        img = render_log_image(render, p)
        cv2.imwrite(str(outdir / f"wall17_vmax{scale}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        mean, corr = compare(img[:, :, 0].astype(float), ref)
        print(f"{scale:>6} {mean:>9.1f} {ref.mean():>9.1f} {corr:>6.3f}", flush=True)
    print("\nsaved renders to vmax_calib/  (pick the scale whose pkg_mean ~ ref_mean)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
