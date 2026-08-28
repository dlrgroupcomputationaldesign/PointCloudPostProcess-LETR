"""Smallest possible run of the package: a CSV in, geometry out.

    python para_test/run_minimal.py
    python para_test/run_minimal.py src/laram_inference_prediction.csv --walls

Everything is defaulted -- survey basis estimated from the cloud's own wall
points, histogram clustering, IRLS plane fit, raster boundary with coverage
merging, Otsu wall thresholding. Contrast with test.py, which spells out every
parameter so they can be tuned and A/B'd; this is what a caller actually needs.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "post_process_src"))

from post_process import post_process  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("csv", nargs="?", default="src/laram_inference_prediction.csv")
ap.add_argument("--out", default=None, help="snapshot dir (default: para_test_output/<stem>_minimal)")
ap.add_argument("--walls", action="store_true", help="also run LETR line segmentation (slow)")
ap.add_argument("--model", default="checkpoints/checkpoint0024.pth")
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

if not args.quiet:
    # The package logs to "Infer" with propagate=False and an in-memory handler
    # bound for blob storage, so nothing reaches the console without this.
    log = logging.getLogger("Infer")
    log.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("   [pp] %(message)s"))
    log.addHandler(h)

csv = Path(args.csv) if Path(args.csv).is_absolute() else REPO / args.csv
out = Path(args.out) if args.out else REPO / "para_test_output" / f"{csv.stem}_minimal"
out.mkdir(parents=True, exist_ok=True)

print(f"reading {csv.name}")
df = pd.read_csv(csv, low_memory=False)

# The only parameter: where snapshots go. Omit it and they go nowhere (the
# package writes to blob storage when a logging_blob_location is passed).
params = {"LOCAL_OUTPUT_DIR": str(out)}

floors, floor_bboxz, floor_levels = post_process.run_floors(df, params)
print(f"\n{len(floors['floors'])} floor(s)   levels {[round(v, 2) for v in floor_levels]}")
for i, (f, (zmin, zmax)) in enumerate(zip(floors["floors"], floor_bboxz), 1):
    print(f"   floor {i}: z {zmin:7.2f} .. {zmax:7.2f}   {len(f['edgePoints']):4d} edge points")

ceilings, ceiling_levels = post_process.run_ceilings(df, params)
print(f"\n{len(ceilings['ceilings'])} ceiling(s)   levels {[round(v, 2) for v in ceiling_levels]}")
for i, c in enumerate(ceilings["ceilings"], 1):
    print(f"   ceiling {i}: {len(c['edgePoints']):4d} edge points")

if args.walls:
    model = REPO / args.model
    if not model.exists():
        print(f"\n{args.model} not found -- skipping walls")
    else:
        print(f"\nrunning walls on {len(floor_bboxz)} level(s) (slow)...")
        walls = post_process.run_walls(df, params, floor_bboxz, str(model))
        print(f"{len(walls['walls'])} wall(s)")

print(f"\nsnapshots -> {out}")
