"""Prove (or disprove) that the harness clusters identically to the pipeline.

The experiments here reimplement the feature construction rather than calling
``cluster_floor_ceiling``, so they can only be trusted if they agree with it.
This runs BOTH on the same points and compares label-for-label.

It also isolates the one knob that genuinely changes DBSCAN's answer:
subsampling. DBSCAN is a density algorithm -- ``min_samples`` neighbours within
``eps`` -- so thinning the cloud makes every neighbourhood sparser and the same
eps behaves like a smaller one. Any harness that subsamples is NOT the pipeline.

    python verify_matches_pipeline.py                  # full-cloud equivalence
    python verify_matches_pipeline.py --sample-sizes 20000 80000 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from lab import LABELS, VARIANTS, cluster_metrics, fmt_metrics, z_profile

REPO = Path(__file__).resolve().parent.parent


def pipeline_labels(pts, col, eps, min_samples, prefer_source=True):
    """Call the REAL cluster_floor_ceiling, reconstructing labels from its output.

    cluster_floor_ceiling returns a dict of point arrays, not labels, and drops
    noise plus clusters at or below min_points. To compare against raw DBSCAN
    labels we ask it for everything (min_points=-1) and map each returned array
    back onto row indices via an exact xyz+rgb lookup.
    """
    if prefer_source and str(REPO / "post_process_src") not in sys.path:
        sys.path.insert(0, str(REPO / "post_process_src"))
    import pandas as pd
    from post_process.utils.floor_ceiling_util import cluster_floor_ceiling
    import post_process.utils.floor_ceiling_util as fcu

    df = pd.DataFrame(
        np.hstack([pts, col * 255.0]), columns=["x", "y", "z", "r", "g", "b"]
    )
    # min_points=-1 keeps every cluster, so the comparison sees DBSCAN's own
    # output rather than the pipeline's post-filter.
    out = cluster_floor_ceiling(df, eps, min_samples, type="floor",
                                blobs=None, min_points=-1)

    key = {}
    for i, row in enumerate(np.hstack([pts, col * 255.0])):
        key[tuple(np.round(row, 9))] = i

    labels = np.full(len(pts), -1, dtype=int)
    for lab, arr in out.items():
        for row in arr:
            j = key.get(tuple(np.round(row, 9)))
            if j is not None:
                labels[j] = lab
    return labels, Path(fcu.__file__)


def agreement(a, b):
    """Fraction of point PAIRS that both labellings group the same way.

    Cluster ids are arbitrary, so labels cannot be compared directly. The
    adjusted Rand index compares partitions, which is what actually matters.
    """
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="laramie")
    ap.add_argument("--kind", default="Floor", choices=["Floor", "Ceiling", "Wall"])
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--sample-sizes", type=int, nargs="+",
                    default=[20000, 80000, 0],
                    help="0 means the full cloud (what the pipeline uses)")
    args = ap.parse_args()

    from lab import load
    from sklearn.cluster import DBSCAN

    print(f"=== {args.dataset}/{args.kind}  eps={args.eps} "
          f"min_samples={args.min_samples} ===\n")

    # --- part 1: same points, both implementations -------------------------
    pts, col = load(args.dataset, args.kind, sample=20000)
    feats, _ = VARIANTS["current"](pts, col)
    mine = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(feats)
    theirs, src = pipeline_labels(pts, col, args.eps, args.min_samples)

    ari = agreement(mine, theirs)
    identical = bool((mine == theirs).all())
    print(f"[equivalence] pipeline code: {src}")
    print(f"[equivalence] adjusted Rand index = {ari:.6f}"
          f"   labels identical: {identical}")
    print("  1.000000 means the harness partitions points exactly as the "
          "pipeline does.\n" if ari > 0.9999 else
          "  BELOW 1.0 -- the harness does NOT reproduce the pipeline.\n")

    # --- part 2: what subsampling costs ------------------------------------
    print("[subsampling] same eps, different point counts:")
    full_pts, full_col = load(args.dataset, args.kind, sample=None, verbose=False)
    n_storeys, _, _ = z_profile(full_pts)
    print(f"  full cloud is {len(full_pts):,} points, {n_storeys} storey(s)\n")

    for n in args.sample_sizes:
        p, c = (full_pts, full_col) if n == 0 else load(
            args.dataset, args.kind, sample=n, verbose=False)
        f, _ = VARIANTS["current"](p, c)
        lab = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(f)
        # Scale the keep-threshold with the cloud so 'kept' stays comparable.
        min_points = 10000 if n == 0 else max(10, int(10000 * n / len(full_pts)))
        m = cluster_metrics(lab, p, min_points)
        tag = "FULL (pipeline)" if n == 0 else f"sample {n:,}"
        print(f"  {tag:18} {fmt_metrics(m)}")


if __name__ == "__main__":
    main()
