"""Sweep eps for each feature variant and score the resulting clusters.

The k-distance study says where the knee is; this says whether clustering there
actually produces good slabs. Scoring follows what the pipeline does with the
result (see lab.cluster_metrics): clusters larger than min_points survive, and
each is RANSAC-fitted as one plane -- so clusters should be FEW and THIN in z.

A thick cluster means two storeys merged; a huge cluster count means one slab
shattered. Both are failures the current single default cannot distinguish.

    python dbscan_experiment/sweep.py
    python dbscan_experiment/sweep.py --variants current physical --sample 40000
"""

import argparse

import numpy as np

from lab import DATASETS, OUT, VARIANTS, cluster_metrics, fmt_metrics, load, z_profile

# eps ranges differ per variant because eps means a different thing in each:
# std devs, multiples of RMS radius, or feet.
DEFAULT_GRIDS = {
    "current": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
    "xyz_zscore": [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    "isotropic": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    "physical": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--kind", default="Floor", choices=["Floor", "Ceiling", "Wall"])
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--sample", type=int, default=40000,
                    help="subsample size; DBSCAN is O(n log n) at best")
    ap.add_argument("--eps", type=float, nargs="+", default=None,
                    help="override the per-variant eps grid")
    ap.add_argument("--min-points-frac", type=float, default=0.0367,
                    help="min cluster size as a fraction of points, so the "
                         "pipeline's 10000-of-272000 threshold scales with "
                         "the subsample (default 10000/272000)")
    args = ap.parse_args()

    from sklearn.cluster import DBSCAN

    for ds in args.datasets:
        print(f"\n{'=' * 78}\n=== {ds} / {args.kind} ===")
        pts, col = load(ds, args.kind, sample=args.sample)
        min_points = max(10, int(args.min_points_frac * len(pts)))
        print(f"  min cluster size for 'kept': {min_points:,}")

        n_storeys, peak_z, min_gap = z_profile(pts)
        print(f"  z-profile: {n_storeys} storey(s) at z={[round(p, 2) for p in peak_z]}"
              f"   smallest gap {min_gap:.2f}")
        print(f"  -> a correct clustering keeps ~{n_storeys} cluster(s); "
              f"eps must stay below the gap or storeys merge")

        for vname in args.variants:
            feats, desc = VARIANTS[vname](pts, col)
            print(f"\n  [{vname}] {desc}")
            grid = args.eps if args.eps else DEFAULT_GRIDS[vname]
            for eps in grid:
                labels = DBSCAN(eps=eps, min_samples=args.min_samples).fit_predict(feats)
                m = cluster_metrics(labels, pts, min_points)
                # Judge against the z-profile, not against a fixed count: one
                # cluster is correct for a single-storey building and wrong for
                # a two-storey one.
                flag = ""
                if m["n_kept"] == 0:
                    flag = "  <- nothing survives"
                elif m["n_kept"] < n_storeys:
                    flag = f"  <- MERGED storeys (expected {n_storeys})"
                elif m["n_kept"] > n_storeys:
                    flag = f"  <- fragmented (expected {n_storeys})"
                else:
                    flag = "  <- ok"
                print(f"    eps {eps:7.3f}  {fmt_metrics(m)}{flag}")


if __name__ == "__main__":
    main()
