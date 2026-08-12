"""k-distance elbow study: does a principled eps exist, and is it portable?

The standard way to choose DBSCAN's eps without guessing: for every point,
measure the distance to its ``min_samples``-th nearest neighbour, sort those
descending, and look for the knee. Points left of the knee are in sparse
regions (noise); the knee is the density where clusters stop being clusters.

Run this per dataset and per feature variant. If the knee lands at the same eps
for both buildings, one global default is defensible and only the value needs
fixing. If it does not, eps has to be derived per dataset -- which is the case
this experiment exists to settle.

    python dbscan_experiment/kdistance.py
    python dbscan_experiment/kdistance.py --variants current isotropic --sample 60000
"""

import argparse

import numpy as np

from lab import DATASETS, OUT, VARIANTS, load


def knee(curve):
    """Index of maximum distance from the chord joining the curve's endpoints.

    The classic 'kneedle' construction: normalise both axes to [0,1], then the
    knee is the point furthest from the straight line between first and last.
    """
    n = len(curve)
    if n < 3:
        return 0
    x = np.linspace(0.0, 1.0, n)
    span = curve[0] - curve[-1]
    if span <= 0:
        return 0
    y = (curve - curve[-1]) / span            # 1 at the left, 0 at the right
    # Distance to the chord y = 1 - x, scaled by 1/sqrt(2) (constant, so ignored).
    return int(np.argmax(np.abs(y - (1.0 - x))))


def k_distances(feats, k):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(feats)
    d, _ = nn.kneighbors(feats)
    return np.sort(d[:, -1])[::-1]            # descending


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--kind", default="Floor", choices=["Floor", "Ceiling", "Wall"])
    ap.add_argument("--min-samples", type=int, default=10,
                    help="k for the k-distance curve; match the pipeline's MIN_SAMPLES")
    ap.add_argument("--sample", type=int, default=60000,
                    help="subsample size (kNN on 300k+ points is slow)")
    ap.add_argument("--plot", default=str(OUT / "kdistance.png"))
    args = ap.parse_args()

    results = {}
    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        pts, col = load(ds, args.kind, sample=args.sample)
        for vname in args.variants:
            feats, desc = VARIANTS[vname](pts, col)
            curve = k_distances(feats, args.min_samples)
            i = knee(curve)
            results[(ds, vname)] = (curve, curve[i])
            print(f"  {vname:11} knee eps = {curve[i]:8.4f}   "
                  f"[p50 {np.median(curve):7.4f}]   {desc}")

    print("\n--- portability: does one eps fit both buildings? ---")
    for vname in args.variants:
        vals = [results[(ds, vname)][1] for ds in args.datasets if (ds, vname) in results]
        if len(vals) > 1:
            ratio = max(vals) / max(min(vals), 1e-12)
            verdict = "portable" if ratio < 1.5 else "NOT portable"
            print(f"  {vname:11} knees {[round(v, 4) for v in vals]}   "
                  f"ratio {ratio:6.2f}x   {verdict}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        OUT.mkdir(exist_ok=True)
        vs = args.variants
        fig, axes = plt.subplots(1, len(vs), figsize=(5 * len(vs), 4), squeeze=False)
        for ax, vname in zip(axes[0], vs):
            for ds in args.datasets:
                if (ds, vname) not in results:
                    continue
                curve, e = results[(ds, vname)]
                ax.plot(np.linspace(0, 100, len(curve)), curve, lw=1.2, label=f"{ds} (knee {e:.3f})")
                ax.axhline(e, ls="--", lw=0.8, alpha=0.5)
            ax.set_title(vname)
            ax.set_xlabel("% of points")
            ax.set_ylabel(f"distance to {args.min_samples}th neighbour")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
