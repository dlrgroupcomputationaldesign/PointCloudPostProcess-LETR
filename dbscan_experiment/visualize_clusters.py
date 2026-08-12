"""See DBSCAN clusters in Open3D -- one window, several eps values side by side.

post_process_endpt.py already visualises clusters, but it is a linear script: to
reach the ceiling eps on line 395 the whole floor stage must run first, and that
opens ~9 blocking windows (draw_geometries + plt.show, per cluster) before the
changed line executes. This does the same clustering with the same feature
construction, straight to a picture, and renders MULTIPLE eps values at once so
they can be compared rather than remembered.

Each cluster gets its own colour; noise is dark grey. With several eps values the
results are laid out along Y, lowest eps nearest the origin.

    python visualize_clusters.py --dataset laramie --eps 0.3 0.5 0.75 1.5
    python visualize_clusters.py --dataset corteva --kind Ceiling --eps 0.5
    python visualize_clusters.py --dataset laramie --variant physical --eps 2 4 8
"""

import argparse

import numpy as np
import open3d as o3d

from lab import DATASETS, VARIANTS, cluster_metrics, fmt_metrics, load, z_profile

NOISE_COLOR = (0.25, 0.25, 0.28)


def cluster_colors(labels):
    """Distinct colour per cluster, dark grey for noise (-1)."""
    import matplotlib.pyplot as plt

    uniq = [l for l in np.unique(labels) if l != -1]
    cmap = plt.get_cmap("tab20")
    lookup = {l: cmap(i % 20)[:3] for i, l in enumerate(uniq)}

    colors = np.empty((len(labels), 3))
    for i, l in enumerate(labels):
        colors[i] = NOISE_COLOR if l == -1 else lookup[l]
    return colors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="laramie",
                    help=f"one of {list(DATASETS)}, or a path to a CSV")
    ap.add_argument("--kind", default="Floor", choices=["Floor", "Ceiling", "Wall"])
    ap.add_argument("--variant", default="current", choices=list(VARIANTS),
                    help="feature construction; 'current' matches the pipeline")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5],
                    help="one or more eps values; several are laid out along Y")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--sample", type=int, default=80000,
                    help="subsample for responsiveness; 0 uses every point")
    ap.add_argument("--min-points-frac", type=float, default=0.0367,
                    help="'kept' threshold as a fraction of points (10000/272000)")
    ap.add_argument("--hide-noise", action="store_true",
                    help="drop noise points instead of drawing them grey")
    ap.add_argument("--gap", type=float, default=1.25,
                    help="spacing between side-by-side results, as a multiple of "
                         "the cloud's Y extent")
    args = ap.parse_args()

    from sklearn.cluster import DBSCAN

    pts, col = load(args.dataset, args.kind, sample=(args.sample or None))
    min_points = max(10, int(args.min_points_frac * len(pts)))

    n_storeys, levels, min_gap = z_profile(pts)
    print(f"  z-profile: {n_storeys} storey(s) at z={[round(v, 2) for v in levels]}"
          f"   smallest gap {min_gap:.2f}")
    print(f"  a correct clustering keeps ~{n_storeys} cluster(s)\n")

    feats, desc = VARIANTS[args.variant](pts, col)
    print(f"  variant '{args.variant}': {desc}\n")

    span_y = float(pts[:, 1].max() - pts[:, 1].min()) * args.gap
    geometries = []

    for k, eps in enumerate(args.eps):
        labels = DBSCAN(eps=eps, min_samples=args.min_samples).fit_predict(feats)
        m = cluster_metrics(labels, pts, min_points)
        verdict = ("ok" if m["n_kept"] == n_storeys
                   else "MERGED" if m["n_kept"] < n_storeys else "fragmented")
        print(f"  eps {eps:7.3f}  {fmt_metrics(m)}  <- {verdict}")

        keep = labels != -1 if args.hide_noise else np.ones(len(labels), bool)
        xyz = pts[keep].copy()
        xyz[:, 1] += k * span_y          # lay results out along Y

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors(labels)[keep])
        geometries.append(pcd)

        # An axis marker at each result's origin, so the row stays readable.
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.05 * span_y,
                origin=(pts[:, 0].min(), pts[:, 1].min() + k * span_y, pts[:, 2].min()),
            )
        )

    if len(args.eps) > 1:
        print(f"\n  laid out along +Y in the order given: "
              f"{', '.join(str(e) for e in args.eps)}")
    print("\nopening Open3D window -- close it to exit.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{args.dataset}/{args.kind} DBSCAN [{args.variant}] "
                    f"eps={args.eps} min_samples={args.min_samples}",
        width=1400,
        height=850,
    )


if __name__ == "__main__":
    main()
