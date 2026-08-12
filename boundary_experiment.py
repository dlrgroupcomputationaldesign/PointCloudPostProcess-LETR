"""Compare floor/ceiling boundary methods (alphashape vs raster) on real data.

Runs on WyomingStateFair_Laramie_inference_output.csv: clusters the floor and
ceiling points (like the production stages), RANSAC-fits each cluster's plane,
then draws BOTH boundaries on the SAME inliers so the only difference is the
boundary method. One comparison PNG per cluster -> boundary_test/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

from post_process_src.post_process.config import PostProcessConfig
from post_process_src.post_process.utils.floor_ceiling_util import (
    cluster_floor_ceiling,
    _boundary_alphashape,
    _boundary_raster,
)

CSV = Path(__file__).resolve().parent / "WyomingStateFair_Laramie_inference_output.csv"
OUT = Path(__file__).resolve().parent / "boundary_test"
OUT.mkdir(exist_ok=True)

LABELS = {"floor": 1.0, "ceiling": 2.0}
MAX_CLUSTERS = 3  # per type, largest first


class _Log:
    def info(self, *a):
        print("   ", *a)


log = _Log()
P = PostProcessConfig().to_parameters()


def ransac_inliers_2d(xyz, dis_thr, ransac_n, num_iter):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _model, inliers = pcd.segment_plane(dis_thr, ransac_n, num_iter)
    return xyz[np.asarray(inliers, dtype=int)][:, :2]


def run(kind, label):
    print(f"\n=== {kind} (label {label}) ===")
    df = pd.read_csv(CSV)
    df = df[df["pred_label"] == label].reset_index(drop=True)
    print(f"{len(df)} points")
    if not len(df):
        return

    eps = P["EPS_F"] if kind == "floor" else P["EPS_C"]
    msamp = P["MIN_SAMPLES_F"] if kind == "floor" else P["MIN_SAMPLES_C"]
    dis = P["DIS_THR_F"] if kind == "floor" else P["DIS_THR_C"]
    rn = P["RANSAC_N_F"] if kind == "floor" else P["RANSAC_N_C"]
    nit = P["NUM_ITER_F"] if kind == "floor" else P["NUM_ITER_C"]
    alpha = P["ALPHA_F"] if kind == "floor" else P["ALPHA_C"]

    clusters = cluster_floor_ceiling(df, eps, msamp, kind, blobs=None, min_points=5000)
    print(f"{len(clusters)} clusters (>5000 pts)")
    order = sorted(clusters, key=lambda k: -len(clusters[k]))[:MAX_CLUSTERS]

    for n in order:
        xyzrgb = clusters[n]
        xyz = xyzrgb[:, :3]
        pts2d = ransac_inliers_2d(xyz, dis, rn, nit)
        al = np.array(_boundary_alphashape(pts2d, alpha, log))
        ra = np.array(_boundary_raster(pts2d, P["BOUNDARY_CELL_F"], P["BOUNDARY_FILL_GAP_F"],
                                       P["BOUNDARY_SIMPLIFY_EPS_FRAC_F"], log))
        print(f"  cluster {n}: {len(xyz)} pts, {len(pts2d)} inliers | "
              f"alphashape {len(al)} verts, raster {len(ra)} verts")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, (name, b, col) in zip(axes, [("alphashape", al, "red"), ("raster (Option B)", ra, "blue")]):
            ax.scatter(pts2d[:, 0], pts2d[:, 1], s=1, c="0.7")
            if len(b):
                ax.plot(b[:, 0], b[:, 1], "-", color=col, linewidth=2)
            ax.set_title(f"{kind} cluster {n} — {name}  ({len(b)} verts)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = OUT / f"{kind}_cluster{n}_compare.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  saved {out.name}")


for kind, label in LABELS.items():
    run(kind, label)
print(f"\ndone -> {OUT}")
