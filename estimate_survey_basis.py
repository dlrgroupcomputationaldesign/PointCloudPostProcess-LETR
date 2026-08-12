"""Estimate SURVEY_BASIS (the axis-aligning rotation) for an e57 scan, via Open3D.

``parameters["SURVEY_BASIS"]`` is the 3x3 matrix the post-process stages use as
``xyz @ SURVEY_BASIS.T`` to move into an axis-aligned frame. This script recovers
it from the cloud itself using three independent Open3D estimators, so you can
see whether they agree before trusting a number.

Estimators
----------
obb      ``get_minimal_oriented_bounding_box(robust=True).R`` -- one call, but it
         fits the convex hull, so furniture and stray points tilt it. Included as
         a baseline / sanity check, not as the recommended answer.
normals  ``estimate_normals()``, keep near-horizontal normals (wall faces), fold
         their azimuths mod 90 deg and take the circular mean. Walls vote in
         proportion to surveyed surface area, so this is the robust one.
patches  ``detect_planar_patches()`` -- explicit planar segments; vertical patches
         vote weighted by patch area. Slow but an honest second opinion.

A building scan is already Z-up, so the only real unknown is yaw about Z; every
estimator is reduced to that single angle (mod 90 deg, since a 90 deg turn only
swaps X and Y).

Usage
-----
    python estimate_survey_basis.py --input LaramieCM.e57
    python estimate_survey_basis.py --input LaramieCM.e57 --methods normals obb
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- io

def stream_xyz(e57_path, stride, chunk_size):
    """Yield XYZ chunks from an e57, keeping every Nth point globally."""
    import pye57

    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    e57 = pye57.E57(str(e57_path))
    stride = max(1, int(stride))
    total_seen = 0
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                n = reader.read()
                if n <= 0:
                    break
                xyz = np.column_stack(
                    (data["cartesianX"][:n], data["cartesianY"][:n], data["cartesianZ"][:n])
                ).astype(np.float64, copy=False)
                first = (stride - total_seen % stride) % stride
                total_seen += n
                if first < len(xyz):
                    yield xyz[first::stride]
    finally:
        e57.close()


def load_cloud(e57_path, stride, voxel, chunk_size):
    """Stride-sample the e57, then voxel-downsample so density is uniform.

    Uniform density matters: raw scans are dozens of times denser near the
    scanner, which would let one corner of one room outvote the whole building.
    """
    chunks = list(stream_xyz(e57_path, stride, chunk_size))
    if not chunks:
        raise SystemExit("no points read")
    xyz = np.vstack(chunks)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd = pcd.voxel_down_sample(voxel)
    print(f"  {len(xyz):,} sampled points -> {len(pcd.points):,} voxels @ {voxel} m")
    return pcd


def wall_slab(pcd, low_frac, high_frac):
    """Crop to a mid-height slab so walls vote and floors/ceilings do not."""
    xyz = np.asarray(pcd.points)
    z_lo, z_hi = np.percentile(xyz[:, 2], [1, 99])
    span = z_hi - z_lo
    lo, hi = z_lo + low_frac * span, z_lo + high_frac * span
    box = o3d.geometry.AxisAlignedBoundingBox(
        (-np.inf, -np.inf, lo), (np.inf, np.inf, hi)
    )
    slab = pcd.crop(box)
    print(f"  z {z_lo:.2f}..{z_hi:.2f} m; slab {lo:.2f}..{hi:.2f} m keeps "
          f"{len(slab.points):,}/{len(xyz):,}")
    return slab


# -------------------------------------------------------------------- estimators

def yaw_from_azimuths(azimuths, weights=None):
    """Circular mean of angles that are only meaningful mod 90 deg.

    Wall normals point at 0/90/180/270 deg for the same wall grid, so the angles
    are quadrupled before averaging (a mod-90 fold) and the mean divided back.
    """
    a4 = 4.0 * azimuths
    if weights is None:
        weights = np.ones_like(azimuths)
    s = np.sum(weights * np.sin(a4))
    c = np.sum(weights * np.cos(a4))
    yaw = np.rad2deg(np.arctan2(s, c) / 4.0) % 90.0
    # Concentration in [0,1]: 1 = every wall agrees, 0 = no dominant grid.
    strength = float(np.hypot(s, c) / np.sum(weights))
    return float(yaw), strength


def yaw_via_normals(slab, radius, max_nn, horiz_tol):
    """Estimate yaw from the normals of near-vertical (wall) surfaces."""
    slab.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    n = np.asarray(slab.normals)
    wall = np.abs(n[:, 2]) < horiz_tol          # normal roughly horizontal => wall
    print(f"  wall-like normals: {wall.sum():,}/{len(n):,} "
          f"(|nz| < {horiz_tol})")
    if wall.sum() < 500:
        raise SystemExit("too few wall normals; raise --horiz-tol or lower --stride")
    az = np.arctan2(n[wall, 1], n[wall, 0])
    return yaw_from_azimuths(az)


def yaw_via_obb(pcd):
    """Yaw from the minimal oriented bounding box -- the naive one-call answer."""
    obb = pcd.get_minimal_oriented_bounding_box(robust=True)
    R = np.asarray(obb.R)
    # Columns of R are the box axes in world coords. Drop the one closest to Z
    # (the up axis) and read the yaw off the most horizontal remaining axis.
    up_idx = int(np.argmax(np.abs(R[2, :])))
    horiz = [i for i in range(3) if i != up_idx]
    axis = R[:, horiz[0]]
    yaw = np.rad2deg(np.arctan2(axis[1], axis[0])) % 90.0
    tilt = np.rad2deg(np.arccos(min(1.0, abs(R[2, up_idx]))))
    print(f"  obb up-axis tilt off Z: {tilt:.2f} deg (large => hull is skewed)")
    return float(yaw), float(np.asarray(obb.extent).prod())


def yaw_via_patches(slab, normal_variance, min_points):
    """Yaw from detected planar patches, each weighted by its area."""
    patches = slab.detect_planar_patches(
        normal_variance_threshold_deg=normal_variance,
        coplanarity_deg=75.0,
        outlier_ratio=0.75,
        min_plane_edge_length=0.0,
        min_num_points=min_points,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
    )
    az, area = [], []
    for p in patches:
        R = np.asarray(p.R)
        normal = R[:, 2]                       # patch normal is the 3rd axis
        if abs(normal[2]) > 0.2:               # skip floors / ceilings
            continue
        e = np.asarray(p.extent)
        az.append(np.arctan2(normal[1], normal[0]))
        area.append(float(np.sort(e)[-1] * np.sort(e)[-2]))
    print(f"  {len(patches)} patches, {len(az)} vertical")
    if len(az) < 3:
        raise SystemExit("too few vertical patches; loosen --normal-variance")
    return yaw_from_azimuths(np.array(az), np.array(area))


# ------------------------------------------------------------------------ output

def yaw_matrix(yaw_deg):
    """Rotation B with ``xyz @ B`` axis-aligned -- i.e. SURVEY_BASIS.

    The stages align with ``xyz @ SURVEY_BASIS`` and rotate back out with
    ``xyz @ SURVEY_BASIS.T``; walls.py passing ``SURVEY_BASIS.T`` into
    point_axis_align (which transposes again) is what makes the two cancel.
    """
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def print_basis(R):
    print('    "SURVEY_BASIS": [')
    for row, axis in zip(R, "XYZ"):
        print(f"        [{row[0]: .10f}, {row[1]: .10f}, {row[2]: .10f}],   # {axis}")
    print("    ],")


def _row(label, ext):
    print(f"  {label:22}  X {ext[0]:7.2f}  Y {ext[1]:7.2f}  Z {ext[2]:7.2f} m   "
          f"footprint {ext[0] * ext[1]:8.1f} m^2   volume {np.prod(ext):9.0f} m^3")


def verify_against_obb(pcd, R):
    """Cross-check the yaw by comparing axis-aligned and oriented boxes.

    ``get_axis_aligned_bounding_box`` carries no rotation -- it just bounds the
    points in whatever frame they sit in. That makes it useless for *finding* the
    yaw but ideal for *checking* it: the OBB is the tightest box at any
    orientation, so once the cloud is correctly rotated its AABB should collapse
    onto the OBB. Residual gap is the alignment error.
    """
    xyz = np.asarray(pcd.points)

    aabb_before = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    obb = np.sort(np.asarray(pcd.get_minimal_oriented_bounding_box(robust=True).extent))[::-1]

    rotated = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz @ R))
    aabb_after = np.asarray(rotated.get_axis_aligned_bounding_box().get_extent())

    _row("AABB before", aabb_before)
    _row("AABB after rotation", aabb_after)
    _row("minimal OBB", obb)

    slack_before = np.prod(aabb_before) / np.prod(obb)
    slack_after = np.prod(aabb_after) / np.prod(obb)
    print(f"\n  AABB/OBB volume ratio:  {slack_before:.3f} before  ->  "
          f"{slack_after:.3f} after  (1.000 = perfectly aligned)")
    print(f"  wasted volume removed:  {100 * (1 - slack_after / slack_before):.1f}%")
    if slack_after > 1.15:
        print("  ^ still loose: the cloud may not be Manhattan, or has outliers "
              "inflating the AABB (the OBB hull ignores fewer of them).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to the .e57 file")
    ap.add_argument("--methods", nargs="+", default=["normals", "obb"],
                    choices=["normals", "obb", "patches"],
                    help="estimators to run; the reported answer comes from the "
                         "first one listed that succeeds")
    ap.add_argument("--stride", type=int, default=40,
                    help="keep every Nth point while streaming (speed knob)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="voxel size in meters (density equalisation)")
    ap.add_argument("--slab", type=float, nargs=2, default=(0.15, 0.80),
                    metavar=("LOW", "HIGH"),
                    help="height fractions of the wall slab used for voting")
    ap.add_argument("--normal-radius", type=float, default=0.25,
                    help="neighbourhood radius for estimate_normals (m)")
    ap.add_argument("--normal-max-nn", type=int, default=30)
    ap.add_argument("--horiz-tol", type=float, default=0.15,
                    help="|nz| below this counts as a wall normal")
    ap.add_argument("--normal-variance", type=float, default=60.0,
                    help="detect_planar_patches normal_variance_threshold_deg")
    ap.add_argument("--min-patch-points", type=int, default=200)
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    print(f"reading {path.name} (stride={args.stride}, voxel={args.voxel} m)...")
    pcd = load_cloud(path, args.stride, args.voxel, args.chunk_size)

    ext = np.asarray(pcd.points).ptp(axis=0)
    print(f"  raw extents  X {ext[0]:.2f}  Y {ext[1]:.2f}  Z {ext[2]:.2f} m")
    if ext[2] > min(ext[0], ext[1]):
        print("  WARNING: Z is not the shortest axis -- confirm the cloud is Z-up.")

    slab = wall_slab(pcd, *args.slab)

    results = {}
    for method in args.methods:
        print(f"\n[{method}]")
        try:
            if method == "normals":
                yaw, strength = yaw_via_normals(
                    slab, args.normal_radius, args.normal_max_nn, args.horiz_tol)
                print(f"  yaw {yaw:.3f} deg   grid strength {strength:.3f} "
                      f"(>0.5 = clean Manhattan building)")
            elif method == "obb":
                yaw, volume = yaw_via_obb(pcd)
                print(f"  yaw {yaw:.3f} deg   box volume {volume:,.0f} m^3")
            else:
                yaw, strength = yaw_via_patches(
                    slab, args.normal_variance, args.min_patch_points)
                print(f"  yaw {yaw:.3f} deg   area-weighted agreement {strength:.3f}")
            results[method] = yaw
        except Exception as exc:  # one estimator failing shouldn't sink the run
            print(f"  failed: {exc}")

    if not results:
        raise SystemExit("every estimator failed")

    if len(results) > 1:
        vals = np.array(list(results.values()))
        # Compare mod 90 deg, wrapping the seam (89.9 and 0.1 differ by 0.2).
        spread = np.ptp((vals - vals[0] + 45.0) % 90.0)
        print(f"\nestimators agree to within {spread:.3f} deg")
        if spread > 2.0:
            print("  ^ they disagree; trust 'normals' and inspect the cloud.")

    chosen = args.methods[0] if args.methods[0] in results else next(iter(results))
    yaw = results[chosen]
    R = yaw_matrix(yaw)

    print(f"\n=== answer (from '{chosen}'): yaw = {yaw:.3f} deg ===")
    verify_against_obb(pcd, R)
    print()
    print_basis(R)


if __name__ == "__main__":
    main()
