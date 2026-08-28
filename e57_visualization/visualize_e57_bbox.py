"""Load an e57, compute its bounding box(es), and show them with Open3D.

Opens a desktop window via ``o3d.visualization.draw_geometries`` containing the
point cloud plus:

  * red   -- ``get_axis_aligned_bounding_box()`` (AABB), the box in the survey frame
  * green -- ``get_minimal_oriented_bounding_box()`` (OBB), the tightest box at any
             orientation (``--obb``)
  * a coordinate frame at the cloud's min corner, sized to the cloud

Pass ``--basis`` to rotate the cloud by a SURVEY_BASIS first (see
``estimate_survey_basis.py``). The AABB then visibly shrinks onto the OBB, which
is the quickest way to eyeball whether a yaw estimate is right.

Usage
-----
    python visualize_e57_bbox.py --input LaramieCM.e57
    python visualize_e57_bbox.py --input LaramieCM.e57 --obb
    python visualize_e57_bbox.py --input LaramieCM.e57 --obb \
        --basis 0.2377202861 0.9713336531 0 -0.9713336531 0.2377202861 0 0 0 1
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def stream_e57(e57_path, stride, chunk_size):
    """Yield (xyz, rgb) chunks from an e57, keeping every Nth point globally."""
    import pye57

    with_color = ["cartesianX", "cartesianY", "cartesianZ",
                  "colorRed", "colorGreen", "colorBlue"]
    plain = ["cartesianX", "cartesianY", "cartesianZ"]

    e57 = pye57.E57(str(e57_path))
    stride = max(1, int(stride))
    total_seen = 0
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            try:
                data, buffers = e57.make_buffers(with_color, chunk_size)
                has_color = True
            except Exception:
                data, buffers = e57.make_buffers(plain, chunk_size)
                has_color = False
            reader = header.points.reader(buffers)
            while True:
                n = reader.read()
                if n <= 0:
                    break
                xyz = np.column_stack((data["cartesianX"][:n],
                                       data["cartesianY"][:n],
                                       data["cartesianZ"][:n])).astype(np.float64)
                if has_color:
                    rgb = np.column_stack((data["colorRed"][:n],
                                           data["colorGreen"][:n],
                                           data["colorBlue"][:n])).astype(np.float64) / 255.0
                else:
                    rgb = np.full((n, 3), 0.6)
                first = (stride - total_seen % stride) % stride
                total_seen += n
                if first < len(xyz):
                    yield xyz[first::stride], rgb[first::stride]
    finally:
        e57.close()


def load_pcd(e57_path, stride, voxel, chunk_size):
    """Read the e57 into a single downsampled Open3D PointCloud."""
    xyz_chunks, rgb_chunks = [], []
    for xyz, rgb in stream_e57(e57_path, stride, chunk_size):
        xyz_chunks.append(xyz)
        rgb_chunks.append(rgb)
    if not xyz_chunks:
        raise SystemExit("no points read")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(xyz_chunks))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(rgb_chunks))
    n_before = len(pcd.points)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    print(f"  {n_before:,} points -> {len(pcd.points):,} after voxel {voxel} m")
    return pcd


def describe(box, label):
    # AABB exposes get_extent(); OrientedBoundingBox only has the .extent property.
    ext = np.asarray(box.extent if hasattr(box, "extent") else box.get_extent())
    print(f"  {label:10}  extent  X {ext[0]:7.2f}  Y {ext[1]:7.2f}  Z {ext[2]:7.2f} m"
          f"   volume {np.prod(ext):9.0f} m^3")
    return ext


def _cube_rotations():
    """The 24 proper rotations that map a cube onto itself (signed permutations)."""
    from itertools import permutations, product

    mats = []
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            g = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                g[row, col] = s
            if np.linalg.det(g) > 0:          # keep proper rotations, drop mirrors
                mats.append(g)
    return mats


def rotation_between_boxes(obb):
    """Decompose the AABB->OBB rotation into an axis relabel plus a real rotation.

    Takes only the OBB: the AABB contributes nothing but the identity frame.

    The AABB's frame *is* the world frame, so the rotation between the two boxes
    is exactly ``obb.R`` (its columns are the OBB's axes in world coordinates).
    But Open3D numbers the OBB's axes arbitrarily, so ``obb.R`` typically also
    contains a 90 deg axis swap that carries no geometric meaning. Right-
    multiplying by a signed permutation ``G`` relabels the box's own axes without
    moving the box; picking the ``G`` that maximises the trace gives the smallest
    equivalent rotation -- the part that is actually a misalignment.
    """
    R = np.asarray(obb.R)
    print("\n  obb.R (raw, columns = OBB axes in world coords):")
    for row in R:
        print(f"      [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")

    # Angle of the raw matrix, via the rotation-angle identity trace(R)=1+2cos(t).
    raw_angle = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"  raw rotation angle: {raw_angle:.3f} deg  (inflated by the axis relabel)")

    best = max(_cube_rotations(), key=lambda g: np.trace(R @ g))
    Rc = R @ best
    angle = np.rad2deg(np.arccos(np.clip((np.trace(Rc) - 1) / 2, -1, 1)))
    print("\n  after relabelling the OBB's axes:")
    for row in Rc:
        print(f"      [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
    print(f"  true rotation angle: {angle:.3f} deg")

    # Split into yaw about Z and any residual tilt of the box's up axis.
    yaw = np.rad2deg(np.arctan2(Rc[1, 0], Rc[0, 0]))
    tilt = np.rad2deg(np.arccos(np.clip(abs(Rc[2, 2]), -1, 1)))
    print(f"  yaw about Z: {yaw:.3f} deg      residual tilt off vertical: {tilt:.3f} deg")
    if tilt > 1.0:
        print("  ^ non-trivial tilt: the OBB is leaning, usually from outliers.")

    # Rc's columns are the box axes in world coords, so world -> box is Rc.T on
    # column vectors -- which for the row-vector stack here is `xyz @ Rc`. That
    # matches the stages' convention (aligned = xyz @ SURVEY_BASIS), so B = Rc.
    print("\n  SURVEY_BASIS implied by the OBB (B = Rc):")
    for row, axis in zip(Rc, "XYZ"):
        print(f"      [{row[0]: .10f}, {row[1]: .10f}, {row[2]: .10f}],   # {axis}")
    return Rc, yaw, tilt


def build_geometries(pcd, show_obb, gray):
    """Return the geometry list for draw_geometries: cloud, boxes, axes."""
    if gray:
        pcd.paint_uniform_color([0.55, 0.55, 0.58])

    geometries = [pcd]

    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1.0, 0.0, 0.0)          # boxes render as wireframes
    geometries.append(aabb)
    describe(aabb, "AABB")
    print(f"  {'':10}  min {np.asarray(aabb.min_bound).round(2)}"
          f"   max {np.asarray(aabb.max_bound).round(2)}")

    if show_obb:
        obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        obb.color = (0.0, 1.0, 0.0)
        geometries.append(obb)
        describe(obb, "OBB")
        ratio = np.prod(aabb.get_extent()) / np.prod(obb.extent)
        print(f"  AABB/OBB volume ratio {ratio:.3f}  (1.000 = already axis-aligned)")
        rotation_between_boxes(obb)

    # Axes at the min corner, 10% of the cloud's largest dimension.
    size = 0.1 * float(np.max(aabb.get_extent()))
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=aabb.min_bound)
    )
    return geometries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to the .e57 file")
    ap.add_argument("--stride", type=int, default=20,
                    help="keep every Nth point while streaming (speed knob)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="voxel downsample size in meters; 0 disables")
    ap.add_argument("--obb", action="store_true",
                    help="also draw the minimal oriented bounding box in green")
    ap.add_argument("--basis", type=float, nargs=9, default=None,
                    metavar="R",
                    help="9 numbers, row-major SURVEY_BASIS; rotates the cloud "
                         "(xyz @ B) before boxing it")
    ap.add_argument("--align-obb", action="store_true",
                    help="rotate the cloud by the basis derived from obb.R, so "
                         "the OBB becomes axis-aligned. No numbers to copy.")
    ap.add_argument("--gray", action="store_true",
                    help="ignore e57 RGB and draw the cloud in flat gray")
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    print(f"reading {path.name}...")
    pcd = load_pcd(path, args.stride, args.voxel, args.chunk_size)

    R = None
    if args.align_obb:
        print("\ncomputing the OBB's own rotation to align by...")
        R, _, _ = rotation_between_boxes(pcd.get_minimal_oriented_bounding_box(robust=True))
    elif args.basis is not None:
        R = np.asarray(args.basis, dtype=np.float64).reshape(3, 3)

    if R is not None:
        # Same convention as the post-process stages: aligned = xyz @ SURVEY_BASIS
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) @ R)
        print(f"\n  rotated the cloud by this basis (yaw "
              f"{np.rad2deg(np.arctan2(R[1, 0], R[0, 0])):.3f} deg)")

    geometries = build_geometries(pcd, args.obb, args.gray)

    print("\nopening Open3D window -- close it to exit.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{path.name} + bbox",
        width=1280,
        height=800,
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()
