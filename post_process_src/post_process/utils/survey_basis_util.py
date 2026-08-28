"""Derive SURVEY_BASIS from the post-process input CSV itself.

``parameters["SURVEY_BASIS"]`` is the rotation the stages apply as
``xyz @ SURVEY_BASIS`` to reach an axis-aligned frame (and ``@ SURVEY_BASIS.T``
to rotate results back out). Normally it is supplied by whoever registered the
survey. This module recovers it from the labelled point cloud instead, so a scan
can be processed without that value in hand.

Why the CSV and not the e57
---------------------------
The CSV is already segmented, so ``pred_label == Wall`` selects exactly the
surfaces that define the building's grid -- no height-slab heuristic needed to
keep floors and ceilings from voting. It is also the data the pipeline actually
consumes, so the answer cannot drift from a differently-scaled source file.

Method
------
Estimate a normal per wall point, keep the near-horizontal ones (a wall's normal
is horizontal), and take the circular mean of their azimuths folded mod 90 deg --
all four faces of an orthogonal grid describe the same rotation. Walls vote in
proportion to surveyed area, so furniture and clutter cannot outvote them.

Scale-independent: works whether the CSV is in feet or metres, because the
neighbourhood radius is derived from the cloud's own diagonal.

Usage
-----
Normally nothing calls this directly: leave ``SURVEY_BASIS`` unset (or None) and
the floor/ceiling/wall stages estimate it via stages.common.resolve_survey_basis,
which memoises the result so the three stages do not each re-estimate it.

To compute one explicitly::

    from post_process.utils.survey_basis_util import survey_basis_from_df
    parameters["SURVEY_BASIS"] = survey_basis_from_df(df)
"""

import numpy as np


WALL_LABEL = 3  # post_process.labels.DEFAULT_LABELS = (Other, Floor, Ceiling, Wall)


def _wall_points(df, parameters=None):
    """Return the XYZ of wall-labelled points, or all points if none are labelled."""
    try:
        from ..labels import label_mask
        mask = label_mask(df, "Wall", parameters).to_numpy()
    except Exception:
        mask = (df["pred_label"] == WALL_LABEL).to_numpy()

    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if mask.sum() < 500:
        print(f"  [survey_basis] only {mask.sum()} wall points; "
              f"falling back to a mid-height slab of all points")
        z = xyz[:, 2]
        lo, hi = np.percentile(z, [1, 99])
        span = hi - lo
        mask = (z >= lo + 0.15 * span) & (z <= lo + 0.80 * span)
    return xyz[mask]


def _fold_azimuths(azimuths):
    """Circular mean of angles meaningful only mod 90 deg.

    Wall normals point at 0/90/180/270 deg for one and the same grid, so the
    angles are quadrupled before averaging and the result divided back down.
    Returns (yaw_deg in [0,90), concentration in [0,1]).
    """
    a4 = 4.0 * azimuths
    s, c = np.sin(a4).mean(), np.cos(a4).mean()
    yaw = np.rad2deg(np.arctan2(s, c) / 4.0) % 90.0
    return float(yaw), float(np.hypot(s, c))


def _yaw_at_radius(xyz, radius, horiz_tol, max_nn):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    n = np.asarray(pcd.normals)
    wall = np.abs(n[:, 2]) < horiz_tol       # horizontal normal => vertical surface
    if wall.sum() < 200:
        return None
    yaw, strength = _fold_azimuths(np.arctan2(n[wall, 1], n[wall, 0]))
    return yaw, strength, int(wall.sum())


def yaw_matrix(yaw_deg):
    """Rotation B such that ``xyz @ B`` is axis-aligned.

    The convention throughout: stages align with ``xyz @ SURVEY_BASIS`` and
    rotate results back out with ``xyz @ SURVEY_BASIS.T``. point_axis_align
    applies the matrix exactly as given, so the value returned here is what the
    stages consume unchanged.

    Getting the direction backwards yields a matrix that rotates the building
    the wrong way by twice the yaw. It is still a valid rotation -- orthonormal,
    determinant +1 -- so nothing errors and the result looks plausible while
    being wrong. The footprint check in _basis_from_points is what catches it.
    """
    a = np.deg2rad(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def estimate_yaw(xyz, horiz_tol=0.15, max_nn=30, verbose=True):
    """Estimate the building's yaw (deg, mod 90) from wall points.

    Runs at three neighbourhood radii and takes the median. Agreement across
    radii is the confidence signal: a real Manhattan grid is scale-stable, noise
    is not.
    """
    diagonal = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    # 0.5% of the diagonal lands near 0.3 m for a typical building, in whatever
    # units the cloud uses -- this is what makes the function unit-agnostic.
    base = 0.005 * diagonal

    results = []
    for factor in (0.5, 1.0, 2.0):
        out = _yaw_at_radius(xyz, base * factor, horiz_tol, max_nn)
        if out is None:
            continue
        yaw, strength, n_wall = out
        results.append((yaw, strength))
        if verbose:
            print(f"  [survey_basis] radius {base * factor:6.3f}: "
                  f"yaw {yaw:7.3f} deg   strength {strength:.3f}   "
                  f"({n_wall:,} wall normals)")

    if not results:
        raise RuntimeError("no wall normals found; cannot estimate SURVEY_BASIS")

    yaws = np.array([r[0] for r in results])
    strengths = np.array([r[1] for r in results])
    # Median on the mod-90 circle: re-centre on the first value before averaging
    # so a 89.9/0.1 pair is not treated as 89.8 deg apart.
    centred = (yaws - yaws[0] + 45.0) % 90.0
    yaw = float((yaws[0] + np.median(centred) - 45.0) % 90.0)
    spread = float(np.ptp(centred))

    if verbose:
        print(f"  [survey_basis] consensus yaw {yaw:.3f} deg "
              f"(spread {spread:.3f} deg, mean strength {strengths.mean():.3f})")
        if spread > 2.0:
            print("  [survey_basis] WARNING: radii disagree; inspect the cloud.")
        if strengths.mean() < 0.4:
            print("  [survey_basis] WARNING: weak grid -- the building may not be "
                  "rectilinear, so axis alignment is ill-defined.")
    return yaw, spread, float(strengths.mean())


def _basis_from_points(xyz, long_axis="x", verbose=True, as_list=True):
    """Shared tail: yaw -> long-axis choice -> matrix. Used by both sources."""
    yaw, _, _ = estimate_yaw(xyz, verbose=verbose)

    extent = (xyz @ yaw_matrix(yaw)).ptp(axis=0)
    wants_swap = (long_axis == "x") == (extent[1] > extent[0])
    if wants_swap:
        yaw = (yaw - 90.0) % 360.0
        extent = extent[[1, 0, 2]]
        if verbose:
            print(f"  [survey_basis] rotated a further -90 deg to put the long "
                  f"side on {long_axis.upper()}")

    B = yaw_matrix(yaw)
    before = xyz.ptp(axis=0)
    area_before = before[0] * before[1]
    area_after = extent[0] * extent[1]
    gain = 100 * (1 - area_after / area_before)

    if verbose:
        print(f"  [survey_basis] footprint {area_before:,.0f} -> {area_after:,.0f} "
              f"sq units ({gain:.1f}% tighter)")
        print(f"  [survey_basis] final yaw {yaw % 360:.3f} deg")

    # A correct basis can only shrink the axis-aligned footprint. No gain means
    # the cloud was already square to the axes and identity is the right answer;
    # it is also what you see when the basis was estimated from a DIFFERENT
    # building than the one being processed, so it is worth shouting about.
    if gain < 0.5:
        print(f"  [survey_basis] WARNING: footprint barely changed ({gain:.1f}%). "
              f"This cloud is already axis-aligned -- use the identity matrix. "
              f"If you expected a rotation, check that the basis was estimated "
              f"from the same building as the data being processed.")

    return [list(row) for row in B] if as_list else B


def survey_basis_from_df(df, parameters=None, long_axis="x", verbose=True,
                         as_list=True):
    """Compute SURVEY_BASIS from a post-process input dataframe.

    Args:
        df: input dataframe with x/y/z and pred_label columns.
        parameters: optional parameters dict (only used for a custom LABEL_DICT).
        long_axis: "x" or "y" -- which axis the building's longer side lands on.
            Rotating by yaw or yaw-90 both axis-align the cloud; this picks
            between them deterministically instead of leaving it to chance.
        as_list: return a plain nested list (JSON-serialisable) rather than an
            ndarray, so it can be dropped straight into ``parameters``.

    Returns:
        3x3 rotation matrix, rows X/Y/Z.
    """
    xyz = _wall_points(df, parameters)
    if verbose:
        print(f"  [survey_basis] {len(xyz):,} wall points (from labels)")
    return _basis_from_points(xyz, long_axis, verbose, as_list)


def _e57_wall_points(e57_path, stride, voxel, slab, chunk_size, verbose):
    """Stream an e57 and return points from a mid-height slab.

    The e57 carries no segmentation, so walls cannot be selected by label the way
    they can from the CSV. A mid-height slab is the stand-in: it drops the floor
    and ceiling, which would otherwise swamp the vote with horizontal normals.
    Voxel downsampling first is essential -- raw scans are far denser near the
    scanner, so without it one corner of one room outvotes the whole building.
    """
    import open3d as o3d
    import pye57

    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    e57 = pye57.E57(str(e57_path))
    stride = max(1, int(stride))
    total_seen = 0
    chunks = []
    try:
        for scan_idx in range(e57.scan_count):
            header = e57.get_header(scan_idx)
            data, buffers = e57.make_buffers(fields, chunk_size)
            reader = header.points.reader(buffers)
            while True:
                n = reader.read()
                if n <= 0:
                    break
                arr = np.column_stack((data["cartesianX"][:n],
                                       data["cartesianY"][:n],
                                       data["cartesianZ"][:n])).astype(np.float64)
                first = (stride - total_seen % stride) % stride
                total_seen += n
                if first < len(arr):
                    chunks.append(arr[first::stride])
    finally:
        e57.close()

    if not chunks:
        raise RuntimeError(f"no points read from {e57_path}")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(chunks)))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    xyz = np.asarray(pcd.points)

    z = xyz[:, 2]
    z_lo, z_hi = np.percentile(z, [1, 99])
    span = z_hi - z_lo
    mask = (z >= z_lo + slab[0] * span) & (z <= z_lo + slab[1] * span)
    if verbose:
        print(f"  [survey_basis] {total_seen:,} sampled -> {len(xyz):,} voxels "
              f"-> {mask.sum():,} in slab")
    return xyz[mask]


def survey_basis_from_e57(e57_path, long_axis="x", stride=60, voxel=0.05,
                          slab=(0.15, 0.80), chunk_size=2_000_000,
                          verbose=True, as_list=True):
    """Compute SURVEY_BASIS straight from an e57, with no CSV involved.

    Same estimator as survey_basis_from_df; only the point selection differs
    (height slab instead of pred_label == Wall). Yaw is scale- and shift-
    invariant, so it does not matter that the e57 is in metres and unshifted
    while the CSV is in feet and moved to the origin -- both give the same angle.

    Slower than the CSV path: it streams the whole file (~1-2 min for a 2.7 GB
    scan) where the CSV is already in memory.
    """
    if verbose:
        print(f"  [survey_basis] streaming {e57_path} (stride={stride}, "
              f"voxel={voxel})")
    xyz = _e57_wall_points(e57_path, stride, voxel, slab, chunk_size, verbose)
    return _basis_from_points(xyz, long_axis, verbose, as_list)
