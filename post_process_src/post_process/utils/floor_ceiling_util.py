import pandas as pd
import numpy as np
from shapely.geometry import LineString
from io import BytesIO
from PIL import Image
import cv2
from ..runtime import logger
from .blob_util import (
    plot_inliers_with_obb_html_bytes,
    plot_plane_inliers_outliers_html_bytes,
    snapshot_plotly_html_bytes,
    upload_html_bytes_to_blob,
    upload_image_array_to_blob,
    upload_matplotlib_fig_to_blob,
    write_figure_output,
    write_html_output,
    write_image_output,
)


def _boundary_alphashape(points_2d, alpha_value, logger):
    """Original concave-hull boundary. Returns a list of (x, y) exterior coords."""
    import alphashape

    alpha_shape = alphashape.alphashape(points_2d, alpha_value)
    if alpha_shape.geom_type == "Polygon":
        return list(alpha_shape.exterior.coords)
    if alpha_shape.geom_type == "MultiPolygon":
        return list(max(alpha_shape.geoms, key=lambda p: p.area).exterior.coords)
    logger.info("Alpha shape is not a Polygon or MultiPolygon.")
    return []


_E57_CACHE = {}


def load_e57_in_csv_frame(path, scale, basis=None, voxel=0.3,
                          chunk_size=2_000_000, logger_=None):
    """Stream an e57 into the CSV's coordinate frame, downsampled and cached.

    The segmented CSV is a heavy downsample of the scan: on one ceiling band it
    held 24,619 points where the e57 has ~17 million. That is why the CSV's
    occupancy grid is 71.7% void and shatters into hundreds of blobs -- there
    simply are not enough points to form a connected surface. Measured directly,
    the e57 fills 89.1% of those void cells, so they are a downsampling artifact
    rather than occlusion; only the remaining 11% are true scanner shadows.

    Points are voxel-downsampled to ``voxel`` (cloud units) on the way in. The
    boundary only needs roughly one point per raster cell, so keeping the full
    density would cost gigabytes to no benefit.

    The frame conversion is ``(raw - raw_min) * scale @ basis``: the e57 is in
    metres and unshifted, the CSV in native units with its minimum at the
    origin. raw_min is accumulated from the FULL stream, not the downsampled
    set, so the offset does not drift by a voxel.

    Cached per (path, scale, voxel) because fit_ceiling_floor runs once per
    cluster and re-streaming a 2.8 GB file each time would dominate the stage.
    """
    key = (str(path), float(scale), float(voxel))
    hit = _E57_CACHE.get(key)
    if hit is not None:
        return hit

    import open3d as o3d
    import pye57

    voxel_raw = max(float(voxel) / max(float(scale), 1e-9), 1e-9)
    fields = ["cartesianX", "cartesianY", "cartesianZ"]
    e57 = pye57.E57(str(path))
    raw_min, kept, total = None, [], 0
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
                total += n
                chunk_min = arr.min(axis=0)
                raw_min = chunk_min if raw_min is None else np.minimum(raw_min, chunk_min)
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))
                kept.append(np.asarray(pcd.voxel_down_sample(voxel_raw).points))
    finally:
        e57.close()

    if not kept:
        raise RuntimeError(f"no points read from {path}")

    pts = np.vstack(kept)
    # Chunks are downsampled independently, so dedupe across their boundaries.
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pts = np.asarray(pcd.voxel_down_sample(voxel_raw).points)
    pts = (pts - raw_min) * float(scale)
    if basis is not None:
        pts = pts @ np.asarray(basis, dtype=float)

    if logger_ is not None:
        logger_.info("e57 boundary source: %s -> %d of %d points at voxel %.2f",
                     path, len(pts), total, voxel)
    _E57_CACHE[key] = pts
    return pts


def xy_point_spacing(points_2d, k=6, sample=20000, seed=0):
    """Median distance to the k-th XY neighbour -- the raster's natural scale.

    Subsampled because this only needs to be approximate and kNN on 100k+
    points is not free. Deterministic via a fixed seed so the derived boundary
    parameters do not wobble between runs.
    """
    from sklearn.neighbors import NearestNeighbors

    pts = np.asarray(points_2d)
    if len(pts) > sample:
        idx = np.random.default_rng(seed).choice(len(pts), sample, replace=False)
        pts = pts[idx]
    if len(pts) <= k:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k).fit(pts)
    dist, _ = nn.kneighbors(pts)
    return float(np.median(dist[:, -1]))


def resolve_boundary_opts(points_2d, boundary_opts, logger=None):
    """Replace "auto" raster knobs with values derived from the point spacing.

    All three raster parameters collapse to one measurable input, the inlier
    point spacing:

      cell     ~= spacing. Below it the occupancy grid speckles -- at cell=0.25
                 against 0.87 ft spacing only 30% of cells are hit and the image
                 shatters into 4,644 blobs. Above it, detail is lost.
      fill_gap  = 2 x cell. There is a hard floor at about that (below it the
                 mask fragments) and above it nothing changes: morphological
                 closing is dilate-then-erode, so a wider kernel fills bigger
                 interior holes without moving the outer boundary.
      simplify  = 1.5 x cell, ABSOLUTE. Raster vertices quantise to cell centres
                 and the contour stair-steps at +/- cell/2, so the tolerance
                 should erase steps at that scale and nothing larger.

    On simplify specifically: the legacy BOUNDARY_SIMPLIFY_EPS_FRAC_F is a
    fraction of the contour perimeter, which makes the tolerance scale with
    building size for no geometric reason -- the same 0.001 is 0.60 ft on one
    of our buildings and 1.18 ft on a larger one. That is why no single
    fraction ports. An absolute length tied to the cell does.
    """
    opts = dict(boundary_opts or {})
    wants = [k for k in ("cell", "fill_gap", "simplify_abs")
             if isinstance(opts.get(k), str) and str(opts[k]).lower() == "auto"]
    if not wants:
        return opts

    spacing = xy_point_spacing(points_2d)
    if spacing <= 0:
        spacing = float(opts.get("cell") if isinstance(opts.get("cell"), (int, float))
                        else 0.25)

    if "cell" in wants:
        opts["cell"] = spacing
    cell = float(opts["cell"])
    if "fill_gap" in wants:
        opts["fill_gap"] = 2.0 * cell
    if "simplify_abs" in wants:
        opts["simplify_abs"] = 1.5 * cell

    if logger is not None:
        logger.info(
            "boundary auto: spacing %.3f -> cell %.3f, fill_gap %.3f, simplify %.3f",
            spacing, cell, float(opts["fill_gap"]),
            float(opts["simplify_abs"]) if opts.get("simplify_abs") is not None else float("nan"),
        )
    return opts


def _fill_exterior(mask):
    """Filled RETR_EXTERNAL contour(s) of a boolean mask.

    Enclosed holes disappear, which is the point: ``edgePoints`` is a single
    outer ring, so a void inside a component is ALREADY spanned by the output.
    Measuring area on filled exteriors is what stops interior holes from being
    counted as something that needs bridging -- the mistake the old growth loop
    made, since it grew the kernel to close holes the contour ignored anyway.
    """
    out = np.zeros(mask.shape, np.uint8)
    found, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if found:
        cv2.drawContours(out, found, -1, 1, cv2.FILLED)
    return out.astype(bool)


def _close_bool(mask, radius_cells):
    """Morphological close with an elliptical kernel of an INTEGER cell radius."""
    r = int(radius_cells)
    if r <= 0:
        return np.array(mask, bool)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(np.asarray(mask, np.uint8), cv2.MORPH_CLOSE,
                            ker).astype(bool)


def _min_gap_cells(mask_a, mask_b):
    """Smallest boundary-to-boundary distance between two masks, in cells."""
    if not mask_a.any() or not mask_b.any():
        return float("inf")
    if (mask_a & mask_b).any():
        return 0.0
    # distanceTransform returns, for every NON-ZERO pixel, the distance to the
    # nearest ZERO pixel -- so inverting A turns it into "distance to A".
    dt = cv2.distanceTransform((~mask_a).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[mask_b].min())


def _min_connecting_radius(mask_a, mask_b, start, max_radius):
    """Smallest integer kernel radius whose closing joins the two masks.

    Consecutive integers, not the old geometric ladder. Growing by 1.5x from
    2 x cell could only ever try radii 2, 3, 4, 7, 10, 15, 23 -- so a gap
    needing 6 cells was bridged with 7 and one needing 16 with 23, each time
    claiming more area than the connection required.

    Every trial restarts from the same input. Closing the previous result would
    compound kernels, and the radius reported would be one that was never
    applied on its own.
    """
    trial = mask_a | mask_b
    for r in range(max(1, int(start)), int(max_radius) + 1):
        closed = _close_bool(trial, r)
        _, lab = cv2.connectedComponents(closed.astype(np.uint8))
        if np.intersect1d(np.unique(lab[mask_a]), np.unique(lab[mask_b])).size:
            return r, closed
    return None, None


def _evidence_merge(original, cell, base_gap, opts, logger, type="ceiling"):
    """Component merging judged by evidence, replacing the grow-until-one loop.

    The old rule grew one global kernel until ``len(contours) == 1``. That gives
    every detached speck unbounded leverage over the whole boundary: measured on
    one ceiling, six components holding 0.24% of the points dragged the kernel
    from 4 to 15 cells and added ~4,650 hull cells, of which only 8.9% had any
    support in the raw scan. On another ceiling the same rule was RIGHT, welding
    two halves of one sparsely-scanned surface. Component count cannot tell
    those apart, because it measures the shape of the mask rather than the
    evidence behind it.

    So each detached component is judged on its own:

      A_bridge   = area the merge invents, beyond either component's own
                   filled exterior. The geometric cost.
      E_merge    = candidate's ORIGINAL occupied cells / newly claimed area.
                   The primary test: what fraction of what we are about to
                   claim is actually evidenced?
      I_area     = A_bridge / union area. How much belongs to neither side.
      D_Q        = candidate's occupied cells / its own exterior area.

    Support is always counted on the PRE-morphology mask: cells created by
    closing are claimed geometry, not evidence. Occupied cells rather than raw
    points, because point density tracks scanner distance and overlap, not area.

    Returns (accepted_mask, report) with the mask already exterior-filled.
    """
    max_gap = opts.get("merge_max_gap", "auto")
    if isinstance(max_gap, str) and str(max_gap).lower() == "auto":
        # Deliberately permissive: this is a safety rail, not the decision. The
        # spec's point is that size alone must not accept or reject, so the
        # evidence tests below should be what binds. PROVISIONAL -- not yet
        # measured across projects.
        max_gap = 24.0 * cell
    max_gap = float(max_gap)
    min_eff = float(opts.get("merge_min_efficiency", 0.10))
    max_infl = float(opts.get("merge_max_area_inflation", 0.25))
    min_density = float(opts.get("component_min_density", 0.10))
    min_frac = float(opts.get("component_min_support_frac", 0.005))
    max_radius = int(opts.get("merge_max_radius_cells",
                              int(np.ceil(max_gap / cell)) + 2))

    base_mask = _close_bool(original, int(round(base_gap / cell)))
    n, lab = cv2.connectedComponents(base_mask.astype(np.uint8))
    total_support = int(original.sum())
    report = {"components": n - 1, "accepted": 0, "contained": 0,
              "noise": 0, "rejected": [], "bridge": None, "merged": []}

    if n <= 2 or total_support == 0:
        return _fill_exterior(base_mask), report

    comps = []
    for i in range(1, n):
        m = lab == i
        comps.append({"id": i, "mask": m, "support": int((original & m).sum())})
    # Main component by EVIDENCE, not by exterior area -- a large hollow
    # contour must not outrank a smaller densely-supported one.
    comps.sort(key=lambda c: -c["support"])
    main, candidates = comps[0], comps[1:]

    # Components too faint to ever pass the support test are dropped once, up
    # front. Support comes from the immutable original mask, so re-testing them
    # every pass could not change the outcome -- and one of these ceilings has
    # 1,356 components, most of them single cells.
    noise = [c for c in candidates if c["support"] / total_support < min_frac]
    candidates = [c for c in candidates if c["support"] / total_support >= min_frac]
    report["noise"] = len(noise)

    accepted = _fill_exterior(main["mask"])
    component_union = accepted.copy()
    logger.info(
        "%s boundary: %d components, main holds %d cells (%.1f%% of evidence); "
        "%d below %.1f%% support dropped as noise, %d candidate(s) to judge",
        type, n - 1, main["support"], 100 * main["support"] / total_support,
        len(noise), 100 * min_frac, len(candidates),
    )

    final_rejects = []
    while candidates:
        trials, rejects, contained = [], [], []
        for cand in candidates:
            q = _fill_exterior(cand["mask"])
            if not (q & ~accepted).any():
                # Sitting inside an enclosed hole of the accepted region, which
                # the outer ring already spans. Nothing to bridge.
                contained.append(cand)
                continue

            gap_cells = _min_gap_cells(accepted, q)
            support = cand["support"]
            a_q = int(q.sum())
            rec = {
                "id": cand["id"], "support": support, "a_q": a_q,
                "density": support / max(a_q, 1),
                "frac": support / max(total_support, 1),
                "gap_cells": gap_cells, "gap_world": gap_cells * cell,
            }
            if rec["gap_world"] > max_gap:
                rec["reason"] = f"gap {rec['gap_world']:.2f} > {max_gap:.2f}"
                rejects.append(rec)
                continue

            r, merged = _min_connecting_radius(
                accepted, q, int(np.ceil(gap_cells / 2.0)), max_radius)
            if merged is None:
                rec["reason"] = f"no connection within {max_radius} cells"
                rejects.append(rec)
                continue

            merged = _fill_exterior(merged)
            a_c, a_m = int(accepted.sum()), int(merged.sum())
            a_union = int((accepted | q).sum())
            bridge = max(0, a_m - a_union)
            rec.update({
                "radius": r, "a_union": a_union, "a_m": a_m, "bridge": bridge,
                "inflation": bridge / max(a_union, 1),
                "efficiency": support / max(a_m - a_c, 1),
                "merged": merged, "q": q,
            })

            fails = []
            if rec["efficiency"] < min_eff:
                fails.append(f"efficiency {rec['efficiency']:.3f}<{min_eff}")
            if rec["inflation"] > max_infl:
                fails.append(f"inflation {rec['inflation']:.3f}>{max_infl}")
            if rec["density"] < min_density:
                fails.append(f"density {rec['density']:.3f}<{min_density}")
            if fails:
                rec["reason"] = "; ".join(fails)
                rejects.append(rec)
            else:
                trials.append(rec)

        for c in contained:
            candidates.remove(c)
            report["contained"] += 1

        if not trials:
            final_rejects = rejects
            break

        # Order-independent: every candidate is re-scored against the grown
        # region each pass, and the cheapest-per-unit-evidence wins.
        best = max(trials, key=lambda t: (t["efficiency"],
                                          -t["bridge"] / max(t["support"], 1)))
        accepted = best["merged"]
        component_union |= best["q"]
        candidates = [c for c in candidates if c["id"] != best["id"]]
        report["accepted"] += 1
        report["merged"].append(best["id"])
        logger.info(
            "%s boundary candidate %d: ACCEPT  support=%d (%.2f%%) gap=%.2f "
            "radius=%d bridge=%d inflation=%.3f efficiency=%.3f",
            type, best["id"], best["support"], 100 * best["frac"],
            best["gap_world"], best["radius"], best["bridge"],
            best["inflation"], best["efficiency"],
        )

    for rec in final_rejects:
        # A significant region rejected is a different event from a speck
        # rejected, and the log has to say which: omitting a real ceiling piece
        # is a conservative choice, but it IS a choice and should be visible.
        loud = rec["frac"] >= min_frac
        logger.info(
            "%s boundary candidate %d: REJECT%s support=%d (%.2f%%) gap=%.2f "
            "%sbridge=%s inflation=%s efficiency=%s -- %s",
            type, rec["id"], "  <-- SIGNIFICANT DISCONNECTED REGION " if loud else "  ",
            rec["support"], 100 * rec["frac"], rec["gap_world"],
            f"radius={rec['radius']} " if "radius" in rec else "",
            rec.get("bridge", "-"),
            f"{rec['inflation']:.3f}" if "inflation" in rec else "-",
            f"{rec['efficiency']:.3f}" if "efficiency" in rec else "-",
            rec["reason"],
        )
    report["rejected"] = final_rejects
    report["bridge"] = accepted & ~component_union
    report["component_union"] = component_union
    return accepted, report


def _coverage_merge(original, cell, base_gap, opts, logger, type="ceiling"):
    """Merge the nearest component repeatedly until the outline covers the evidence.

    A revision of the legacy loop that keeps its simplicity and replaces both of
    its arbitrary parts. Legacy grows ONE GLOBAL kernel until
    ``len(contours) == 1``, bounded at 30 x cell: a connectivity target and a
    magic number, neither of which says anything about whether the boundary is
    right. Here instead:

      * merges are LOCAL -- the nearest component is bridged with the smallest
        radius that actually reaches it, so annexing one piece no longer
        inflates the entire boundary the way one global kernel does;
      * the target is COVERAGE of the original occupied cells. "The outline
        should contain essentially all the evidence" is a claim about the
        result; "exactly one contour" is a claim about the mask;
      * more than one contour is an acceptable outcome. Detached noise is left
        detached rather than chased.

    Coverage separates cases connectivity could not. On one building's two
    ceilings: the upper's main component already holds 99.77% of the evidence,
    so a 99% target stops immediately and the six specks that dragged the legacy
    kernel to 15 x cell are never pursued. The lower's main component holds
    93.9%, and the missing 6% is real ceiling thinly spread across the footprint
    -- the same target keeps merging and the outline expands to span the
    building, which is what the raw scan says is correct there. One number, and
    it moves in the right direction on both.

    The tail is stopped by a KNEE, not a distance limit: merging stops when the
    next bridge costs sharply more area per newly covered cell than the merges
    before it. That is self-calibrating, so it needs no threshold tracking scan
    density -- which is what makes the per-candidate thresholds in
    _evidence_merge vary between projects.
    """
    target = float(opts.get("coverage_target", 0.99))
    knee = float(opts.get("coverage_knee", 4.0))
    min_hist = int(opts.get("coverage_knee_min_merges", 3))
    max_gap = opts.get("merge_max_gap", "auto")
    if isinstance(max_gap, str) and str(max_gap).lower() == "auto":
        max_gap = 24.0 * cell
    max_gap = float(max_gap)
    max_radius = int(np.ceil(max_gap / cell)) + 2

    base_mask = _close_bool(original, int(round(base_gap / cell)))
    n, lab = cv2.connectedComponents(base_mask.astype(np.uint8))
    total = int(original.sum())
    report = {"components": n - 1, "accepted": 0, "contained": 0, "noise": 0,
              "rejected": [], "bridge": None, "merged": [], "stop": "one component"}
    if n <= 2 or total == 0:
        return _fill_exterior(base_mask), report

    comps = []
    for i in range(1, n):
        m = lab == i
        comps.append({"id": i, "mask": m, "support": int((original & m).sum())})
    comps.sort(key=lambda c: -c["support"])
    accepted = _fill_exterior(comps[0]["mask"])
    remaining = comps[1:]
    component_union = accepted.copy()
    covered = int((original & accepted).sum())

    logger.info(
        "%s boundary: %d components, main covers %.2f%% of evidence "
        "(target %.1f%%)", type, n - 1, 100.0 * covered / total, 100.0 * target,
    )

    costs, stopped = [], None
    while remaining and covered / total < target:
        # One distance transform per pass serves every candidate: it gives the
        # distance to the accepted region for every cell, so each component's
        # gap is just a min over its own cells.
        dt = cv2.distanceTransform((~accepted).astype(np.uint8), cv2.DIST_L2, 5)
        gaps = [float(dt[c["mask"]].min()) for c in remaining]
        k = int(np.argmin(gaps))
        cand, gap_cells = remaining[k], gaps[k]
        q = _fill_exterior(cand["mask"])

        if int((original & (accepted | q)).sum()) <= covered:
            remaining.pop(k)          # already inside the outline
            report["contained"] += 1
            continue

        if gap_cells * cell > max_gap:
            stopped = (f"nearest remaining component is {gap_cells * cell:.2f} "
                       f"away, past the {max_gap:.2f} rail")
            break

        r, merged = _min_connecting_radius(
            accepted, q, int(np.ceil(gap_cells / 2.0)), max_radius)
        if merged is None:
            remaining.pop(k)
            continue

        merged = _fill_exterior(merged)
        bridge = max(0, int(merged.sum()) - int((accepted | q).sum()))
        gained = int((original & merged).sum()) - covered
        cost = bridge / max(gained, 1)

        # The knee. Compared against the MEDIAN of previous merges rather than
        # the last one, so a single awkward bridge does not end the run.
        if len(costs) >= min_hist and cost > knee * float(np.median(costs)):
            stopped = (f"cost knee -- next bridge {cost:.1f} cells per covered "
                       f"cell vs median {float(np.median(costs)):.1f}")
            break

        costs.append(cost)
        accepted, covered = merged, covered + gained
        component_union |= q
        report["accepted"] += 1
        report["merged"].append(cand["id"])
        remaining.pop(k)

    if stopped is None:
        stopped = ("coverage target reached" if covered / total >= target
                   else "no components left")
    report["stop"] = stopped
    report["noise"] = len(remaining)
    report["bridge"] = accepted & ~component_union
    report["component_union"] = component_union

    contours, _ = cv2.findContours(accepted.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    logger.info(
        "%s boundary: merged %d component(s), coverage %.2f%%, %d left detached, "
        "%d contour(s) -- stopped: %s",
        type, report["accepted"], 100.0 * covered / total, report["noise"],
        len(contours), stopped,
    )
    return accepted, report


def _e57_plane_points(e57_opts, plane_model, distance_threshold, points_2d,
                      logger, type="floor"):
    """XY of raw-scan points lying on the fitted plane, or None.

    Selection is by distance to the plane, so the scan needs no labels -- which
    also sidesteps the segmentation being unreliable on exactly these surfaces.
    Used two ways: as the boundary's point source, and as HELD-OUT EVIDENCE for
    scoring connect modes, since the CSV is a ~0.1% downsample and cannot say
    whether a filled cell is real ceiling.
    """
    try:
        dense = load_e57_in_csv_frame(
            e57_opts["path"], e57_opts.get("scale", 1.0),
            e57_opts.get("basis"), voxel=e57_opts.get("voxel", 0.3),
            logger_=logger,
        )
        normal = np.asarray(plane_model[:3], dtype=float)
        norm = float(np.linalg.norm(normal))
        resid = np.abs(dense @ normal + float(plane_model[3])) / max(norm, 1e-12)
        picked = dense[resid <= float(distance_threshold)]
        # Stay inside the CSV cluster's own footprint: the plane is infinite and
        # would otherwise pick up the same elevation building-wide.
        margin = float(e57_opts.get("margin", 2.0))
        lo = points_2d.min(axis=0) - margin
        hi = points_2d.max(axis=0) + margin
        picked = picked[np.all((picked[:, :2] >= lo) & (picked[:, :2] <= hi), axis=1)]
        if len(picked) < max(100, len(points_2d) // 10):
            logger.info("%s: only %d dense points on this plane; not usable",
                        type, len(picked))
            return None
        return picked[:, :2]
    except Exception as exc:          # missing file, bad path, pye57 absent
        logger.info("%s: e57 evidence unavailable (%s)", type, exc)
        return None


def _score_against_evidence(region, evidence):
    """F1 of a claimed region against independently observed ceiling cells.

    Precision and recall are BOTH needed and neither works alone. A mode that
    claims almost nothing scores near-perfect precision with poor recall; one
    that claims the whole bounding box scores the reverse. F1 ranks them without
    anyone having to decide in advance which failure matters more.

      precision -- of the area claimed, how much is real surface
      recall    -- of the real surface, how much was captured
    """
    b, e = int(region.sum()), int(evidence.sum())
    if b == 0 or e == 0:
        return 0.0, 0.0, 0.0
    hit = int((region & evidence).sum())
    p, r = hit / b, hit / e
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r


def _component_diagnostic(original, report, accepted):
    """Colour-coded QA image separating evidence from claimed geometry.

    The distinction the old boundary mask could not show: which cells came from
    the points and which were invented by morphology. Drawn back to front so
    the original evidence stays visible on top of everything claimed.

      amber   bridge area -- created by closing, belongs to no component
      blue    the main component's filled exterior
      green   accepted candidates
      white   ORIGINAL occupied cells (the actual evidence)
    """
    h, w = original.shape
    img = np.zeros((h, w, 3), np.uint8)
    img[accepted] = (55, 55, 60)
    bridge = report.get("bridge")
    if bridge is not None:
        img[bridge] = (20, 140, 200)
    union = report.get("component_union")
    if union is not None:
        img[union & accepted] = (180, 90, 40)
    img[original] = (255, 255, 255)
    return np.flipud(img)


def _boundary_raster(points_2d, cell, fill_gap, simplify_eps_frac, logger,
                     mask_sink=None, simplify_abs=None, connect=False,
                     connect_max_cells=30.0, connect_mode=None,
                     merge_opts=None, component_sink=None, type="floor",
                     evidence_2d=None, auto_modes=("off", "legacy", "coverage"),
                     auto_fallback="legacy"):
    """Occupancy-grid + morphological-close boundary (fills sparse-density voids).

    Rasterizes the projected points at ``cell`` resolution, closes gaps up to
    ``fill_gap`` wide so low-density patches don't carve the outline, then takes
    the largest external contour and simplifies it. ``cell``/``fill_gap`` are in
    the same units as ``points_2d``. Returns a closed list of (x, y) coords, or
    [] if no contour is found (caller falls back to alphashape).
    """
    if len(points_2d) < 3:
        return []

    cell = max(float(cell), 1e-6)
    merge_opts = dict(merge_opts or {})

    # Which connect strategy. The legacy boolean still selects between the old
    # growth loop and no connecting at all, so existing callers are unaffected.
    mode = str(connect_mode or ("legacy" if connect else "off")).lower()

    # Padding must cover the LARGEST kernel any trial will use, not the initial
    # fill_gap. Sizing it from fill_gap alone (pad = 3 cells) meant a 15-cell
    # kernel got clipped by the array edge: morphology went asymmetric near the
    # border and area falsely saturated -- laramie's ceilings reported identical
    # area at 15x, 20x and 30x because the mask had hit the wall, not because
    # the shape had converged.
    max_gap = merge_opts.get("merge_max_gap", "auto")
    if isinstance(max_gap, str) and str(max_gap).lower() == "auto":
        max_gap = 24.0 * cell
    merge_radius = int(np.ceil(float(max_gap) / cell)) + 2   # evidence / coverage
    legacy_radius = int(np.ceil(connect_max_cells))          # the growth loop's bound
    if mode == "auto":
        # Auto runs SEVERAL modes on this grid, so the pad has to cover the
        # largest kernel any of them might use. Sizing it for the merge modes
        # alone clipped legacy's growth, making legacy-under-auto behave
        # differently from legacy on its own -- which would corrupt the very
        # comparison auto exists to make.
        candidate_radius = max(merge_radius, legacy_radius)
    elif mode in ("evidence", "coverage"):
        candidate_radius = merge_radius
    else:
        candidate_radius = legacy_radius
    worst_radius = max(int(np.ceil(float(fill_gap) / cell)), candidate_radius)
    pad = worst_radius + 1

    xmin, ymin = points_2d[:, 0].min(), points_2d[:, 1].min()
    ix = np.floor((points_2d[:, 0] - xmin) / cell).astype(int)
    iy = np.floor((points_2d[:, 1] - ymin) / cell).astype(int)
    grid = np.zeros((int(iy.max()) + 1 + 2 * pad, int(ix.max()) + 1 + 2 * pad), np.uint8)
    grid[iy + pad, ix + pad] = 255
    original = grid > 0

    def _close(gap):
        kk = max(1, int(round(gap / cell)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1))
        img = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, ker)
        found, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return img, found

    def _produce(m):
        """Mask + contours for one connect mode. Returns (closed, contours, report)."""
        if m in ("evidence", "coverage"):
            merge_fn = _evidence_merge if m == "evidence" else _coverage_merge
            acc, rep = merge_fn(original, cell, fill_gap, merge_opts, logger,
                                type=type)
            img = (acc * 255).astype(np.uint8)
            found, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            return img, found, rep

        img, found = _close(fill_gap)
        # Legacy: grow one global kernel until the mask is a single region. It
        # is right when a surface really is one sparsely-scanned sheet and wrong
        # when it is not, and it cannot tell the difference -- which is what
        # "auto" below settles with held-out evidence.
        if m == "legacy" and len(found) > 1:
            gap = applied = fill_gap
            for _ in range(24):
                gap *= 1.5
                if gap > connect_max_cells * cell:
                    break
                img, found = _close(gap)
                applied = gap
                if len(found) <= 1:
                    break
            # Report the gap actually APPLIED. Logging `gap` reported a value
            # 1.5x past the bound whenever the loop exited on it, i.e. a radius
            # that was never used.
            logger.info(
                "Raster boundary: grew fill_gap %.2f -> %.2f (%.0f x cell) to "
                "reach %d region(s)", fill_gap, applied, applied / cell,
                len(found),
            )
        return img, found, None

    def _claimed(found):
        """The region the boundary will actually enclose: largest contour, filled."""
        out = np.zeros(original.shape, np.uint8)
        if found:
            cv2.drawContours(out, [max(found, key=cv2.contourArea)], -1, 1,
                             cv2.FILLED)
        return out.astype(bool)

    report = None
    produced = False
    if mode == "auto":
        # Run the candidates and let INDEPENDENT evidence choose. This is the
        # one decision that can use the raw scan: a merge rule has to score each
        # candidate incrementally, before the run ends, but picking between two
        # FINISHED boundaries can afford a global check. Extraction is ~0.0 s, so
        # trying three modes is free next to the plane fit that preceded them.
        evidence = None
        if evidence_2d is not None and len(evidence_2d):
            ex = np.floor((np.asarray(evidence_2d)[:, 0] - xmin) / cell).astype(int) + pad
            ey = np.floor((np.asarray(evidence_2d)[:, 1] - ymin) / cell).astype(int) + pad
            ok = ((ex >= 0) & (ey >= 0)
                  & (ex < grid.shape[1]) & (ey < grid.shape[0]))
            evidence = np.zeros(original.shape, bool)
            evidence[ey[ok], ex[ok]] = True

        if evidence is None or not evidence.any():
            # No evidence means no basis to choose on. Falling back to a named
            # mode is honest; inventing a proxy criterion and calling the result
            # "selected" would not be.
            mode = str(auto_fallback).lower()
            logger.info(
                "Raster boundary: connect=auto needs raw-scan evidence to "
                "choose and none is available -- using %s", mode)
        else:
            best = None
            for m in auto_modes:
                img_m, found_m, rep_m = _produce(m)
                f1, prec, rec = _score_against_evidence(_claimed(found_m), evidence)
                logger.info(
                    "Raster boundary [auto] %-8s F1 %.3f (precision %.3f, "
                    "recall %.3f), %d contour(s)", m, f1, prec, rec, len(found_m))
                # Rank on F1 rounded to 3dp, then on FEWER CONTOURS. Modes can
                # tie exactly -- coverage that merges nothing encloses the same
                # region as off -- and leaving that to iteration order picks
                # arbitrarily. Fewer contours is the right tiebreak because only
                # the largest is kept: every other one is silently discarded.
                key = (round(f1, 3), -len(found_m))
                if best is None or key > best[0]:
                    best = (key, m, img_m, found_m, rep_m)
            _, mode, closed, contours, report = best
            produced = True
            logger.info("Raster boundary [auto] -> %s", mode)

    if not produced:
        closed, contours, report = _produce(mode)

    if component_sink is not None and report is not None:
        component_sink(_component_diagnostic(original, report, closed > 0))

    if mask_sink is not None:
        # Flip so the QA image reads with +y up, like the boundary plots.
        mask_sink(np.dstack([np.flipud(closed)] * 3))

    if not contours:
        logger.info("Raster boundary: no contour found.")
        return []

    c = max(contours, key=cv2.contourArea)

    # Closing reduces fragmentation but does not prevent it: one dataset still
    # produced 41 separate contours. Keeping only the largest is silent data
    # loss, so say what was dropped.
    if len(contours) > 1:
        areas = sorted((cv2.contourArea(x) * cell * cell for x in contours),
                       reverse=True)
        logger.info(
            "Raster boundary: %d contours, kept largest (%.0f sq units), "
            "dropped %d totalling %.0f sq units",
            len(contours), areas[0], len(areas) - 1, sum(areas[1:]),
        )

    # An absolute tolerance wins when given: a fraction of the perimeter scales
    # with building size, so the same fraction means different geometry on
    # different projects.
    if simplify_abs is not None:
        eps = max(float(simplify_abs), 0.0)
    else:
        eps = max(float(simplify_eps_frac), 0.0) * cv2.arcLength(c, True)
    if eps > 0:
        c = cv2.approxPolyDP(c, eps, True)

    px = c[:, 0, :].astype(float)
    coords = [
        (xmin + (col - pad + 0.5) * cell, ymin + (row - pad + 0.5) * cell)
        for col, row in px
    ]
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

def point_axis_align(df, survey_basis):
    """Rotate a frame's xyz into the axis-aligned working frame.

    ``survey_basis`` is SURVEY_BASIS exactly as configured, applied directly with
    no hidden transpose: the alignment is ``xyz @ SURVEY_BASIS``, and results
    rotate back out with ``@ SURVEY_BASIS.T``, so in-then-out is the identity and
    output stays in the input cloud's frame.

    Getting that direction backwards rotates the building the wrong way by TWICE
    the yaw, and nothing raises -- a transposed rotation is still orthonormal
    with determinant +1, so the geometry looks plausible. The check that catches
    it is the footprint: a correct basis can only SHRINK the axis-aligned
    bounding area, while a transposed one enlarges it.
    """
    xyz = df[['x', 'y', 'z']].values
    rotated_xyz = xyz @ np.asarray(survey_basis)

    # Convert back to DataFrame
    df_rotated = pd.DataFrame(rotated_xyz, columns=['x', 'y', 'z'])
    df_concate = pd.concat([df_rotated, df.iloc[:, 3:]], axis=1)

    return df_concate

# Grey for DBSCAN noise, then a categorical palette for the clusters. Kept
# explicit (rather than a colormap lookup) so the same cluster id always draws
# the same colour across eps values, which makes two snapshots comparable.
_NOISE_RGB = (90, 90, 96)
_CLUSTER_RGB = (
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
)


def label_colors(labels):
    """Per-point uint8 RGB: grey for noise (-1), one colour per cluster."""
    ids = [l for l in np.unique(labels) if l != -1]
    lookup = {l: _CLUSTER_RGB[i % len(_CLUSTER_RGB)] for i, l in enumerate(ids)}
    out = np.empty((len(labels), 3), dtype=np.uint8)
    for i, l in enumerate(labels):
        out[i] = _NOISE_RGB if l == -1 else lookup[l]
    return out


def detect_levels(z, bin_size=0.25, min_frac=0.02, min_gap=1.64,
                  min_points_frac=0.02):
    """Find storey elevations by splitting the z-histogram at genuine voids.

    Levels are separated by EMPTY SPACE, not by how far apart their peaks are.
    A gap is direct evidence of separation -- nothing connects the storeys --
    whereas peak distance is indirect and, worse, merges transitively: each peak
    is compared to the previous one, so a chain of closely spaced peaks collapses
    across an arbitrary span. On one building's ceiling that chained all eight
    runs into a single "level" at z=16.25, a height with no points at all,
    sitting in the void between two real ceilings at 12.4 and 24.2.

    Measured across three buildings the gap criterion holds over a 12.5x range
    of ``min_gap`` against ~3x for the peak-distance rule it replaces, because
    its two bounds are physically independent: below, the density dips inside a
    single tiered level (~0.16 m); above, the minimum habitable floor-to-floor
    less the levels' own point spread (~1.4 m).

    The elevation returned is the COUNT-WEIGHTED centroid of each group, so a
    sparse tier cannot pull it the way an unweighted mean of peaks does.

    All distances are in the cloud's own units (feet for these datasets).
    """
    z = np.asarray(z, dtype=float)
    lo, hi = z.min(), z.max()
    nbins = max(2, int(np.ceil((hi - lo) / max(bin_size, 1e-9))))
    counts, edges = np.histogram(z, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    runs = _occupied_runs(counts, min_frac)
    if not runs:
        return [float(np.median(z))]

    # Join runs whose separation is too small to be a storey. Unlike a peak
    # chain this cannot propagate: a void either exceeds min_gap or it does not.
    groups = [list(runs[0])]
    for r in runs[1:]:
        if centres[r[0]] - centres[groups[-1][-1]] < min_gap:
            groups[-1].extend(r)
        else:
            groups.append(list(r))

    kept = [g for g in groups if counts[g].sum() >= min_points_frac * len(z)]
    if not kept:
        kept = [max(groups, key=lambda g: counts[g].sum())]

    return [float(np.average(centres[g], weights=counts[g])) for g in kept]


def _occupied_runs(counts, min_frac):
    """Contiguous stretches of bins holding at least min_frac of the tallest."""
    occupied = counts >= counts.max() * min_frac
    runs, run = [], []
    for i, on in enumerate(occupied):
        if on:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    if run:
        runs.append(run)
    return runs


def dominant_mode_range(z, bin_size=0.25, min_frac=0.02):
    """(lo, hi) of the densest contiguous band of heights.

    A level's points are not all its surface. On one of our buildings the upper
    level's "Floor" label held four horizontal sheets a foot apart, and the two
    lowest were the storey's CEILING misclassified as floor -- 21% of the points
    there also carried a Ceiling label, against 1% at the real floor.

    The real surface is the densest one, so the fix is to keep only the run of
    bins containing the tallest bin and discard the rest. That reuses the same
    occupancy threshold the level split already uses, so it adds no parameter,
    and it is a no-op on a clean single-surface level.
    """
    z = np.asarray(z, dtype=float)
    nbins = max(2, int(np.ceil((z.max() - z.min()) / max(bin_size, 1e-9))))
    counts, edges = np.histogram(z, bins=nbins)
    runs = _occupied_runs(counts, min_frac)
    if not runs:
        return float(z.min()), float(z.max())

    peak = int(np.argmax(counts))
    for r in runs:
        if r[0] <= peak <= r[-1]:
            return float(edges[r[0]]), float(edges[r[-1] + 1])
    biggest = max(runs, key=lambda r: counts[r].sum())
    return float(edges[biggest[0]]), float(edges[biggest[-1] + 1])


def histogram_cluster_labels(pts, bin_size=0.25, min_frac=0.02,
                             min_storey_sep=8.0, dominant_only=False,
                             logger_=None, min_gap=1.64,
                             min_points_frac=0.02):
    """Assign each point to a storey by elevation; -1 for points far from any.

    Boundaries sit at the midpoints between detected levels. Points further than
    half a storey from EVERY level are labelled noise -- that is the histogram
    equivalent of DBSCAN's noise class, and it matters because segmentation
    labels are not clean: one of our datasets has ~1% of its "Floor" points
    sitting near ceiling height, which would otherwise be swept into a slab.
    """
    levels = detect_levels(pts[:, 2], bin_size, min_frac, min_gap,
                           min_points_frac)
    if len(levels) == 1:
        labels = np.zeros(len(pts), dtype=int)
    else:
        bounds = [(levels[i] + levels[i + 1]) / 2.0 for i in range(len(levels) - 1)]
        labels = np.digitize(pts[:, 2], bounds)

    distance = np.min(
        np.abs(pts[:, 2][:, None] - np.asarray(levels)[None, :]), axis=1
    )
    labels = np.where(distance > min_storey_sep / 2.0, -1, labels)

    # Keep only the densest band within each level -- see dominant_mode_range.
    if dominant_only:
        for k in range(len(levels)):
            member = labels == k
            if member.sum() < 100:
                continue
            lo, hi = dominant_mode_range(pts[member, 2], bin_size, min_frac)
            drop = member & ((pts[:, 2] < lo) | (pts[:, 2] > hi))
            if drop.any():
                labels[drop] = -1
                if logger_ is not None:
                    logger_.info(
                        "level %d: kept densest band z %.2f..%.2f, dropped "
                        "%d of %d points outside it", k, lo, hi,
                        int(drop.sum()), int(member.sum()),
                    )
    return labels, levels


def plot_z_histogram(z, levels, bin_size, min_frac, min_storey_sep, type="floor",
                     min_gap=None):
    """The 1-D vertical histogram behind the level split, as a QA figure.

    Height runs up the y-axis so the plot reads like a building section. It
    shows every decision the splitter made: the peak-detection floor
    (min_frac), the elevations it settled on, the slice boundaries at their
    midpoints, and the +/- half-storey band outside which points are dropped as
    noise. If a level looks wrong, this says whether the cause was the peak
    threshold, the merge rule, or the data itself.
    """
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    lo, hi = float(z.min()), float(z.max())
    nbins = max(2, int(np.ceil((hi - lo) / max(bin_size, 1e-9))))
    counts, edges = np.histogram(z, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(centres, counts, height=bin_size, color="0.55", linewidth=0)

    threshold = counts.max() * min_frac
    ax.axvline(threshold, color="tab:orange", ls=":", lw=1.4,
               label=f"peak threshold ({min_frac:.0%} of max)")

    half = min_storey_sep / 2.0
    for i, level in enumerate(levels):
        ax.axhline(level, color="tab:red", lw=1.6,
                   label="detected level" if i == 0 else None)
        ax.axhspan(level - half, level + half, color="tab:green", alpha=0.08,
                   label="kept (within half a storey)" if i == 0 else None)
        ax.annotate(f" z={level:.2f}", (counts.max() * 0.62, level),
                    color="tab:red", fontsize=9, va="bottom")

    for i in range(len(levels) - 1):
        boundary = (levels[i] + levels[i + 1]) / 2.0
        ax.axhline(boundary, color="tab:blue", ls="--", lw=1.2,
                   label="slice boundary" if i == 0 else None)

    ax.set_xlabel("point count")
    ax.set_ylabel("height (cloud units)")
    # Name the parameter that actually splits levels (min_gap), not the one that
    # only sets the noise radius -- they are different knobs and the title used
    # to advertise the wrong one.
    gap_txt = f"split at voids > {min_gap:.2f}" if min_gap else "split at voids"
    ax.set_title(f"{type}: 1-D height histogram\n"
                 f"{len(levels)} level(s)  |  bin {bin_size:.2f}  |  {gap_txt}"
                 f"  |  noise beyond {min_storey_sep / 2:.2f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def cluster_floor_ceiling(df, eps, min_samples, type, blobs=None, min_points=10000,
                          local_dir=None, method="histogram", hist_opts=None,
                          dominant_only=True):
    """Split floor/ceiling points into one cluster per storey.

    ``method="histogram"`` (default) slices the z-histogram: measured on three
    datasets it fits planes at least as well as DBSCAN and runs in ~10 ms on a
    full cloud versus minutes, with no subsampling needed. ``method="dbscan"``
    keeps the original 6-D (xyz+rgb) density clustering, which can also separate
    two disjoint surfaces that share a height -- something elevation slicing
    cannot do.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    pts = df[["x", "y", "z"]].to_numpy()
    col_u8 = df[["r", "g", "b"]].to_numpy()
    col = col_u8 / 255.0

    # NOTE: this snapshot is the INPUT to clustering, drawn in the scan's own
    # colours -- it is written before DBSCAN runs and carries no cluster
    # information. The DBSCAN result is the separate "_dbscan" snapshot below.
    if blobs is not None or local_dir:
        write_html_output(
            snapshot_plotly_html_bytes(
                points_xyz=pts,
                colors_rgb=col_u8,     # uint8 [0,255]
                point_size=2,
            ),
            f"{type}_cluster.html",
            blobs=blobs,
            local_dir=local_dir,
        )

    if str(method).lower() == "dbscan":
        feats = StandardScaler().fit_transform(np.hstack([pts, col]))  # xyz + rgb
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
        logger.info("%s clustering: DBSCAN eps=%s min_samples=%s", type, eps,
                    min_samples)
    else:
        opts = hist_opts or {}
        bin_size = opts.get("bin_size", 0.25)
        min_frac = opts.get("min_frac", 0.02)
        min_storey_sep = opts.get("min_storey_sep", 8.0)
        labels, levels = histogram_cluster_labels(
            pts, bin_size=bin_size, min_frac=min_frac,
            min_storey_sep=min_storey_sep, dominant_only=dominant_only,
            logger_=logger, min_gap=opts.get("min_gap", 1.64),
            min_points_frac=opts.get("min_points_frac", 0.02),
        )
        logger.info("%s clustering: histogram, %d level(s) at z=%s", type,
                    len(levels), [round(v, 2) for v in levels])
        if blobs is not None or local_dir:
            write_figure_output(
                plot_z_histogram(pts[:, 2], levels, bin_size, min_frac,
                                 min_storey_sep, type,
                                 min_gap=opts.get("min_gap")),
                f"{type}_zhistogram.png", blobs, local_dir,
            )

    # build cluster dict (skip noise and small clusters)
    cluster_dict = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        m = labels == lab
        if m.sum() <= min_points:
            continue
        cluster_dict[int(lab)] = np.hstack([pts[m], col_u8[m]])  # xyz + rgb(0-255)

    # The clustering result itself: every point coloured by its label, INCLUDING
    # the noise and the below-min_points clusters that cluster_dict drops. Those
    # discards are exactly what you need to see when tuning eps -- a merge and a
    # shatter both leave the surviving clusters looking plausible on their own.
    if blobs is not None or local_dir:
        n_noise = int((labels == -1).sum())
        n_small = len([l for l in np.unique(labels)
                       if l != -1 and (labels == l).sum() <= min_points])
        logger.info(
            "%s DBSCAN: %d clusters, %d kept, %d dropped as small, %.1f%% noise",
            type, len(np.unique(labels[labels != -1])), len(cluster_dict),
            n_small, 100.0 * n_noise / max(1, len(labels)),
        )
        write_html_output(
            snapshot_plotly_html_bytes(
                points_xyz=pts,
                colors_rgb=label_colors(labels),
                point_size=2,
            ),
            f"{type}_dbscan.html",
            blobs=blobs,
            local_dir=local_dir,
        )

    return cluster_dict

def fit_plane_irls(points, iterations=20, tukey_c=4.685, tol=1e-9):
    """Robust plane fit by iteratively reweighted least squares. Deterministic.

    Fits z = a*x + b*y + c, which for floors and ceilings is not a restriction
    (a vertical plane is not one) and makes the near-horizontal constraint
    automatic rather than a rejection test.

    Tukey's biweight gives gross outliers weight exactly zero rather than merely
    down-weighting them, so points from another surface cannot drag the plane.
    The scale is re-estimated each pass from the median absolute deviation,
    which no outlier can move.

    Chosen over RANSAC because it is REPRODUCIBLE. Measured over five repeats,
    the RANSAC-derived threshold on a tiered band swung 16x (0.136 to 2.223)
    while this returned an identical answer every time. It is also insensitive
    to its starting point -- six different seeds converge to the same plane.

    Returns (normal, d, scale) for the plane n.p + d = 0, normal unit-length and
    pointing up, plus the robust residual scale.
    """
    pts = np.asarray(points, dtype=float)
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    z = pts[:, 2]
    w = np.ones(len(pts))
    coef = np.array([0.0, 0.0, float(np.median(z))])
    scale = 0.0

    for _ in range(max(1, iterations)):
        try:
            new_coef, *_ = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)
        except np.linalg.LinAlgError:
            break
        shift = float(np.max(np.abs(new_coef - coef)))
        coef = new_coef
        residual = z - A @ coef
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        if scale < 1e-12:
            break
        u = residual / (tukey_c * scale)
        w = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
        if shift < tol:          # converged -- further passes change nothing
            break

    a, b, c = coef
    normal = np.array([-a, -b, 1.0])
    norm = float(np.linalg.norm(normal))
    return normal / norm, float(-c / norm), float(scale)


def _plane_residuals(points, model):
    """Perpendicular distance from each point to a plane (a,b,c,d)."""
    normal = np.asarray(model[:3], dtype=float)
    return np.abs(points @ normal + float(model[3])) / np.linalg.norm(normal)


def _curve_knee(residual, hi, samples=60):
    """Threshold where inlier-fraction-vs-threshold stops rising steeply.

    Widening the threshold recruits points quickly while it is still crossing
    the slab's own thickness, then flattens once the slab is captured and
    further widening only sweeps in unrelated structure. The knee is that
    transition: the point furthest from the chord joining the curve's ends,
    with both axes normalised to [0, 1].
    """
    grid = np.linspace(hi / samples, hi, samples)
    frac = np.array([np.mean(residual <= t) for t in grid])
    x = (grid - grid[0]) / max(grid[-1] - grid[0], 1e-12)
    y = (frac - frac[0]) / max(frac[-1] - frac[0], 1e-12)
    return float(grid[int(np.argmax(y - x))])


def required_iterations(inlier_frac, ransac_n, probability=0.999):
    """Iterations needed to draw one all-inlier sample with `probability`.

    N = log(1 - p) / log(1 - w**n). At the inlier ratios these clouds produce
    (0.75-0.98) this lands in the tens, against a configured 1000.
    """
    w = min(max(float(inlier_frac), 1e-6), 1.0 - 1e-9)
    denom = np.log1p(-(w ** int(ransac_n)))
    if denom >= -1e-12:
        return 1
    return int(np.ceil(np.log(1.0 - probability) / denom))


def auto_ransac_params(points, ransac_n, passes=2, lo=None, hi=None,
                       probability=0.999, logger_=None):
    """Derive (distance_threshold, num_iterations) from the points themselves.

    The threshold needs a plane to measure residuals against, and the plane
    needs a threshold, so it is iterated: fit generously, read the knee, refit.
    Two passes suffice where the band really is one plane. Where it is not --
    a tiered or sloped deck -- the knee moves between passes and the clamp is
    what keeps the result sane, so `lo`/`hi` are not optional in practice.
    """
    import open3d as o3d

    # Start wide enough to span the band, so the first plane is not fit to a
    # sliver of the slab.
    threshold = float(np.percentile(np.abs(points[:, 2] - np.median(points[:, 2])), 95))
    threshold = max(threshold, 1e-6)
    knee = threshold

    for _ in range(max(1, passes)):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        model, _ = pcd.segment_plane(threshold, int(ransac_n), 200)
        residual = _plane_residuals(points, model)
        knee = _curve_knee(residual, hi=float(np.percentile(residual, 99)))
        threshold = knee

    clamped = threshold
    if lo is not None:
        clamped = max(clamped, float(lo))
    if hi is not None:
        clamped = min(clamped, float(hi))
    if logger_ is not None and abs(clamped - threshold) > 1e-9:
        logger_.info("auto DIS_THR: knee %.3f clamped to %.3f -- the band may "
                     "not be a single plane", threshold, clamped)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, inliers = pcd.segment_plane(clamped, int(ransac_n), 200)
    w = len(inliers) / max(1, len(points))
    return clamped, required_iterations(w, ransac_n, probability), w


def _resolve_auto(distance_threshold, num_iterations, xyz, ransac_n, auto_opts,
                  logger, type):
    """Replace "auto" values with data-derived ones; pass others through."""
    wants_thr = isinstance(distance_threshold, str) and distance_threshold.lower() == "auto"
    wants_iter = isinstance(num_iterations, str) and str(num_iterations).lower() == "auto"
    if not (wants_thr or wants_iter):
        return distance_threshold, num_iterations

    opts = auto_opts or {}
    thr, iters, w = auto_ransac_params(
        xyz, ransac_n,
        lo=opts.get("min_threshold"),
        hi=opts.get("max_threshold"),
        logger_=logger,
    )
    safety = float(opts.get("iteration_safety", 4.0))
    if wants_thr:
        distance_threshold = thr
    if wants_iter:
        num_iterations = max(int(opts.get("min_iterations", 50)),
                             int(np.ceil(iters * safety)))
    logger.info("%s auto RANSAC: DIS_THR=%.3f NUM_ITER=%s (inliers %.1f%%, "
                "ransac_n=%s)", type, float(distance_threshold), num_iterations,
                100.0 * w, ransac_n)
    return distance_threshold, num_iterations


def fit_ceiling_floor(
    df,
    distance_threshold,
    ransac_n,
    num_iterations,
    alpha_value,
    logger,
    type='floor',
    blobs=None,
    snapshot_idx=None,
    boundary_opts=None,
    auto_opts=None,
    local_dir=None,
    plane_method="irls",
    e57_opts=None,
    ):
    import matplotlib.pyplot as plt
    import open3d as o3d

    # Boundary method + raster knobs (default = original alphashape behaviour).
    boundary_opts = boundary_opts or {}
    boundary_method = str(boundary_opts.get("method", "alphashape")).lower()

    # Snapshots go to blob when configured, else to local_dir, else nowhere.
    emit_snapshots = blobs is not None or bool(local_dir)
    mask_sink = component_sink = None
    if emit_snapshots and boundary_method == "raster":
        def mask_sink(image_rgb):
            write_image_output(image_rgb, f"{type}_boundary_mask_{snapshot_idx}.png",
                               blobs, local_dir)

        def component_sink(image_rgb):
            write_image_output(image_rgb,
                               f"{type}_boundary_components_{snapshot_idx}.png",
                               blobs, local_dir)

    # Extract XYZ and RGB columns
    xyz = df[['x', 'y', 'z']].values  # Point coordinates
    rgb = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values (0-1)
    points = df[['x', 'y']].values
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # Assign RGB colors

    if str(plane_method).lower() == "irls":
        # Deterministic robust fit; the inlier threshold comes from the fit's
        # own residual scale unless one was configured.
        normal, d, scale = fit_plane_irls(xyz)
        if isinstance(distance_threshold, str) or distance_threshold is None:
            distance_threshold = max(4.685 * scale, 1e-6)
            if auto_opts:
                lo, hi = auto_opts.get("min_threshold"), auto_opts.get("max_threshold")
                if lo is not None:
                    distance_threshold = max(distance_threshold, float(lo))
                if hi is not None:
                    distance_threshold = min(distance_threshold, float(hi))
        plane_model = [normal[0], normal[1], normal[2], d]
        residual = np.abs(xyz @ normal + d)
        inliers = np.where(residual <= distance_threshold)[0].tolist()
        logger.info(
            "%s IRLS plane: tilt %.2f deg, scale %.3f, DIS_THR %.3f, "
            "%d/%d inliers (%.1f%%)", type,
            float(np.degrees(np.arccos(min(1.0, abs(normal[2]))))), scale,
            float(distance_threshold), len(inliers), len(xyz),
            100.0 * len(inliers) / max(1, len(xyz)),
        )
    else:
        # DIS_THR / NUM_ITER may be "auto", in which case they are read off this
        # cluster's own residual distribution rather than configured globally.
        distance_threshold, num_iterations = _resolve_auto(
            distance_threshold, num_iterations, xyz, ransac_n, auto_opts, logger, type
        )
        plane_model, inliers = pcd.segment_plane(distance_threshold,
                                                 ransac_n,
                                                 num_iterations)

    # Extract plane parameters (ax + by + cz + d = 0)
    a, b, c, d = plane_model
    # print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    # Extract inlier and outlier points
    inlier_cloud = pcd.select_by_index(inliers)  # Points belonging to the plane
    outlier_cloud = pcd.select_by_index(inliers, invert=True)  # Points not in the plane

    # Color the plane points (for visualization)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for plane
    outlier_cloud.paint_uniform_color([0, 0, 1.0])  # Blue for non-plane

    # Visualize the result
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Plane Fitting")
    if emit_snapshots:
        pts = np.asarray(pcd.points)  # (N, 3)
        inliers = np.asarray(inliers, dtype=int)
        inlier_pts = pts[inliers]  # plane points
        outlier_pts = np.delete(pts, inliers, axis=0)

        snapshot_html_bytes = plot_plane_inliers_outliers_html_bytes(
            inlier_pts,
            outlier_pts,
            point_size=2,
        )
        write_html_output(snapshot_html_bytes,
                          f"{type}_ransac_{snapshot_idx}.html", blobs, local_dir)

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    bbox = inlier_cloud.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green box
    bbox_zmin = bbox.get_min_bound()[2]  # Compute the center
    bbox_zmax = bbox.get_max_bound()[2]

    # o3d.visualization.draw_geometries([inlier_cloud, bbox])
    if emit_snapshots:
        inlier_pts = np.asarray(inlier_cloud.points)  # (Ni, 3)

        # Oriented bounding box from inliers
        bbox = inlier_cloud.get_oriented_bounding_box()
        bbox.color = (0, 1, 0)

        html_bytes = plot_inliers_with_obb_html_bytes(inlier_pts, bbox, point_size=2)
        write_html_output(html_bytes, f"{type}_planefit_{snapshot_idx}.html",
                          blobs, local_dir)

    floor_points = np.asarray(pcd.points)[inliers]
    floor_points_2d = floor_points[:, :2]

    # Optionally re-select the boundary's points from the ORIGINAL scan instead
    # of the segmented CSV. Selection is by distance to the plane just fitted,
    # so the e57 needs no labels -- which also sidesteps the segmentation being
    # unreliable on exactly these surfaces (one ceiling's own label fitted at
    # 0.712 median residual while floor-labelled points at the same height fitted
    # at 0.023). The plane and the levels still come from the CSV.
    dense_2d = None
    if e57_opts and e57_opts.get("path"):
        source = str(e57_opts.get("source", "csv")).lower()
        connect_mode_req = str((boundary_opts or {}).get("connect_mode") or "").lower()
        # Load the scan when it is the boundary SOURCE, and also when the
        # connect mode is "auto" -- which needs it as held-out evidence to score
        # the candidate modes against. Streaming is cached per (path, scale,
        # voxel), so wanting it for both costs one pass.
        if source == "e57" or connect_mode_req == "auto":
            dense_2d = _e57_plane_points(
                e57_opts, plane_model, distance_threshold, floor_points_2d,
                logger, type,
            )
        if source == "e57" and dense_2d is not None:
            logger.info(
                "%s boundary from e57: %d CSV inliers -> %d dense points "
                "within %.3f of the plane", type, len(floor_points_2d),
                len(dense_2d), float(distance_threshold),
            )
            floor_points_2d = dense_2d
            if connect_mode_req == "auto":
                # The two features are individually fine and together circular:
                # auto would score the boundary against the very points it was
                # just built from. Precision and recall both approach 1 for any
                # mode that traces tightly, "off" wins trivially, and the scores
                # look excellent while measuring nothing. Evidence has to be
                # HELD OUT. Dropping it here makes auto take its documented
                # no-evidence path rather than report confident nonsense.
                logger.info(
                    "%s: connect_mode='auto' with BOUNDARY_SOURCE='e57' would "
                    "score the boundary against its own input -- evidence must "
                    "be held out, so auto is disabled for this surface",
                    type,
                )
                dense_2d = None

    # Boundary extraction. "raster" fills sparse-density voids (no zigzag);
    # "alphashape" is the original concave hull. Raster falls back to alphashape
    # if it fails to find a contour, so behaviour degrades gracefully.
    # `used` records what ACTUALLY produced the boundary, including the silent
    # raster -> alphashape fallback. Both QA figures used to be titled "Alpha
    # Shape" unconditionally, so a raster boundary was labelled as a method it
    # had not used -- and the fallback, which changes the geometry completely,
    # left no trace on the image at all.
    if boundary_method == "raster":
        opts = resolve_boundary_opts(floor_points_2d, boundary_opts, logger)
        connect_mode = str(opts.get("connect_mode")
                           or ("legacy" if opts.get("connect") else "off")).lower()
        boundary_coords = _boundary_raster(
            floor_points_2d,
            opts.get("cell", 0.25),
            opts.get("fill_gap", 1.0),
            opts.get("simplify_eps_frac", 0.02),
            logger,
            mask_sink,
            simplify_abs=opts.get("simplify_abs"),
            connect=bool(opts.get("connect", False)),
            connect_mode=connect_mode,
            merge_opts=opts,
            component_sink=component_sink,
            type=type,
            # Held-out evidence for connect="auto". None everywhere else, and
            # None here too when no scan is configured -- in which case auto
            # says so and falls back rather than guessing.
            evidence_2d=dense_2d,
            auto_fallback=opts.get("auto_fallback", "legacy"),
        )
        used = (f"raster, connect={connect_mode}, cell="
                f"{float(opts.get('cell', 0.0)):.3f}, simplify="
                f"{float(opts.get('simplify_abs') or 0.0):.3f}")
        if len(boundary_coords) < 3:
            logger.info("Raster boundary empty; falling back to alphashape.")
            boundary_coords = _boundary_alphashape(floor_points_2d, alpha_value, logger)
            used = f"raster FOUND NOTHING -> alphashape, alpha={alpha_value}"
    else:
        boundary_coords = _boundary_alphashape(floor_points_2d, alpha_value, logger)
        used = f"alphashape, alpha={alpha_value}"

    # Plot if boundary was found
    if boundary_coords:
        boundary_array = np.array(boundary_coords)
        fig, ax = plt.subplots()
        ax.scatter(points[:, 0], points[:, 1], s=10, label=f"{type} points")
        ax.plot(boundary_array[:, 0], boundary_array[:, 1], "r-", linewidth=2, label="Boundary")
        ax.scatter(boundary_array[:, 0], boundary_array[:, 1], s=1, color="red", label="Corner Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{type} {snapshot_idx} boundary\n{used}", fontsize=9)
        ax.legend()
        ax.grid(True)
        ax.axis("equal")

        write_figure_output(fig, f"{type}_boundary_{snapshot_idx}.png",
                            blobs, local_dir)
    else:
        logger.info("No valid boundary found.")
    
    line = LineString(boundary_coords)
    simplified = line.simplify(tolerance=0.8)  # tweak tolerance

    corner_coords = np.array(simplified.coords)

    extruded_coords = []
    for x, y in corner_coords:
        extruded_coords.append([x, y, bbox_zmin])
        extruded_coords.append([x, y, bbox_zmax])

    # Plot
    fig, ax = plt.subplots()
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.scatter(corner_coords[:, 0], corner_coords[:, 1], color="red", label="Corner Points")
    # These are the points that actually reach edgePoints, so the title names
    # BOTH simplifications: the configurable one inside the boundary method, and
    # the fixed 0.8 applied here afterwards. The second is not a parameter and
    # is coarser than the derived tolerance whenever cell < 0.533, in which case
    # it -- not BOUNDARY_SIMPLIFY_* -- is what set this vertex count.
    plt.title(f"{type} {snapshot_idx} edge points -- {len(corner_coords)} vertices\n"
              f"{used}, then simplify(0.8)", fontsize=9)

    # write_figure_output closes the figure even when no sink is configured --
    # these are created one per cluster in a loop, so leaving them open leaks.
    write_figure_output(fig, f"{type}_edgepoints_{snapshot_idx}.png",
                        blobs, local_dir)

    return bbox_zmin, bbox_zmax, extruded_coords

def sorted_merged_floor_ceiling_plane(bbox_arr):
    plane_arr = sorted(bbox_arr, key=lambda x: x[1]) #sort by zmin
    return plane_arr

def points_between_level(bbox1, bbox2, xyzrgb):
    z_min_floor = bbox1[1]
    z_max_floor = bbox2[0]
    # Filter points between the floors
    filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor) & (xyzrgb[:, 2] < z_max_floor)]
    return filtered_points

def project_points_to_floor(filtered_points, bins):
    import matplotlib.pyplot as plt

    # Project to XY plane (ignore Z)
    xy_projected = filtered_points[:, :2]  # Keep only x, y
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(xy_projected[:, 0], xy_projected[:, 1], bins=bins)

    # Plot histogram
    plt.figure(figsize=(10, 8))
    plt.imshow(hist.T, origin='lower', cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.axis('off')

    # Save plot to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    img_bytes.seek(0) 
    img = Image.open(img_bytes)
    img_array = np.array(img)
    bgr_arr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return xy_projected, x_edges, y_edges, bgr_arr





