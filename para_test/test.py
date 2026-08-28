"""Floor-stage parameter test: 1-D height histogram clustering + auto plane fit.

Set PROJECT below and run from anywhere:

    python para_test/test.py

Outputs (point cloud, clustering, RANSAC, boundary) are written under
para_test_output/<project>/ instead of blob storage, because LOCAL_OUTPUT_DIR is
set and no logging_blob_location is passed.

To add a project, add an entry to PROJECTS. The survey basis is estimated from
the cloud's own wall points unless you supply a registered one.
"""

import math
import sys
from pathlib import Path

import pandas as pd

# Run from any working directory: resolve everything against the repo root, and
# make the repo root importable so `survey_basis` is found.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from post_process import post_process
from survey_basis import survey_basis_from_df, survey_basis_from_e57

# The package logs to an in-memory handler destined for blob storage, with
# propagate=False -- so nothing reaches the console. Attach a stream handler so
# the stage diagnostics (auto-derived parameters, dropped contours, clamped
# thresholds) are visible while tuning.
import logging

_pp_log = logging.getLogger("Infer")
if not any(isinstance(h, logging.StreamHandler) for h in _pp_log.handlers):
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("   [pp] %(message)s"))
    _pp_log.addHandler(_h)

# ---------------------------------------------------------------- projects

PROJECT = "laramie"          # <-- switch project here

# Walls run the LETR line-segmentation model, which takes a minute or two per
# level on CPU. Turn off to iterate on floors/ceilings alone.
RUN_WALLS = True
LINE_SEG_MODEL = "checkpoints/checkpoint0024.pth"

# How the raster boundary handles a mask that comes out in several pieces:
#
#   "off"       one fixed close at fill_gap, then keep the largest contour.
#               Never invents area; loses every fragment but the biggest.
#
#   "legacy"    grow ONE GLOBAL kernel until len(contours) == 1, bounded at
#               30 x cell. Both parts are arbitrary: connectivity says nothing
#               about whether the boundary is right, and one global kernel means
#               annexing a single speck inflates the whole outline.
#
#   "evidence"  judge each detached component separately, accepting a merge only
#               if the original occupied cells it contributes justify the area
#               the bridge invents. Principled, but five thresholds, three of
#               them measured against quantities that scale with scan density --
#               so they do not port between projects.
#
#   "coverage"  merge the NEAREST component repeatedly until the outline
#               contains ~all the evidence. Local merges like "evidence", but
#               ONE target instead of five thresholds, and the tail is stopped
#               by a cost knee -- the next bridge costing far more area per
#               newly covered cell than the ones before it -- rather than by a
#               distance limit, so it calibrates itself to the project.
#
# Why coverage looks promising on the two ceilings we understand: laramie's
# upper main component already holds 99.77% of the evidence, so a 99% target
# stops immediately and the six specks that dragged the legacy kernel to
# 15 x cell are never pursued. The lower one holds 93.9%, and the missing 6% is
# real ceiling spread thinly across the footprint -- so the same target keeps
# merging, which is what the raw scan says is correct there. "evidence" gets the
# first right (1.06x -> 1.02x the floor beneath) and the second badly wrong
# (0.94x -> 0.54x, 50 -> 368 edge points), because it judges those scattered
# fragments one at a time and each is individually too small to pass.
#
# NONE of this is settled. Turn COMPARE on to see every mode side by side.
CONNECT_MODE_C = "coverage"
CONNECT_MODE_F = "coverage"
COMPARE_CONNECT_MODES = True    # also run every other mode and print them all

# Where SURVEY_BASIS comes from when a project has no "basis" entry:
#   "e57" -- mid-height slab of the raw scan. Cross-checked against the two
#            registered bases we have, this lands 0.028 deg (laramie) and
#            0.008 deg (corteva) off, versus 0.064 / 0.066 from the CSV. More
#            points and no dependence on segmentation. Streams the whole file
#            once (~1-2 min for 2.7 GB), then cached to survey_basis.json.
#   "csv" -- pred_label == Wall. Instant, slightly less accurate.
# A project's "basis" entry always wins over both.
SURVEY_BASIS_SOURCE = "e57"

PROJECTS = {
    "laramie": {
        "e57": "src/raw_e57/LaramieCM.e57",
        "csv": "src/input_csv/laram_inference_prediction.csv",
        # Registered basis. Delete this line to estimate it from the cloud
        # instead -- the estimate lands within ~0.06 deg of this value.
        # "basis": [
        #     [-0.23829087279918065, 0.9711938323221605, 0.0],
        #     [-0.9711938323221605, -0.23829087279918065, 0.0],
        #     [0.0, 0.0, 0.9999999999999998],
        # ],
    },
    "corteva": {
                "e57": "src/raw_e57/052124_YorkTest-Luke.e57",
                "csv": "src/input_csv/00-10231-20_CortevaYorkTest_Output.csv",
                # "basis": [
                #             [1, 0, 0],
                #             [0, 1, 0],
                #             [0, 0, 1],
                #         ],
                },
    "ty": {"e57": "src/raw_e57/Ty_50mm.e57",
            "csv": "src/input_csv/ty_inference_prediction.csv",
            # "basis": [
            #             [0.7075731030386642, -0.706640151602098, 0.0],
            #             [ 0.706640151602098, 0.7075731030386642, 0.0],
            #             [0.0, 0.0, 1.0],
            #         ],
            },
    "maxwell": {"e57": "src/raw_e57/MaxwellHall.e57",
        "csv": "src/input_csv/maxwell_inference_prediction.csv",
        # "basis": [
        #             [0.7075731030386642, -0.706640151602098, 0.0],
        #             [ 0.706640151602098, 0.7075731030386642, 0.0],
        #             [0.0, 0.0, 1.0],
        #         ],
        },
    "conroe": {"e57": "src/raw_e57/Conroe.e57",
        "csv": "src/input_csv/conroe_inference_prediction.csv",
        # "basis": [
        #             [0.7075731030386642, -0.706640151602098, 0.0],
        #             [ 0.706640151602098, 0.7075731030386642, 0.0],
        #             [0.0, 0.0, 1.0],
        #         ],
        }
}

# --------------------------------------------------------------- parameters

project = PROJECTS[PROJECT]
csv_path = REPO / project["csv"]
out_dir = REPO / "para_test_output" / PROJECT

print(f"[{PROJECT}] reading {csv_path.name}")
df_pred = pd.read_csv(csv_path, low_memory=False)

def resolve_survey_basis():
    """Registered basis if given, else estimate it -- caching the e57 result.

    Streaming a 2.7 GB scan on every tuning run would dominate the stage, and
    the answer only changes when the file does, so it is cached beside the
    outputs and keyed on the e57's path and modification time.
    """
    import json

    if "basis" in project:
        print(f"[{PROJECT}] using the registered survey basis")
        return project["basis"]

    if SURVEY_BASIS_SOURCE == "e57" and "e57" in project:
        e57_path = REPO / project["e57"]
        if not e57_path.exists():
            print(f"[{PROJECT}] {e57_path.name} not found; falling back to the CSV")
        else:
            cache_file = out_dir / "survey_basis.json"
            stamp = {"e57": str(e57_path), "mtime": e57_path.stat().st_mtime}
            if cache_file.exists():
                cached = json.loads(cache_file.read_text())
                if all(cached.get(k) == v for k, v in stamp.items()):
                    print(f"[{PROJECT}] survey basis from cache "
                          f"({cache_file.relative_to(REPO)})")
                    return cached["basis"]
            print(f"[{PROJECT}] estimating survey basis from {e57_path.name} "
                  f"(streams the file once)...")
            basis = survey_basis_from_e57(str(e57_path))
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({**stamp, "basis": basis}, indent=2))
            return basis

    print(f"[{PROJECT}] estimating survey basis from CSV wall points...")
    return survey_basis_from_df(df_pred)


survey_basis = resolve_survey_basis()
_yaw = math.degrees(math.atan2(survey_basis[1][0], survey_basis[0][0]))
print(f"[{PROJECT}] SURVEY_BASIS yaw {_yaw:.3f} deg ({_yaw % 90:.3f} mod 90)")

internal_params = {
    "SURVEY_BASIS": survey_basis,

    # --- clustering: 1-D height histogram ---------------------------------
    # "histogram" slices the z-histogram into one cluster per storey. It scored
    # at least as well as DBSCAN on every dataset tested and needs no eps.
    # Switch to "dbscan" to compare; EPS_F/MIN_SAMPLES_F are ignored otherwise.
    "CLUSTER_METHOD_F": "histogram",   # or "dbscan"
    "EPS_F": 0.5,            # only used when CLUSTER_METHOD_F == "dbscan"
    "MIN_SAMPLES_F": 10,     # only used when CLUSTER_METHOD_F == "dbscan"

    # Histogram knobs, in METRES (converted internally to the cloud's units).
    # MIN_STOREY_SEP_M is the one that matters: peaks closer than this are
    # treated as one storey. Too small and a sloped/tiered level splits into
    # several floors; too large and a real mezzanine is absorbed below.
    "MIN_STOREY_SEP_M": 2.4,
    "HIST_BIN_M": 0.08,
    "HIST_MIN_FRAC": 0.02,

    # Within each level keep only the DENSEST band of heights and discard the
    # rest. That removes surfaces mislabelled into this class -- on laramie the
    # upper "Floor" level held the storey's ceiling (21% of points there also
    # carried a Ceiling label, vs 1% at the real floor). No-op on clean levels.
    "DOMINANT_BAND_ONLY_F": True,
    "DOMINANT_BAND_ONLY_C": True,

    # --- plane fitting ----------------------------------------------------
    # "irls" is a deterministic robust fit (Tukey biweight, zero weight to gross
    # outliers). Chosen over RANSAC because it is reproducible: over 5 repeats
    # the RANSAC-derived threshold on a tiered band swung 16x while IRLS gave an
    # identical answer every time. RANSAC_N_F/NUM_ITER_F are ignored under irls.
    "PLANE_METHOD_F": "irls",        # or "ransac"
    "PLANE_METHOD_C": "irls",

    # "auto" takes DIS_THR from the fit's own robust residual scale (4.685 x
    # sigma, the Tukey cutoff), clamped below. Set a number here to pin it.
    "DIS_THR_F": "auto",
    "AUTO_DIS_THR_MIN_M": 0.02,   # floor on the derived threshold
    "AUTO_DIS_THR_MAX_M": 0.6,    # ceiling -- clamps non-planar (tiered) bands

    "RANSAC_N_F": 3,     # 3 points define a plane; the most robust choice
    # NOT "auto": Open3D's segment_plane already terminates adaptively via its
    # `probability` argument, so this is only an upper bound it rarely reaches.
    # Raising it costs almost nothing; deriving it adds a wasted preliminary fit.
    "NUM_ITER_F": 1000,

    # --- boundary ---------------------------------------------------------
    # "raster" rasterises the inliers, morphologically closes small voids, and
    # traces the outer contour. Measured against alphashape it is ~0.0s vs
    # 13-34s with equal or better area accuracy, and it does not depend on
    # point density the way alpha does.
    "BOUNDARY_METHOD_F": "raster",   # or "alphashape"
    "ALPHA_F": 0.15,                 # only used when method == "alphashape"

    # All three raster knobs derive from one measurable quantity, the inlier
    # point spacing. Set numbers instead to pin them:
    #   cell     ~= spacing   (below it the occupancy grid speckles)
    #   fill_gap  = 2 x cell  (hard floor; above it nothing changes)
    #   simplify  = 1.5 x cell, ABSOLUTE, not a fraction of perimeter
    "BOUNDARY_CELL_F": "auto",
    "BOUNDARY_FILL_GAP_F": "auto",
    "BOUNDARY_SIMPLIFY_F": "auto",
    "BOUNDARY_CONNECT_F": True,   # floors CAN have real courtyards/openings
    # Legacy fraction-of-perimeter tolerance, ignored when BOUNDARY_SIMPLIFY_F
    # is set. It scales with building size, so no single value ports across
    # projects -- 0.001 is 0.60 ft on one of these clouds and 1.18 ft on another.
    "BOUNDARY_SIMPLIFY_EPS_FRAC_F": 0.02,

    # --- ceilings: same treatment ------------------------------------------
    # MIN_STOREY_SEP_M / HIST_* / AUTO_DIS_THR_* above are shared by both
    # stages; only these per-stage keys need repeating.
    "CLUSTER_METHOD_C": "histogram",
    "EPS_C": 0.5,            # only used when CLUSTER_METHOD_C == "dbscan"
    "MIN_SAMPLES_C": 10,
    "DIS_THR_C": "auto",
    "RANSAC_N_C": 3,
    "NUM_ITER_C": 1000,
    "ALPHA_C": 0.15,                 # only used when method == "alphashape"
    "BOUNDARY_METHOD_C": "raster",
    "BOUNDARY_CELL_C": "auto",
    "BOUNDARY_FILL_GAP_C": "auto",
    "BOUNDARY_SIMPLIFY_C": "auto",
    # A ceiling is ONE continuous surface, so its interior voids are occlusion
    # shadows -- fixtures, ducts, structure blocking line of sight -- not real
    # openings. "connect" grows fill_gap until the mask is a single region.
    # Without it laramie's ceiling fragmented into 213 blobs and the boundary
    # traced whichever happened to be largest.
    "BOUNDARY_CONNECT_C": True,
    
    "BOUNDARY_SIMPLIFY_EPS_FRAC_C": 0.02,

    # --- component merging (BOUNDARY_CONNECT_MODE == "evidence") ------------
    # Set from the switches at the top of the file. When the mode is "off" or
    # "legacy" every threshold below is ignored.
    "BOUNDARY_CONNECT_MODE_C": CONNECT_MODE_C,
    "BOUNDARY_CONNECT_MODE_F": CONNECT_MODE_F,

    # EVERY THRESHOLD HERE IS PROVISIONAL. They have not been fitted to anything
    # -- they are placeholders picked to be permissive on distance and strict on
    # evidence, so the efficiency test rather than the gap test decides. Before
    # hardening any of them, run all projects and check that accepted and
    # rejected candidates actually separate; picking finals from one building is
    # the mistake these metrics exist to prevent.

    # Distance rail, "auto" -> 24 x cell. Deliberately loose: size alone should
    # not accept or reject a component.
    "BOUNDARY_MERGE_MAX_GAP_C": "auto",

    # THE primary test: candidate's original occupied cells / newly claimed
    # output area. 0.10 means at least a tenth of what the merge claims must be
    # evidenced. On laramie both ceiling-0 candidates scored 0.042 and were
    # refused; on maxwell the accepted one scored 0.873.
    "BOUNDARY_MERGE_MIN_EFFICIENCY_C": 0.10,

    # Cap on bridge area as a fraction of the two components' own union -- area
    # belonging to neither side.
    "BOUNDARY_MERGE_MAX_AREA_INFLATION_C": 0.25,

    # A candidate that is mostly hollow contour is not evidence for anything.
    "BOUNDARY_COMPONENT_MIN_DENSITY_C": 0.10,

    # Below this share of the total occupied cells a component is noise and is
    # dropped once, up front. Support is computed on the pre-morphology mask so
    # it cannot change between passes -- and one laramie ceiling has 276
    # components, 273 of them below this line.
    "BOUNDARY_COMPONENT_MIN_SUPPORT_FRAC_C": 0.005,

    # --- "coverage" mode ----------------------------------------------------
    # The whole stopping rule, replacing BOTH of legacy's arbitrary parts
    # (len(contours) == 1, and the 30 x cell bound). Merge the nearest component
    # repeatedly until the outline contains this share of the original occupied
    # cells. 0.99 separates laramie's two ceilings correctly: the upper's main
    # component already holds 99.79% so nothing is pursued, the lower's holds
    # 94.59% so merging continues.
    "BOUNDARY_COVERAGE_TARGET_C": 0.99,
    "BOUNDARY_COVERAGE_TARGET_F": 0.99,

    # Tail stop. Halt when the next bridge costs this many times the MEDIAN
    # area-per-newly-covered-cell of the merges already made -- a ratio against
    # the run's own history, so it needs no threshold tracking scan density.
    # CAVEAT, measured: on laramie's lower ceiling this fires at 95.2% coverage
    # (next bridge 282 cells/covered cell vs a median of 4.9) and the boundary
    # lands at 0.57x the floor, where legacy reaches 0.94x. The cost is measured
    # per CSV cell, and that surface is only 34% occupied -- so claiming real
    # ceiling looks expensive precisely where the sampling is thin. Raising this
    # would not fix it honestly; the fragmentation is a BOUNDARY_CELL problem.
    "BOUNDARY_COVERAGE_KNEE_C": 4.0,
    "BOUNDARY_COVERAGE_KNEE_F": 4.0,

    # Merges required before the knee may fire -- a median of one is noise.
    "BOUNDARY_COVERAGE_KNEE_MIN_C": 3,
    "BOUNDARY_COVERAGE_KNEE_MIN_F": 3,

    # The distance rail also bounds "coverage": merging stops if the nearest
    # remaining component is further than this. On corteva that is what stops it
    # (nearest 25.07 against a 22.28 rail), not the coverage target.
    "BOUNDARY_MERGE_MAX_GAP_F": "auto",

    # --- boundary point source ---------------------------------------------
    # "csv" (default) traces the segmented inliers. "e57" re-selects the
    # boundary's points from the RAW scan by distance to the already-fitted
    # plane -- no labels needed, so it also sidesteps bad segmentation.
    # Measured on laramie's lower ceiling: the CSV grid was 71.7% void and the
    # e57 fills 89.1% of those cells, so the holes are a downsampling artifact,
    # not occlusion. Costs ~2 min to stream the file once (then cached).
    "BOUNDARY_SOURCE_C": "csv",
    "BOUNDARY_SOURCE_F": "csv",
    "BOUNDARY_E57_PATH": str(REPO / project.get("e57", "LaramieCM.e57")),
    "BOUNDARY_E57_VOXEL": 0.3,   # one point per raster cell is plenty
    "BOUNDARY_E57_MARGIN": 2.0,  # clip to the CSV cluster's footprint + this
    # The frame conversion assumes csv == (e57_metres - e57_min) * this scale,
    # then rotated by SURVEY_BASIS. Inferred from matching extents, not
    # recorded -- a wrong offset shifts the boundary with no obvious symptom.
    "POINT_CLOUD_TO_POST_PROCESSING_SCALE": 3.280839895013123,

    # --- walls -------------------------------------------------------------
    # Walls are found by looking DOWN: points between consecutive floors are
    # projected to a top-down density image, thresholded, and fed to the LETR
    # line-segmentation model. Detected lines are buffered into wall polygons.
    #
    # Note this consumes floor_bboxz -- the wall slab is literally the gap
    # between one floor's zmax and the next floor's zmin -- so every floor
    # change above moves the wall input.

    # Histogram resolution of the top-down image. 300 bins over this building is
    # 0.55 ft/bin against 0.61 ft point spacing, so ~37% of bins are empty and
    # walls read as dotted rather than solid lines. Coarser bins gave far more
    # detections (100 bins -> 8x the segments) but risk merging nearby walls,
    # so this is left alone pending a visual check.
    "PROJECTED_BINS": 300,

    # Working resolution for thresholding -- NOT the model's input size, which
    # is set by test_size=1100 inside img_process_model_input. At 400 the image
    # is downsampled 5.8x and then upsampled ~6x again for the model, and
    # INTER_AREA averaging deletes thin lines before INT_THR sees them.
    # Measured: 400 -> 1200 took detections from 38 to 52 on one slab.
    "RESIZE_WIDTH": 1200,

    # "auto" uses Otsu, which reads the split off the image's own histogram.
    # A fixed number is a value on a colormapped, re-rendered, resized image --
    # meaningless physically and invalid the moment RESIZE_WIDTH changes.
    "INT_THR": "auto",

    # Closing kernel that repairs single-pixel dropouts along a wall. The
    # shipped value was 1, which is a no-op: dilating and eroding by a 1x1
    # structuring element returns the input unchanged.
    "WALL_MORPH_KERNEL": 3,

    # ONE angular tolerance for "is this line axis-aligned", replacing the
    # VERT_THR / HORI_THR pair -- they were reciprocals both encoding
    # tan(5.71 deg). Lines outside it are discarded, which is also why a wrong
    # SURVEY_BASIS destroys wall detection rather than merely skewing it.
    "WALL_ANGLE_TOL_DEG": 5.71,

    # "auto" converts WALL_THICKNESS_M into resized-image pixels, which is the
    # space the line buffers live in. The shipped BUFFER_THR of 2 px built
    # 1.65 ft walls around a measured 0.92 ft (11 in) reality, so each polygon
    # swallowed ~80% more width than the wall occupies.
    "BUFFER_THR": "auto",
    "WALL_THICKNESS_M": 0.279,   # 11 in, measured on laramie

    # A property of the LETR checkpoint rather than the building; left fixed.
    "SCORE_THR": 0.55,

    # No logging_blob_location below, so snapshots land here instead.
    "LOCAL_OUTPUT_DIR": str(out_dir),
}

# ------------------------------------------------------------------- run


def ring_area(edge_points):
    """Shoelace area of an edgePoints ring, in the cloud's squared units."""
    pts = [(p["x"], p["y"]) for p in edge_points]
    if len(pts) < 3:
        return 0.0
    return 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1]
                         - pts[(i + 1) % len(pts)][0] * pts[i][1]
                         for i in range(len(pts))))


def paired_floor_area(ceiling_pts, floors):
    """Area of the nearest floor BELOW this ceiling, or nan.

    A storey's ceiling should span roughly its floor, and the floor is fitted
    from different points by the same code -- so this is an independent check on
    the boundary, and the cheap half of the two validators. It needs no e57
    streaming and no extra parameters. On laramie it is what flags an over-grown
    ceiling immediately: a roof larger than the floor under it is suspect.
    """
    if not ceiling_pts:
        return float("nan")
    zc = sum(p["z"] for p in ceiling_pts) / len(ceiling_pts)
    below = [(zc - z, a) for a, z in floors if z < zc]
    return min(below)[1] if below else float("nan")


CONNECT_MODES = ("off", "legacy", "evidence", "coverage", "auto")


def run_floors_with(mode, label):
    params = dict(internal_params)
    params["BOUNDARY_CONNECT_MODE_F"] = mode
    out, bboxz, levels = post_process.run_floors(df_pred, params)
    print(f"\n[{PROJECT}] {len(out['floors'])} floor(s)  [connect={label}]")
    areas = []
    for i, (floor, (zmin, zmax)) in enumerate(zip(out["floors"], bboxz), 1):
        pts = floor["edgePoints"]
        area = ring_area(pts)
        zmid = sum(p["z"] for p in pts) / len(pts) if pts else 0.0
        areas.append((area, zmid))
        print(f"   floor {i}: z {zmin:7.2f} .. {zmax:7.2f}  "
              f"(thickness {zmax - zmin:5.2f})  "
              f"{len(pts):4d} edge points   area {area:9,.0f}")
    print(f"[{PROJECT}] floor levels: {[round(v, 2) for v in levels]}")
    return out, bboxz, levels, areas


floor_output, floor_bboxz, floor_level, floor_areas = run_floors_with(
    CONNECT_MODE_F, CONNECT_MODE_F)


def run_ceilings_with(mode, label):
    params = dict(internal_params)
    params["BOUNDARY_CONNECT_MODE_C"] = mode
    out, levels = post_process.run_ceilings(df_pred, params)
    print(f"\n[{PROJECT}] {len(out['ceilings'])} ceiling(s)  [connect={label}]")
    for i, c in enumerate(out["ceilings"], 1):
        area = ring_area(c["edgePoints"])
        ref = paired_floor_area(c["edgePoints"], floor_areas)
        ratio = area / ref if ref and ref == ref else float("nan")
        print(f"   ceiling {i}: {len(c['edgePoints']):4d} edge points   "
              f"area {area:9,.0f}   vs floor below {ratio:5.2f}x")
    print(f"[{PROJECT}] ceiling levels: {[round(v, 2) for v in levels]}")
    return out, levels


ceiling_output, ceiling_level = run_ceilings_with(CONNECT_MODE_C, CONNECT_MODE_C)

if COMPARE_CONNECT_MODES:
    # Reference runs only. floor_output / ceiling_output above stay the ones the
    # rest of the script uses, so walls still consume the floors from the mode
    # actually selected -- and every ceiling here is measured against THAT same
    # floor reference, so the ratios are comparable across modes.
    #
    # Read "vs floor below" and the edge point count TOGETHER: a ratio far from
    # 1.0 means the boundary disagrees with the floor under it, and a high vertex
    # count means it is tracing a ragged web rather than a room outline. Either
    # number alone can look fine while the boundary is wrong -- on laramie's
    # lower ceiling "off" gives 0.54x with 368 vertices, which is both.
    print(f"\n{'=' * 62}\n[{PROJECT}] A/B reference runs (not used downstream)")
    for other in CONNECT_MODES:
        if other != CONNECT_MODE_F:
            print(f"\n{'-' * 62}\n[{PROJECT}] floors, connect={other}")
            run_floors_with(other, other)
    for other in CONNECT_MODES:
        if other != CONNECT_MODE_C:
            print(f"\n{'-' * 62}\n[{PROJECT}] ceilings, connect={other}")
            run_ceilings_with(other, other)
    print(f"{'=' * 62}")

if RUN_WALLS:
    model_path = REPO / LINE_SEG_MODEL
    if not model_path.exists():
        print(f"\n[{PROJECT}] {LINE_SEG_MODEL} not found -- skipping walls")
    else:
        print(f"\n[{PROJECT}] running walls on {len(floor_bboxz)} level(s) "
              f"(LETR inference, slow)...")
        wall_output = post_process.run_walls(
            df_pred,
            internal_params,
            floor_bboxz,
            str(model_path),
        )
        walls = wall_output["walls"]
        print(f"\n[{PROJECT}] {len(walls)} wall(s)")
        by_level = {}
        for w in walls:
            by_level.setdefault(w["levelIndex"], []).append(w)
        for lvl in sorted(by_level):
            ws = by_level[lvl]
            heights = [w["zRange"]["max"] - w["zRange"]["min"] for w in ws]
            print(f"   level {lvl}: {len(ws):3d} wall(s)   "
                  f"height min {min(heights):5.2f} / median "
                  f"{sorted(heights)[len(heights) // 2]:5.2f} / max {max(heights):5.2f}")
        if not walls:
            print("   none -- check the 'line segmentation: N of 1000' lines above; "
                  "0 means the projected image was blank or SCORE_THR too high")

print(f"[{PROJECT}] snapshots -> {out_dir}")
