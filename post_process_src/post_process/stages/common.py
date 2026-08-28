from collections.abc import Mapping
from typing import Any

import numpy as np

from ..utils.blob_util import setup_blob_clients


def get_parameter(
    parameters: Mapping[str, Any],
    key: str,
    fallback_key: str | None = None,
) -> Any:
    if key in parameters:
        return parameters[key]
    if fallback_key is not None:
        return parameters[fallback_key]
    return parameters[key]


_BASIS_CACHE: dict = {}


def resolve_survey_basis(df, parameters):
    """SURVEY_BASIS as configured, or estimated from the cloud's own walls.

    A basis is normally supplied by whoever registered the survey. When it is
    absent -- ``None``, or the string "auto" -- it is recovered from the
    wall-labelled points instead, so a scan can be processed without that value
    in hand. Measured against the two registered matrices available, the CSV
    estimate lands 0.064 and 0.066 degrees off.

    An IDENTITY matrix counts as supplied, not as missing. It is a legitimate
    answer for a building already square to the axes (one of ours is), so it
    must not silently trigger estimation.

    Memoised because run_floors, run_ceilings and run_walls each need it and
    each receives the same frame; estimating normals at three radii on a few
    hundred thousand wall points is not something to do three times. The key is
    the wall points' count and extent, which is stable for a given cloud and
    cheap to compute.
    """
    basis = parameters.get("SURVEY_BASIS")
    if basis is not None and not (isinstance(basis, str)
                                  and str(basis).lower() == "auto"):
        return basis

    from ..runtime import logger
    from ..utils.survey_basis_util import survey_basis_from_df

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    key = (len(xyz), tuple(np.round(xyz.min(axis=0), 3)),
           tuple(np.round(xyz.max(axis=0), 3)))
    hit = _BASIS_CACHE.get(key)
    if hit is not None:
        return hit

    logger.info("SURVEY_BASIS not supplied -- estimating it from wall points")
    basis = survey_basis_from_df(df, parameters, verbose=False)
    yaw = float(np.degrees(np.arctan2(basis[1][0], basis[0][0])))
    logger.info("SURVEY_BASIS estimated: yaw %.3f deg (%.3f mod 90)",
                yaw, yaw % 90.0)
    _BASIS_CACHE[key] = basis
    return basis


def make_blob_factory(logging_blob_location):
    if logging_blob_location:
        return setup_blob_clients(logging_blob_location)
    return None


def auto_ransac_options(parameters):
    """Guard rails for DIS_THR="auto" / NUM_ITER="auto", metres -> cloud units.

    The clamp matters: the knee is only well defined when the cluster really is
    one plane. A tiered or sloped level has no single plane, its knee wanders
    between passes, and without a ceiling the threshold can grow until it
    accepts the whole tier and hides the problem behind a 98% inlier rate.

    iteration_safety multiplies the textbook iteration count, which assumes the
    inlier ratio is known exactly; it is only estimated here.
    """
    units_per_meter = float(parameters.get("POINT_CLOUD_TO_POST_PROCESSING_SCALE", 1.0))
    return {
        "min_threshold": float(parameters.get("AUTO_DIS_THR_MIN_M", 0.02)) * units_per_meter,
        "max_threshold": float(parameters.get("AUTO_DIS_THR_MAX_M", 0.5)) * units_per_meter,
        "iteration_safety": float(parameters.get("AUTO_ITER_SAFETY", 4.0)),
        "min_iterations": int(parameters.get("AUTO_MIN_ITER", 50)),
    }


def e57_boundary_options(parameters, suffix):
    """Config for taking boundary points from the raw scan instead of the CSV.

    The CSV is a heavy downsample -- measured at ~0.1% of the e57 on one ceiling
    band -- which is why its occupancy grid is mostly void and fragments. Points
    are re-selected from the e57 by distance to the already-fitted plane, so no
    segmentation of the scan is needed.

    The frame conversion assumes the CSV was produced as
    ``(e57_metres - e57_min) * POINT_CLOUD_TO_POST_PROCESSING_SCALE`` and then
    rotated by SURVEY_BASIS. That matches how these datasets were built, but it
    is inferred rather than recorded -- a wrong offset shifts the boundary
    without any obvious symptom, so check the first result against the CSV one.
    """
    source = str(parameters.get(f"BOUNDARY_SOURCE_{suffix}",
                                parameters.get("BOUNDARY_SOURCE", "csv"))).lower()
    path = parameters.get(f"BOUNDARY_E57_PATH_{suffix}",
                          parameters.get("BOUNDARY_E57_PATH"))
    # Gate on the PATH, not the source. connect_mode="auto" traces the CSV but
    # still needs the scan as held-out evidence to score candidates against, so
    # returning None whenever source != "e57" silently starved it -- auto fell
    # back to a named mode every time and never scored anything.
    if not path:
        return None
    return {
        "source": source,
        "path": path,
        "scale": float(parameters.get("POINT_CLOUD_TO_POST_PROCESSING_SCALE", 1.0)),
        "basis": parameters.get("SURVEY_BASIS"),
        # One point per raster cell is enough; the raster cell tracks the point
        # spacing, so ~0.3 native units keeps the grid dense without the memory.
        "voxel": float(parameters.get("BOUNDARY_E57_VOXEL", 0.3)),
        # Clip to the CSV cluster's footprint: the fitted plane is infinite and
        # would otherwise sweep in every surface at that elevation building-wide.
        "margin": float(parameters.get("BOUNDARY_E57_MARGIN", 2.0)),
    }


def _merge_gap(parameters, suffix):
    """The merge distance rail in the cloud's units, from metres.

    An explicit BOUNDARY_MERGE_MAX_GAP_* (a number, not "auto") still wins, so
    callers pinning it in cloud units are unaffected.
    """
    explicit = parameters.get(f"BOUNDARY_MERGE_MAX_GAP_{suffix}")
    if explicit is not None and not (isinstance(explicit, str)
                                     and str(explicit).lower() == "auto"):
        return float(explicit)
    units_per_meter = float(parameters.get("POINT_CLOUD_TO_POST_PROCESSING_SCALE", 1.0))
    return float(parameters.get(f"BOUNDARY_MERGE_MAX_GAP_M_{suffix}", 8.0)) * units_per_meter


def merge_options(parameters, suffix):
    """Evidence-aware component-merging controls for the raster boundary.

    ``BOUNDARY_CONNECT_MODE_*`` selects the strategy:

      "off"       one fixed close at fill_gap, keep the largest contour
      "legacy"    grow one global kernel until len(contours) == 1
      "evidence"  judge each detached component on the evidence it contributes
                  against the output area the merge would invent

    Defaulting to legacy/off from the existing BOUNDARY_CONNECT_* boolean keeps
    every current caller byte-identical; "evidence" is opt-in.

    THE THRESHOLDS BELOW ARE PROVISIONAL. They are placeholders chosen to be
    permissive on gap and strict on evidence, so that the efficiency test rather
    than the distance test decides. They have not been measured across projects,
    and picking finals from one building is exactly the mistake the metrics
    exist to avoid -- run all three and look at whether accepted and rejected
    candidates actually separate before hardening any of these.
    """
    connect = bool(parameters.get(f"BOUNDARY_CONNECT_{suffix}", False))
    mode = parameters.get(f"BOUNDARY_CONNECT_MODE_{suffix}",
                          parameters.get("BOUNDARY_CONNECT_MODE"))
    return {
        "connect_mode": str(mode).lower() if mode else ("legacy" if connect else "off"),
        # Safety rail, not the decision -- in METRES, converted here.
        #
        # It used to derive from the cell (24 x cell), which meant it tracked
        # POINT SPACING, i.e. how the building was scanned. Two projects got
        # rails 2x apart in physical distance for the same setting: maxwell
        # 11.1 units (~3.4 m) against corteva 22.3 units (~6.8 m), purely
        # because corteva's scan is sparser. But "how far apart can two pieces
        # of one surface legitimately be" is a question about the BUILDING.
        #
        # 8 m is provisional and chosen against the measured gaps: it permits
        # laramie's lower ceiling (~2.5 m, where connecting is right) and
        # corteva's nearest unmerged component (7.6 m), and rejects conroe's
        # 23-24 m outliers. It still rejects corteva's 11.3 m component holding
        # 24.9% of the evidence -- which may be wrong, and is the single case
        # most worth checking before this value is fixed.
        "merge_max_gap": _merge_gap(parameters, suffix),
        # Primary test: supported cells / newly claimed area.
        "merge_min_efficiency": float(
            parameters.get(f"BOUNDARY_MERGE_MIN_EFFICIENCY_{suffix}", 0.10)),
        # Cap on area belonging to neither component.
        "merge_max_area_inflation": float(
            parameters.get(f"BOUNDARY_MERGE_MAX_AREA_INFLATION_{suffix}", 0.25)),
        # A candidate that is mostly hollow contour is not evidence.
        "component_min_density": float(
            parameters.get(f"BOUNDARY_COMPONENT_MIN_DENSITY_{suffix}", 0.10)),
        # Below this share of total evidence a component is noise, dropped once
        # up front -- support cannot change, and one ceiling has 1,356 of them.
        "component_min_support_frac": float(
            parameters.get(f"BOUNDARY_COMPONENT_MIN_SUPPORT_FRAC_{suffix}", 0.005)),

        # --- "coverage" mode only ---------------------------------------
        # Share of the original occupied cells the final outline should contain.
        # This is the whole stopping rule, replacing both of legacy's arbitrary
        # parts (len(contours) == 1 and the 30 x cell bound). 0.99 was chosen
        # because it separates the two ceilings we understand: one main
        # component already holds 99.77% so nothing is pursued, the other holds
        # 93.9% with the rest spread across the footprint so merging continues.
        "coverage_target": float(
            parameters.get(f"BOUNDARY_COVERAGE_TARGET_{suffix}", 0.99)),
        # Stop when the next bridge costs this many times the MEDIAN area-per-
        # covered-cell of the merges already made. A ratio against the run's own
        # history, so it does not need to track scan density the way an absolute
        # efficiency threshold does.
        "coverage_knee": float(
            parameters.get(f"BOUNDARY_COVERAGE_KNEE_{suffix}", 4.0)),
        # Merges needed before the knee can fire -- a median of one is noise.
        "coverage_knee_min_merges": int(
            parameters.get(f"BOUNDARY_COVERAGE_KNEE_MIN_{suffix}", 3)),

        # --- "auto" mode only -------------------------------------------
        # Which mode to use when connect_mode is "auto" but no raw scan is
        # configured. Auto scores candidates against held-out scan evidence; with
        # none available there is nothing independent to choose on, so it names a
        # fallback and says so rather than inventing a proxy criterion.
        "auto_fallback": str(
            parameters.get(f"BOUNDARY_AUTO_FALLBACK_{suffix}", "legacy")).lower(),
    }


def cluster_method(parameters, suffix):
    """Which splitter to use: "histogram" (default) or "dbscan".

    Checks the stage-specific key first (CLUSTER_METHOD_F / _C), then the shared
    CLUSTER_METHOD. The default lives here so both stages agree; get_parameter
    raises KeyError rather than taking a default, hence the explicit .get chain.
    """
    return str(
        parameters.get(
            f"CLUSTER_METHOD_{suffix}",
            parameters.get("CLUSTER_METHOD", "histogram"),
        )
    ).lower()


def histogram_options(parameters):
    """Histogram-clustering knobs, converted from metres to the cloud's units.

    The physical defaults are expressed in METRES and multiplied by
    POINT_CLOUD_TO_POST_PROCESSING_SCALE (native-units-per-metre, 3.2808 for the
    feet-based CSVs), so the same configuration works on a metric cloud. This
    mirrors how the opening stage handles its physical size filters.

    MIN_STOREY_SEP_M is the important one: peaks closer together than this are
    treated as one storey. Too small and a sloped or tiered level splits into
    several floors; too large and a genuine mezzanine is absorbed into the deck
    below. 2.4 m is roughly a floor-to-floor height.
    """
    units_per_meter = float(parameters.get("POINT_CLOUD_TO_POST_PROCESSING_SCALE", 1.0))
    return {
        "bin_size": float(parameters.get("HIST_BIN_M", 0.08)) * units_per_meter,
        "min_frac": float(parameters.get("HIST_MIN_FRAC", 0.02)),
        "min_storey_sep": float(parameters.get("MIN_STOREY_SEP_M", 2.4)) * units_per_meter,
        # Levels split at VOIDS of this width, not at peak distance. Bounded
        # below by density dips inside one tiered level (~0.16 m) and above by
        # the minimum habitable floor-to-floor less the levels' point spread
        # (~1.4 m), so 0.5 m sits mid-window with wide margin either side.
        "min_gap": float(parameters.get("MIN_LEVEL_GAP_M", 0.5)) * units_per_meter,
        "min_points_frac": float(parameters.get("MIN_LEVEL_POINTS_FRAC", 0.02)),
    }


def scalar_mode(values) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")

    unique_values, counts = np.unique(values, return_counts=True)
    return float(unique_values[np.argmax(counts)])

