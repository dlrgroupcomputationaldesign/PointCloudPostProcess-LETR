from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .labels import DEFAULT_LABELS


DEFAULT_SURVEY_BASIS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

@dataclass(frozen=True)
class BoundaryConfig:
    """Boundary extraction from the fitted plane's inliers.

    - "raster" (default): rasterize to an occupancy grid, close it to repair
      sparse voids, then trace the largest external contour. Density-independent
      and effectively instant. Measured against alphashape it gave equal or
      better area in ~0.0 s versus 13-34 s.
    - "alphashape": the original concave hull. Density-sensitive, so the outline
      zigzags where projected point density drops, and alpha is dangerous --
      at 1.0 one dataset fragmented into 42 pieces, silently discarding ~5,900
      real points.

    The three raster knobs default to "auto" and derive from ONE measurable
    quantity, the inlier point spacing: cell ~= spacing, fill_gap = 2 x cell,
    simplify = 1.5 x cell (absolute). Pinning them to numbers is supported but
    no single fixed value ports between buildings.

    ``connect_mode`` decides what happens when the mask comes out in pieces --
    see stages.common.merge_options. NOTE on the "coverage" default: it is the
    best single choice across floors AND ceilings, but it is not uniformly the
    best. Measured, it avoids the 40-77% floor inflation "legacy" produces,
    while on one sparse ceiling it reaches 0.57x the floor area where "legacy"
    reaches 0.94x. Every stop it makes is logged with a reason.
    """
    method: str = "raster"           # raster | alphashape
    cell: Any = "auto"               # grid resolution; "auto" -> point spacing
    fill_gap: Any = "auto"           # closing radius; "auto" -> 2 x cell
    simplify_abs: Any = "auto"       # approxPolyDP tolerance; "auto" -> 1.5 x cell
    simplify_eps_frac: float = 0.02  # legacy fraction-of-perimeter; ignored when simplify_abs is set
    connect_mode: str = "coverage"   # off | legacy | evidence | coverage | auto
    coverage_target: float = 0.99
    coverage_knee: float = 4.0
    coverage_knee_min: int = 3
    # Distance rail for the merge modes. merge_max_gap_m is in METRES and is the
    # one to set; merge_max_gap stays for callers pinning it in cloud units and
    # wins when it is a number rather than "auto". NOTE this rail does NOT apply
    # to "legacy", which uses its own hard-coded connect_max_cells = 30 x cell.
    merge_max_gap_m: float = 8.0
    merge_max_gap: Any = "auto"

    # "auto" runs off/legacy/coverage and scores each against HELD-OUT evidence
    # from the raw scan -- precision (of what I claim, how much is real surface)
    # and recall (of the real surface, how much did I capture), ranked by F1.
    # A merge rule has to judge each candidate incrementally, before the run
    # ends; choosing between finished boundaries can afford a global check, and
    # that is the whole reason it can use evidence the modes themselves cannot.
    #
    # It needs PostProcessConfig.scan_path. Without one there is nothing
    # independent to choose on, so it says so and uses auto_fallback rather than
    # inventing a proxy criterion.
    #
    # Measured across two buildings it picks DIFFERENTLY per surface -- legacy
    # where a sparse ceiling needs aggressive connecting (recall 0.55 -> 0.98),
    # coverage where legacy over-claims (precision 0.43). No single mode wins
    # everywhere, which is the case for having it.
    auto_fallback: str = "legacy"

    # Where the traced points come from. "e57" re-selects them from the raw scan
    # by distance to the fitted plane, needing no labels. Independent of the
    # evidence used by connect_mode="auto", which loads the scan either way.
    source: str = "csv"              # csv | e57


@dataclass(frozen=True)
class SurfaceConfig:
    """Shared floor/ceiling settings. Both stages take the same treatment.

    cluster_method "histogram" slices the z-histogram into one cluster per
    storey. It scored at least as well as DBSCAN on every dataset tested, runs
    in 8-16 ms against minutes, and has no eps to tie to anything physical --
    on one building no eps could both chain a level and refuse to bridge levels.

    plane_method "irls" is a deterministic robust fit. Over five repeats on the
    same points a RANSAC-derived threshold swung 16x on a tiered band while IRLS
    returned an identical answer every time.
    """
    cluster_method: str = "histogram"   # histogram | dbscan
    plane_method: str = "irls"          # irls | ransac
    dominant_band_only: bool = True
    eps: float = 0.5                    # dbscan only
    min_samples: int = 10               # dbscan only
    distance_threshold: Any = "auto"    # "auto" -> the fit's own robust scale
    ransac_n: int = 3                   # ransac only; 3 points define a plane
    num_iterations: int = 1000          # ransac only; an upper bound it rarely reaches
    alpha: float = 0.15                 # alphashape only
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)


@dataclass(frozen=True)
class FloorConfig(SurfaceConfig):
    pass


@dataclass(frozen=True)
class CeilingConfig(SurfaceConfig):
    pass


@dataclass(frozen=True)
class HistogramConfig:
    """Level detection knobs, in METRES (converted via the units-per-metre scale).

    min_level_gap_m is the important one: levels split at VOIDS of this width,
    not at peak distance. Its two bounds are physically independent -- density
    dips inside one tiered level (~0.16 m) below, minimum habitable
    floor-to-floor less the levels' own spread (~1.4 m) above -- which is why a
    fixed value ports. Level count held over a 12.5x range of it.
    """
    hist_bin_m: float = 0.08
    hist_min_frac: float = 0.02
    min_storey_sep_m: float = 2.4
    min_level_gap_m: float = 0.5
    min_level_points_frac: float = 0.02
    auto_dis_thr_min_m: float = 0.02
    auto_dis_thr_max_m: float = 0.6


@dataclass(frozen=True)
class WallConfig:
    """Top-down line segmentation (LETR).

    resize_width is the WORKING resolution, not the model's input size -- that
    is test_size=1100 inside img_process_model_input. It decides what survives
    thresholding: shrinking averages a thin wall into the floor before the model
    ever sees it. 1200 is a laramie-sized default; measured, detection needs
    roughly 3-5 pixels across the thinnest wall, so a building 4x longer needs
    proportionally more (one 669-unit slab fell to 1.65 px and returned 11 of
    1000 segments).

    intensity_threshold "auto" uses Otsu, reading the split off the image's own
    histogram. A fixed number is a value on a colormapped, re-rendered, resized
    image -- it has no physical meaning and shifts whenever anything upstream
    changes.

    buffer_threshold "auto" converts wall_thickness_m into resized-image pixels.
    The old fixed 2 px built 1.65 ft walls around a measured 0.92 ft reality.
    """
    projected_bins: int = 300
    resize_width: int = 1200
    intensity_threshold: Any = "auto"    # "auto" -> Otsu
    score_threshold: float = 0.55        # a checkpoint property, not a building one
    angle_tol_deg: float = 5.71          # replaces the vertical/horizontal pair
    morph_kernel: int = 3                # 1 is a no-op (dilate+erode by 1x1)
    buffer_threshold: Any = "auto"       # "auto" -> from wall_thickness_m
    wall_thickness_m: float = 0.279      # 11 in
    vertical_threshold: float | None = None    # legacy; wins over angle_tol_deg
    horizontal_threshold: float | None = None  # legacy


@dataclass(frozen=True)
class OpeningConfig:
    enabled: bool = False

    # Detector selection (resolved by detectors.registry.build_detector).
    # "grounding_dino_hf"  -> HuggingFace transformers port (default; no CUDA build)
    # "grounding_dino"     -> original IDEA-Research package (needs CUDA toolkit + --no-build-isolation)
    detector: str = "grounding_dino_hf"
    # A list runs each prompt separately and combines the results (better than one
    # combined ". "-joined caption in practice); a single string is one call.
    gd_text_prompt: Sequence[str] | str = ("door", "window", "opening")
    gd_box_threshold: float = 0.35
    gd_text_threshold: float = 0.25
    # IoU for combining boxes across prompts (None/0 = keep all; pure union).
    gd_nms_iou: float = 0.5
    # Original-package backend: only the .pth is required; the config .py defaults
    # to GroundingDINO_SwinT_OGC.py bundled in the installed groundingdino package.
    gd_weights_path: str | None = None
    gd_config_path: str | None = None
    # HF-port backend: model id / local dir.
    gd_model_id: str = "IDEA-Research/grounding-dino-tiny"      

    # Physical size sanity filter applied to detected boxes (metres; converted to
    # native units via e57_to_csv_scale before comparison).
    min_width: float = 0.3
    max_width: float = 4.0
    min_height: float = 0.3
    max_height: float = 4.0
    min_wall_points: int = 200
    # Reject boxes covering more than this fraction of the wall image (a box that
    # is ~the whole wall is the wall, not an opening).
    max_coverage: float = 0.7
    # Aspect ratio = width / height; rejects implausibly wide/flat or thin boxes.
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0

    # Attach the cloud points inside each detected opening's box to the output
    # (like floor/wall points). The per-wall point buffer is reservoir-capped to
    # bound memory; set collect_points False to skip it on very large clouds.
    collect_points: bool = True
    max_collected_points_per_wall: int = 300_000

    # Log-density image rendering (see utils.opening_image_util). image_bin_m is in
    # METRES; wall geometry is in native CSV units (feet here), so it is converted
    # via e57_to_csv_scale (which doubles as the native-units-per-meter factor).
    # Like the experiment, counts accumulate at the finer image_fine_bin_m and are
    # block-summed to image_bin_m for rendering (0.025 -> 0.05 = factor 2).
    image_bin_m: float = 0.05
    image_fine_bin_m: float = 0.025
    image_cell_px: int = 4
    image_transform: str = "log1p"  # log1p | sqrt | raw
    image_clip_pct: float | None = None  # None -> vmax = max(log1p(count))
    image_gamma: float | None = None
    # Brightness: multiplies vmax (>1 lightens, <1 darkens). Default 1.8 with
    # clip_pct=None reproduces the brighter output_log_img render (calibrated on
    # Laramie wall_17: package mean ~153 vs reference ~150), which detects better.
    image_vmax_scale: float = 1.8
    # Post-render enhancement, applied to the uint8 grayscale in this order:
    # denoise -> CLAHE -> unsharp. All default to off so the render is unchanged.
    # Gaussian denoise to kill sparse-wall speckle before sharpening (px sigma at
    # the rendered resolution; 0 = off).
    image_denoise_sigma: float = 0.0
    # CLAHE local-contrast equalization: clip limit (0/None = off) and tile count
    # per axis. Makes openings pop against locally-varying wall density.
    image_clahe_clip: float = 0.0
    image_clahe_tile: int = 8
    # Unsharp mask to crispen opening edges: amount (0 = off) and blur sigma (px).
    # out = img + amount * (img - gaussian(img, sigma)).
    image_unsharp_amount: float = 0.0
    image_unsharp_sigma: float = 1.0
    # Render source:
    # - "density"     production log-density render with vmax clip, brightness
    #                 scaling and optional denoise/CLAHE/unsharp.
    # - "intensity"   mean per-cell reflectance, where the dense cloud carries an
    #                 intensity field. Separates glass/openings from wall by
    #                 material but needs good coverage; per wall, if the fraction
    #                 of populated render cells is below image_intensity_min_coverage,
    #                 that wall falls back to density. Sources without an intensity
    #                 field (and the no-dense-cloud path) also fall back to density.
    # - "raw_density" pure log1p + gray_r normalized over the full data range,
    #                 with no clip / vmax_scale / gamma / enhancement. Matches the
    #                 output of opening_detection_exp/export_all_walls_log_images.py.
    image_source: str = "density"  # density | intensity | raw_density
    image_intensity_clip_pct: float | None = 99.0
    image_intensity_min_coverage: float = 0.35
    image_cmap: str = "gray_r"  # gray_r (openings bright) | gray
    wall_distance_tolerance: float = 0.15
    # When no blob location is set, write the log/detection images + JSON here so
    # local runs are inspectable. None = don't write images locally.
    local_output_dir: str | None = None

    # Optional dense point source streamed from an E57 file (meters). The stream is
    # mapped into native CSV units via (point - offset) * scale.
    dense_source_path: str | None = None
    dense_wall_ids: Sequence[str] | None = None
    dense_wall_margin: float = 0.25  # metres (MARGIN_M)
    point_cloud_chunk_size: int = 2_000_000  # CHUNK_SIZE
    # Native-units-per-meter (3.2808 for feet, 1.0 for meters). Also drives the
    # meters->native conversion for image_bin_m and the size filters above.
    point_cloud_to_post_processing_scale: float = 3.280839895013123
    # E57->CSV translation. If annotation_csv_path is set, the offset is computed
    # as xyz_min_ft * (1/scale) like the experiment; otherwise this explicit value
    # (or the E57 header min) is used.
    point_cloud_to_csv_offset: Sequence[float] | None = None
    annotation_csv_path: str | None = None
    annotation_chunk_size: int = 500_000  # ANNOTATION_CHUNK_SIZE


@dataclass(frozen=True)
class PostProcessConfig:
    # None means "estimate it from the cloud's wall points". An explicit
    # identity matrix is a legitimate answer for a building already square to
    # the axes, so it counts as supplied and does NOT trigger estimation.
    survey_basis: Sequence[Sequence[float]] | None = None
    labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LABELS)
    floor: FloorConfig = field(default_factory=FloorConfig)
    ceiling: CeilingConfig = field(default_factory=CeilingConfig)
    wall: WallConfig = field(default_factory=WallConfig)
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    openings: OpeningConfig = field(default_factory=OpeningConfig)

    # The raw scan, used two ways: as the boundary point source when
    # BoundaryConfig.source is "e57", and as held-out evidence when
    # connect_mode is "auto". Streaming is cached per (path, scale, voxel), so
    # wanting it for both costs one pass. None disables both.
    scan_path: str | None = None
    scan_voxel: float = 0.3    # one point per raster cell is enough
    scan_margin: float = 2.0   # clip to the cluster's footprint + this

    def _surface(self, cfg, s: str) -> dict[str, Any]:
        """Per-stage keys for a floor/ceiling, suffixed _F or _C."""
        b = cfg.boundary
        return {
            f"CLUSTER_METHOD_{s}": cfg.cluster_method,
            f"PLANE_METHOD_{s}": cfg.plane_method,
            f"DOMINANT_BAND_ONLY_{s}": cfg.dominant_band_only,
            f"EPS_{s}": cfg.eps,
            f"MIN_SAMPLES_{s}": cfg.min_samples,
            f"DIS_THR_{s}": cfg.distance_threshold,
            f"RANSAC_N_{s}": cfg.ransac_n,
            f"NUM_ITER_{s}": cfg.num_iterations,
            f"ALPHA_{s}": cfg.alpha,
            f"BOUNDARY_METHOD_{s}": b.method,
            f"BOUNDARY_CELL_{s}": b.cell,
            f"BOUNDARY_FILL_GAP_{s}": b.fill_gap,
            f"BOUNDARY_SIMPLIFY_{s}": b.simplify_abs,
            f"BOUNDARY_SIMPLIFY_EPS_FRAC_{s}": b.simplify_eps_frac,
            f"BOUNDARY_CONNECT_MODE_{s}": b.connect_mode,
            f"BOUNDARY_COVERAGE_TARGET_{s}": b.coverage_target,
            f"BOUNDARY_COVERAGE_KNEE_{s}": b.coverage_knee,
            f"BOUNDARY_COVERAGE_KNEE_MIN_{s}": b.coverage_knee_min,
            f"BOUNDARY_MERGE_MAX_GAP_{s}": b.merge_max_gap,
            f"BOUNDARY_MERGE_MAX_GAP_M_{s}": b.merge_max_gap_m,
            f"BOUNDARY_AUTO_FALLBACK_{s}": b.auto_fallback,
            f"BOUNDARY_SOURCE_{s}": b.source,
        }

    def to_parameters(self) -> dict[str, Any]:
        """Return the legacy parameter dictionary used by the current stages."""
        h = self.histogram
        parameters = {
            "SURVEY_BASIS": ([list(row) for row in self.survey_basis]
                             if self.survey_basis is not None else None),
            "LABELS": list(self.labels),
            "EPS": self.floor.eps,
            "MIN_SAMPLES": self.floor.min_samples,
            **self._surface(self.floor, "F"),
            **self._surface(self.ceiling, "C"),
            "HIST_BIN_M": h.hist_bin_m,
            "HIST_MIN_FRAC": h.hist_min_frac,
            "MIN_STOREY_SEP_M": h.min_storey_sep_m,
            "MIN_LEVEL_GAP_M": h.min_level_gap_m,
            "MIN_LEVEL_POINTS_FRAC": h.min_level_points_frac,
            "AUTO_DIS_THR_MIN_M": h.auto_dis_thr_min_m,
            "AUTO_DIS_THR_MAX_M": h.auto_dis_thr_max_m,
            "BOUNDARY_E57_PATH": self.scan_path,
            "BOUNDARY_E57_VOXEL": self.scan_voxel,
            "BOUNDARY_E57_MARGIN": self.scan_margin,
            "PROJECTED_BINS": self.wall.projected_bins,
            "RESIZE_WIDTH": self.wall.resize_width,
            "INT_THR": self.wall.intensity_threshold,
            "SCORE_THR": self.wall.score_threshold,
            "WALL_ANGLE_TOL_DEG": self.wall.angle_tol_deg,
            "WALL_MORPH_KERNEL": self.wall.morph_kernel,
            "BUFFER_THR": self.wall.buffer_threshold,
            "WALL_THICKNESS_M": self.wall.wall_thickness_m,
            "OPENINGS_ENABLED": self.openings.enabled,
            "OPENING_DETECTOR": self.openings.detector,
            "OPENING_GD_TEXT_PROMPT": (
                self.openings.gd_text_prompt
                if isinstance(self.openings.gd_text_prompt, str)
                else list(self.openings.gd_text_prompt)
            ),
            "OPENING_GD_BOX_THRESHOLD": self.openings.gd_box_threshold,
            "OPENING_GD_TEXT_THRESHOLD": self.openings.gd_text_threshold,
            "OPENING_GD_NMS_IOU": self.openings.gd_nms_iou,
            "OPENING_GD_CONFIG_PATH": self.openings.gd_config_path,
            "OPENING_GD_WEIGHTS_PATH": self.openings.gd_weights_path,
            "OPENING_GD_MODEL_ID": self.openings.gd_model_id,
            "OPENING_MIN_WIDTH": self.openings.min_width,
            "OPENING_MAX_WIDTH": self.openings.max_width,
            "OPENING_MIN_HEIGHT": self.openings.min_height,
            "OPENING_MAX_HEIGHT": self.openings.max_height,
            "OPENING_MIN_WALL_POINTS": self.openings.min_wall_points,
            "OPENING_MAX_COVERAGE": self.openings.max_coverage,
            "OPENING_MIN_ASPECT_RATIO": self.openings.min_aspect_ratio,
            "OPENING_MAX_ASPECT_RATIO": self.openings.max_aspect_ratio,
            "OPENING_COLLECT_POINTS": self.openings.collect_points,
            "OPENING_MAX_COLLECTED_POINTS_PER_WALL": self.openings.max_collected_points_per_wall,
            "OPENING_IMAGE_BIN_M": self.openings.image_bin_m,
            "OPENING_IMAGE_FINE_BIN_M": self.openings.image_fine_bin_m,
            "OPENING_IMAGE_CELL_PX": self.openings.image_cell_px,
            "OPENING_IMAGE_TRANSFORM": self.openings.image_transform,
            "OPENING_IMAGE_CLIP_PCT": self.openings.image_clip_pct,
            "OPENING_IMAGE_GAMMA": self.openings.image_gamma,
            "OPENING_IMAGE_VMAX_SCALE": self.openings.image_vmax_scale,
            "OPENING_IMAGE_DENOISE_SIGMA": self.openings.image_denoise_sigma,
            "OPENING_IMAGE_CLAHE_CLIP": self.openings.image_clahe_clip,
            "OPENING_IMAGE_CLAHE_TILE": self.openings.image_clahe_tile,
            "OPENING_IMAGE_UNSHARP_AMOUNT": self.openings.image_unsharp_amount,
            "OPENING_IMAGE_UNSHARP_SIGMA": self.openings.image_unsharp_sigma,
            "OPENING_IMAGE_SOURCE": self.openings.image_source,
            "OPENING_IMAGE_INTENSITY_CLIP_PCT": self.openings.image_intensity_clip_pct,
            "OPENING_IMAGE_INTENSITY_MIN_COVERAGE": self.openings.image_intensity_min_coverage,
            "OPENING_IMAGE_CMAP": self.openings.image_cmap,
            "OPENING_WALL_DISTANCE_TOLERANCE": self.openings.wall_distance_tolerance,
            "OPENING_LOCAL_OUTPUT_DIR": self.openings.local_output_dir,
            "OPENING_DENSE_SOURCE_PATH": self.openings.dense_source_path,
            "OPENING_DENSE_WALL_IDS": (
                list(self.openings.dense_wall_ids)
                if self.openings.dense_wall_ids is not None
                else None
            ),
            "OPENING_DENSE_WALL_MARGIN": self.openings.dense_wall_margin,
            "OPENING_POINT_CLOUD_CHUNK_SIZE": self.openings.point_cloud_chunk_size,
            "POINT_CLOUD_TO_POST_PROCESSING_SCALE": self.openings.point_cloud_to_post_processing_scale,
            "OPENING_POINT_CLOUD_TO_CSV_OFFSET": (
                list(self.openings.point_cloud_to_csv_offset)
                if self.openings.point_cloud_to_csv_offset is not None
                else None
            ),
            "OPENING_ANNOTATION_CSV_PATH": self.openings.annotation_csv_path,
            "OPENING_ANNOTATION_CHUNK_SIZE": self.openings.annotation_chunk_size,
        }
        # Emitted only when explicitly set, so they stay a deliberate override
        # rather than a default that silently outranks WALL_ANGLE_TOL_DEG.
        if self.wall.vertical_threshold is not None:
            parameters["VERT_THR"] = self.wall.vertical_threshold
        if self.wall.horizontal_threshold is not None:
            parameters["HORI_THR"] = self.wall.horizontal_threshold
        return parameters


def coerce_parameters(parameters: Mapping[str, Any] | PostProcessConfig) -> dict[str, Any]:
    if isinstance(parameters, PostProcessConfig):
        return parameters.to_parameters()

    merged = PostProcessConfig().to_parameters()
    merged.update(dict(parameters))
    return merged
