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
    """Boundary extraction from the projected RANSAC inliers.

    - "alphashape": concave hull (the original); density-sensitive, so the
      boundary can zigzag where the projected point density drops.
    - "raster": rasterize to an occupancy grid, morphologically close it to fill
      sparse voids, then take the largest external contour. Density-independent.
    Raster knobs are in the SAME units as the cloud (feet for these datasets).
    """
    method: str = "alphashape"  # alphashape | raster
    cell: float = 0.25            # grid resolution (units per pixel)
    fill_gap: float = 0.8         # max void width to fill (morph-close kernel)
    simplify_eps_frac: float = 0.02  # approxPolyDP epsilon as a fraction of perimeter


@dataclass(frozen=True)
class FloorConfig:
    eps: float = 0.5
    min_samples: int = 10
    distance_threshold: float = 1.0
    ransac_n: int = 10
    num_iterations: int = 1000
    alpha: float = 2.0
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)


@dataclass(frozen=True)
class CeilingConfig:
    eps: float = 0.5
    min_samples: int = 10
    distance_threshold: float = 1.0
    ransac_n: int = 10
    num_iterations: int = 1000
    alpha: float = 2.0
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)


@dataclass(frozen=True)
class WallConfig:
    projected_bins: int = 300
    resize_width: int = 400
    intensity_threshold: int = 20
    score_threshold: float = 0.6
    vertical_threshold: float = 10.0
    horizontal_threshold: float = 0.1
    buffer_threshold: float = 2.0


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
    max_coverage: float = 0.85
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
    survey_basis: Sequence[Sequence[float]] = field(
        default_factory=lambda: DEFAULT_SURVEY_BASIS
    )
    labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LABELS)
    floor: FloorConfig = field(default_factory=FloorConfig)
    ceiling: CeilingConfig = field(default_factory=CeilingConfig)
    wall: WallConfig = field(default_factory=WallConfig)
    openings: OpeningConfig = field(default_factory=OpeningConfig)

    def to_parameters(self) -> dict[str, Any]:
        """Return the legacy parameter dictionary used by the current stages."""
        parameters = {
            "SURVEY_BASIS": [list(row) for row in self.survey_basis],
            "LABELS": list(self.labels),
            "EPS": self.floor.eps,
            "MIN_SAMPLES": self.floor.min_samples,
            "EPS_F": self.floor.eps,
            "MIN_SAMPLES_F": self.floor.min_samples,
            "DIS_THR_F": self.floor.distance_threshold,
            "RANSAC_N_F": self.floor.ransac_n,
            "NUM_ITER_F": self.floor.num_iterations,
            "ALPHA_F": self.floor.alpha,
            "BOUNDARY_METHOD_F": self.floor.boundary.method,
            "BOUNDARY_CELL_F": self.floor.boundary.cell,
            "BOUNDARY_FILL_GAP_F": self.floor.boundary.fill_gap,
            "BOUNDARY_SIMPLIFY_EPS_FRAC_F": self.floor.boundary.simplify_eps_frac,
            "EPS_C": self.ceiling.eps,
            "MIN_SAMPLES_C": self.ceiling.min_samples,
            "DIS_THR_C": self.ceiling.distance_threshold,
            "RANSAC_N_C": self.ceiling.ransac_n,
            "NUM_ITER_C": self.ceiling.num_iterations,
            "ALPHA_C": self.ceiling.alpha,
            "BOUNDARY_METHOD_C": self.ceiling.boundary.method,
            "BOUNDARY_CELL_C": self.ceiling.boundary.cell,
            "BOUNDARY_FILL_GAP_C": self.ceiling.boundary.fill_gap,
            "BOUNDARY_SIMPLIFY_EPS_FRAC_C": self.ceiling.boundary.simplify_eps_frac,
            "PROJECTED_BINS": self.wall.projected_bins,
            "RESIZE_WIDTH": self.wall.resize_width,
            "INT_THR": self.wall.intensity_threshold,
            "SCORE_THR": self.wall.score_threshold,
            "VERT_THR": self.wall.vertical_threshold,
            "HORI_THR": self.wall.horizontal_threshold,
            "BUFFER_THR": self.wall.buffer_threshold,
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
        return parameters


def coerce_parameters(parameters: Mapping[str, Any] | PostProcessConfig) -> dict[str, Any]:
    if isinstance(parameters, PostProcessConfig):
        return parameters.to_parameters()

    merged = PostProcessConfig().to_parameters()
    merged.update(dict(parameters))
    return merged
