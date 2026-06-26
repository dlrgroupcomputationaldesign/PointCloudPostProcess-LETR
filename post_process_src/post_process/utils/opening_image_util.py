"""Render per-wall log-density images for opening detection and map detections
back to 3-D.

This is the productionised version of the offline sweep in
``opening_detection_exp/export_all_walls_count_log_img.py``: a wall's points are
histogrammed in its own (length, height) frame, compressed with log1p/sqrt, and
emitted as a grayscale RGB image that a detector (Grounding DINO) consumes.

Convention: the count grid is indexed ``counts[z_idx, s_idx]`` with row 0 at
``z_min`` (bottom of the wall). The rendered *image* is flipped vertically so
row 0 is the top (``z_max``) -- the natural orientation a vision model expects,
with openings the right way up. :func:`box_to_grid_span` undoes that flip when
mapping a detector's pixel box back into ``(s, z)`` spans.
"""

import numpy as np


def downsample_sum(img, factor):
    """Block-sum a 2-D array by an integer factor (coarsen a fine histogram).

    Trims any remainder rows/cols that don't fill a full block. Mirrors the
    experiment's accumulate-fine-then-block-sum approach (0.025 m -> 0.05 m).
    """
    factor = max(1, int(factor))
    if factor == 1:
        return img
    h, w = img.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    if h2 == 0 or w2 == 0:
        return img
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).sum(axis=(1, 3))


def count_grid_dims(frame, bin_m):
    """Number of (rows=z, cols=s) cells for a wall frame at ``bin_m`` resolution."""
    width = frame["s_max"] - frame["s_min"]
    height = frame["z_max"] - frame["z_min"]
    n_s = max(1, int(np.ceil(width / bin_m)))
    n_z = max(1, int(np.ceil(height / bin_m)))
    return n_z, n_s


def _enhance_grayscale(img8, parameters):
    """Optional photo-like enhancement of the uint8 grayscale before flip/upscale.

    Applied in order: Gaussian denoise -> CLAHE local-contrast -> unsharp mask.
    Each step is skipped when its strength parameter is 0/None, so the default
    config returns ``img8`` unchanged. Boundaries are crispened *after* denoising
    so speckle isn't amplified by the unsharp step.
    """
    denoise_sigma = float(parameters.get("OPENING_IMAGE_DENOISE_SIGMA", 0.0) or 0.0)
    clahe_clip = float(parameters.get("OPENING_IMAGE_CLAHE_CLIP", 0.0) or 0.0)
    unsharp_amount = float(parameters.get("OPENING_IMAGE_UNSHARP_AMOUNT", 0.0) or 0.0)

    if denoise_sigma <= 0.0 and clahe_clip <= 0.0 and unsharp_amount <= 0.0:
        return img8

    import cv2

    out = img8
    if denoise_sigma > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), denoise_sigma)

    if clahe_clip > 0.0:
        tile = max(1, int(parameters.get("OPENING_IMAGE_CLAHE_TILE", 8) or 8))
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
        out = clahe.apply(out)

    if unsharp_amount > 0.0:
        sigma = float(parameters.get("OPENING_IMAGE_UNSHARP_SIGMA", 1.0) or 1.0)
        blurred = cv2.GaussianBlur(out, (0, 0), sigma)
        out = cv2.addWeighted(out, 1.0 + unsharp_amount, blurred, -unsharp_amount, 0.0)

    return np.ascontiguousarray(out, dtype=np.uint8)


def render_log_image(counts, parameters):
    """Turn a raw count grid into an HxWx3 uint8 RGB image for the detector.

    Mirrors the offline ``_render_variant`` baseline: log1p/sqrt/raw compression,
    optional percentile clip and gamma tone curve, and ``gray_r`` (openings
    bright) or ``gray`` polarity. The result is vertically flipped so the image
    reads top-down, then optionally upscaled by an integer ``cell_px`` factor
    (nearest-neighbour) so small walls aren't a handful of pixels.
    """
    transform = str(parameters["OPENING_IMAGE_TRANSFORM"])
    arr = counts.astype(np.float64)
    if transform == "raw":
        pass
    elif transform == "log1p":
        arr = np.log1p(arr)
    elif transform == "sqrt":
        arr = np.sqrt(arr)
    else:
        raise ValueError("unknown OPENING_IMAGE_TRANSFORM: {}".format(transform))

    clip_pct = parameters.get("OPENING_IMAGE_CLIP_PCT")
    pos = arr[arr > 0]
    if clip_pct is not None and pos.size:
        vmax = float(np.percentile(pos, float(clip_pct)))
    else:
        vmax = float(arr.max()) if arr.size else 1.0
    # Brightness lever: a scale > 1 raises vmax so typical density maps lighter
    # (gray_r), reproducing brighter renders; < 1 darkens. Default 1.0 = no change.
    vmax *= float(parameters.get("OPENING_IMAGE_VMAX_SCALE", 1.0) or 1.0)
    if vmax <= 0.0:
        vmax = 1.0

    norm = np.clip(arr / vmax, 0.0, 1.0)
    gamma = parameters.get("OPENING_IMAGE_GAMMA")
    if gamma:
        norm = norm ** float(gamma)

    cmap = str(parameters["OPENING_IMAGE_CMAP"])
    if cmap == "gray_r":
        gray = 1.0 - norm  # low density (openings) -> bright
    elif cmap == "gray":
        gray = norm
    else:
        raise ValueError("unknown OPENING_IMAGE_CMAP: {}".format(cmap))

    img8 = (gray * 255.0).astype(np.uint8)
    img8 = _enhance_grayscale(img8, parameters)
    img8 = np.flipud(img8)  # row 0 -> top (z_max)

    cell_px = max(1, int(parameters["OPENING_IMAGE_CELL_PX"]))
    if cell_px > 1:
        img8 = np.repeat(np.repeat(img8, cell_px, axis=0), cell_px, axis=1)

    return np.stack([img8, img8, img8], axis=-1)


def render_raw_log_image(counts, parameters):
    """Match the output of ``export_all_walls_log_images.py``: pure log1p +
    gray_r, normalized over the full data range, with no clip/vmax_scale/gamma
    and no denoise/CLAHE/unsharp enhancement.

    Mirrors what a matplotlib ``imshow(log1p(counts), cmap='gray_r',
    origin='lower', interpolation='nearest')`` would save to disk, including
    the flipud so row 0 of the returned raster is z_max (top of the image).
    Useful when comparing the production renderer to the experiment script's
    raw output without tuning any of the production levers.
    """
    arr = np.log1p(counts.astype(np.float64))
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    norm = np.clip(arr / vmax, 0.0, 1.0)

    cmap = str(parameters.get("OPENING_IMAGE_CMAP", "gray_r"))
    if cmap == "gray_r":
        gray = 1.0 - norm
    elif cmap == "gray":
        gray = norm
    else:
        raise ValueError("unknown OPENING_IMAGE_CMAP: {}".format(cmap))

    img8 = (gray * 255.0).astype(np.uint8)
    img8 = np.flipud(img8)  # row 0 -> top (z_max), matches matplotlib origin='lower'

    cell_px = max(1, int(parameters.get("OPENING_IMAGE_CELL_PX", 1)))
    if cell_px > 1:
        img8 = np.repeat(np.repeat(img8, cell_px, axis=0), cell_px, axis=1)

    return np.stack([img8, img8, img8], axis=-1)


def render_intensity_image(intensity_sum, counts, parameters):
    """Render a per-cell MEAN intensity (reflectance) image, oriented like
    :func:`render_log_image`.

    ``intensity_sum`` and ``counts`` are the (already render-binned) grids of
    summed intensity and point count. The mean is taken per populated cell;
    empty cells (no return -- openings, occlusions) render black. High
    reflectance maps to bright so openings/glass read as dark rectangles, which
    is the natural-photo polarity a detector expects. The same denoise/CLAHE/
    unsharp levers and ``cell_px`` upscaling as the density path are applied.
    """
    counts = counts.astype(np.float64)
    nonempty = counts > 0
    mean = np.zeros_like(counts)
    mean[nonempty] = intensity_sum[nonempty] / counts[nonempty]

    clip_pct = parameters.get("OPENING_IMAGE_INTENSITY_CLIP_PCT")
    pos = mean[nonempty]
    if clip_pct is not None and pos.size:
        vmax = float(np.percentile(pos, float(clip_pct)))
    else:
        vmax = float(mean.max()) if mean.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    norm = np.clip(mean / vmax, 0.0, 1.0)
    gamma = parameters.get("OPENING_IMAGE_GAMMA")
    if gamma:
        norm = norm ** float(gamma)

    img8 = (norm * 255.0).astype(np.uint8)
    img8[~nonempty] = 0  # openings / occlusions stay black
    img8 = _enhance_grayscale(img8, parameters)
    img8 = np.flipud(img8)  # row 0 -> top (z_max)

    cell_px = max(1, int(parameters["OPENING_IMAGE_CELL_PX"]))
    if cell_px > 1:
        img8 = np.repeat(np.repeat(img8, cell_px, axis=0), cell_px, axis=1)

    return np.stack([img8, img8, img8], axis=-1)


def box_to_grid_span(box_xyxy, frame, n_z, n_s, cell_px):
    """Map a detector pixel box back to ``(s_min, s_max, z_min, z_max)`` in metres.

    Inverts the upscale and the vertical flip applied in :func:`render_log_image`.
    Returns ``None`` if the span collapses to nothing.
    """
    cell_px = max(1, int(cell_px))
    x0, y0, x1, y1 = (v / cell_px for v in box_xyxy)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    s_lo, s_hi = frame["s_min"], frame["s_max"]
    z_lo, z_hi = frame["z_min"], frame["z_max"]
    ds = (s_hi - s_lo) / n_s
    dz = (z_hi - z_lo) / n_z

    cand_s_min = np.clip(s_lo + x0 * ds, s_lo, s_hi)
    cand_s_max = np.clip(s_lo + x1 * ds, s_lo, s_hi)
    # Image row 0 is z_max (flipped), so larger pixel-y means smaller z.
    cand_z_max = np.clip(z_hi - y0 * dz, z_lo, z_hi)
    cand_z_min = np.clip(z_hi - y1 * dz, z_lo, z_hi)

    if cand_s_max <= cand_s_min or cand_z_max <= cand_z_min:
        return None
    return float(cand_s_min), float(cand_s_max), float(cand_z_min), float(cand_z_max)


_CATEGORY_COLORS = {
    "door": (0, 200, 0),
    "window": (0, 128, 255),
    "opening": (255, 0, 0),
}


def annotate_detections(image_rgb, annotations):
    """Draw labelled boxes on a copy of ``image_rgb``.

    ``annotations`` is an iterable of dicts with ``box_xyxy`` (pixel coords on
    this image), ``category`` and ``score``. Replaces the original
    ``groundingdino.util.inference.annotate`` so we don't depend on that package.
    """
    import cv2

    canvas = np.ascontiguousarray(image_rgb).copy()
    for ann in annotations:
        x0, y0, x1, y1 = (int(round(v)) for v in ann["box_xyxy"])
        color = _CATEGORY_COLORS.get(ann.get("category"), (255, 255, 0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        text = "{} {:.2f}".format(ann.get("category", "?"), ann.get("score", 0.0))
        cv2.putText(
            canvas,
            text,
            (x0, max(0, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
