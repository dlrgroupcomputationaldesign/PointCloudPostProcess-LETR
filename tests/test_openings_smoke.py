"""Fast smoke tests for the opening-detection stage -- no heavy data, no ML deps.

Exercises the config plumbing, the detector registry (lazy, so no groundingdino
or transformers import), and the render -> detect-box -> 3-D mapping round-trip
against a synthetic wall. Run with: pytest -q
"""

import numpy as np
import pytest

from post_process.config import PostProcessConfig
from post_process.detectors import Detection, build_detector
from post_process.stages import openings as op
from post_process.utils.opening_image_util import (
    box_to_grid_span,
    downsample_sum,
    render_log_image,
)


def _axis_aligned_wall(wid="1", length=4.0, height=3.0):
    """A simple wall: `length` m along x, `height` m along z, thin in y."""
    return {
        "id": wid,
        "bbox": [
            {"x": 0.0, "y": 0.0, "z": 0.0}, {"x": length, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": height}, {"x": length, "y": 0.0, "z": height},
            {"x": 0.0, "y": 0.1, "z": 0.0}, {"x": length, "y": 0.1, "z": 0.0},
            {"x": 0.0, "y": 0.1, "z": height}, {"x": length, "y": 0.1, "z": height},
        ],
        "footprint": [
            {"x": 0.0, "y": 0.0}, {"x": length, "y": 0.0},
            {"x": length, "y": 0.1}, {"x": 0.0, "y": 0.1},
        ],
        "zRange": {"min": 0.0, "max": height},
    }


def test_config_exposes_opening_parameters():
    params = PostProcessConfig().to_parameters()
    for key in (
        "OPENING_DETECTOR",
        "OPENING_GD_WEIGHTS_PATH",
        "OPENING_IMAGE_BIN_M",
        "POINT_CLOUD_TO_POST_PROCESSING_SCALE",
        "OPENING_MIN_WIDTH",
    ):
        assert key in params
    assert params["OPENING_DETECTOR"] == "grounding_dino_hf"


def test_registry_resolves_both_backends_lazily():
    params = PostProcessConfig().to_parameters()
    assert type(build_detector(params)).__name__ == "GroundingDinoHFDetector"

    original = dict(params, OPENING_DETECTOR="grounding_dino")
    assert type(build_detector(original)).__name__ == "GroundingDinoDetector"


def test_registry_rejects_unknown_detector():
    with pytest.raises(ValueError):
        build_detector({"OPENING_DETECTOR": "does-not-exist"})


def test_fine_grid_dims_then_block_sum_to_render():
    # 4 m x 3 m. Fine bin 0.025 -> 120 x 160 cells; block-sum factor 2 -> 60 x 80.
    params = PostProcessConfig().to_parameters()
    params["POINT_CLOUD_TO_POST_PROCESSING_SCALE"] = 1.0  # treat wall coords as metres
    frame = op._wall_frame(_axis_aligned_wall())
    fine = op._empty_count_grid(frame, params)
    assert fine.shape == (120, 160)
    assert op._render_bin_factor(params) == 2
    render = downsample_sum(fine, op._render_bin_factor(params))
    assert render.shape == (60, 80)


def test_render_produces_upscaled_rgb():
    params = PostProcessConfig().to_parameters()
    params["POINT_CLOUD_TO_POST_PROCESSING_SCALE"] = 1.0
    render = np.zeros((60, 80), dtype=np.uint32)
    render[10:20, 30:50] = 5  # some occupied render cells
    img = render_log_image(render, params)
    cell = params["OPENING_IMAGE_CELL_PX"]
    assert img.shape == (60 * cell, 80 * cell, 3)
    assert img.dtype == np.uint8


def test_box_maps_back_to_expected_3d_span():
    params = PostProcessConfig().to_parameters()
    params["POINT_CLOUD_TO_POST_PROCESSING_SCALE"] = 1.0
    wall = _axis_aligned_wall()
    frame = op._wall_frame(wall)
    fine = op._empty_count_grid(frame, params)
    n_z, n_s = downsample_sum(fine, op._render_bin_factor(params)).shape  # render grid (60, 80)
    cell = params["OPENING_IMAGE_CELL_PX"]

    # Pixel box over the bottom 40 of 60 rows, cols 10..30 (image row 0 = top).
    det = Detection(
        label="a door",
        score=0.9,
        box_xyxy=(10 * cell, (n_z - 40) * cell, 30 * cell, n_z * cell),
    )
    cand = op._detection_to_candidate(det, wall, frame, n_z, n_s, params)
    assert cand is not None
    assert cand["class"] == "door"
    assert cand["bottomZ"] == pytest.approx(0.0, abs=0.05)
    assert cand["topZ"] == pytest.approx(2.0, abs=0.05)   # 40/60 * 3 m
    assert cand["width"] == pytest.approx(1.0, abs=0.05)  # 20/80 * 4 m
    assert len(cand["bbox"]) == 8


def test_run_openings_dense_npy_with_model_and_offset(tmp_path):
    """End-to-end via the API-style signature: dense .npy cloud + xyz_offset +
    opening_detection_model, with a stub detector. Verifies the model is injected
    and that images + JSON land in the local output dir."""
    import numpy as np

    wall = _axis_aligned_wall(wid="1", length=4.0, height=3.0)
    wall_output = {"walls": [wall], "points": []}

    # Dense points on the wall plane (scale=1.0, offset=0 -> identity transform).
    pts = np.column_stack([
        np.random.RandomState(0).uniform(0, 4, 20000),
        np.full(20000, 0.05),
        np.random.RandomState(1).uniform(0, 3, 20000),
    ]).astype(np.float64)
    npy_path = tmp_path / "cloud.npy"
    np.save(npy_path, pts)

    params = PostProcessConfig().to_parameters()
    params["POINT_CLOUD_TO_POST_PROCESSING_SCALE"] = 1.0
    params["OPENING_LOCAL_OUTPUT_DIR"] = str(tmp_path / "out")

    captured = {}

    def fake_build_detector(p):
        captured.update(p)

        class _Stub:
            def detect(self, image_rgb):
                h, w = image_rgb.shape[:2]
                return [Detection("a door", 0.9, (w * 0.4, h * 0.5, w * 0.6, h * 0.98))]

        return _Stub()

    op.build_detector = fake_build_detector
    try:
        out = op.run_openings(
            None,
            params,
            wall_output,
            point_cloud_path=str(npy_path),
            xyz_offset=[0.0, 0.0, 0.0],
            opening_detection_model="fake_weights.pth",
            logging_blob_location=None,
        )
    finally:
        op.build_detector = build_detector  # restore

    # model path was injected for the detector to load
    assert captured["OPENING_GD_WEIGHTS_PATH"] == "fake_weights.pth"
    # produced the expected output structure
    assert set(out) == {"points", "doors", "windows", "openings"}
    assert len(out["doors"]) == 1
    # images written locally
    assert (tmp_path / "out" / "wall_1_log.png").exists()
    assert (tmp_path / "out" / "wall_1_detected.png").exists()


def test_to_original_coordinates_inverts_scale_and_shift():
    from post_process.utils.coordinate_util import to_original_coordinates

    scale, offset = 2.0, [10.0, 20.0, 5.0]

    def to_post(o):  # forward transform the package applies: post = orig*scale - offset
        return [o[0] * scale - offset[0], o[1] * scale - offset[1], o[2] * scale - offset[2]]

    px, py, pz = to_post([3.0, 4.0, 1.0])
    combined = {
        "points": [{"location": {"x": px, "y": py, "z": pz}}],
        "floors": [{"edgePoints": [{"x": px, "y": py, "z": pz}]}],
        "ceilings": [{"edgePoints": [{"x": px, "y": py, "z": pz}]}],
        "walls": [{
            "bbox": [{"x": px, "y": py, "z": pz}],
            "footprint": [{"x": px, "y": py}],
            "zRange": {"min": pz, "max": pz},
        }],
        "levels": [{"id": "1", "zMode": pz}],
        "doors": [{
            "bbox": [{"x": px, "y": py, "z": pz}],
            "bottomZ": pz, "topZ": pz,
            "width": 1.0 * scale, "height": 2.0 * scale,  # lengths invert as /scale
        }],
    }

    out = to_original_coordinates(combined, scale, offset)
    loc = out["points"][0]["location"]
    assert (loc["x"], loc["y"], loc["z"]) == pytest.approx((3.0, 4.0, 1.0))
    assert out["walls"][0]["footprint"][0]["x"] == pytest.approx(3.0)
    assert out["walls"][0]["zRange"]["max"] == pytest.approx(1.0)
    assert out["levels"][0]["zMode"] == pytest.approx(1.0)
    door = out["doors"][0]
    assert door["bottomZ"] == pytest.approx(1.0)
    assert door["width"] == pytest.approx(1.0)
    assert door["height"] == pytest.approx(2.0)
    # input dict not mutated (deep copy)
    assert combined["levels"][0]["zMode"] == pz


def test_box_to_grid_span_inverts_flip_and_cell_px():
    params = PostProcessConfig().to_parameters()
    params["POINT_CLOUD_TO_POST_PROCESSING_SCALE"] = 1.0
    frame = op._wall_frame(_axis_aligned_wall())
    fine = op._empty_count_grid(frame, params)
    n_z, n_s = downsample_sum(fine, op._render_bin_factor(params)).shape
    # Full-image box must recover the full (s, z) extent.
    span = box_to_grid_span((0, 0, n_s * 4, n_z * 4), frame, n_z, n_s, 4)
    s_min, s_max, z_min, z_max = span
    assert s_min == pytest.approx(frame["s_min"], abs=1e-6)
    assert s_max == pytest.approx(frame["s_max"], abs=1e-6)
    assert z_min == pytest.approx(frame["z_min"], abs=1e-6)
    assert z_max == pytest.approx(frame["z_max"], abs=1e-6)
