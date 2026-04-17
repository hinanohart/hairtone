"""Mask builder smoke tests — verify shapes, ranges, and edge cases."""

import numpy as np

from hairtone.masks import build_hair_mask


def _fake_inputs(h: int = 96, w: int = 128) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    bgr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    hair = np.zeros((h, w), dtype=np.float32)
    hair[h // 4 : 3 * h // 4, :] = 0.9
    skin = np.zeros((h, w), dtype=np.float32)
    skin[: h // 4, :] = 0.8
    cloth = np.zeros((h, w), dtype=np.float32)
    cloth[-h // 8 :, :] = 0.6
    return {"bgr": bgr, "hair": hair, "skin": skin, "cloth": cloth}


def test_mask_shape_and_range() -> None:
    inp = _fake_inputs()
    out = build_hair_mask(inp["bgr"], inp["hair"], inp["skin"], inp["cloth"])
    assert out.mask.shape == inp["bgr"].shape[:2]
    assert out.mask.dtype == np.float32
    assert out.mask.min() >= 0.0
    assert out.mask.max() <= 1.0


def test_mask_nonzero_inside_hair_region() -> None:
    inp = _fake_inputs()
    out = build_hair_mask(inp["bgr"], inp["hair"], inp["skin"], inp["cloth"])
    hair_center = out.mask[40:72, :].mean()
    assert hair_center > 0.1


def test_mask_zero_when_no_hair_probability() -> None:
    inp = _fake_inputs()
    inp["hair"] = np.zeros_like(inp["hair"])
    out = build_hair_mask(inp["bgr"], inp["hair"], inp["skin"], inp["cloth"])
    # Without hair probability, the region-grow seed is empty — mask stays ~0.
    assert out.mask.mean() < 0.05


def test_mask_artifacts_include_aux_maps() -> None:
    inp = _fake_inputs()
    out = build_hair_mask(inp["bgr"], inp["hair"], inp["skin"], inp["cloth"])
    assert out.skin_score.shape == out.mask.shape
    assert out.skin_nearby.shape == out.mask.shape
