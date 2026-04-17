"""Exercise the LAB colour-transfer code path end-to-end without torch."""

import numpy as np
import pytest

from hairtone.presets import get_preset
from hairtone.recolor import recolor


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_recolor_rejects_bad_strength() -> None:
    img = _make_image()
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    preset = get_preset("blue")
    with pytest.raises(ValueError):
        recolor(img, mask, preset, strength=-0.1)
    with pytest.raises(ValueError):
        recolor(img, mask, preset, strength=1.1)


def test_recolor_rejects_shape_mismatch() -> None:
    img = _make_image()
    bad_mask = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        recolor(img, bad_mask, get_preset("blue"))


def test_recolor_empty_mask_returns_copy() -> None:
    img = _make_image()
    mask = np.zeros(img.shape[:2], dtype=np.float32)
    out = recolor(img, mask, get_preset("blue"))
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)
    # Must be a copy — mutating `out` must not touch `img`.
    out[0, 0] = [0, 0, 0]
    assert img[0, 0].tolist() != [0, 0, 0] or np.array_equal(out, img) is False


def test_recolor_full_mask_changes_pixels() -> None:
    img = _make_image(32, 32)
    mask = np.ones(img.shape[:2], dtype=np.float32)
    out = recolor(img, mask, get_preset("blue"), strength=1.0)
    assert out.shape == img.shape
    # Blue recolor must move the mean blue channel up (BGR → channel 0).
    assert out[..., 0].mean() > img[..., 0].mean()


def test_recolor_strength_zero_preserves_image() -> None:
    img = _make_image()
    mask = np.ones(img.shape[:2], dtype=np.float32)
    out = recolor(img, mask, get_preset("purple"), strength=0.0)
    np.testing.assert_allclose(out, img, atol=1)


def test_recolor_respects_skin_nearby_shape() -> None:
    img = _make_image()
    mask = np.ones(img.shape[:2], dtype=np.float32)
    skin = np.full(img.shape[:2], 0.5, dtype=np.float32)
    out = recolor(img, mask, get_preset("red"), skin_nearby=skin)
    assert out.shape == img.shape
