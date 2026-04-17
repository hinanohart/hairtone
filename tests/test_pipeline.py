"""Integration tests for the pipeline that use a stub backend (no torch)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from hairtone.backend import SegmentationResult
from hairtone.pipeline import recolor_file, recolor_image


class _StubBackend:
    """Return an all-hair probability for the centre stripe of the image."""

    def segment(self, bgr: np.ndarray) -> SegmentationResult:
        h, w = bgr.shape[:2]
        hair = np.zeros((h, w), dtype=np.float32)
        hair[h // 4 : 3 * h // 4, :] = 0.95  # confident hair stripe
        skin = np.zeros((h, w), dtype=np.float32)
        skin[: h // 4, :] = 0.8  # forehead/skin stripe
        cloth = np.zeros((h, w), dtype=np.float32)
        return SegmentationResult(hair_union=hair, skin=skin, cloth=cloth)


@pytest.fixture
def sample_bgr() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(30, 230, size=(128, 96, 3), dtype=np.uint8)


def test_recolor_image_end_to_end(sample_bgr) -> None:
    out = recolor_image(sample_bgr, "blue", backend=_StubBackend())
    assert out.shape == sample_bgr.shape
    # Middle stripe (hair zone) must gain blue energy.
    mid = out[32:96, :, 0].mean()
    baseline = sample_bgr[32:96, :, 0].mean()
    assert mid > baseline


def test_recolor_image_validates_shape() -> None:
    with pytest.raises(ValueError):
        recolor_image(np.zeros((16, 16), dtype=np.uint8), "blue", backend=_StubBackend())


def test_recolor_image_rejects_unknown_preset(sample_bgr) -> None:
    with pytest.raises(KeyError):
        recolor_image(sample_bgr, "not-a-colour", backend=_StubBackend())


def test_recolor_file_round_trips(tmp_path: Path, sample_bgr: np.ndarray) -> None:
    src = tmp_path / "src.png"
    out = tmp_path / "out.png"
    cv2.imwrite(str(src), sample_bgr)
    returned = recolor_file(src, "red", out=out, backend=_StubBackend())
    assert returned == out.resolve()
    assert out.is_file()
    decoded = cv2.imread(str(out))
    assert decoded is not None
    assert decoded.shape == sample_bgr.shape


def test_recolor_file_missing_src_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        recolor_file(tmp_path / "no.png", "blue", out=tmp_path / "out.png", backend=_StubBackend())
