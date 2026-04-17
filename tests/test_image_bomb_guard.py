"""Verify the decompression-bomb cap rejects oversized inputs."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from hairtone.backend import SegmentationResult
from hairtone.pipeline import MAX_PIXELS, recolor_file, recolor_image


class _StubBackend:
    def segment(self, bgr):
        h, w = bgr.shape[:2]
        return SegmentationResult(
            hair_union=np.zeros((h, w), dtype=np.float32),
            skin=np.zeros((h, w), dtype=np.float32),
            cloth=np.zeros((h, w), dtype=np.float32),
        )


def test_recolor_image_rejects_too_large() -> None:
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too large"):
        recolor_image(small, "blue", backend=_StubBackend(), max_pixels=50)


def test_recolor_image_accepts_below_cap() -> None:
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    out = recolor_image(small, "blue", backend=_StubBackend(), max_pixels=200)
    assert out.shape == small.shape


def test_max_pixels_default_is_tight_but_useful() -> None:
    # 64 megapixel cap matches the documented MAX_PIXELS constant.
    assert MAX_PIXELS == 64_000_000


def test_recolor_file_propagates_cap(tmp_path: Path) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), img)
    with pytest.raises(ValueError, match="too large"):
        recolor_file(
            src, "blue", out=tmp_path / "out.png",
            backend=_StubBackend(), max_pixels=50,
        )


def test_recolor_file_validates_jpeg_quality(tmp_path: Path) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    src = tmp_path / "src.png"
    cv2.imwrite(str(src), img)
    with pytest.raises(ValueError, match="jpeg_quality"):
        recolor_file(
            src, "blue", out=tmp_path / "out.jpg",
            backend=_StubBackend(), jpeg_quality=200,
        )
