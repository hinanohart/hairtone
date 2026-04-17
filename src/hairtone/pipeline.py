"""High-level entry points: image-in/image-out and file-in/file-out."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hairtone.backend import HairtoneBackend
from hairtone.masks import build_hair_mask
from hairtone.presets import Preset, get_preset
from hairtone.recolor import recolor
from hairtone.torch_backend import TorchSegFormerBiSeNetBackend

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


def recolor_image(
    bgr: NDArray[np.uint8],
    preset: Preset | str,
    *,
    backend: HairtoneBackend | None = None,
    strength: float = 0.85,
) -> NDArray[np.uint8]:
    """Re-colour a single BGR image in memory."""
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"Expected a BGR image of shape (H, W, 3), got {bgr.shape}")
    p = get_preset(preset) if isinstance(preset, str) else preset

    chosen_backend = backend or TorchSegFormerBiSeNetBackend()
    seg = chosen_backend.segment(bgr)

    artefacts = build_hair_mask(bgr, seg.hair_union, seg.skin, seg.cloth)
    return recolor(
        bgr, artefacts.mask, p, skin_nearby=artefacts.skin_nearby, strength=strength
    )


def recolor_file(
    src: str | Path,
    preset: Preset | str,
    *,
    out: str | Path,
    backend: HairtoneBackend | None = None,
    strength: float = 0.85,
    jpeg_quality: int = 92,
) -> Path:
    """Re-colour an image file and write the result.

    Returns the resolved output path on success.
    """
    import cv2

    src_path = Path(src).expanduser().resolve()
    out_path = Path(out).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"source image not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(src_path))
    if bgr is None:
        raise ValueError(f"cv2 could not decode {src_path}")

    result = recolor_image(bgr, preset, backend=backend, strength=strength)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(
            str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )
    else:
        ok = cv2.imwrite(str(out_path), result)
    if not ok:
        raise OSError(f"cv2.imwrite failed for {out_path}")
    log.info("wrote %s", out_path)
    return out_path
