"""Three-zone hair mask with region growing and skin-aware suppression.

Given the UNION hair probability from :mod:`hairtone.backend` plus per-class
skin/cloth probabilities, :func:`build_hair_mask` returns a soft mask in
``[0, 1]`` suitable for blending with :mod:`hairtone.recolor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaskArtifacts:
    """The mask plus the by-products the re-colouring stage needs."""

    mask: NDArray[np.float32]
    skin_score: NDArray[np.float32]
    skin_nearby: NDArray[np.float32]


def _zone_mask(
    raw_prob: NDArray[np.float32],
    color_match: NDArray[np.float32],
    mid_match: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compose a 3-zone mask: trust core, soft-filter mid, colour-filter edge."""
    import numpy as np

    mask = np.zeros_like(raw_prob, dtype=np.float32)

    core_zone = raw_prob > 0.6
    mask[core_zone] = raw_prob[core_zone]

    mid_zone = (raw_prob > 0.3) & (raw_prob <= 0.6)
    mask[mid_zone] = raw_prob[mid_zone] * mid_match[mid_zone]

    edge_zone = (raw_prob > 0.05) & (raw_prob <= 0.3)
    mask[edge_zone] = raw_prob[edge_zone] * color_match[edge_zone]
    return mask


def _core_hair_stats(
    img_lab: NDArray[np.float64], raw_prob: NDArray[np.float32]
) -> tuple[float, float, float, float, float, float] | None:
    """Return hair-colour mean + clamped std inside the confident core.

    Falls back to ``prob > 0.5`` when the strict 0.7 core has fewer than 100
    pixels, and returns ``None`` when no confident hair pixels exist at all
    (pure-background image) so callers can skip the mask entirely.
    """
    import numpy as np

    core = raw_prob > 0.7
    if np.sum(core) < 100:
        core = raw_prob > 0.5
    if not np.any(core):
        return None
    hair_L = float(img_lab[:, :, 0][core].mean())
    hair_A = float(img_lab[:, :, 1][core].mean())
    hair_B = float(img_lab[:, :, 2][core].mean())
    hair_L_std = max(float(img_lab[:, :, 0][core].std()), 10.0)
    hair_A_std = max(float(img_lab[:, :, 1][core].std()), 3.0)
    hair_B_std = max(float(img_lab[:, :, 2][core].std()), 3.0)
    return hair_L, hair_A, hair_B, hair_L_std, hair_A_std, hair_B_std


def _skin_aux_maps(
    img_lab: NDArray[np.float64],
    sf_skin: NDArray[np.float32],
    hair_L: float,
    hair_A: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return (skin_score, skin_nearby) maps used downstream by the recolor step."""
    import cv2
    import numpy as np

    skin_score = (
        np.clip((img_lab[:, :, 0] - hair_L) / 30.0, 0, 2)
        * np.clip((hair_A - img_lab[:, :, 1]) / 8.0, 0, 2)
    ).astype(np.float32)

    skin_binary = (sf_skin > 0.3).astype(np.uint8)
    k_skin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    skin_nearby = cv2.dilate(skin_binary, k_skin, iterations=1).astype(np.float32)
    skin_nearby = cv2.GaussianBlur(skin_nearby, (15, 15), 5)
    return skin_score, skin_nearby


def _region_grow_boost(
    mask: NDArray[np.float32],
    raw_prob: NDArray[np.float32],
    dA: NDArray[np.float64],
    dB: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Promote pixels that match the core hair colour near a confident seed."""
    import cv2
    import numpy as np

    seed = (mask > 0.3).astype(np.uint8)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    grown = cv2.dilate(seed, k_grow, iterations=1)
    dist_from_seed = cv2.distanceTransform(1 - seed, cv2.DIST_L2, 5)
    grow_fade = np.clip(1.0 - dist_from_seed / 50.0, 0.0, 1.0).astype(np.float32)
    ab_dist = np.sqrt(dA ** 2 + dB ** 2)
    color_match_ab = np.clip(1.0 - ab_dist / 2.5, 0.0, 1.0).astype(np.float32)
    grown_mask: NDArray[np.float32] = np.asarray(grown > 0, dtype=np.float32)
    grow_value = color_match_ab * grow_fade * grown_mask * 0.8
    added = int(np.sum((grow_value > 0.1) & (raw_prob < 0.05)))
    log.debug("region_grow: +%d px", added)
    out: NDArray[np.float32] = np.maximum(mask, grow_value).astype(np.float32)
    return out


def _guarantee_core_transition(
    mask: NDArray[np.float32],
    raw_prob: NDArray[np.float32],
    color_match: NDArray[np.float32],
    skin_score: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Blur the confident hair core so the mask has a clean rim near it."""
    import cv2
    import numpy as np

    core_soft: NDArray[np.float32] = np.asarray(
        cv2.GaussianBlur(np.asarray(raw_prob > 0.6, dtype=np.float32), (5, 5), 1.5),
        dtype=np.float32,
    )
    core_soft = core_soft * np.asarray(color_match > 0.3, dtype=np.float32)
    core_soft = core_soft * np.asarray(skin_score < 1.0, dtype=np.float32)
    out: NDArray[np.float32] = np.maximum(mask, core_soft * 0.8).astype(np.float32)
    return out


def _cleanup_noise(mask: NDArray[np.float32]) -> NDArray[np.float32]:
    """Morphological open+close + bilateral filter to kill speckle."""
    import cv2
    import numpy as np

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u8 = (mask * 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k, iterations=1)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k, iterations=1)
    cleaned_f = u8.astype(np.float32) / 255.0
    cleaned_f = cv2.bilateralFilter(cleaned_f, d=9, sigmaColor=0.15, sigmaSpace=4)
    out: NDArray[np.float32] = np.nan_to_num(cleaned_f, nan=0.0).astype(np.float32)
    return out


def _attenuate_skin_cloth(
    mask: NDArray[np.float32],
    raw_prob: NDArray[np.float32],
    sf_skin: NDArray[np.float32],
    sf_cloth: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Pull the mask down on pixels the parsers label as skin/cloth.

    Applied *last* (after the core-transition boost) so that hairline pixels
    deep inside the confident core are never attenuated below the parser's
    own hair probability, even when they are near a skin boundary.
    """
    import numpy as np

    hair_trust = np.clip(raw_prob * 2.0, 0.0, 1.0).astype(np.float32)
    cloth_suppress = np.clip(
        1.0 - sf_cloth * 0.5 * (1 - hair_trust), 0.5, 1.0
    ).astype(np.float32)
    skin_suppress = np.clip(
        1.0 - sf_skin * 0.3 * (1 - hair_trust), 0.7, 1.0
    ).astype(np.float32)
    return (mask * skin_suppress * cloth_suppress).astype(np.float32)


def _erode_boundary(mask: NDArray[np.float32]) -> NDArray[np.float32]:
    """Pull the mask boundary in by ~3 px via distance-transform feathering."""
    import cv2
    import numpy as np

    binary = (mask > 0.3).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    edge_fade = np.clip(dist / 3.0, 0.0, 1.0).astype(np.float32)
    return np.clip(mask * edge_fade, 0.0, 1.0).astype(np.float32)


def build_hair_mask(
    img_bgr: NDArray[np.uint8],
    raw_prob: NDArray[np.float32],
    sf_skin: NDArray[np.float32],
    sf_cloth: NDArray[np.float32],
) -> MaskArtifacts:
    """Turn the UNION probability map into a soft 3-zone hair mask."""
    import cv2
    import numpy as np

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    stats = _core_hair_stats(img_lab, raw_prob)
    if stats is None:
        zero = np.zeros(img_bgr.shape[:2], dtype=np.float32)
        return MaskArtifacts(mask=zero, skin_score=zero, skin_nearby=zero.copy())
    hair_L, hair_A, hair_B, lL_std, lA_std, lB_std = stats

    dL = (img_lab[:, :, 0] - hair_L) / lL_std
    dA = (img_lab[:, :, 1] - hair_A) / lA_std
    dB = (img_lab[:, :, 2] - hair_B) / lB_std
    color_dist = np.sqrt(dL ** 2 + dA ** 2 + dB ** 2)
    color_match = np.clip(1.0 - color_dist / 3.5, 0.0, 1.0).astype(np.float32)
    mid_match = np.clip(color_match + 0.4, 0.0, 1.0).astype(np.float32)

    mask = _zone_mask(raw_prob, color_match, mid_match)
    skin_score, skin_nearby = _skin_aux_maps(img_lab, sf_skin, hair_L, hair_A)
    mask = _region_grow_boost(mask, raw_prob, dA, dB)
    mask = _cleanup_noise(mask)
    mask = _guarantee_core_transition(mask, raw_prob, color_match, skin_score)
    # Suppression runs *after* the core-transition guarantee so skin-adjacent
    # core hair is never attenuated below the parser's own hair probability.
    mask = _attenuate_skin_cloth(mask, raw_prob, sf_skin, sf_cloth)
    mask = _erode_boundary(mask)

    return MaskArtifacts(mask=mask, skin_score=skin_score, skin_nearby=skin_nearby)
