"""LAB Reinhard colour transfer with a warm-shift hedge on skin pixels.

The key insight (compared with vanilla Reinhard) is that mask leakage onto
skin is handled not by *removing* the spill but by *shifting the effective
target colour toward warm skin tones* so the leakage looks like shadow
rather than dyed skin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hairtone.presets import Preset

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# LAB anchors for the warm-shift heuristic (empirical, tuned on portraits).
_WARM_A = 148.0
_WARM_B = 128.0
_SKIN_A = 130.0
_SKIN_B = 140.0


def _fade_gradient(mask: NDArray[np.float32], exponent: float) -> NDArray[np.float64]:
    import numpy as np

    fade = np.clip((mask - 0.2) / 0.6, 0.0, 1.0)
    return fade.astype(np.float64) ** exponent


def recolor(
    img_bgr: NDArray[np.uint8],
    mask: NDArray[np.float32],
    preset: Preset,
    *,
    skin_nearby: NDArray[np.float32] | None = None,
    strength: float = 0.85,
) -> NDArray[np.uint8]:
    """Apply ``preset`` to ``img_bgr`` weighted by ``mask``.

    Parameters
    ----------
    img_bgr:
        Source image in OpenCV BGR uint8.
    mask:
        Soft hair mask in ``[0, 1]`` float32 (same HxW as ``img_bgr``).
    preset:
        Target colour (LAB tuple).
    skin_nearby:
        Optional soft map of "near skin" pixels. When provided, the warm
        shift only activates within this region so that hair highlights are
        not accidentally dulled.
    strength:
        Final blend ratio between original and re-coloured image (0..1).
    """
    import cv2
    import numpy as np

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0.0, 1.0]")
    if mask.shape != img_bgr.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image {img_bgr.shape[:2]}"
        )

    target_L, target_A, target_B = preset.lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    hair = mask > 0.5
    if not np.any(hair):
        return img_bgr.copy()

    sL = img_lab[:, :, 0][hair].mean()
    sA = img_lab[:, :, 1][hair].mean()
    sB = img_lab[:, :, 2][hair].mean()

    fade = _fade_gradient(mask, 1.5)

    danger_A = max(0.0, _SKIN_A - target_A)
    danger_B = max(0.0, target_B - _SKIN_B)
    ab_danger = float(np.sqrt(danger_A ** 2 + danger_B ** 2))
    ab_fade = fade.copy()
    if ab_danger > 5:
        ab_fade = ab_fade ** (1.5 + min(ab_danger, 40) * 0.03)

    L_shift = abs(target_L - sL)
    L_danger = max(0.0, L_shift - 10.0)
    L_fade = fade.copy()
    if L_danger > 0:
        L_fade = L_fade ** (1.0 + L_danger * 0.04)

    # Warm-shift: estimate "skin-like" pixels from local LAB deviation.
    pix_L = img_lab[:, :, 0]
    pix_A = img_lab[:, :, 1]
    L_bright = np.clip((pix_L - sL - 20.0) / 30.0, 0.0, 1.0)
    A_cool = np.clip((sA - pix_A) / 12.0, 0.0, 1.0)
    skin_like = L_bright * A_cool

    warm_factor = np.clip((skin_like - 0.2) * 5.0, 0.0, 1.0)
    if skin_nearby is not None:
        warm_factor = warm_factor * skin_nearby
    warm_factor = cv2.GaussianBlur(
        warm_factor.astype(np.float32), (9, 9), 3
    ).astype(np.float64)

    a_corr = max(0.0, _WARM_A - target_A)
    b_corr = _WARM_B - target_B
    eff_target_A = target_A + a_corr * warm_factor
    eff_target_B = target_B + b_corr * warm_factor
    L_warm_suppress = (1.0 - warm_factor) ** 2

    result_lab = img_lab.copy()
    target_L_map = sL + (target_L - sL) * L_fade * L_warm_suppress
    result_lab[:, :, 0] = (img_lab[:, :, 0] - sL) + target_L_map
    target_A_map = sA + (eff_target_A - sA) * ab_fade * 0.7
    result_lab[:, :, 1] = (img_lab[:, :, 1] - sA) * 0.7 + target_A_map
    target_B_map = sB + (eff_target_B - sB) * ab_fade * 0.8
    result_lab[:, :, 2] = (img_lab[:, :, 2] - sB) * 0.8 + target_B_map
    result_lab = np.clip(result_lab, 0, 255)
    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # 30% luminance compensation
    og_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    rg_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lr = np.clip(np.where(rg_gray > 1, og_gray / rg_gray, 1.0), 0.5, 2.0)
    hsv = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + 0.3 * (lr - 1.0)), 0, 255)
    result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask3 = np.stack([mask] * 3, axis=-1).astype(np.float64)
    blended = (
        img_bgr.astype(np.float64) * (1.0 - mask3 * strength)
        + result_bgr.astype(np.float64) * mask3 * strength
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
