"""Segmentation backend abstraction.

The reference pipeline combines two face-parsing networks — SegFormer
(``jonathandinu/face-parsing``) and a BiSeNet trained on CelebAMask-HQ —
and returns a union hair-probability map together with per-class skin and
clothing probabilities. The abstraction lets users swap in their own models
without touching the colour-transfer code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SegmentationResult:
    """Soft probability maps returned by a :class:`HairtoneBackend`.

    Shapes are all ``(H, W)`` of ``float32`` in ``[0, 1]``.
    """

    hair_union: NDArray[np.float32]
    skin: NDArray[np.float32]
    cloth: NDArray[np.float32]


class HairtoneBackend(Protocol):
    """Anything that can turn a BGR image into per-pixel soft masks."""

    def segment(self, bgr: NDArray[np.uint8]) -> SegmentationResult: ...
