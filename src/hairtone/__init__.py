"""hairtone — hair re-colouring with a UNION (SegFormer + BiSeNet) mask.

Public API:
    recolor_file(src, preset, *, out) -> Path
    recolor_image(bgr, preset, *, backend=None) -> ndarray
    Preset, PRESETS, HairtoneBackend, TorchSegFormerBiSeNetBackend
"""

from hairtone.backend import HairtoneBackend
from hairtone.pipeline import recolor_file, recolor_image
from hairtone.presets import PRESETS, Preset, get_preset, list_preset_names
from hairtone.torch_backend import TorchSegFormerBiSeNetBackend

__all__ = [
    "PRESETS",
    "HairtoneBackend",
    "Preset",
    "TorchSegFormerBiSeNetBackend",
    "get_preset",
    "list_preset_names",
    "recolor_file",
    "recolor_image",
]

__version__ = "0.1.2"
