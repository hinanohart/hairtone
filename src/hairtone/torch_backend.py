"""Reference implementation: SegFormer + BiSeNet on CPU/GPU via PyTorch.

Depends on ``torch``, ``opencv-python``, ``transformers``, and — optionally —
a local BiSeNet checkpoint. If ``bisenet_weights`` is ``None`` the backend
falls back to SegFormer-only segmentation, which is slightly less aggressive
on flyaway hair strands.

The backend lazily loads network weights on first use and caches them on the
instance, so re-using one backend for many images (e.g. a folder of portraits
or a video frame stream) pays the model-load cost only once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hairtone.backend import SegmentationResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

# SegFormer face-parsing class IDs (jonathandinu/face-parsing uses the
# CelebAMask-HQ convention).
_SF_HAIR = 13
_SF_HAT = 14
_SF_SKIN = 1
_SF_CLOTH = 18

_DEFAULT_SEGFORMER_ID = "jonathandinu/face-parsing"


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "hairtone's torch backend requires 'torch' and 'transformers'.\n"
            "Install with: pip install 'hairtone[torch]'"
        ) from err


@dataclass
class TorchSegFormerBiSeNetBackend:
    """Combined SegFormer + (optional) BiSeNet segmentation with model caching.

    Parameters
    ----------
    segformer_id:
        HuggingFace repo ID for the SegFormer face-parsing checkpoint.
    bisenet_weights:
        Path to a BiSeNet ``.pth`` checkpoint compatible with the
        CelebAMask-HQ network layout. If ``None``, BiSeNet is skipped.
    bisenet_module:
        Python module path that exposes a ``BiSeNet`` class with the
        ``n_classes`` constructor argument. Default ``"bisenet_model"``.
    device:
        Torch device string. Defaults to ``"cuda"`` if available, else ``"cpu"``.
    """

    segformer_id: str = _DEFAULT_SEGFORMER_ID
    bisenet_weights: Path | None = None
    bisenet_module: str = "bisenet_module"
    device: str | None = None

    _device_cache: str | None = field(default=None, init=False, repr=False)
    _sf_proc: Any = field(default=None, init=False, repr=False)
    _sf_model: Any = field(default=None, init=False, repr=False)
    _bisenet_model: Any = field(default=None, init=False, repr=False)
    _bisenet_mean: Any = field(default=None, init=False, repr=False)
    _bisenet_std: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _require_torch()

    # --- device + lazy loaders ------------------------------------------------

    def _resolved_device(self) -> str:
        if self._device_cache is not None:
            return self._device_cache
        if self.device:
            self._device_cache = self.device
            return self.device
        import torch

        self._device_cache = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device_cache

    def _ensure_segformer(self) -> tuple[Any, Any]:
        """Load SegFormer on first call and cache it on the instance."""
        if self._sf_model is not None and self._sf_proc is not None:
            return self._sf_proc, self._sf_model
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        device = self._resolved_device()
        self._sf_proc = SegformerImageProcessor.from_pretrained(self.segformer_id)
        model = SegformerForSemanticSegmentation.from_pretrained(self.segformer_id)
        model.eval()
        model.to(device)
        self._sf_model = model
        log.debug("SegFormer %s loaded on %s", self.segformer_id, device)
        return self._sf_proc, self._sf_model

    def _ensure_bisenet(self) -> Any | None:
        """Return the cached BiSeNet, loading it once if weights were provided."""
        if self.bisenet_weights is None:
            return None
        if self._bisenet_model is not None:
            return self._bisenet_model

        import importlib

        import torch

        weights_path = Path(self.bisenet_weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"BiSeNet weights not found: {weights_path}")
        try:
            module = importlib.import_module(self.bisenet_module)
        except ImportError as err:
            raise ImportError(
                f"BiSeNet module {self.bisenet_module!r} not importable. "
                "Install it or pass bisenet_module=<your module path>."
            ) from err
        if not hasattr(module, "BiSeNet"):
            raise AttributeError(
                f"Module {self.bisenet_module!r} has no attribute 'BiSeNet'."
            )
        net = module.BiSeNet(n_classes=19)
        # weights_only=True refuses pickled Python objects, so a malicious
        # .pth file cannot execute code during deserialization.
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        device = self._resolved_device()
        net.to(device)

        self._bisenet_model = net
        self._bisenet_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            1, 3, 1, 1
        ).to(device)
        self._bisenet_std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            1, 3, 1, 1
        ).to(device)
        log.debug("BiSeNet weights %s loaded on %s", weights_path, device)
        return self._bisenet_model

    # --- public API -----------------------------------------------------------

    def segment(self, bgr: NDArray[np.uint8]) -> SegmentationResult:
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        device = self._resolved_device()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        proc, sf = self._ensure_segformer()
        pil_image = Image.fromarray(rgb)
        inputs = proc(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = sf(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        sf_hair = cv2.resize(probs[_SF_HAIR], (w, h), interpolation=cv2.INTER_LINEAR)
        sf_hat = cv2.resize(probs[_SF_HAT], (w, h), interpolation=cv2.INTER_LINEAR)
        skin = cv2.resize(probs[_SF_SKIN], (w, h), interpolation=cv2.INTER_LINEAR)
        cloth = cv2.resize(probs[_SF_CLOTH], (w, h), interpolation=cv2.INTER_LINEAR)
        hair_combined = np.clip(sf_hair + sf_hat * 0.5, 0.0, 1.0)

        bisenet = self._ensure_bisenet()
        if bisenet is not None:
            bi_prob = self._bisenet_probs(rgb, h, w, bisenet)
            hair_union = np.maximum(hair_combined, bi_prob).astype(np.float32)
        else:
            hair_union = hair_combined.astype(np.float32)
            log.info("BiSeNet weights not provided — using SegFormer-only mask.")

        return SegmentationResult(
            hair_union=hair_union,
            skin=skin.astype(np.float32),
            cloth=cloth.astype(np.float32),
        )

    # --- helpers --------------------------------------------------------------

    def _bisenet_probs(
        self, rgb: NDArray[np.uint8], h: int, w: int, net: Any
    ) -> NDArray[np.float32]:
        import cv2
        import numpy as np
        import torch

        device = self._resolved_device()
        mean = self._bisenet_mean
        std = self._bisenet_std

        accum: NDArray[np.float64] = np.zeros((h, w), dtype=np.float64)
        for scale in (512, 768):
            inp = cv2.resize(rgb, (scale, scale), interpolation=cv2.INTER_LANCZOS4)
            t = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            t = (t.to(device) - mean) / std
            with torch.no_grad():
                bp = torch.softmax(net(t)[0], dim=1)[0, 17].cpu().numpy()
            accum += cv2.resize(bp, (w, h), interpolation=cv2.INTER_LINEAR)
        return (accum / 2.0).astype(np.float32)
