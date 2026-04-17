"""Vendored BiSeNet architecture for face parsing.

Source: https://github.com/zllrunning/face-parsing.PyTorch (MIT License,
Copyright (c) 2019 zll). Only the inference-relevant classes are vendored,
and the ImageNet pretrained download of ``resnet18`` is disabled because
hairtone always loads the CelebAMask-HQ checkpoint afterwards anyway.

Full original LICENSE text is preserved in ``licenses/zllrunning-MIT.txt``
at the repo root as required by the MIT license.
"""

from hairtone._vendor.bisenet.model import BiSeNet

__all__ = ["BiSeNet"]
