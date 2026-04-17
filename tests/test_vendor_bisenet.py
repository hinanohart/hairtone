"""Smoke-test the vendored BiSeNet architecture without downloading weights."""

import pytest


def test_vendor_bisenet_importable() -> None:
    pytest.importorskip("torch")
    from hairtone._vendor.bisenet import BiSeNet

    net = BiSeNet(n_classes=19)
    assert hasattr(net, "cp")
    assert hasattr(net, "ffm")
    assert hasattr(net, "conv_out")


def test_vendor_bisenet_forward_shape() -> None:
    torch = pytest.importorskip("torch")
    from hairtone._vendor.bisenet import BiSeNet

    net = BiSeNet(n_classes=19).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64)
        out, out16, out32 = net(x)
    assert out.shape == (1, 19, 64, 64)
    assert out16.shape == (1, 19, 64, 64)
    assert out32.shape == (1, 19, 64, 64)
