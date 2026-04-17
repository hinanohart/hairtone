"""Console script: ``hairtone photo.jpg blue --out out.jpg``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hairtone import __version__
from hairtone.pipeline import recolor_file
from hairtone.presets import PRESETS, list_preset_names


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hairtone",
        description="Recolour hair with a UNION face-parsing mask + LAB Reinhard.",
    )
    p.add_argument("src", type=Path, help="Input image (any OpenCV-readable format)")
    p.add_argument(
        "preset",
        choices=[*list_preset_names(), "all"],
        help="Preset key (see --list-presets), or 'all' for every preset",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. When the preset is 'all', this becomes the output *directory*.",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.85,
        help="Blend strength between original and recoloured image (0..1, default 0.85)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality when the output extension is .jpg/.jpeg (default 92)",
    )
    p.add_argument(
        "--bisenet-weights",
        type=Path,
        default=None,
        help="Path to a BiSeNet CelebAMask-HQ .pth checkpoint. Optional; SegFormer-only fallback is used when absent.",
    )
    p.add_argument(
        "--bisenet-module",
        type=str,
        default="hairtone._vendor.bisenet",
        help="Dotted Python module exposing a BiSeNet(n_classes=19) class. Defaults to the vendored copy.",
    )
    p.add_argument(
        "--segformer-revision",
        type=str,
        default=None,
        help="Pin the HuggingFace SegFormer repo to a specific commit SHA for reproducibility.",
    )
    p.add_argument(
        "--list-presets",
        action="store_true",
        help="Print all preset keys with their hex reference and exit.",
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress logging below WARNING"
    )
    p.add_argument(
        "--version", action="version", version=f"hairtone {__version__}"
    )
    return p


def _print_presets() -> None:
    print(f"{'key':<14}  {'name':<14}  hex")
    print(f"{'---':<14}  {'----':<14}  ---")
    for preset in PRESETS.values():
        print(f"{preset.name:<14}  {preset.pretty_name:<14}  {preset.hex_reference}")


def _default_out_for_single(src: Path, preset: str) -> Path:
    return src.with_name(f"{src.stem}_{preset}{src.suffix}")


def _default_out_dir_for_all(src: Path) -> Path:
    return src.with_name(f"{src.stem}_hairtone")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.list_presets:
        _print_presets()
        return 0

    try:
        # Deferred backend construction so --list-presets works without torch.
        from hairtone.torch_backend import TorchSegFormerBiSeNetBackend

        backend = TorchSegFormerBiSeNetBackend(
            bisenet_weights=args.bisenet_weights,
            bisenet_module=args.bisenet_module,
            segformer_revision=args.segformer_revision,
        )
        if args.preset == "all":
            out_dir = args.out or _default_out_dir_for_all(args.src)
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in list_preset_names():
                dest = out_dir / f"{args.src.stem}_{name}{args.src.suffix}"
                recolor_file(
                    args.src,
                    name,
                    out=dest,
                    backend=backend,
                    strength=args.strength,
                    jpeg_quality=args.jpeg_quality,
                )
        else:
            dest = args.out or _default_out_for_single(args.src, args.preset)
            recolor_file(
                args.src,
                args.preset,
                out=dest,
                backend=backend,
                strength=args.strength,
                jpeg_quality=args.jpeg_quality,
            )
    except (FileNotFoundError, ValueError, KeyError, ImportError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
