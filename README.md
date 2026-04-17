# hairtone

**Portrait hair re-colouring with a UNION face-parsing mask and LAB colour
transfer.**

```bash
hairtone photo.jpg blue --out blue.jpg
hairtone photo.jpg all --out out_dir/
```

Before / after (drop your own portrait under `examples/`):

```
 photo.jpg  ──►  SegFormer∪BiSeNet ──►  3-zone soft mask ──►  LAB Reinhard
                 (confident hair)       (core/mid/edge)       (+ warm-shift)
                                                                      │
                                                                      ▼
                                                                 photo_blue.jpg
```

## What makes it different

Most hair-recolour scripts either (a) paint a flat colour inside a hard
segmentation mask, producing visible edges on flyaways and skin spill, or
(b) run a full diffusion model and burn 10 s / 12 GB of VRAM per image.
`hairtone` sits in the middle:

- **UNION mask.** SegFormer (`jonathandinu/face-parsing`) and a BiSeNet
  trained on CelebAMask-HQ are combined with a per-pixel max. When one
  network misses the top-left hair strand, the other usually catches it.
- **Three-zone soft mask.** Confident core pixels are trusted, mid pixels
  are gently filtered by colour distance, edge pixels are only kept if
  their LAB distance to the *core hair colour* is small. This turns the
  binary segmentation into a matte that handles fly-aways.
- **Region growing.** Pixels near the confident seed that match the core
  hair chroma get added back — covers strands the segmentation networks
  cut off.
- **Warm-shift colour transfer.** Instead of fighting mask spill onto skin
  by eroding the mask (which ruins the hairline), `hairtone` shifts the
  *effective target colour* toward warm skin tones on skin-like pixels.
  Blue hair on a skin spill ends up as "shadow" rather than "dyed cheek".
- **CPU friendly.** The critical path is OpenCV + NumPy. A GPU is only
  useful for the SegFormer / BiSeNet forward passes, and SegFormer alone
  runs in seconds on modern CPUs.

The algorithm is fully deterministic — same image + same preset + same
weights ⇒ same output — which matters for pipelines that version-control
their re-colourings.

## Install

```bash
pip install hairtone                 # CLI + numpy + OpenCV + Pillow only
pip install "hairtone[torch]"        # + torch + transformers (recommended)
```

### Model weights

`hairtone` does **not** bundle weights. The first time you run it,
HuggingFace `transformers` will download SegFormer automatically. For the
optional BiSeNet pass, grab a CelebAMask-HQ checkpoint (e.g. from
[zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch),
MIT-licensed) and point `--bisenet-weights` at it:

```bash
hairtone photo.jpg blue \
    --out photo_blue.jpg \
    --bisenet-weights ./79999_iter.pth
```

Without `--bisenet-weights` the pipeline still works — it just falls back
to SegFormer-only segmentation.

## Python API

```python
import cv2
from hairtone import recolor_image, TorchSegFormerBiSeNetBackend

backend = TorchSegFormerBiSeNetBackend(bisenet_weights=None)  # SegFormer only
bgr = cv2.imread("photo.jpg")
out = recolor_image(bgr, "blue", backend=backend, strength=0.85)
cv2.imwrite("photo_blue.jpg", out)
```

Custom backend? Implement the `HairtoneBackend` protocol:

```python
from hairtone.backend import HairtoneBackend, SegmentationResult

class MyBackend:
    def segment(self, bgr) -> SegmentationResult:
        ...
```

and pass it to `recolor_image(..., backend=MyBackend())`. The colour
transfer, mask builder, and CLI work unchanged.

## Presets

Run `hairtone --list-presets` for the full list with hex references. The
19 presets cover blondes (`blonde`, `honey`, `strawberry`), pastels
(`pastel_pink`, `lavender`, `mint`), primaries (`red`, `blue`, `green`,
`purple`), jewel tones (`hotpink`, `teal`, `turquoise`, `coral`), and
metallics (`silver`, `ash`).

Each preset is a LAB tuple; you can define your own:

```python
from hairtone.presets import Preset
emerald = Preset("emerald", "Emerald", lab=(155, 96, 165), hex_reference="#2d8a55")
```

## CLI

```
usage: hairtone [-h] [--out OUT] [--strength STRENGTH]
                [--jpeg-quality JPEG_QUALITY]
                [--bisenet-weights BISENET_WEIGHTS]
                [--list-presets] [--quiet] [--version]
                src {all,ash,ash,blonde,...}
```

- `hairtone photo.jpg blue` → writes `photo_blue.jpg` next to the source.
- `hairtone photo.jpg all --out out_dir/` → writes every preset.
- `hairtone photo.jpg blue --out blue.png --strength 0.7` → lower blend.

## Project layout

```
src/hairtone/
    __init__.py           # public API
    backend.py            # HairtoneBackend Protocol + SegmentationResult
    torch_backend.py      # SegFormer ∪ (optional) BiSeNet reference impl
    masks.py              # 3-zone mask + region grow + skin suppression
    recolor.py            # LAB Reinhard with warm-shift
    pipeline.py           # recolor_image / recolor_file entry points
    presets.py            # 19 named LAB presets
    cli.py                # hairtone console script
tests/                    # stub-backend tests, no weights needed
```

## Development

```bash
git clone https://github.com/hinanohart/hairtone
cd hairtone
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch]"
pytest
ruff check .
mypy src
```

Tests use a stub backend and **do not** download any weights, so CI is fast
and offline-safe.

## License

- Code: **MIT** (see [LICENSE](LICENSE)).
- SegFormer checkpoint: **Apache 2.0**.
- BiSeNet checkpoint (user-supplied): typically **MIT**.

See the Third-party notices in `LICENSE` for the authoritative list.
