# Known Limitations

These are intentional design choices where a stricter fix would hurt
usability or usefulness. They are surfaced here so users can judge
whether hairtone fits their threat model / use case.

## L1 — Default SegFormer weights are non-commercial

`jonathandinu/face-parsing` is licensed for **non-commercial research and
educational use only** (the CelebAMask-HQ dataset it was trained on is
research-only). Commercial users must either retrain on a permissively
licensed dataset or plug a different checkpoint / backend in through the
`HairtoneBackend` protocol. hairtone's own code is MIT.

## L2 — `HairtoneBackend` runs third-party code in-process

The Protocol is an official extension seam. Any `class MyBackend:`
implementation imported into your program executes as regular Python
code with full privileges — the same trust model as any `pip install`'d
package. Do **not** accept a backend class name from untrusted user
input; load only backends you wrote or reviewed.

## L3 — `bisenet_module` accepts any importable module name

`TorchSegFormerBiSeNetBackend(bisenet_module="some.module")` calls
`importlib.import_module(...)` with the user-supplied string. An attacker
who can drop a `.py` file on `sys.path` **and** control this argument
can execute arbitrary code at import time. In practice both of those
require prior local access. The default value is `hairtone._vendor.bisenet`
(vendored in-tree), so the common CLI path has no such surface.

## L4 — `recolor_file` / `analyze` follow symlinks and write arbitrary output paths

Both helpers resolve paths with `Path.expanduser().resolve()` for CLI /
trusted-caller use. If you embed them in a network service, validate
`src` and `out` against an allow-listed directory before calling in —
otherwise a client with control over those arguments can read any file
the server process can read (via symlink, though only the recoloured
pixels come back) or overwrite any file it can write.

## L5 — Decompression-bomb cap is process-wide constant

`recolor_image(..., max_pixels=MAX_PIXELS)` caps inputs at ~64 MP to
prevent out-of-memory crashes on adversarial PNGs. Legitimate
high-resolution workflows can raise the cap per call or pre-resize
images before handing them to hairtone.

## L6 — No temporal / video consistency

Frame-to-frame coherence is **not** addressed. Using hairtone on every
frame of a video will produce visible flicker on hair regions where the
segmentation wavers. A future backend or post-process pass could smooth
this out; for now, apply per-frame at your own risk.

## L7 — Warm-shift heuristic is tuned on lightly pigmented skin

The `_WARM_A = 148`, `_WARM_B = 128`, and L/A deviation thresholds in
`recolor.py` were empirically tuned on a fair-skin portrait set. On
darker skin tones the "spill looks like shadow" effect may be
under-strength or over-strength; PRs with demographic-balanced tuning
data are welcome.
