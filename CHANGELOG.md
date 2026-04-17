# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · semver.

## [0.1.1] — 2026-04-18

### Fixed
- **License accuracy.** SegFormer checkpoint
  (`jonathandinu/face-parsing`) is now correctly documented as **non-commercial
  research & educational use only** (previously misstated as Apache 2.0).
  This is a license *documentation* fix; the code stays MIT.
- **Out-of-the-box BiSeNet flow.** The advertised `--bisenet-weights` path
  now works without any extra setup: a zllrunning-derived BiSeNet
  architecture is vendored under `hairtone._vendor.bisenet` (MIT,
  attribution preserved in `licenses/zllrunning-MIT.txt`). New
  `--bisenet-module` CLI flag lets advanced users swap it out.
- README usage synopsis no longer contains a duplicate preset name.

### Added
- `SECURITY.md` with private-disclosure instructions and trust model.
- `KNOWN_LIMITATIONS.md` documenting the non-commercial checkpoint,
  plug-in trust boundary, symlink following, decompression-bomb cap,
  and demographic tuning caveat.
- Decompression-bomb cap (`MAX_PIXELS = 64 MP`) on `recolor_image` /
  `recolor_file`; user-overridable via `max_pixels=...`.
- `jpeg_quality` argument validation (must be in [0, 100]).
- Explicit `use_safetensors=True`, `trust_remote_code=False`, and
  `revision=` support on the SegFormer loader (safer, pinnable).
- Wrapping of BiSeNet `torch.load` errors with a friendly message
  instructing users to re-save as `torch.save(net.state_dict(), …)`.
- Citations section (Reinhard 2001, SegFormer 2021, BiSeNet 2018,
  CelebAMask-HQ 2020).
- Dependabot config for `pip` and `github-actions`.
- GitHub Actions pinned to commit SHAs; explicit minimal `permissions`
  block on the CI workflow.
- `--bisenet-module` and `--segformer-revision` CLI flags.

### Changed
- `torch>=2.2` floor (was `>=2.0`) so hardened `weights_only` semantics
  are reliable across the tested range.

## [0.1.0] — 2026-04-18

First public release.

### Added
- SegFormer ∪ BiSeNet UNION face-parsing backend (PyTorch / HuggingFace),
  with SegFormer-only fallback when no BiSeNet checkpoint is provided.
- Three-zone soft hair mask (core / mid / edge) with core-colour statistics,
  region growing, skin-aware attenuation, bilateral smoothing, and
  boundary erosion.
- Warm-shift LAB Reinhard transfer that hides mask spill on skin by
  shifting the effective target colour toward warm tones instead of trying
  to *remove* the spill.
- 19 named presets covering blondes, pastels, primary, jewel tones, and
  metallics, each with a hex reference for reviewability.
- Typed public API (``recolor_image``, ``recolor_file``, ``Preset``,
  ``HairtoneBackend``) and a ``hairtone`` CLI with ``--list-presets``,
  ``--all``, and ``--bisenet-weights``.
- Deterministic stub-backend tests — the full test suite runs without
  downloading model weights.
