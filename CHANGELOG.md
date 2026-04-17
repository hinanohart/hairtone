# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · semver.

## [0.1.2] — 2026-04-18

### Fixed
- **License compliance.** `pip install hairtone` now receives the full
  MIT text for the vendored BiSeNet architecture, as required by MIT's
  "shall be included in all copies" clause. The license ships three
  ways: (1) `src/hairtone/_vendor/bisenet/LICENSE` is installed
  alongside the vendored source, (2) `NOTICE.md` inlines the full MIT
  text, (3) `licenses/zllrunning-MIT.txt` remains in the sdist. v0.1.1
  wheels carried only a *reference* to `licenses/zllrunning-MIT.txt`
  that was missing from the wheel, which we consider a packaging bug.
- Each vendored BiSeNet file now carries a prominent modification
  notice ("Modified 2026-04-18 for hairtone: ...") so reviewers can see
  what differs from upstream without diffing.

### Changed
- README preset taxonomy now lists all 19 presets (`pink`, `orange`,
  `cyan` previously omitted from the brief summary).

### Added
- README badges (PyPI / Python / License / CI / CodeQL / Downloads).
- `KNOWN_LIMITATIONS.md` L8 (unverified BiSeNet checkpoint digest),
  L9 (user-defined `Preset` NaN/Inf is not validated), and L10 (first
  run SegFormer download UX).
- CHANGELOG compare links.

## [0.1.1] — 2026-04-18

### Changed
- **License**: switched from MIT to **Apache License 2.0** for
  consistency with the author's other OSS projects (mosaicraft, yuragi)
  and to obtain an explicit patent-grant clause — the industry standard
  for ML/AI libraries. Third-party notices moved to `NOTICE.md` so the
  `LICENSE` file is a pristine Apache-2.0 template that GitHub detects
  correctly.
- `torch>=2.2` floor (was `>=2.0`) so hardened `weights_only` semantics
  are reliable across the tested range.

### Fixed
- **License accuracy.** SegFormer checkpoint
  (`jonathandinu/face-parsing`) is now correctly documented as **non-commercial
  research & educational use only** (previously misstated as Apache 2.0).
  This is a license *documentation* fix; the project-wide code license
  is Apache-2.0 (see the Changed section above).
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

## [0.1.0] — 2026-04-18

First public release.

[0.1.2]: https://github.com/hinanohart/hairtone/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/hinanohart/hairtone/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/hinanohart/hairtone/releases/tag/v0.1.0

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
