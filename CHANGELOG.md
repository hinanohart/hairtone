# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · semver.

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
