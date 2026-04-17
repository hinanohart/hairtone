# Third-party notices

`hairtone` itself is released under the [MIT License](LICENSE). This
document lists third-party components it vendors or interoperates with
at run time; users are responsible for honouring each license.

## Vendored (shipped with this repository)

### BiSeNet architecture code

Location: `src/hairtone/_vendor/bisenet/`.
A clean inference-only subset of
https://github.com/zllrunning/face-parsing.PyTorch, released under the
**MIT License**. The verbatim upstream LICENSE text is preserved at
[`licenses/zllrunning-MIT.txt`](licenses/zllrunning-MIT.txt).

## User-downloaded at run time (NOT shipped)

### SegFormer face-parsing checkpoint

HuggingFace repo: `jonathandinu/face-parsing`.
Per its HuggingFace model card, released **for non-commercial research
and educational purposes only**. The model is fine-tuned from
`nvidia/mit-b5` (which carries NVIDIA's own restrictions) on the
**CelebAMask-HQ** dataset (research-only).

Commercial users must replace this checkpoint — either retrain on a
permissive dataset or plug in a different backend via the
`HairtoneBackend` protocol.

### BiSeNet CelebAMask-HQ checkpoint

Typical source: `zllrunning/face-parsing.PyTorch` `79999_iter.pth`.
Distributed under **MIT**, but its training data (CelebAMask-HQ) is
research-only.

## Python dependencies at run time

- **transformers** (`huggingface/transformers`) — Apache 2.0.
- **torch** — BSD-3-Clause (with component-specific licences).
- **opencv-python** — Apache 2.0 (with third-party notices).
- **numpy** — BSD-3-Clause.
- **Pillow** — HPND.

Check each project's own `LICENSE` file for the authoritative terms.
