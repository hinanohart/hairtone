# Third-party notices

`hairtone` itself is released under the [Apache License 2.0](LICENSE).
This document lists third-party components it vendors or interoperates
with at run time; users are responsible for honouring each license.

## Vendored (shipped with this repository)

### BiSeNet architecture code

Location: `src/hairtone/_vendor/bisenet/` (shipped with the wheel).

A clean inference-only subset of
https://github.com/zllrunning/face-parsing.PyTorch, released under the
**MIT License**. The verbatim upstream LICENSE text is copied both as
`src/hairtone/_vendor/bisenet/LICENSE` (inside every wheel and sdist) and
as `licenses/zllrunning-MIT.txt` (sdist only, for source-archive
reviewers), and is also reproduced in full below:

```
MIT License

Copyright (c) 2019 zll

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

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
