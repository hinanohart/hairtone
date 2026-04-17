# Security Policy

## Supported versions

Only the latest `0.1.x` series receives security fixes.

## Reporting a vulnerability

Please report security issues **privately** via GitHub's
[Security Advisories](https://github.com/hinanohart/hairtone/security/advisories/new).
Do **not** open a public issue.

- Acknowledgement target: **7 days**.
- Coordinated disclosure window: **90 days** by default; shorter if the
  issue is already being exploited in the wild.

## Trust model

hairtone is a **library + CLI** and has no network-reachable attack surface
of its own. The following surfaces are intentional plug-in seams that run
third-party code in the caller's process — only load components you wrote
or trust:

- `HairtoneBackend` Protocol (arbitrary segmentation code).
- `TorchSegFormerBiSeNetBackend(bisenet_module=...)` imports any
  installed Python module; defaults to the vendored
  `hairtone._vendor.bisenet`.

See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) for the full list.

## Hardenings already in place

- `torch.load(..., weights_only=True)` on user-supplied BiSeNet
  checkpoints (refuses pickled Python objects).
- `from_pretrained(..., use_safetensors=True, trust_remote_code=False)`
  on SegFormer (refuses pickled `.bin` weights).
- Decompression-bomb cap (`MAX_PIXELS = 64_000_000`) in
  `recolor_image`/`recolor_file`.
- Input validation on image shape, strength range, JPEG quality,
  and preset keys.
- Dependency floors pin known-CVE minima (Pillow ≥ 11.1.0,
  transformers ≥ 4.53.0, opencv-python ≥ 4.8.1.78, torch ≥ 2.2).
