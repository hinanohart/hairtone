# Examples

Drop a portrait (`.jpg` / `.png`) in this directory (all image files are
`.gitignore`'d) and run:

```bash
hairtone examples/you.jpg blue --out examples/you_blue.jpg
hairtone examples/you.jpg all  --out examples/out/
```

The folder stays untracked on purpose — portraits are personal and
licensing is messy. Commit only synthetic or Creative-Commons reference
images if you need to ship demo assets with a PR.
