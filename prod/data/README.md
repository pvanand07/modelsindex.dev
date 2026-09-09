# MODELINDEX production data

Generated browser data. Rebuild it from the repository root:

```bash
python scripts/build_prod_data.py
```

Do not edit the JSON files by hand.

- `manifest.json` — schema/provenance, coverage, calibration gates, and caveats.
- `gpus.json` — hardware specs plus the coefficients needed for browser-side estimates.
- `models.json` — one canonical Ollama tag per unique weight digest, aliases, model
  metadata, capability signals, VRAM curves, registry `pushed_at` (last tag push),
  and the Ollama library description.
- `library.json` — per-family `description` and README markdown from `ollama.com/library/{name}`.
- `quality.json` — optional GPU-independent intelligence scores (`q_base`, `q_file`) keyed by
  canonical model ref. Built from `data/quality/`; missing refs fall back to a size prior.

## Accuracy labels

- `estimate.calibrated` means the GPU speed formula was fitted to measurements. It
  does not mean every model was benchmarked.
- `vram_digest_calibrated` means that weight digest has a measured residual offset.
  Other models use the validated global VRAM formula.
- Use-case `signals` come from structural metadata and conservative model-name hints.
  They are not benchmark scores.
- `q_file` is a vendor-benchmark mix times a seed quant fidelity factor. It is not an
  Artificial Analysis proprietary score and is not available for every catalog family.
- VRAM assumes f16 KV. Partial-offload speed is not modeled.

The web UI loads these files from `../prod/data/` when the repository root is served.
Production hosting should gzip or Brotli-compress JSON responses.
