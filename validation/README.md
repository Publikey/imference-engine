# GPU validation harness

Load each engine's **base model**, render one image, report pass/fail. Built for
eyeballing pipelines + renders on a GPU instance, one engine at a time — a
failure in one never blocks the others.

This is separate from the `tests/test_e2e_*.py` pytest smokes (which need
pre-downloaded local single-file weights and are per-engine). This harness
downloads the base model from a config and produces images + a JSON report.

## Run

```bash
# install the engine(s) you want to validate (+ torch for your CUDA)
pip install -e ".[sdxl,sd15,zimage,flux,chroma,qwenimage,anima,dev]"

# gated repos (FLUX.1-dev) need auth
huggingface-cli login

# validate everything the config lists
python validation/validate.py

# or a subset, to a chosen output dir
python validation/validate.py --engines sdxl,flux --outdir /tmp/renders

# see the plan without loading torch / downloading
python validation/validate.py --dry-run
```

Renders land in `validation/renders/<engine>_seed42.png` and a machine-readable
`validation/renders/report.json`. Exit code is non-zero if any selected engine
failed, so it also works as a CI gate.

## Configure — `base_models.yaml`

Each top-level key is a backend name. Edit it to match your setup:

- **`weights_local`** — point at a `.safetensors` (or diffusers dir for Anima)
  you already have; bypasses the download entirely. The fastest path if you've
  pulled weights from Civitai/HF.
- **`filename`** — the single-file weight to download from `repo`. Entries marked
  `(VERIFY)` are best-effort — confirm the exact filename on the HF repo, or use
  `weights_local`. Notably: **Z-Image / Chroma / Qwen-Image** base transformers
  may be sharded or named differently than the guess; Qwen-Image in particular is
  usually sharded in the base repo, so `weights_local` is the reliable route.
- **`offload`** — `true` for the 8–20B models (FLUX, Chroma, Qwen-Image) so they
  fit consumer VRAM via `enable_model_cpu_offload`.
- **`base_model`** — shared-component repo for transformer-only checkpoints
  (FLUX/Chroma/Qwen/Z-Image). Downloaded from HF on first use.

## Notes per engine

| Engine | Base model | Gotcha |
|---|---|---|
| `sdxl` | SDXL 1.0 base | single-file, straightforward |
| `sd15` | SD 1.5 | single-file, light (~2 GB) |
| `zimage` | Z-Image-Turbo | confirm the transformer filename; 8-step turbo |
| `flux` | FLUX.1-dev | **gated** (login); ~24 GB → `offload: true` |
| `chroma` | Chroma1-HD | confirm filename; real CFG (negative used) |
| `qwenimage` | Qwen-Image | base transformer sharded → prefer `weights_local`; 20B |
| `anima` | Anima-Base (diffusers repo) | **Modular pipeline** — the `__call__` kwargs are the unverified seam; if it errors on an unexpected kwarg, trim it in `imference_engine/anima/backend.py` |

## When something fails

`report.json` records the `error` and full `traceback` per engine. For the newer
backends (FLUX, Chroma, Qwen-Image, Anima) the likely first failures are class
names or `__call__` kwargs that differ from the diffusers build you installed —
each backend's module docstring points at the single spot to adjust. Send the
traceback and it's a quick fix.
