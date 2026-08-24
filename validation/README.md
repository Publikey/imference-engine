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
- **`filename`** — the single-file weight to download from `repo`. The shipped
  entries are all validated (see status below); swap in `weights_local` to
  validate a specific community checkpoint instead. Z-Image / Qwen-Image use the
  Comfy-Org transformer-only single files (their official base repos are
  multi-file), with `base_model` supplying the shared components.
- **`offload`** — `true` for the 8–20B models (FLUX, Chroma, Qwen-Image) so they
  fit consumer VRAM via `enable_model_cpu_offload`.
- **`base_model`** — shared-component repo for transformer-only checkpoints
  (FLUX/Chroma/Qwen/Z-Image). Downloaded from HF on first use.

## Status — all 7 validated on 0.39; 0.40 re-run pending

Every engine below has been validated end-to-end (base model → rendered image) on
**diffusers 0.39** (RTX PRO 5000 Blackwell, torch 2.12). The pins have since
moved to **diffusers 0.40.0** (the H3-unlocking fold — see `pyproject.toml`);
re-run `validate.py` on 0.40 before tagging a release from it.

| Engine | Base model | Notes |
|---|---|---|
| `sdxl` | SDXL 1.0 base | ✅ single-file |
| `sd15` | SD 1.5 | ✅ single-file, light (~2 GB) |
| `zimage` | Z-Image-Turbo (Comfy-Org transformer) | ✅ 8-step turbo, shift 3.0 |
| `flux` | FLUX.1-dev | ✅ **gated** (`hf auth login`); ~24 GB → `offload: true` |
| `chroma` | Chroma1-HD | ✅ real CFG (negative used); trends saturated at high CFG |
| `qwenimage` | Qwen-Image (Comfy-Org transformer, 40.9 GB) | ✅ 20B, `offload: true`; slow |
| `anima` | Anima-Base (diffusers repo) | ✅ **Modular pipeline**; t2i only (no img2img) |

## Wan 2.2 (video) — separate harness

Wan is not an image `Engine` backend; it's its own `WanEngine` (GGUF-MoE video).
Validate it with **`validate_wan.py`** (needs `pip install -e ".[wan,dev]"`):

```bash
python validation/validate_wan.py --list                       # builtin variants (no torch)
python validation/validate_wan.py --variant wan22-t2v-lightning # text-to-video (4-step)
python validation/validate_wan.py --variant wan22-i2v-lightning --image in.png  # image-to-video
python validation/validate_wan.py --frames 33                  # lighter/faster smoke
```

Outputs `renders/wan_<variant>_seed42.mp4` + a sample frame PNG. Heavy: A14B
experts are ~15 GB GGUF each (×2) + shared UMT5/VAE (~11.5 GB); `WAN_PROFILE=auto`
picks the quant from VRAM/RAM, offload keeps VRAM ≈ one expert (~17 GB).

**Status: re-validated end-to-end on diffusers 0.39** (RTX PRO 5000 Blackwell,
torch 2.12) — t2v and i2v both render. (Pins have since moved to 0.40.0 —
re-run `validate_wan.py` on it, like the image suite.) Two fixes came out of it: the UMT5 input
embedding is re-tied on load (transformers 5.1 left it zero-init → the encoder
ignored the prompt), and the i2v default GGUF is now **bullerwins** — the
QuantStack i2v GGUF renders mush on this stack (its `patch_embedding` dequantizes
wrong; the same official model from bullerwins is clean). See
[`../imference_engine/wan/README.md`](../imference_engine/wan/README.md) →
*Built-in variants*. The harness surfaces engine INFO logs (CDN pulls, the UMT5
re-tie, LoRA-applied, DIAG lines) by default; `-q` silences them. `--flow-shift` /
`--guidance` / `--guidance2` / `--no-lora` / `--gguf-repo` / `--gguf-*-template`
override a variant's recipe from the CLI for A/B testing.

## Stage base components onto R2 — `stage_r2.py`

Push each backend's **shared base components** onto the R2 (S3-compatible) CDN
mirror, so workers / imference-desktop with `IMAGE_MODEL_CDN=<bucket-url>` load
them straight from R2 and never touch HuggingFace — immune to a repo going gated
or being removed. Pure I/O, **no GPU** (the model is never loaded): it pulls the
exact `BASE_PATTERNS` the backend uses (never the transformer weights the
checkpoint replaces), writes the `.manifest.json` the CDN reader expects, and
uploads `<prefix>/<repo>/<file>` idempotently.

```bash
pip install -e ".[runtime,stage]"          # stage = boto3
hf auth login                              # once, for the gated FLUX base
export R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
export R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...  R2_BUCKET=gen-models

# --prefix is the path between the bucket root and <repo> (matches IMAGE_MODEL_CDN).
# With bucket gen-models + CDN at .../image, keys become image/<repo>/<file>.
python validation/stage_r2.py --prefix image           # all CDN-wired bases (flux, chroma, sd15, qwenimage, anima)
python validation/stage_r2.py --prefix image --rm      # delete each local dir after upload (disk-tight streaming)
python validation/stage_r2.py --prefix image --dry-run # print the plan + resolved repos, touch nothing
```

Best run on the same remote instance as `validate.py`: fat pipe to HF **and**
Cloudflare, HF auth already set up for gated FLUX, and `--rm` streams one base at
a time so the disk never holds them all. Uploads are resumable (an object already
present with the same size is skipped).

> **anima** is the whole modular repo (DiT + Qwen3 encoder + text conditioner +
> VAE), staged as one tree. Its loader resolves a repo-id `weights_path` through
> `local_repo_dir` before `ModularPipeline.from_pretrained`, so with
> `IMAGE_MODEL_CDN` set it loads from R2 like the others.

**Wan GGUF (`--wan-gguf`).** Wan reads GGUF experts as *direct files*
(`<WAN_MODEL_CDN>/<repo>/<file>`, no manifest), so a separate mode uploads named
files to `<prefix>/<repo>/<file>` — e.g. mirror the bullerwins i2v experts (the
QuantStack i2v mirror renders mush; see `wan/README.md`):

```bash
python validation/stage_r2.py --prefix wan22 --rm \
  --wan-gguf bullerwins/Wan2.2-I2V-A14B-GGUF \
  --files wan2.2_i2v_high_noise_14B_Q8_0.gguf,wan2.2_i2v_low_noise_14B_Q8_0.gguf
```

## When something fails

`report.json` records the `error` and full `traceback` per engine. For the newer
backends (FLUX, Chroma, Qwen-Image, Anima) the likely first failures are class
names or `__call__` kwargs that differ from the diffusers build you installed —
each backend's module docstring points at the single spot to adjust. Send the
traceback and it's a quick fix.
