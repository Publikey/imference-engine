# SD 1.5 engine (`imference_engine.sd15`)

A deliberately tiny, single-model engine for running plain **Stable Diffusion
1.5** text-to-image. It's the working local snippet wrapped in a class — nothing
more. Meant for short-lived "for fun" use, not the production multi-model path.

What it intentionally does **not** have: multi-model LRU, offline model tree,
CDN mirror, scheduler swapping, VAE substitution, img2img, batching, LoRA,
negative prompts. If you need any of that, use `imference_engine.engine.Engine`.

## Usage

```python
from imference_engine.sd15 import SD15Engine

engine = SD15Engine().load()          # loads the pipeline once
image = engine.generate(
    prompt="a girl laying on grass",  # required
    height=512,                       # default 512
    width=512,                        # default 512
    num_inference_steps=25,           # default 25
    guidance_scale=7.5,               # default 7.5
)
image.save("cursed_sd15.png")
```

The safety checker is disabled (it otherwise blanks images to black). Device is
auto-detected (cuda → mps → cpu); fp16 on CUDA, fp32 elsewhere.

## Install

Needs the runtime extra (torch + diffusers + transformers + Pillow):

```
pip install "imference-engine[runtime]"
```

The checkpoint is pulled from the Hugging Face Hub on first load
(`stable-diffusion-v1-5/stable-diffusion-v1-5`). Override the repo with
`SD15Engine(model_id="...")` to run a 1.5 fine-tune.
