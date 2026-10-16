# Editing tools — catalog and wire contracts

One catalog of small editing tools, one implementation per tool
(`imference_engine.tools`), two places to run it: the Desktop sidecar (local)
and runqy workers (cloud). This file is the contract between the five repos
(engine, gen-image-workers, imference API, imference-desktop, Imference-mcp).

## Catalog

Each tool is a JSON manifest in `imference_engine/tools/manifests/<id>.json`:

| key | meaning |
| --- | --- |
| `id`, `version`, `name`, `description`, `category` (`image` / `video` / `generation`) | identity |
| `mcp_name` | MCP tool name (snake_case) |
| `inputs` | `{name: {type: image\|mask\|video, required, multiple, min_items, max_items, max_megapixels, max_mb, max_seconds}}` |
| `params` | `{name: {type: int\|float\|bool\|string\|enum\|points, default, required, min, max, values, max_length, pattern, max_items, description}}` |
| `outputs` | `{name: {type: image\|mask\|video\|text\|json, format}}` |
| `weights` | `[{name, path, size_mb, sha256, source, when: {param: value}}]` — `path` is relative to the tools mirror |
| `runtimes` | `{local: [cpu, cuda, mps, rocm, directml], cloud_queue: tools-cpu\|tools-gpu\|generation\|null, slow_on_cpu, min_ram_gb, min_vram_gb, needs_engine}` |
| `cost.credits` | price per run in credits (1 credit = $0.001). `0` on `generation` tools = the price of the selected model |
| `license` | `{spdx, source, verified, cloud_override?}` — a tool reaches a cloud queue only when `verified` and permissive, or when `cloud_override` records an owner decision to accept the risk (`cloud_allowed`) |
| `available` | `false` = declared, not implemented (hidden everywhere) |

`python -m imference_engine.tools export` prints the public catalog
(`{"version": 1, "tools": [...]}`, no weight sources or hashes, plus
`weights_mb`, `weight_files` (CDN-relative `path`, `size_mb`, and the param
value it is fetched for, `when`) and `cloud_available`). The API, Desktop and MCP ship a copy of that JSON and must
re-export it when manifests change.

### Param validation (identical on every surface)

- unknown param -> `invalid_input`; missing `required` -> `invalid_input`;
  absent -> `default` (or null).
- `int`: integer (a string of digits or an integral float is accepted), within `min`/`max`.
- `float`: finite number within `min`/`max`.
- `bool`: true/false (also "true"/"false"/"1"/"0").
- `string`: `max_length`, `pattern` (full match).
- `enum`: one of `values` (string form of a value accepted: `"4"` for `4`).
- `points`: list of `{x, y, label}` (or `[x, y, label]`), label 0/1, at most `max_items`.

### Error codes

`invalid_input`, `too_large` (client errors: never charged, never retried),
`unsupported_runtime`, `weights_missing`, `oom` (cloud can succeed: Desktop
offers the cloud with its price — never silently), `timeout`, `internal`.

### Progress stages

`downloading_weights`, `loading`, `running`, `encoding`, each with a 0..1
fraction.

## Engine API

```python
from imference_engine.tools import catalog, run_tool, ToolError, release
res = run_tool(tool_id, inputs, params, output_dir=..., on_progress=fn,
               device="auto", engine=None, timeout=None)
res.to_dict()  # {tool_id, tool_version, outputs: [{name, type, format, path, text?, data?, width?, height?}],
               #  params, duration_ms, runtime, stages}
release()      # drop warm ONNX sessions / Florence-2
```

`inputs` values are local paths (a list for `multiple` inputs). Generation
tools (`needs_engine`) take `engine=<loaded Engine>` and `params.model` = a
registered model name (a single-model engine uses its only model).

Weights: `TOOLS_MODEL_CACHE` (default `$HF_HOME/tools` or
`~/.cache/imference/tools`), mirror `TOOLS_MODEL_CDN` (fallback
`$IMAGE_MODEL_CDN/tools`). `validation/stage_tools_r2.py` builds the mirror.

## Cloud: runqy task (queues `tools-cpu`, `tools-gpu`)

Enqueued by the API (`POST <RUNQY_ENDPOINT>` envelope `{queue, data, timeout}`)
with `timeout: 0` — the per-queue `active_timeout` rules:

```json
{
  "id": "<request_id>",
  "webhook_url": "https://imference.com/webhook",
  "tool_id": "remove-background",
  "tool_version": "1.0.0",
  "inputs": {"image": "https://<signed input url>"},
  "params": {"model": "birefnet-lite", "output": "png-alpha"}
}
```

Inputs are always URLs (never inline bytes). Multiple inputs are a list of URLs.

Worker result (returned AND POSTed to `webhook_url`, header
`X-Webhook-Secret: $WEBHOOK_SECRET` when set), same envelope as generation:

```json
{
  "id": "<request_id>",
  "tool_id": "remove-background", "tool_version": "1.0.0",
  "duration_ms": 812, "runtime": "onnx:CUDAExecutionProvider",
  "data": [{"url": "https://<blob>/tools/<uuid>.png", "seed": "", "name": "image",
            "type": "image", "format": "png", "width": 1024, "height": 1024}],
  "errors": []
}
```

Text outputs (`describe`) are uploaded as a `.txt` blob and also inlined as
`"text"` on the data item. On failure: `"data": []` and
`"errors": [{"error": "<message>", "code": "oom", "batch_index": -1}]`.
Outputs go to the workers' Azure Blob container (`AZURE_STORAGE_CONTAINER`,
default `ephemeral`, 24 h lifecycle rule). The worker logs `tool_id`,
`tool_version`, duration and input size for pricing.

## Cloud: generation tools (`vary`, `inpaint`, `outpaint`, `instruct-edit`)

They run on the existing image workers of the selected model (its normal
queue).
`inpaint` is disabled (`available: false`, not in the catalog) until it gets a
per-step latent-blend implementation; the example below shows the payload shape. The API enqueues a normal generation request for the model plus:

```json
{"id": "...", "webhook_url": "...", "model_name": "<model>", "prompt": "...",
 "tool_id": "inpaint", "tool_version": "1.0.0",
 "tool_inputs": {"image": "https://...", "mask": "https://..."},
 "tool_params": {"strength": 0.9, "feather": 12, "seed": 42}}
```

An im-engine worker that sees `tool_id` calls
`run_tool(tool_id, inputs, params, engine=<its engine>)` with
`params.model = model_name` and answers with the result envelope above
(`seed` = the seed used). Workers whose engine predates tools answer
`unsupported_runtime`.

`instruct-edit` needs a model whose engine is `qwenimage-edit` (e.g. catalog
model `qwen-image-edit-2511`); `vary` / `inpaint` / `outpaint` need a model
that is NOT an edit model. Its prompt is `tool_params.instruction` (there is
no `prompt` param); `tool_inputs` may carry `references` (a list of up to 2
URLs) next to `image`. Served by the `qwen-image-edit-im-engine` worker.

## API (imference)

- `GET /api/tools` — public catalog JSON (`{"version", "tools": [...]}`, each
  with `price_credits` and `price_usd`; generation tools carry `price: "model"`).
- `GET /tools`, `GET /tools/{id}` — server-rendered catalog pages (SEO).
- `POST /tools/{tool_id}` (Bearer API key, credits) and
  `POST /ondemand/tools/{tool_id}` (x402). Body: JSON
  `{"inputs": {"image": "<https url | data: URI>"}, "params": {...}}` or
  multipart (`image=<file>`, …, `params=<json>`). Non-URL inputs are uploaded
  by the API to a private blob container and passed as short-lived signed URLs;
  they are deleted when the task ends. Response `{"request_id", "kind": "tool",
  "tool_id", "credits"}`. Result: the existing `GET /status` /
  `GET /ondemand/status`, whose `data` gains `Outputs` (all items) and `Text`.
- Credits are refunded when a tool request fails.

## Desktop

- Local runs: a dedicated tools sidecar (`sidecar/tools_main.py`, runqy stdio
  protocol, no generation model needed) started on first use and stopped when
  idle. Task payload `{"op": "run", "tool_id", "inputs": {name: path},
  "params", "output_dir", "device"}`; `{"op": "release"}`; `{"op": "catalog"}`.
  Result `{"ok": true, "result": <ToolResult.to_dict()>}` or
  `{"ok": false, "error": {code, message}}`. Progress on stderr:
  `IMTOOL_PROGRESS {"task_id", "stage", "fraction"}`.
- Generation tools run in the generation sidecar (`tool_id` in its payload).
- Local HTTP API for agents (MCP `run_on: local`): `127.0.0.1:<port>`,
  `Authorization: Bearer <token>`, discovery file
  `<UserConfigDir>/imference-desktop-go/local-api.json` = `{"port", "token",
  "pid"}`. `GET /v1/tools`, `POST /v1/tools/{tool_id}` with
  `{"inputs": {name: <path | https url | data: URI>}, "params"}` ->
  `{"tool_id", "outputs": [{name, type, format, path, text?}], "duration_ms"}`
  or `{"error": {code, message}}`.

## MCP

One MCP tool per catalog entry (`mcp_name`), input = the tool's inputs
(local path, URL or data URI) + params + `run_on: local | cloud | auto`
(`auto` = local when Desktop's local API answers, else cloud).
