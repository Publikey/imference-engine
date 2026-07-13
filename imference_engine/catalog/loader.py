"""Declarative model catalog — parse a ``models.yml`` into ``ModelConfig``.

Replaces one-by-one ``Engine.register_model(...)`` calls with a declarative
registry. Each row carries the load metadata (engine, weights, base_model) plus
the per-model generation defaults (layer 2 of the precedence chain — see
``catalog/defaults.py``). This is the image-side equivalent of the Wan
sub-package's ``presets.py``, YAML-driven instead of a hard-coded dict.

Pure metadata: parsing opens NO weight files and imports no torch, so a catalog
can be inspected on a machine without the runtime extras.

Schema (``version: 1``):

    version: 1
    models:
      - name: z-image-turbo          # required, unique
        engine: zimage               # required (a backend key)
        weights: /cache/z.safetensors  # required (local path OR repo/URL)
        base_model: Tongyi-MAI/Z-Image-Turbo   # optional
        defaults:                    # optional (layer 2), all keys optional
          num_steps: 8
          guidance_scale: 1.0
          backend_options:           # explicit sub-block for engine-specific knobs
            shift: 3.0
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Union

from imference_engine.catalog.defaults import GenerationDefaults


class CatalogError(ValueError):
    """Raised on any malformed catalog: bad structure, missing/duplicate field,
    unknown engine, or an unknown/ill-typed ``defaults`` key."""


@dataclass
class ModelConfig:
    """One catalog row. ``defaults`` is layer 2 of the precedence chain."""

    name: str
    engine: str
    weights: str
    base_model: Optional[str] = None
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)


# Per-field coercion for the defaults block. YAML already yields native
# int/float/str, but this normalizes quoted scalars ("8" -> 8) and rejects
# nonsense ("abc" as num_steps) with a clear error instead of failing deep in
# diffusers. backend_options is handled separately (free dict).
_NUMERIC_INT = ("num_steps", "width", "height", "clip_skip")
_NUMERIC_FLOAT = ("guidance_scale", "strength")
_STRING = ("scheduler", "negative_prompt")


def load(
    source: Union[str, Path],
    *,
    known_engines: Optional[set] = None,
) -> list[ModelConfig]:
    """Parse a catalog YAML file into a list of ``ModelConfig``.

    ``known_engines`` (typically ``set(engine._backends)``) enables strict
    validation that each row's ``engine`` is registered; pass ``None`` to skip
    that check (structure-only, for standalone catalog inspection). Raises
    ``CatalogError`` on any problem, aggregating file context into the message.
    """
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise CatalogError(f"cannot read catalog {path}: {e}") from e
    return loads(text, known_engines=known_engines, origin=str(path))


def loads(
    text: str,
    *,
    known_engines: Optional[set] = None,
    origin: str = "<string>",
) -> list[ModelConfig]:
    """Parse catalog YAML from a string. See ``load`` for semantics."""
    import yaml

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CatalogError(f"{origin}: invalid YAML: {e}") from e

    if doc is None:
        return []
    if not isinstance(doc, dict):
        raise CatalogError(f"{origin}: top level must be a mapping, got {type(doc).__name__}")

    version = doc.get("version", 1)
    if version != 1:
        raise CatalogError(f"{origin}: unsupported catalog version {version!r} (expected 1)")

    raw_models = doc.get("models")
    if raw_models is None:
        raise CatalogError(f"{origin}: missing required 'models' list")
    if not isinstance(raw_models, list):
        raise CatalogError(f"{origin}: 'models' must be a list, got {type(raw_models).__name__}")

    configs: list[ModelConfig] = []
    seen: set = set()
    for i, entry in enumerate(raw_models):
        ctx = f"{origin}: models[{i}]"
        if not isinstance(entry, dict):
            raise CatalogError(f"{ctx}: each entry must be a mapping, got {type(entry).__name__}")

        name = _require_str(entry, "name", ctx)
        if _row_kind(entry, ctx) != "image":
            continue  # video rows are read by load_video(); skip here
        if name in seen:
            raise CatalogError(f"{ctx}: duplicate model name {name!r}")
        seen.add(name)
        ctx = f"{origin}: model {name!r}"

        engine = _require_str(entry, "engine", ctx)
        if known_engines is not None and engine not in known_engines:
            raise CatalogError(
                f"{ctx}: unknown engine {engine!r}. Known: {sorted(known_engines)}"
            )
        weights = _require_str(entry, "weights", ctx)

        base_model = entry.get("base_model")
        if base_model is not None and not isinstance(base_model, str):
            raise CatalogError(f"{ctx}: 'base_model' must be a string, got {type(base_model).__name__}")

        known_keys = {"kind", "name", "engine", "weights", "base_model", "defaults"}
        extra = set(entry) - known_keys
        if extra:
            raise CatalogError(f"{ctx}: unknown key(s) {sorted(extra)}. Allowed: {sorted(known_keys)}")

        defaults = _parse_defaults(entry.get("defaults"), ctx)
        configs.append(ModelConfig(
            name=name, engine=engine, weights=weights,
            base_model=base_model, defaults=defaults,
        ))

    return configs


def _row_kind(entry: dict, ctx: str) -> str:
    kind = entry.get("kind", "image")
    if kind not in ("image", "video"):
        raise CatalogError(f"{ctx}: 'kind' must be image|video, got {kind!r}")
    return kind


@dataclass
class VideoModelConfig:
    """One video catalog row. ``spec`` is the arch-specific passthrough (base_repo,
    gguf_repo, loras, flow_shift, gguf_* for Wan) — the video backend converts it
    to its variant type. Generic here so LTX etc. carry their own spec fields."""

    name: str
    arch: str            # video backend key ("wan" | "ltx"); = the row's `engine`
    mode: str            # "t2v" | "i2v"
    spec: dict = field(default_factory=dict)


def load_video(
    source: Union[str, Path],
    *,
    known_archs: Optional[set] = None,
) -> list[VideoModelConfig]:
    """Parse the ``kind: video`` rows of a catalog into ``VideoModelConfig``.

    ``known_archs`` (typically ``set(engine._backends)`` on the video engine)
    validates each row's arch. Image rows are ignored (read by ``load``)."""
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise CatalogError(f"cannot read catalog {path}: {e}") from e
    return loads_video(text, known_archs=known_archs, origin=str(path))


def loads_video(
    text: str,
    *,
    known_archs: Optional[set] = None,
    origin: str = "<string>",
) -> list[VideoModelConfig]:
    """Parse video rows from catalog YAML. See ``load_video``."""
    import yaml

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise CatalogError(f"{origin}: invalid YAML: {e}") from e
    if doc is None:
        return []
    if not isinstance(doc, dict):
        raise CatalogError(f"{origin}: top level must be a mapping, got {type(doc).__name__}")
    raw_models = doc.get("models")
    if not isinstance(raw_models, list):
        raise CatalogError(f"{origin}: 'models' must be a list")

    configs: list[VideoModelConfig] = []
    seen: set = set()
    for i, entry in enumerate(raw_models):
        ctx = f"{origin}: models[{i}]"
        if not isinstance(entry, dict):
            raise CatalogError(f"{ctx}: each entry must be a mapping, got {type(entry).__name__}")
        name = _require_str(entry, "name", ctx)
        if _row_kind(entry, ctx) != "video":
            continue  # image rows read by load()
        if name in seen:
            raise CatalogError(f"{ctx}: duplicate model name {name!r}")
        seen.add(name)
        ctx = f"{origin}: model {name!r}"

        arch = _require_str(entry, "engine", ctx)
        if known_archs is not None and arch not in known_archs:
            raise CatalogError(f"{ctx}: unknown video arch {arch!r}. Known: {sorted(known_archs)}")
        mode = _require_str(entry, "mode", ctx)
        if mode not in ("t2v", "i2v"):
            raise CatalogError(f"{ctx}: 'mode' must be t2v|i2v, got {mode!r}")

        spec = {k: v for k, v in entry.items()
                if k not in ("kind", "name", "engine", "mode")}
        configs.append(VideoModelConfig(name=name, arch=arch, mode=mode, spec=spec))

    return configs


def _require_str(entry: dict, key: str, ctx: str) -> str:
    val = entry.get(key)
    if val is None:
        raise CatalogError(f"{ctx}: missing required '{key}'")
    if not isinstance(val, str) or not val:
        raise CatalogError(f"{ctx}: '{key}' must be a non-empty string, got {val!r}")
    return val


def _parse_defaults(raw, ctx: str) -> GenerationDefaults:
    if raw is None:
        return GenerationDefaults()
    if not isinstance(raw, dict):
        raise CatalogError(f"{ctx}: 'defaults' must be a mapping, got {type(raw).__name__}")

    valid = {f.name for f in fields(GenerationDefaults)}
    unknown = set(raw) - valid
    if unknown:
        raise CatalogError(
            f"{ctx}: unknown defaults key(s) {sorted(unknown)}. Allowed: {sorted(valid)}"
        )

    kwargs: dict = {}
    for key in _NUMERIC_INT:
        if raw.get(key) is not None:
            kwargs[key] = _coerce(raw[key], int, key, ctx)
    for key in _NUMERIC_FLOAT:
        if raw.get(key) is not None:
            kwargs[key] = _coerce(raw[key], float, key, ctx)
    for key in _STRING:
        if raw.get(key) is not None:
            if not isinstance(raw[key], str):
                raise CatalogError(f"{ctx}: defaults.{key} must be a string, got {raw[key]!r}")
            kwargs[key] = raw[key]

    bo = raw.get("backend_options")
    if bo is not None:
        if not isinstance(bo, dict):
            raise CatalogError(
                f"{ctx}: defaults.backend_options must be a mapping, got {type(bo).__name__}"
            )
        kwargs["backend_options"] = dict(bo)

    return GenerationDefaults(**kwargs)


def _coerce(value, typ, key: str, ctx: str):
    # bool is an int subclass — reject it explicitly so `num_steps: true` fails
    # loudly instead of silently becoming 1.
    if isinstance(value, bool):
        raise CatalogError(f"{ctx}: defaults.{key} must be {typ.__name__}, got bool {value!r}")
    try:
        return typ(value)
    except (TypeError, ValueError) as e:
        raise CatalogError(f"{ctx}: defaults.{key} must be {typ.__name__}, got {value!r}") from e
