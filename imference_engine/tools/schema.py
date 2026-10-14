"""Tool manifests: the declarative contract of one editing tool.

A manifest is a JSON document (one per tool, shipped in ``tools/manifests/``)
that says what a tool takes, what it returns, which weights it needs, where it
can run and what it costs. It is the single source of truth for every surface:

- the engine validates params against it before running anything,
- Desktop reads ``runtimes`` to badge a tool Local / Local (slow) / Cloud,
- the cloud worker reads ``runtimes.cloud_queue`` to know which tools it serves,
- the API publishes ``GET /tools`` from it (id, params, price),
- the MCP server builds one MCP tool per manifest.

Pure metadata: parsing imports no torch / onnxruntime / Pillow.

Param types: ``int`` / ``float`` (``min``/``max``), ``bool``, ``string``
(``max_length``, ``pattern``), ``enum`` (``values``), ``points`` (a list of
``{"x", "y", "label"}`` clicks, pixel coordinates in the input image, label 1 =
include / 0 = exclude). Input types: ``image``, ``mask`` (an image, white =
selected), ``video``; ``multiple: true`` takes a list (``min_items`` /
``max_items``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from imference_engine.tools.errors import INVALID_INPUT, ToolError

PARAM_TYPES = ("int", "float", "bool", "string", "enum", "points")
INPUT_TYPES = ("image", "mask", "video")
OUTPUT_TYPES = ("image", "mask", "video", "text", "json")
LOCAL_RUNTIMES = ("cpu", "cuda", "mps", "rocm", "directml")
CLOUD_QUEUES = ("tools-cpu", "tools-light", "generation")
CATEGORIES = ("image", "video", "generation")

# Licenses a tool may ship to a cloud queue with. Anything else (or an
# unverified license) keeps the tool local-only — see ToolManifest.cloud_allowed.
PERMISSIVE_SPDX = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "LGPL-2.1-or-later",
    "LGPL-2.1", "LGPL-3.0", "ISC", "Zlib", "n/a", "model",
}


class ManifestError(ValueError):
    """A malformed manifest (a bug in the catalog, not in a request)."""


@dataclass
class ParamSpec:
    name: str
    type: str
    default: Any = None
    required: bool = False
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    values: Optional[list] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    max_items: Optional[int] = None

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        for k in ("default", "description", "min", "max", "values", "max_length", "pattern", "max_items"):
            v = getattr(self, k)
            if v is not None and v != "":
                d[k] = v
        if self.required:
            d["required"] = True
        return d

    def coerce(self, value: Any) -> Any:
        """Validate + normalize one value. Raises ToolError(invalid_input)."""
        name = self.name

        def bad(msg: str) -> ToolError:
            return ToolError(INVALID_INPUT, f"param {name!r}: {msg}", field=name)

        t = self.type
        if t == "int":
            if isinstance(value, bool):
                raise bad("expected an integer")
            if isinstance(value, str):
                try:
                    value = int(value.strip())
                except ValueError:
                    raise bad(f"expected an integer, got {value!r}") from None
            if isinstance(value, float):
                if not value.is_integer():
                    raise bad(f"expected an integer, got {value!r}")
                value = int(value)
            if not isinstance(value, int):
                raise bad(f"expected an integer, got {type(value).__name__}")
            self._check_range(value, bad)
            return value
        if t == "float":
            if isinstance(value, bool):
                raise bad("expected a number")
            if isinstance(value, str):
                try:
                    value = float(value.strip())
                except ValueError:
                    raise bad(f"expected a number, got {value!r}") from None
            if not isinstance(value, (int, float)):
                raise bad(f"expected a number, got {type(value).__name__}")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise bad("expected a finite number")
            self._check_range(value, bad)
            return value
        if t == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "on", "false", "0", "no", "off"):
                return value.strip().lower() in ("true", "1", "yes", "on")
            if isinstance(value, int) and value in (0, 1):
                return bool(value)
            raise bad(f"expected a boolean, got {value!r}")
        if t == "string":
            if not isinstance(value, str):
                raise bad(f"expected a string, got {type(value).__name__}")
            if self.max_length is not None and len(value) > self.max_length:
                raise bad(f"longer than {self.max_length} characters")
            if self.pattern and not re.fullmatch(self.pattern, value):
                raise bad(f"does not match {self.pattern}")
            return value
        if t == "enum":
            if value not in (self.values or []):
                # Accept "2" for an int enum (query strings, CLI).
                for v in self.values or []:
                    if str(v) == str(value):
                        return v
                raise bad(f"expected one of {self.values}, got {value!r}")
            return value
        if t == "points":
            if isinstance(value, str):
                import json
                try:
                    value = json.loads(value)
                except ValueError:
                    raise bad("expected a JSON list of {x, y, label}") from None
            if not isinstance(value, list):
                raise bad("expected a list of {x, y, label}")
            if self.max_items is not None and len(value) > self.max_items:
                raise bad(f"at most {self.max_items} points")
            out = []
            for i, p in enumerate(value):
                if isinstance(p, (list, tuple)) and len(p) in (2, 3):
                    p = {"x": p[0], "y": p[1], "label": p[2] if len(p) == 3 else 1}
                if not isinstance(p, dict) or "x" not in p or "y" not in p:
                    raise bad(f"point {i} must be {{x, y, label}}")
                try:
                    x, y = float(p["x"]), float(p["y"])
                    label = int(p.get("label", 1))
                except (TypeError, ValueError):
                    raise bad(f"point {i} has non-numeric coordinates") from None
                if label not in (0, 1):
                    raise bad(f"point {i} label must be 0 or 1")
                out.append({"x": x, "y": y, "label": label})
            return out
        raise bad(f"unknown param type {t!r}")  # pragma: no cover - parse rejects it

    def _check_range(self, value, bad) -> None:
        if self.min is not None and value < self.min:
            raise bad(f"must be >= {self.min}")
        if self.max is not None and value > self.max:
            raise bad(f"must be <= {self.max}")


@dataclass
class InputSpec:
    name: str
    type: str
    required: bool = True
    multiple: bool = False
    min_items: int = 1
    max_items: Optional[int] = None
    max_megapixels: Optional[float] = None
    max_mb: Optional[float] = None
    max_seconds: Optional[float] = None
    description: str = ""

    def to_dict(self) -> dict:
        d: dict = {"type": self.type, "required": self.required}
        for k in ("description", "max_megapixels", "max_mb", "max_seconds", "max_items"):
            v = getattr(self, k)
            if v not in (None, ""):
                d[k] = v
        if self.multiple:
            d["multiple"] = True
            d["min_items"] = self.min_items
        return d


@dataclass
class OutputSpec:
    name: str
    type: str
    format: Optional[str] = None
    description: str = ""

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.format:
            d["format"] = self.format
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class WeightSpec:
    """One weight file. ``path`` is relative to the tools CDN mirror root (and
    to the local cache). ``when`` restricts it to a param value, e.g.
    ``{"model": "isnet"}`` — only fetched when that variant is requested."""

    name: str
    path: str
    size_mb: float = 0
    sha256: str = ""
    source: str = ""
    when: dict = field(default_factory=dict)

    def applies(self, params: dict) -> bool:
        return all(params.get(k) == v for k, v in self.when.items())

    def to_dict(self) -> dict:
        d = {"name": self.name, "path": self.path, "size_mb": self.size_mb, "sha256": self.sha256}
        if self.source:
            d["source"] = self.source
        if self.when:
            d["when"] = dict(self.when)
        return d


@dataclass
class Runtimes:
    local: list = field(default_factory=list)
    cloud_queue: Optional[str] = None
    slow_on_cpu: bool = False
    min_vram_gb: float = 0
    min_ram_gb: float = 0
    needs_engine: bool = False   # runs a generation model (vary / inpaint): needs an Engine

    def to_dict(self) -> dict:
        return {
            "local": list(self.local), "cloud_queue": self.cloud_queue,
            "slow_on_cpu": self.slow_on_cpu, "min_vram_gb": self.min_vram_gb,
            "min_ram_gb": self.min_ram_gb, "needs_engine": self.needs_engine,
        }


@dataclass
class License:
    spdx: str
    source: str = ""
    verified: bool = False
    # Owner decision to serve the tool in the cloud although the license is
    # not verified / not permissive (the risk is accepted, recorded here
    # instead of faking ``verified``). The text says who decided and why.
    cloud_override: str = ""

    def to_dict(self) -> dict:
        d = {"spdx": self.spdx, "source": self.source, "verified": self.verified}
        if self.cloud_override:
            d["cloud_override"] = self.cloud_override
        return d


@dataclass
class ToolManifest:
    id: str
    version: str
    name: str
    description: str
    category: str
    inputs: dict
    params: dict
    outputs: dict
    runtimes: Runtimes
    license: License
    weights: list = field(default_factory=list)
    credits: float = 0
    available: bool = True
    mcp_name: str = ""

    # -- validation -------------------------------------------------------

    def validate_params(self, params: Optional[dict]) -> dict:
        """Validate a request's params, fill defaults. Returns a NEW dict.

        Unknown keys are rejected (a typo must not silently run with the
        default), required params must be present.
        """
        params = dict(params or {})
        unknown = sorted(set(params) - set(self.params))
        if unknown:
            raise ToolError(INVALID_INPUT, f"unknown param(s) {unknown} for tool {self.id!r}; "
                            f"allowed: {sorted(self.params)}", field=unknown[0])
        out: dict = {}
        for name, spec in self.params.items():
            if name in params and params[name] is not None:
                out[name] = spec.coerce(params[name])
            elif spec.required:
                raise ToolError(INVALID_INPUT, f"missing required param {name!r}", field=name)
            elif spec.default is not None:
                out[name] = spec.default
            else:
                out[name] = None
        return out

    def validate_input_names(self, inputs: dict) -> None:
        """Check the input names / arity (not the content — see tools.io)."""
        unknown = sorted(set(inputs) - set(self.inputs))
        if unknown:
            raise ToolError(INVALID_INPUT, f"unknown input(s) {unknown} for tool {self.id!r}; "
                            f"expected: {sorted(self.inputs)}", field=unknown[0])
        for name, spec in self.inputs.items():
            val = inputs.get(name)
            if val is None or val == [] or val == "":
                if spec.required:
                    raise ToolError(INVALID_INPUT, f"missing required input {name!r}", field=name)
                continue
            if spec.multiple:
                if not isinstance(val, (list, tuple)):
                    raise ToolError(INVALID_INPUT, f"input {name!r} takes a list", field=name)
                if len(val) < spec.min_items:
                    raise ToolError(INVALID_INPUT, f"input {name!r} needs at least {spec.min_items} items", field=name)
                if spec.max_items is not None and len(val) > spec.max_items:
                    raise ToolError(INVALID_INPUT, f"input {name!r} takes at most {spec.max_items} items", field=name)
            elif isinstance(val, (list, tuple)):
                raise ToolError(INVALID_INPUT, f"input {name!r} takes a single file", field=name)

    def weights_for(self, params: dict) -> list:
        return [w for w in self.weights if w.applies(params)]

    @property
    def cloud_allowed(self) -> bool:
        """A tool reaches a cloud queue only with a verified permissive license,
        or an explicit owner override (``license.cloud_override``)."""
        if self.runtimes.cloud_queue is None:
            return False
        if self.license.cloud_override:
            return True
        return self.license.verified and self.license.spdx in PERMISSIVE_SPDX

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "mcp_name": self.mcp_name,
            "inputs": {k: v.to_dict() for k, v in self.inputs.items()},
            "params": {k: v.to_dict() for k, v in self.params.items()},
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "weights": [w.to_dict() for w in self.weights],
            "runtimes": self.runtimes.to_dict(),
            "cost": {"credits": self.credits},
            "license": self.license.to_dict(),
            "available": self.available,
        }

    def public_dict(self) -> dict:
        """What ``GET /tools`` / the MCP server publish: no weight paths."""
        d = self.to_dict()
        d.pop("weights")
        d["weights_mb"] = round(sum(w.size_mb for w in self.weights), 1)
        return d


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


def parse_manifest(doc: Any, *, origin: str = "<manifest>") -> ToolManifest:
    """Parse + structurally validate one manifest dict. Raises ManifestError."""
    if not isinstance(doc, dict):
        raise ManifestError(f"{origin}: manifest must be an object")

    def req(key: str, typ=str):
        if key not in doc:
            raise ManifestError(f"{origin}: missing {key!r}")
        v = doc[key]
        if not isinstance(v, typ):
            raise ManifestError(f"{origin}: {key!r} must be {typ}")
        return v

    known = {"id", "version", "name", "description", "category", "inputs", "params", "outputs",
             "weights", "runtimes", "cost", "license", "available", "mcp_name", "$comment"}
    extra = set(doc) - known
    if extra:
        raise ManifestError(f"{origin}: unknown key(s) {sorted(extra)}")

    tid = req("id")
    if not _ID_RE.match(tid):
        raise ManifestError(f"{origin}: bad id {tid!r} (lowercase, digits, dashes)")
    ctx = f"{origin} ({tid})"
    category = req("category")
    if category not in CATEGORIES:
        raise ManifestError(f"{ctx}: category must be one of {CATEGORIES}")

    inputs = {}
    for name, spec in req("inputs", dict).items():
        if not isinstance(spec, dict) or spec.get("type") not in INPUT_TYPES:
            raise ManifestError(f"{ctx}: input {name!r} needs a type in {INPUT_TYPES}")
        inputs[name] = InputSpec(
            name=name, type=spec["type"], required=bool(spec.get("required", True)),
            multiple=bool(spec.get("multiple", False)), min_items=int(spec.get("min_items", 1)),
            max_items=spec.get("max_items"), max_megapixels=spec.get("max_megapixels"),
            max_mb=spec.get("max_mb"), max_seconds=spec.get("max_seconds"),
            description=spec.get("description", ""),
        )
    if not inputs:
        raise ManifestError(f"{ctx}: a tool needs at least one input")

    params = {}
    for name, spec in doc.get("params", {}).items():
        if not isinstance(spec, dict) or spec.get("type") not in PARAM_TYPES:
            raise ManifestError(f"{ctx}: param {name!r} needs a type in {PARAM_TYPES}")
        if spec["type"] == "enum" and not spec.get("values"):
            raise ManifestError(f"{ctx}: enum param {name!r} needs values")
        p = ParamSpec(
            name=name, type=spec["type"], default=spec.get("default"),
            required=bool(spec.get("required", False)), description=spec.get("description", ""),
            min=spec.get("min"), max=spec.get("max"), values=spec.get("values"),
            max_length=spec.get("max_length"), pattern=spec.get("pattern"),
            max_items=spec.get("max_items"),
        )
        if p.default is not None:
            try:
                p.coerce(p.default)
            except ToolError as e:
                raise ManifestError(f"{ctx}: default of {name!r} is invalid: {e}") from None
        params[name] = p

    outputs = {}
    for name, spec in req("outputs", dict).items():
        if not isinstance(spec, dict) or spec.get("type") not in OUTPUT_TYPES:
            raise ManifestError(f"{ctx}: output {name!r} needs a type in {OUTPUT_TYPES}")
        outputs[name] = OutputSpec(name=name, type=spec["type"], format=spec.get("format"),
                                   description=spec.get("description", ""))

    weights = []
    for i, w in enumerate(doc.get("weights", [])):
        if not isinstance(w, dict) or not w.get("name") or not w.get("path"):
            raise ManifestError(f"{ctx}: weights[{i}] needs name + path")
        when = w.get("when", {}) or {}
        for k in when:
            if k not in params:
                raise ManifestError(f"{ctx}: weights[{i}].when refers to unknown param {k!r}")
        weights.append(WeightSpec(name=w["name"], path=w["path"], size_mb=float(w.get("size_mb", 0)),
                                  sha256=w.get("sha256", "") or "", source=w.get("source", ""), when=when))

    rt = req("runtimes", dict)
    local = list(rt.get("local", []))
    bad_rt = [r for r in local if r not in LOCAL_RUNTIMES]
    if bad_rt:
        raise ManifestError(f"{ctx}: unknown local runtime(s) {bad_rt}")
    queue = rt.get("cloud_queue")
    if queue is not None and queue not in CLOUD_QUEUES:
        raise ManifestError(f"{ctx}: cloud_queue must be one of {CLOUD_QUEUES}")
    runtimes = Runtimes(local=local, cloud_queue=queue, slow_on_cpu=bool(rt.get("slow_on_cpu", False)),
                        min_vram_gb=float(rt.get("min_vram_gb", 0)), min_ram_gb=float(rt.get("min_ram_gb", 0)),
                        needs_engine=bool(rt.get("needs_engine", False)))

    lic = req("license", dict)
    if not lic.get("spdx"):
        raise ManifestError(f"{ctx}: license.spdx is required")
    license_ = License(spdx=lic["spdx"], source=lic.get("source", ""), verified=bool(lic.get("verified", False)),
                       cloud_override=str(lic.get("cloud_override") or ""))

    cost = doc.get("cost", {}) or {}
    return ToolManifest(
        id=tid, version=req("version"), name=req("name"), description=req("description"),
        category=category, inputs=inputs, params=params, outputs=outputs, runtimes=runtimes,
        license=license_, weights=weights, credits=float(cost.get("credits", 0)),
        available=bool(doc.get("available", True)),
        mcp_name=doc.get("mcp_name") or tid.replace("-", "_"),
    )
