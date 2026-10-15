"""The tool catalog: every manifest shipped in ``tools/manifests/*.json``.

    from imference_engine.tools import catalog
    catalog.get("remove-background")      # -> ToolManifest
    catalog.for_queue("tools-gpu")      # tools a cloud worker of that queue serves
    catalog.export()                      # JSON-able list for GET /tools / MCP

Unavailable tools (``available: false``, e.g. declared but not implemented
yet) are hidden from ``all()`` / ``for_queue()`` / ``export()`` unless asked.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Optional, Union

from imference_engine.tools.errors import INVALID_INPUT, ToolError
from imference_engine.tools.schema import ManifestError, ToolManifest, parse_manifest

CATALOG_VERSION = 1


class ToolCatalog:
    def __init__(self, manifests: list) -> None:
        self._tools: dict = {}
        for m in manifests:
            if m.id in self._tools:
                raise ManifestError(f"duplicate tool id {m.id!r}")
            self._tools[m.id] = m
        mcp = [m.mcp_name for m in manifests]
        dup = {n for n in mcp if mcp.count(n) > 1}
        if dup:
            raise ManifestError(f"duplicate mcp_name(s) {sorted(dup)}")

    @classmethod
    def from_dir(cls, path: Union[str, Path]) -> "ToolCatalog":
        path = Path(path)
        manifests = []
        for f in sorted(path.glob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except ValueError as e:
                raise ManifestError(f"{f}: invalid JSON: {e}") from e
            manifests.append(parse_manifest(doc, origin=f.name))
        return cls(manifests)

    @classmethod
    def builtin(cls) -> "ToolCatalog":
        root = resources.files("imference_engine.tools") / "manifests"
        manifests = []
        for f in sorted(root.iterdir(), key=lambda p: p.name):
            if f.name.endswith(".json"):
                manifests.append(parse_manifest(json.loads(f.read_text(encoding="utf-8")), origin=f.name))
        return cls(manifests)

    def get(self, tool_id: str, *, include_unavailable: bool = False) -> ToolManifest:
        m = self._tools.get(tool_id)
        if m is None or (not m.available and not include_unavailable):
            raise ToolError(INVALID_INPUT, f"unknown tool {tool_id!r}", field="tool_id")
        return m

    def by_mcp_name(self, name: str) -> ToolManifest:
        for m in self.all():
            if m.mcp_name == name:
                return m
        raise ToolError(INVALID_INPUT, f"unknown tool {name!r}", field="tool_id")

    def all(self, *, include_unavailable: bool = False) -> list:
        return [m for m in self._tools.values() if m.available or include_unavailable]

    def ids(self) -> list:
        return [m.id for m in self.all()]

    def for_queue(self, queue: str) -> list:
        """Tools a cloud worker on ``queue`` serves: right queue AND a verified
        permissive license (the license gate of the plan)."""
        return [m for m in self.all() if m.runtimes.cloud_queue == queue and m.cloud_allowed]

    def export(self, *, public: bool = True) -> dict:
        tools = [m.public_dict() if public else m.to_dict() for m in self.all()]
        for t, m in zip(tools, self.all()):
            t["cloud_available"] = m.cloud_allowed
        return {"version": CATALOG_VERSION, "tools": tools}


_BUILTIN: Optional[ToolCatalog] = None


def builtin() -> ToolCatalog:
    global _BUILTIN
    if _BUILTIN is None:
        _BUILTIN = ToolCatalog.builtin()
    return _BUILTIN


# Module-level conveniences over the built-in catalog.
def get(tool_id: str) -> ToolManifest:
    return builtin().get(tool_id)


def all() -> list:  # noqa: A001 - mirrors ToolCatalog.all
    return builtin().all()


def for_queue(queue: str) -> list:
    return builtin().for_queue(queue)


def export(*, public: bool = True) -> dict:
    return builtin().export(public=public)
