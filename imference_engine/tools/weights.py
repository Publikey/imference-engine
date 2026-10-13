"""Tool weights: local cache + CDN mirror + sha256 check.

Small ONNX weights (2–200 MB) live flat under ``<cache>/<path>``. A missing file
is pulled from ``<cdn>/<path>`` (``TOOLS_MODEL_CDN``, falling back to
``IMAGE_MODEL_CDN`` + ``/tools``) with the engine's parallel downloader —
plain HTTP, so it works under ``HF_HUB_OFFLINE=1``; HuggingFace is never
touched. A manifest ``sha256`` is verified once after download and a
``.sha256`` stamp next to the file skips re-hashing on later loads.

Env:
    TOOLS_MODEL_CACHE   cache root (default: $HF_HOME/tools or ~/.cache/imference/tools)
    TOOLS_MODEL_CDN     mirror root, e.g. https://gen-models.example/tools
    IMAGE_MODEL_CDN     fallback mirror root (``/tools`` appended)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from imference_engine.runtime.env import env_str
from imference_engine.tools.errors import WEIGHTS_MISSING, ToolError
from imference_engine.tools.schema import WeightSpec

logger = logging.getLogger(__name__)


def cache_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = env_str("TOOLS_MODEL_CACHE")
    if env:
        return Path(env)
    hf = env_str("HF_HOME")
    if hf:
        return Path(hf) / "tools"
    return Path.home() / ".cache" / "imference" / "tools"


def cdn_root(override: Optional[str] = None) -> Optional[str]:
    if override:
        return override.rstrip("/")
    env = env_str("TOOLS_MODEL_CDN")
    if env:
        return env.rstrip("/")
    img = env_str("IMAGE_MODEL_CDN")
    if img:
        return img.rstrip("/") + "/tools"
    return None


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _verified(path: Path, sha: str) -> bool:
    stamp = path.with_name(path.name + ".sha256")
    if stamp.is_file():
        try:
            if stamp.read_text().strip() == sha and stamp.stat().st_mtime >= path.stat().st_mtime:
                return True
        except OSError:
            pass
    ok = sha256_file(path) == sha
    if ok:
        try:
            stamp.write_text(sha)
        except OSError:
            pass
    return ok


def local_path(w: WeightSpec, *, cache_dir: Optional[str] = None) -> Path:
    return cache_root(cache_dir) / w.path


def is_present(w: WeightSpec, *, cache_dir: Optional[str] = None) -> bool:
    return local_path(w, cache_dir=cache_dir).is_file()


def ensure(
    weights: list,
    *,
    cache_dir: Optional[str] = None,
    cdn: Optional[str] = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
    downloader: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """Make every weight available locally; return ``{name: Path}``.

    Raises ``ToolError(weights_missing)`` when a file is absent and no mirror is
    configured, the download fails, or the checksum does not match.
    """
    out: dict = {}
    todo = [w for w in weights if not local_path(w, cache_dir=cache_dir).is_file()]
    total_mb = sum(max(w.size_mb, 1) for w in todo) or 1
    done_mb = 0.0
    base = cdn_root(cdn)
    for w in weights:
        dest = local_path(w, cache_dir=cache_dir)
        if not dest.is_file():
            if not base:
                raise ToolError(WEIGHTS_MISSING, f"weight {w.path} not in {cache_root(cache_dir)} and no "
                                "TOOLS_MODEL_CDN / IMAGE_MODEL_CDN mirror is configured")
            url = f"{base}/{w.path}"
            if on_progress:
                on_progress("downloading_weights", done_mb / total_mb)
            logger.info("Downloading tool weight %s -> %s", url, dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if downloader is not None:
                    downloader(url, str(dest))
                else:
                    from imference_engine.runtime.download import download_parallel
                    download_parallel(url, str(dest), threads=4, progress=False, chunk_mb=16)
            except ToolError:
                raise
            except Exception as e:  # network / HTTP / disk
                for p in (dest, Path(str(dest) + ".part")):
                    try:
                        p.unlink()
                    except OSError:
                        pass
                raise ToolError(WEIGHTS_MISSING, f"could not download {url}: {e}") from e
            done_mb += max(w.size_mb, 1)
        if w.sha256:
            if not _verified(dest, w.sha256):
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise ToolError(WEIGHTS_MISSING, f"checksum mismatch for {w.path} (file removed, retry)")
        else:
            logger.debug("No sha256 in manifest for %s; skipping verification", w.path)
        out[w.name] = dest
    if todo and on_progress:
        on_progress("downloading_weights", 1.0)
    return out
