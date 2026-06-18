"""Back-compat shim — the parallel downloader now lives in ``runtime/download.py``.

Kept so the existing CLI invocation ``python -m imference_engine.wan.download
<url> <dest> [threads]`` (referenced by worker ``start.sh`` bootstrap scripts)
and ``from imference_engine.wan.download import download_parallel`` keep
resolving unchanged.
"""
from __future__ import annotations

from imference_engine.runtime.download import (  # noqa: F401
    _download_single,
    download_parallel,
    main,
)

if __name__ == "__main__":
    main()
