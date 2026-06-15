"""Parallel multi-stream HTTP downloader (saturates CDN bandwidth).

A single GET doesn't fill the link; this fetches N byte-ranges in parallel
(R2/Cloudflare support Range), writes them to a preallocated file via os.pwrite,
verifies the size, and retries. Used by the engine's CDN loader AND by the
worker's start.sh to bootstrap the base tarball.

    python -m imference_engine.wan.download <url> <dest> [threads]
"""
from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (imference-engine wan)"


def download_parallel(url: str, dest: str, threads: int = 8, *, progress: bool = True) -> str:
    """Download ``url`` to ``dest`` in ``threads`` parallel range streams."""
    import concurrent.futures
    import threading
    import time
    import urllib.error
    import urllib.request

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    tmp = dest + ".part"
    label = os.path.basename(dest)

    def _head_total() -> int:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req) as r:
            return int(r.headers.get("Content-Length") or 0)

    last_err = None
    for attempt in range(1, 4):
        try:
            total = _head_total()
            if total <= 0:
                raise IOError("no Content-Length from server")
            got = [0]
            lock = threading.Lock()
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.ftruncate(fd, total)

                def _range(start: int, end: int) -> None:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": _UA, "Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req) as r:
                        off = start
                        while True:
                            buf = r.read(4 * 1024 * 1024)
                            if not buf:
                                break
                            os.pwrite(fd, buf, off)
                            off += len(buf)
                            with lock:
                                got[0] += len(buf)

                n = max(1, threads)
                size = (total + n - 1) // n
                spans = [(i * size, min((i + 1) * size - 1, total - 1))
                         for i in range(n) if i * size < total]
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(spans)) as ex:
                    futs = [ex.submit(_range, s, e) for s, e in spans]
                    while not all(f.done() for f in futs):
                        if progress:
                            d = got[0]
                            print(f"\r  {label}: {d//1024//1024}/{total//1024//1024} MiB "
                                  f"({d*100//total}%) [{len(spans)} streams]   ",
                                  end="", file=sys.stderr, flush=True)
                        time.sleep(0.5)
                    for f in futs:
                        f.result()
                if progress:
                    print("", file=sys.stderr, flush=True)
            finally:
                os.close(fd)
            if os.path.getsize(tmp) != total:
                raise IOError(f"incomplete: {os.path.getsize(tmp)} of {total} bytes")
            os.replace(tmp, dest)
            return dest
        except (urllib.error.URLError, IOError) as e:
            last_err = e
            logger.warning("download attempt %d/3 failed (%s); retrying", attempt, e)
            try:
                os.remove(tmp)
            except OSError:
                pass
            time.sleep(2 * attempt)
    raise RuntimeError(f"parallel download failed for {url}: {last_err}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        sys.exit("usage: python -m imference_engine.wan.download <url> <dest> [threads]")
    url, dest = argv[0], argv[1]
    threads = int(argv[2]) if len(argv) > 2 else int(os.environ.get("WAN_CDN_THREADS", "8"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_parallel(url, dest, threads)


if __name__ == "__main__":
    main()
