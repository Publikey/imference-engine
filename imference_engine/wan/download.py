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


def _download_single(url: str, dest: str, total: int, *, progress: bool = True) -> str:
    """Sequential single-stream download — for servers that ignore Range."""
    import time
    import urllib.request

    tmp = dest + ".part"
    label = os.path.basename(dest)
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    got = 0
    last = 0.0
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            buf = r.read(2 * 1024 * 1024)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if progress and time.time() - last > 0.5:
                last = time.time()
                print(f"\r  {label}: {got//1024//1024}/{total//1024//1024} MiB "
                      f"({got*100//total}%) [1 stream]   ",
                      end="", file=sys.stderr, flush=True)
    if progress:
        print("", file=sys.stderr, flush=True)
    if os.path.getsize(tmp) != total:
        raise IOError(f"incomplete: {os.path.getsize(tmp)} of {total} bytes")
    os.replace(tmp, dest)
    return dest


def download_parallel(url: str, dest: str, threads: int = 8, *,
                      progress: bool = True, chunk_mb: int = 64) -> str:
    """Download ``url`` to ``dest`` with ``threads`` workers pulling fixed-size
    chunks from a queue (work-stealing) — every worker stays busy to the end, so
    there's no slow single-stream tail at the finish."""
    import queue
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

    def _supports_ranges() -> bool:
        # A server that honours Range answers a 1-byte request with 206 +
        # Content-Range. If it returns 200 (whole body), parallel ranges would
        # each re-download the entire file -> overflow/corruption (the 167% bug).
        req = urllib.request.Request(
            url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.getcode() == 206 or bool(r.headers.get("Content-Range"))
        except urllib.error.HTTPError as e:
            return e.code == 206

    last_err = None
    for attempt in range(1, 4):
        try:
            total = _head_total()
            if total <= 0:
                raise IOError("no Content-Length from server")
            if not _supports_ranges():
                logger.info("  %s: server ignores Range -> single-stream", label)
                return _download_single(url, dest, total, progress=progress)
            chunk = max(1, chunk_mb) * 1024 * 1024
            jobs: "queue.Queue" = queue.Queue()
            for start in range(0, total, chunk):
                jobs.put((start, min(start + chunk - 1, total - 1)))

            got = [0]
            lock = threading.Lock()
            errors: list = []
            stop = threading.Event()
            no_range = threading.Event()  # a chunk got 200 (server ignored Range)
            fd = os.open(tmp, os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.ftruncate(fd, total)

                def _worker() -> None:
                    while not stop.is_set():
                        try:
                            start, end = jobs.get_nowait()
                        except queue.Empty:
                            return
                        try:
                            req = urllib.request.Request(
                                url, headers={"User-Agent": _UA,
                                              "Range": f"bytes={start}-{end}"})
                            with urllib.request.urlopen(req) as r:
                                # 200 means the server ignored Range and is
                                # streaming the WHOLE file from byte 0 — writing
                                # it at `start` would be silent corruption. Bail
                                # and let the caller retry as a single stream.
                                if r.getcode() != 206:
                                    no_range.set()
                                    stop.set()
                                    return
                                off = start
                                while True:
                                    buf = r.read(2 * 1024 * 1024)
                                    if not buf:
                                        break
                                    if off + len(buf) - 1 > end:
                                        buf = buf[:end - off + 1]
                                    os.pwrite(fd, buf, off)
                                    off += len(buf)
                                    with lock:
                                        got[0] += len(buf)
                        except Exception as ex:  # noqa: BLE001
                            errors.append(ex)
                            stop.set()
                            return

                n = max(1, min(threads, jobs.qsize()))
                workers = [threading.Thread(target=_worker, daemon=True) for _ in range(n)]
                for w in workers:
                    w.start()
                while any(w.is_alive() for w in workers):
                    if progress:
                        d = got[0]
                        print(f"\r  {label}: {d//1024//1024}/{total//1024//1024} MiB "
                              f"({d*100//total}%) [{n} streams]   ",
                              end="", file=sys.stderr, flush=True)
                    time.sleep(0.5)
                for w in workers:
                    w.join()
                if progress:
                    print("", file=sys.stderr, flush=True)
                if errors:
                    raise errors[0]
            finally:
                os.close(fd)
            if no_range.is_set():
                logger.info("  %s: server ignored Range mid-flight -> single-stream", label)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return _download_single(url, dest, total, progress=progress)
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
