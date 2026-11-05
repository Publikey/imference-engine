"""download_parallel must never accept a range body the server cut short.

http.client's ``read(amt)`` returns b"" (no IncompleteRead) when the connection
closes before Content-Length is satisfied. The .part file is preallocated with
ftruncate, so before the fix a short chunk left a zero-filled hole that still
passed the size check. In prod that zeroed 64 MB of the fp16-fix SDXL VAE:
every image came out the same flat gray, reported as ok.
"""
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from imference_engine.runtime.download import download_parallel

MB = 1024 * 1024


class _TruncatingHandler(BaseHTTPRequestHandler):
    """Range-aware server that cuts the FIRST response for a chosen range short:
    it announces the full Content-Length, writes half, then closes."""
    payload = b""
    cut_start = 0
    cut_done = None  # threading.Event

    def log_message(self, *a):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        start_s, _, end_s = self.headers["Range"][len("bytes="):].partition("-")
        start, end = int(start_s), min(int(end_s), len(self.payload) - 1)
        body = self.payload[start:end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if start == self.cut_start and not self.cut_done.is_set():
            self.cut_done.set()
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


def test_short_range_is_retried_not_zero_filled(tmp_path):
    payload = os.urandom(3 * MB + 123)
    _TruncatingHandler.payload = payload
    _TruncatingHandler.cut_start = 1 * MB  # second chunk
    _TruncatingHandler.cut_done = threading.Event()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _TruncatingHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        dest = tmp_path / "vae.safetensors"
        url = f"http://127.0.0.1:{srv.server_address[1]}/vae.safetensors"
        download_parallel(url, str(dest), threads=2, progress=False, chunk_mb=1)
    finally:
        srv.shutdown()

    assert _TruncatingHandler.cut_done.is_set(), "the truncation never triggered"
    assert dest.read_bytes() == payload
    assert not (tmp_path / "vae.safetensors.part").exists()
