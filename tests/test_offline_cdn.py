"""CDN-manifest population path of runtime.offline.local_repo_dir.

Serves a fake flat mirror over a local Range-aware HTTP server (no torch, no
HuggingFace, no real CDN) and checks the manifest-driven pull lands files in the
flat tree and is idempotent. This is what lets the image side drop the boot
tarball and go fully offline via IMAGE_MODEL_CDN under HF_HUB_OFFLINE=1.

The handler honours HEAD (Content-Length) + Range (206) like R2/Cloudflare, so
download_parallel takes its normal multi-stream path instead of the slow
200->single-stream fallback.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from imference_engine.runtime.offline import local_repo_dir, write_manifest

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
FILES = {
    "model_index.json": '{"_class_name": "StableDiffusionXLPipeline"}',
    "scheduler/scheduler_config.json": '{"_class_name": "EulerScheduler"}',
    "tokenizer/vocab.json": '{"a": 1}',
    "tokenizer/merges.txt": "a b\n",
}


class _RangeHandler(BaseHTTPRequestHandler):
    root = ""

    def log_message(self, *a):  # silence
        pass

    def _path(self):
        return os.path.join(self.root, self.path.lstrip("/"))

    def do_HEAD(self):
        p = self._path()
        if not os.path.isfile(p):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(p)))
        self.end_headers()

    def do_GET(self):
        p = self._path()
        if not os.path.isfile(p):
            self.send_error(404)
            return
        data = open(p, "rb").read()
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start_s, _, end_s = rng[len("bytes="):].partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(data) - 1
            end = min(end, len(data) - 1)
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def test_write_manifest_roundtrip(tmp_path):
    """The prefetch-side writer emits exactly what the CDN reader expects."""
    repo_dir = tmp_path / "repo"
    for rel, content in FILES.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    listed = write_manifest(str(repo_dir))
    assert listed == sorted(FILES)  # sorted, forward-slash, excludes itself
    on_disk = json.loads((repo_dir / ".manifest.json").read_text())
    assert on_disk == sorted(FILES)
    assert ".manifest.json" not in on_disk


@pytest.fixture
def cdn_server(tmp_path):
    """A local HTTP mirror serving <repo>/.manifest.json + the repo files.

    The manifest is produced by the real write_manifest (prefetch path), so this
    fixture also exercises the writer<->reader contract end to end.
    """
    root = tmp_path / "cdn"
    repo_dir = root / REPO
    repo_dir.mkdir(parents=True)
    for rel, content in FILES.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    write_manifest(str(repo_dir))

    _RangeHandler.root = str(root)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        thread.join()


def test_local_repo_dir_cdn_populates_flat_tree(cdn_server, tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_CDN_THREADS", "2")
    cache = tmp_path / "image-tree"

    d = local_repo_dir(
        REPO, ["*"], str(cache), namespace="image", cdn_base=cdn_server,
    )

    assert d == os.path.join(str(cache), REPO)
    for rel, content in FILES.items():
        landed = os.path.join(d, *rel.split("/"))
        assert os.path.isfile(landed), f"{rel} not pulled from CDN"
        assert open(landed).read() == content


def test_local_repo_dir_cdn_idempotent_offline(cdn_server, tmp_path):
    """Second call returns instantly via the sentinel — no server needed."""
    cache = tmp_path / "image-tree"
    local_repo_dir(REPO, ["*"], str(cache), namespace="image", cdn_base=cdn_server)

    # A bogus CDN URL would fail if hit; the sentinel must short-circuit it.
    d = local_repo_dir(
        REPO, ["*"], str(cache), namespace="image",
        cdn_base="http://127.0.0.1:1/never",
    )
    assert os.path.isfile(os.path.join(d, "model_index.json"))
