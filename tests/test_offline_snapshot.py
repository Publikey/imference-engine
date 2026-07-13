"""HuggingFace snapshot path of runtime.offline.local_repo_dir — the completion
marker that makes a killed download (sentinel present, shards missing) self-heal
instead of poisoning the cache.

No torch / no network: snapshot_download is monkeypatched.
"""
from __future__ import annotations

import os
import sys
import types

from imference_engine.runtime.offline import local_repo_dir

REPO = "some/repo"


def _patch_snapshot(monkeypatch, calls):
    """Inject a fake ``huggingface_hub`` so ``from huggingface_hub import
    snapshot_download`` resolves without the real package installed."""
    def fake_snap(repo, allow_patterns=None, local_dir=None):
        calls.append(repo)
        os.makedirs(local_dir, exist_ok=True)
        # simulate a full pull: sentinel + a shard
        open(os.path.join(local_dir, "model_index.json"), "w").close()
        open(os.path.join(local_dir, "big.safetensors"), "w").close()

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = fake_snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)


def test_fresh_pull_writes_marker_then_fast_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls: list = []
    _patch_snapshot(monkeypatch, calls)
    cache = tmp_path / "tree"

    d = local_repo_dir(REPO, ["*"], str(cache), namespace="image")
    assert calls == [REPO]
    assert os.path.isfile(os.path.join(d, ".hf_complete"))

    # Second call: marker present -> no re-download.
    local_repo_dir(REPO, ["*"], str(cache), namespace="image")
    assert calls == [REPO]  # unchanged


def test_truncated_tree_is_repaired_online(tmp_path, monkeypatch):
    """Sentinel present but NO marker (killed pull) -> online re-run repairs it,
    instead of short-circuiting to a broken tree (the FLUX disk-full bug)."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls: list = []
    _patch_snapshot(monkeypatch, calls)
    cache = tmp_path / "tree"
    d = cache / "some" / "repo"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text("{}")  # sentinel only, no shard, no marker

    local_repo_dir(REPO, ["*"], str(cache), namespace="image")
    assert calls == [REPO]  # re-ran despite the sentinel
    assert os.path.isfile(os.path.join(d, "big.safetensors"))  # missing shard filled
    assert os.path.isfile(os.path.join(d, ".hf_complete"))


def test_offline_trusts_sentinel_without_download(tmp_path, monkeypatch):
    """HF_HUB_OFFLINE=1 + sentinel present -> trust the pre-shipped/base-tarball
    tree; never call snapshot_download (can't, and mustn't regress that path)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls: list = []
    _patch_snapshot(monkeypatch, calls)
    cache = tmp_path / "tree"
    d = cache / "some" / "repo"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text("{}")

    out = local_repo_dir(REPO, ["*"], str(cache), namespace="image")
    assert calls == []  # trusted, not downloaded
    assert out == str(d)


def test_marker_beats_offline_check(tmp_path, monkeypatch):
    """A completed tree (marker) fast-skips regardless of HF_HUB_OFFLINE."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls: list = []
    _patch_snapshot(monkeypatch, calls)
    cache = tmp_path / "tree"
    d = cache / "some" / "repo"
    d.mkdir(parents=True)
    (d / ".hf_complete").write_text("")

    local_repo_dir(REPO, ["*"], str(cache), namespace="image")
    assert calls == []
