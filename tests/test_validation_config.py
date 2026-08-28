"""No-torch checks on the GPU validation harness config + plan.

Ensures validation/base_models.yaml stays in sync with the registered backends
and that every entry is well-formed, so the harness can't silently drift from the
engine set or ship a malformed row.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "validation" / "base_models.yaml"
VALIDATE = ROOT / "validation" / "validate.py"


def _config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_covers_exactly_the_registered_backends():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    backends = set(engine._backends)
    configured = set(_config())
    assert configured == backends, (
        f"base_models.yaml drift: missing={backends - configured}, "
        f"extra={configured - backends}"
    )


def test_every_entry_is_well_formed():
    for name, entry in _config().items():
        assert entry.get("mode") in ("single_file", "repo"), f"{name}: bad mode"
        assert entry.get("repo"), f"{name}: missing repo"
        assert entry.get("prompt"), f"{name}: missing prompt"
        if entry["mode"] == "single_file":
            # needs a filename to download OR a local override
            assert entry.get("filename") or entry.get("weights_local"), \
                f"{name}: single_file needs filename or weights_local"


def test_flux_marked_gated_and_offloaded():
    """FLUX.1-dev is gated + heavy — the config must reflect that."""
    flux = _config()["flux"]
    assert flux["offload"] is True
    assert "FLUX.1-dev" in flux["repo"]


def test_anima_uses_repo_mode():
    anima = _config()["anima"]
    assert anima["mode"] == "repo"  # ModularPipeline.from_pretrained on a repo/dir


def test_dry_run_runs_without_torch():
    """--dry-run must print the plan without importing torch/the backends."""
    out = subprocess.run(
        [sys.executable, str(VALIDATE), "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "Validating 8 engine(s)" in out.stdout
    assert "sdxl" in out.stdout and "anima" in out.stdout


def test_dry_run_rejects_unknown_engine():
    out = subprocess.run(
        [sys.executable, str(VALIDATE), "--engines", "nope", "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 2
    assert "unknown engine" in out.stderr
