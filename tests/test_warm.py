"""Warm (deploy-time base-component prefetch). No torch / no network.

warm() lets a worker pull shared base-components at deploy so a fresh pod is
'ready' with the base on disk; the checkpoint stays lazy. Best-effort: a failed
prefetch must NOT crash setup().
"""
import imference_engine.wan.loader as wan_loader
from imference_engine import Engine, RuntimeConfig
from imference_engine.wan import WanEngine, WanRuntimeConfig
from imference_engine.zimage import ZImageBackend


def test_engine_warm_dispatches_and_dedups(monkeypatch):
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    calls = []
    for name, backend in engine._backends.items():
        monkeypatch.setattr(backend, "prefetch_base",
                            lambda base_model=None, _n=name: calls.append((_n, base_model)))

    engine.warm([
        ("sdxl", None), ("sdxl", None),                  # duplicate -> once
        ("zimage", "Tongyi-MAI/Z-Image-Turbo"),
        ("zimage", "Tongyi-MAI/Z-Image"),
        ("nope", None),                                  # unknown backend -> skipped
    ])

    assert calls.count(("sdxl", None)) == 1
    assert ("zimage", "Tongyi-MAI/Z-Image-Turbo") in calls
    assert ("zimage", "Tongyi-MAI/Z-Image") in calls
    assert all(c[0] != "nope" for c in calls)


def test_engine_warm_is_best_effort(monkeypatch):
    """A prefetch failure logs + is skipped — warm() never raises (safe in setup())."""
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()

    def boom(base_model=None):
        raise RuntimeError("cdn down")

    for backend in engine._backends.values():
        monkeypatch.setattr(backend, "prefetch_base", boom)

    engine.warm([("sdxl", None), ("zimage", "x")])  # must not raise


def test_zimage_prefetch_base_noop_without_base_model():
    """No base_model -> no repo to pull, returns without touching the network."""
    ZImageBackend().prefetch_base(None)
    ZImageBackend().prefetch_base()


def test_wan_warm_pulls_shared_and_variant_bases(monkeypatch):
    pulled = []
    monkeypatch.setattr(
        wan_loader, "_local_repo_dir",
        lambda repo, patterns, cache_dir, cdn_base=None: pulled.append(repo))

    eng = WanEngine(runtime=WanRuntimeConfig(
        device="cpu", model_cdn="https://cdn.example/wan22")).load()
    eng.warm()

    from imference_engine.video import WanBackend
    shared = WanBackend.SHARED_BASE_REPO
    assert shared in pulled
    assert pulled.count(shared) == 1            # shared base pulled exactly once
    # built-in variants include an i2v base distinct from the (t2v) shared base
    assert len(set(pulled)) >= 2
