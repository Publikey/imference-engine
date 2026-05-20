"""Unit tests for BatchSizer — pure logic, no GPU required."""
from imference_engine.managers.batch import BatchSizer


GB = 1024 ** 3


def test_get_max_batch_returns_zero_before_profiling():
    b = BatchSizer()
    assert b.get_max_batch_size("sdxl") == 0


def test_profile_computes_max_batch_from_vram_headroom():
    b = BatchSizer()
    max_batch = b.profile_and_compute(
        "sdxl",
        False,
        total_vram=24 * GB,
        baseline_vram=4 * GB,
        peak_vram=6 * GB,
    )
    assert 1 <= max_batch <= BatchSizer.MAX_BATCH_SIZE
    # cached for subsequent calls
    assert b.get_max_batch_size("sdxl") == max_batch


def test_profile_floors_at_one_when_peak_equals_baseline():
    b = BatchSizer()
    max_batch = b.profile_and_compute(
        "sdxl",
        False,
        total_vram=24 * GB,
        baseline_vram=4 * GB,
        peak_vram=4 * GB,
    )
    assert max_batch == 1


def test_profile_caps_at_max_batch_size_env():
    """Even with infinite VRAM headroom we shouldn't exceed MAX_BATCH_SIZE."""
    b = BatchSizer()
    max_batch = b.profile_and_compute(
        "sdxl",
        False,
        total_vram=10_000 * GB,
        baseline_vram=0,
        peak_vram=1,
    )
    assert max_batch == BatchSizer.MAX_BATCH_SIZE


def test_oom_halves_the_cached_batch():
    b = BatchSizer()
    b.profile_and_compute(
        "sdxl",
        False,
        total_vram=24 * GB,
        baseline_vram=4 * GB,
        peak_vram=5 * GB,
    )
    initial = b.get_max_batch_size("sdxl")
    if initial <= 1:
        return  # nothing to halve, but logic still consistent
    new_max = b.report_oom("sdxl", False, initial)
    assert new_max == max(1, initial // 2)
    assert b.get_max_batch_size("sdxl") == new_max


def test_oom_floors_at_one():
    b = BatchSizer()
    new_max = b.report_oom("sdxl", False, 1)
    assert new_max == 1


def test_engine_keys_are_isolated():
    b = BatchSizer()
    b.profile_and_compute(
        "sdxl",
        False,
        total_vram=24 * GB,
        baseline_vram=4 * GB,
        peak_vram=5 * GB,
    )
    # zimage must NOT inherit sdxl's cached value
    assert b.get_max_batch_size("zimage") == 0


def test_img2img_separate_from_t2i():
    b = BatchSizer()
    b.profile_and_compute(
        "sdxl",
        False,
        total_vram=24 * GB,
        baseline_vram=4 * GB,
        peak_vram=5 * GB,
    )
    assert b.get_max_batch_size("sdxl", is_img2img=True) == 0
