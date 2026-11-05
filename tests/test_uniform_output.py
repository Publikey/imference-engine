"""Engine._is_uniform: a flat single-color image (corrupt VAE, broken decode)
is failed per image instead of being returned as a successful result."""
import random

from PIL import Image

from imference_engine.engine import Engine


def test_flat_gray_is_uniform():
    # the exact prod symptom: RGB (147, 140, 134) everywhere
    assert Engine._is_uniform(Image.new("RGB", (832, 1216), (147, 140, 134)))


def test_real_content_is_not_uniform():
    rnd = random.Random(0)
    img = Image.new("RGB", (256, 256))
    img.putdata([(rnd.randrange(256),) * 3 for _ in range(256 * 256)])
    assert not Engine._is_uniform(img)


def test_soft_gradient_is_not_uniform():
    # a near-plain background still has a few levels of variation
    img = Image.linear_gradient("L").resize((512, 512)).point(lambda v: 120 + v // 32)
    assert not Engine._is_uniform(img.convert("RGB"))
