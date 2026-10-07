import torch
from PIL import Image
from torchvision import transforms as T

from src.transforms import denormalize, get_eval_transform, get_train_transform


def _img():
    return Image.new("RGB", (50, 30), (200, 20, 20))


def test_eval_transform_shape_and_determinism():
    tf = get_eval_transform(64)
    a, b = tf(_img()), tf(_img())
    assert a.shape == (3, 64, 64)
    assert torch.equal(a, b)


def test_train_transform_shape():
    assert get_train_transform(64)(_img()).shape == (3, 64, 64)


def test_no_horizontal_flip_or_hue_shift_in_training():
    tf = get_train_transform(64)
    assert not any(isinstance(t, (T.RandomHorizontalFlip, T.RandomVerticalFlip)) for t in tf.transforms)
    jitter = [t for t in tf.transforms if isinstance(t, T.ColorJitter)][0]
    assert jitter.hue is None


def test_eval_transform_has_no_random_ops():
    names = [type(t).__name__ for t in get_eval_transform(64).transforms]
    assert names == ["Resize", "ToTensor", "Normalize"]


def test_grayscale_and_rgba_become_rgb_tensors():
    tf = get_eval_transform(32)
    assert tf(Image.new("L", (10, 10)).convert("RGB")).shape == (3, 32, 32)
    assert tf(Image.new("RGBA", (10, 10)).convert("RGB")).shape == (3, 32, 32)


def test_denormalize_roundtrip():
    x = get_eval_transform(16)(_img())
    d = denormalize(x)
    assert d.min() >= 0 and d.max() <= 1
    assert abs(d[0].mean().item() - 200 / 255) < 0.02
