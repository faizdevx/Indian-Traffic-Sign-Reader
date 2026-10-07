import numpy as np
import pytest
import torch

from src.metrics import (classification_metrics, expected_calibration_error, fit_temperature,
                         softmax_np, topk_accuracy)
from src.model_cnn import SmallCNN
from src.model_resnet import build_resnet18, set_train_mode
from src.utils import count_parameters


@pytest.mark.parametrize("n_classes", [3, 17])
def test_cnn_forward_output_dim(n_classes):
    out = SmallCNN(n_classes)(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, n_classes)


def test_cnn_accepts_other_input_sizes():
    assert SmallCNN(5).eval()(torch.randn(1, 3, 96, 96)).shape == (1, 5)


def test_resnet_forward_output_dim():
    m = build_resnet18(7, mode="finetune", pretrained=False).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 7)


def test_resnet_frozen_only_fc_trainable_and_bn_stays_eval():
    m = build_resnet18(7, mode="frozen", pretrained=False)
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
    set_train_mode(m, "frozen")
    assert not m.bn1.training and m.fc.training
    p = count_parameters(m)
    assert p["trainable"] == 512 * 7 + 7 < p["total"]


def test_resnet_invalid_mode():
    with pytest.raises(ValueError):
        build_resnet18(3, mode="nope", pretrained=False)


def test_one_gradient_step_reduces_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    m = SmallCNN(3)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 3, (8,))
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    crit = torch.nn.CrossEntropyLoss()
    l0 = crit(m(x), y).item()
    for _ in range(15):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    assert crit(m(x), y).item() < l0


def test_topk_accuracy_known_values():
    probs = np.array([[.6, .3, .1], [.1, .3, .6], [.2, .5, .3], [.5, .4, .1]])
    y = np.array([0, 1, 2, 2])
    assert topk_accuracy(probs, y, 1) == 0.25
    assert topk_accuracy(probs, y, 2) == 0.75
    assert topk_accuracy(probs, y, 3) == 1.0
    assert topk_accuracy(probs, y, 99) == 1.0


def test_classification_metrics_hand_computed():
    # preds: [0,0,1,1], truth: [0,1,1,1]  -> acc .75; class0 P=.5 R=1; class1 P=1 R=2/3
    probs = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
    m = classification_metrics(probs, np.array([0, 1, 1, 1]), ["a", "b"])
    assert m["accuracy"] == 0.75
    assert m["macro_precision"] == pytest.approx(0.75)
    assert m["macro_recall"] == pytest.approx((1 + 2 / 3) / 2)
    f0, f1 = 2 * .5 * 1 / 1.5, 2 * 1 * (2 / 3) / (1 + 2 / 3)
    assert m["macro_f1"] == pytest.approx((f0 + f1) / 2)
    assert m["weighted_f1"] == pytest.approx((1 * f0 + 3 * f1) / 4)


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.array([[1.0, 0.0]] * 4)
    assert expected_calibration_error(probs, np.array([0, 0, 0, 0]))["ece"] == pytest.approx(0)
    assert expected_calibration_error(probs, np.array([1, 1, 1, 1]))["ece"] == pytest.approx(1)


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    logits = rng.normal(0, 1, (400, 4)) * 3
    logits[np.arange(400), y] += 4 * (rng.random(400) < 0.6)  # right ~60% of the time but very peaked
    t = fit_temperature(logits, y)
    nll = lambda T: -np.log(softmax_np(logits, T)[np.arange(400), y]).mean()
    assert t > 1 and nll(t) < nll(1.0)
