"""ResNet18 transfer learning (torchvision)."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MODES = ("frozen", "finetune")


def build_resnet18(num_classes: int, mode: str = "finetune", pretrained: bool = True) -> nn.Module:
    """mode='frozen': backbone frozen (BN stats too, see set_train_mode), only new fc trained.
    mode='finetune': all layers trainable.
    pretrained=True loads torchvision's ImageNet-1k weights and RAISES if they cannot be
    obtained - it never silently falls back to random initialisation."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as e:  # network / cache failure
        raise RuntimeError(
            "Could not load ImageNet pretrained ResNet18 weights (download failed or no local "
            "cache). Fix network access, or pass --no-pretrained to train from random init "
            "(this is recorded in the metrics JSON and is NOT transfer learning)."
        ) from e
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if mode == "frozen":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("fc.")
    return model


def set_train_mode(model: nn.Module, mode: str) -> None:
    """Call instead of model.train(): keeps frozen backbone BatchNorm in eval mode so the
    pretrained running statistics are not overwritten."""
    model.train()
    if mode == "frozen":
        for name, m in model.named_children():
            if name != "fc":
                m.eval()
