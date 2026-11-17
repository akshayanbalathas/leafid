"""Model zoo for the experiments.

  simple_cnn        5 conv blocks trained from scratch (baseline)
  res_cnn           small residual network trained from scratch
  resnet50          ImageNet-pretrained ResNet-50, new 185-way head (transfer learning)
  resnet50_scratch  same architecture, random init (isolates the effect of pretraining)
"""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

MODEL_NAMES = ("simple_cnn", "res_cnn", "resnet50", "resnet50_scratch")


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """224 -> 112 -> 56 -> 28 -> 14 -> 7, then global average pool."""

    def __init__(self, num_classes: int, widths=(32, 64, 128, 256, 256), dropout: float = 0.3):
        super().__init__()
        chans = (3, *widths)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans, chans[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(widths[-1], num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class ResCNN(nn.Module):
    """Stem (224 -> 56) then 4 residual stages (56 -> 56 -> 28 -> 14 -> 7)."""

    def __init__(self, num_classes: int, widths=(64, 128, 256, 512), dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 7, 2, 3, bias=False), nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
        )
        stages, c_in = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(ResidualBlock(c_in, w, stride), ResidualBlock(w, w)))
            c_in = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(widths[-1], num_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes)
    if name == "res_cnn":
        return ResCNN(num_classes)
    if name in ("resnet50", "resnet50_scratch"):
        weights = ResNet50_Weights.IMAGENET1K_V2 if name == "resnet50" else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048 -> num_classes, trained from scratch
        return model
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    head = model.fc if hasattr(model, "fc") else model.head
    return list(head.parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head."""
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional stage (7x7 feature maps) used for Grad-CAM."""
    if name.startswith("resnet50"):
        return model.layer4[-1]
    if name == "res_cnn":
        return model.stages[-1]
    return model.features[-1]


def save_checkpoint(path: Path, model: nn.Module, name: str, classes: list[str], extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": name, "classes": classes, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, str, list[str]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["model_name"], ckpt["classes"]
