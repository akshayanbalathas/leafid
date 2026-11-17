"""End-to-end inference: photo -> gate -> classifier -> Grad-CAM -> species card.

This is the single interface the app (and anything else) talks to.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms, display_transform
from .gradcam import GradCAM, overlay
from .models import gradcam_layer, load_checkpoint
from .species_card import CardWriter, pretty


@dataclass
class Prediction:
    is_leaf: bool
    p_leaf: float
    topk: list[tuple[str, float]]
    heatmap: Image.Image | None
    card: str


class Predictor:
    def __init__(self, checkpoint: Path, use_gate: bool = True, use_llm: bool = True,
                 min_confidence: float = 0.30, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.model_name, self.classes = load_checkpoint(checkpoint, self.device)
        self.cam = GradCAM(self.model, gradcam_layer(self.model, self.model_name))
        self.to_tensor = build_transforms(train=False)
        self.to_display = display_transform()
        self.min_confidence = min_confidence
        self.gate = None
        if use_gate:
            from .leaf_gate import LeafGate
            self.gate = LeafGate(device=self.device)
        self.cards = CardWriter(use_llm=use_llm)

    def __call__(self, image: Image.Image, k: int = 3) -> Prediction:
        image = image.convert("RGB")
        is_leaf, p_leaf = self.gate(image) if self.gate else (True, 1.0)
        if not is_leaf:
            return Prediction(False, p_leaf, [], None,
                              "### That doesn't look like a leaf 🍂\nTry a close-up of a single leaf.")

        x = self.to_tensor(image).unsqueeze(0).to(self.device)   # [1, 3, 224, 224]
        cam, logits = self.cam(x)
        probs = logits.softmax(dim=1)[0]                          # [num_classes]
        p, idx = probs.topk(k)
        topk = [(self.classes[i], float(q)) for q, i in zip(p.tolist(), idx.tolist())]
        heatmap = overlay(self.to_display(image), cam)

        if topk[0][1] < self.min_confidence:
            card = ("### Not confident 🤔\nBest guesses: "
                    + ", ".join(f"*{pretty(s)}* ({q:.0%})" for s, q in topk)
                    + "\n\nTry a sharper photo of one leaf against a plain background.")
        else:
            card = self.cards.write(topk)
        return Prediction(True, p_leaf, topk, heatmap, card)
