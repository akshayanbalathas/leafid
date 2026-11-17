"""Grad-CAM (Selvaraju et al., ICCV 2017).

Weights each feature map of a conv layer by the average gradient of the target
class score with respect to that map, sums them, applies ReLU and upsamples:
the result shows which image regions pushed the model toward its prediction.
"""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        layer.register_forward_hook(self._save)

    def _save(self, _module, _inp, out):
        self.activations = out
        out.register_hook(lambda g: setattr(self, "gradients", g))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """x: [1, 3, H, W]. Returns (cam [H, W] in [0, 1], logits [1, C])."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # [1, K, 1, 1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # [1, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [H, W] cam onto an RGB image of the same size."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype(np.uint8)
    heat = Image.fromarray(heat).resize(image.size)
    return Image.blend(image.convert("RGB"), heat, alpha)
