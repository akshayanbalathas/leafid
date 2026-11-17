"""CLIP zero-shot "is this a leaf?" gate (Radford et al., ICML 2021).

The species classifier always outputs one of 185 species, even for a photo of a
cat. This gate scores the image against leaf vs. non-leaf text prompts and lets
the app refuse inputs that are not leaves. No training: pretrained CLIP only.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

LEAF_PROMPTS = [
    "a photo of a leaf",
    "a close-up photo of a tree leaf",
    "a photo of a pressed leaf on a white background",
]
OTHER_PROMPTS = [
    "a photo of a person",
    "a photo of an animal",
    "a photo of a building",
    "a photo of a car",
    "a photo of food",
    "a photo of text or a document",
    "a photo of a flower",
    "a blurry photo",
]


class LeafGate:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", threshold: float = 0.5,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> tuple[bool, float]:
        prompts = LEAF_PROMPTS + OTHER_PROMPTS
        inputs = self.processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        probs = self.model(**inputs).logits_per_image.softmax(dim=-1)[0]
        p_leaf = float(probs[: len(LEAF_PROMPTS)].sum())
        return p_leaf >= self.threshold, p_leaf
