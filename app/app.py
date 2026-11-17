"""Gradio front end.

  python app/app.py --checkpoint checkpoints/resnet50_full.pt --share
  python app/app.py --checkpoint checkpoints/resnet50_full.pt --no-llm --no-gate   # fast, CPU-friendly
"""
import argparse
from pathlib import Path

import gradio as gr

from leafid.predict import Predictor
from leafid.species_card import pretty


def build_ui(predictor: Predictor) -> gr.Blocks:
    def run(image):
        if image is None:
            return {}, None, "Upload a leaf photo to begin."
        pred = predictor(image)
        labels = {pretty(s): p for s, p in pred.topk}
        return labels, pred.heatmap, pred.card

    with gr.Blocks(title="LeafID") as demo:
        gr.Markdown("# 🍁 LeafID\nIdentify a tree species from a photo of its leaf.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Leaf photo", sources=["upload", "webcam"])
                button = gr.Button("Identify", variant="primary")
            with gr.Column():
                labels = gr.Label(num_top_classes=3, label="Top species")
                heatmap = gr.Image(label="Grad-CAM: where the model looked")
        card = gr.Markdown()
        button.click(run, inputs=image, outputs=[labels, heatmap, card])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--share", action="store_true", help="public link (needed on Colab)")
    args = ap.parse_args()

    predictor = Predictor(args.checkpoint, use_gate=not args.no_gate, use_llm=not args.no_llm)
    build_ui(predictor).launch(share=args.share)


if __name__ == "__main__":
    main()
