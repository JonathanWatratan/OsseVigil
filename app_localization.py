#!/usr/bin/env python3
"""Focused Gradio MVP for BTXRD lesion-localization research."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DISCLAIMER = (
    "Research demonstration only — not a medical device. Bounding boxes are model outputs trained to reproduce "
    "BTXRD annotations; they do not diagnose cancer or establish benign versus malignant disease. A negative output "
    "does not mean cancer-free. Clinical assessment may require specialist review, pathology, and additional imaging."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/btxrd_yolov8n_best.pt"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


class DetectorRuntime:
    def __init__(self, checkpoint: Path, confidence: float, device: str):
        from ultralytics import YOLO
        self.model = YOLO(str(checkpoint))
        self.confidence = confidence
        self.device = device or None

    def predict(self, image: Image.Image):
        original = image.convert("RGB")
        result = self.model.predict(np.asarray(original), conf=self.confidence, device=self.device, verbose=False)[0]
        annotated = original.copy(); draw = ImageDraw.Draw(annotated)
        detections = []
        if result.boxes is not None:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(float, box)
                draw.rectangle((x1, y1, x2, y2), outline=(255, 167, 38), width=max(3, original.width // 250))
                label = f"Potential lesion {confidence:.2f}"
                draw.rectangle((x1, max(0, y1 - 24), min(original.width, x1 + 190), y1), fill=(14, 37, 63))
                draw.text((x1 + 4, max(1, y1 - 21)), label, fill="white")
                detections.append({"model_score": round(float(confidence), 4), "x1": round(x1), "y1": round(y1), "x2": round(x2), "y2": round(y2)})
        if detections:
            flag = "### Potential lesion — requires professional review\nA model box crossed the declared display threshold. This is not a diagnosis or malignancy assessment."
        else:
            flag = "### No model box above the display threshold\nThis does **not** rule out a lesion or cancer. Professional review may still be necessary."
        return original, annotated, detections, flag


def build_app(runtime: DetectorRuntime, confidence: float):
    import gradio as gr
    with gr.Blocks(title="OsteoInsight — Bone Lesion Localization") as app:
        gr.Markdown("# OsteoInsight\n### Focused Bone Tumor Localization Research MVP")
        gr.Markdown(f"> **{DISCLAIMER}**")
        gr.Markdown(f"Displayed boxes use a model-score threshold of **{confidence:.2f}**. Scores are uncalibrated model outputs, not clinical certainty.")
        upload = gr.Image(type="pil", label="Upload one bone X-ray")
        run = gr.Button("Locate potential lesion regions", variant="primary")
        flag = gr.Markdown()
        with gr.Row():
            original = gr.Image(label="Original image", interactive=False)
            annotated = gr.Image(label="Model-annotated output", interactive=False)
        detections = gr.JSON(label="Model detections")
        run.click(runtime.predict, upload, [original, annotated, detections, flag])
        with gr.Accordion("Educational context", open=False):
            gr.Markdown(
                "Bone-lesion assessment typically combines clinical history, specialist radiograph interpretation, "
                "additional imaging, and sometimes biopsy/pathology. A detector box only marks image regions similar "
                "to the dataset's annotated lesions; it cannot determine stage, treatment, urgency, or prognosis."
            )
    return app


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Detector checkpoint not found: {args.checkpoint}. Complete BTXRD training first.")
    runtime = DetectorRuntime(args.checkpoint, args.confidence, args.device)
    build_app(runtime, args.confidence).queue().launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
