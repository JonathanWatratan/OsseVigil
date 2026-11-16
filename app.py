#!/usr/bin/env python3
"""LEGACY Gradio classifier retained only for research reproducibility.

This is not the primary OsteoInsight product. Production/frontend inference must
use the BTXRD localization API in api.py. Do not present these category scores as
clinical cancer probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from osteoinsight.common import DISCLAIMER, select_device
from osteoinsight.data import make_transforms
from osteoinsight.modeling import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/osteoinsight_best.pth"))
    parser.add_argument("--exploration-dir", type=Path, default=Path("reports/embeddings"))
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link (do not use with private medical data)")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


class ClassificationRuntime:
    def __init__(self, checkpoint: Path, device: str = "auto"):
        self.device = select_device(device)
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.idx_to_class = {index: name for name, index in self.meta["class_to_idx"].items()}
        _, self.transform = make_transforms(int(self.meta.get("image_size", 224)))

    def predict(self, image: Image.Image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        confidence = float(probs[top])
        warning = (
            "Low-confidence result: categories are close. Treat this output as uncertain."
            if confidence < 0.65 else
            "Confidence is an uncalibrated model probability and is not clinical certainty."
        )
        explanation = (
            f"The image is most similar to the training dataset category “{self.idx_to_class[top]}”. "
            "This category name reflects the source annotation scheme; it is not an independent diagnosis."
        )
        return {self.idx_to_class[i]: float(value) for i, value in enumerate(probs)}, warning, explanation


class ExplorationRuntime:
    def __init__(self, folder: Path, device: str = "auto"):
        import joblib
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = select_device(device)
        self.assignments = pd.read_csv(folder / "cluster_assignments.csv")
        self.embeddings = np.load(folder / "embeddings.npy")
        self.kmeans = joblib.load(folder / "kmeans.joblib")
        weights = ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()
        self.encoder = resnet18(weights=weights)
        self.encoder.fc = torch.nn.Identity()
        self.encoder.to(self.device).eval()

    def explore(self, image: Image.Image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            vector = self.encoder(tensor).cpu().numpy()
        cluster = int(self.kmeans.predict(vector)[0])
        distances = np.linalg.norm(self.embeddings - vector[0], axis=1)
        nearest = np.argsort(distances)[:8]
        gallery = [(self.assignments.iloc[i]["absolute_path"], f"Visual neighbor {rank + 1}") for rank, i in enumerate(nearest)]
        text = (
            f"Visual-similarity cluster {cluster}. Clusters and neighbors reflect pretrained image features; "
            "they are not medically validated diagnoses, findings, or lesion localizations."
        )
        return text, gallery


def build_app(classifier, explorer):
    import gradio as gr

    css = """
    :root { --navy: #071a2d; --cyan: #63d3e5; }
    .gradio-container { max-width: 1180px !important; background: linear-gradient(150deg,#061526,#0c2940); }
    .hero { padding: 24px; border: 1px solid #294761; border-radius: 18px; background: rgba(7,26,45,.88); }
    .hero h1 { color: #f5fbff; letter-spacing: -.03em; margin-bottom: 4px; }
    .hero p, .prose { color: #bfd1df; }
    .disclaimer { border-left: 4px solid #63d3e5; padding: 12px 16px; background: #0b2236; color: #e6f5fb; }
    """
    with gr.Blocks(title="OsteoInsight legacy experiment", css=css, theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate")) as demo:
        gr.HTML("<div class='hero'><h1>Legacy classifier experiment</h1><p>Archived for reproducibility — not the OsteoInsight product inference engine</p></div>")
        gr.Markdown("## ⚠️ Do not interpret these dataset-category scores as cancer probabilities\nThe primary product is the BTXRD lesion-localization detector exposed through `/api/predict`.")
        gr.Markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>")
        if classifier:
            with gr.Tab("Legacy dataset-category experiment"):
                gr.Markdown("Upload an image to compare it with the source-verified categories used during training.")
                with gr.Row():
                    image = gr.Image(type="pil", label="X-ray image", height=460)
                    with gr.Column():
                        probabilities = gr.Label(label="Uncalibrated category probabilities", num_top_classes=8)
                        warning = gr.Markdown()
                        explanation = gr.Markdown()
                run = gr.Button("Analyze image", variant="primary")
                run.click(classifier.predict, image, [probabilities, warning, explanation])
                with gr.Accordion("Batch processing", open=False):
                    files = gr.File(file_count="multiple", file_types=["image"], label="Select multiple images")
                    batch_table = gr.Dataframe(label="Batch results", interactive=False)

                    def batch_predict(items):
                        rows = []
                        for item in items or []:
                            with Image.open(item) as sample:
                                probs, warn, _ = classifier.predict(sample)
                            best = max(probs, key=probs.get)
                            rows.append({"file": Path(item).name, "dataset_category": best, "uncalibrated_probability": probs[best], "warning": warn})
                        return pd.DataFrame(rows)

                    gr.Button("Process batch").click(batch_predict, files, batch_table)
        if explorer:
            with gr.Tab("Visual similarity explorer"):
                gr.Markdown("For datasets without medically verified labels. This view performs no disease classification.")
                with gr.Row():
                    explore_image = gr.Image(type="pil", label="Query image", height=420)
                    with gr.Column():
                        cluster_text = gr.Markdown()
                        gallery = gr.Gallery(label="Nearest visual neighbors", columns=4, height=420)
                gr.Button("Find similar images", variant="primary").click(explorer.explore, explore_image, [cluster_text, gallery])
        if not classifier and not explorer:
            gr.Markdown("No trained checkpoint or completed embedding exploration was found. Run the appropriate pipeline first.")
        gr.Markdown("### Known limitations\nPerformance is specific to the audited dataset and label definitions. Domain shift, acquisition differences, duplicates, weak annotations, and missing patient-level independence can materially change results. Do not use this application for diagnosis or care decisions.")
    return demo


def main():
    args = parse_args()
    classifier = ClassificationRuntime(args.checkpoint, args.device) if args.checkpoint.exists() else None
    exploration_files = [args.exploration_dir / name for name in ["cluster_assignments.csv", "embeddings.npy", "kmeans.joblib"]]
    explorer = ExplorationRuntime(args.exploration_dir, args.device) if all(path.exists() for path in exploration_files) else None
    demo = build_app(classifier, explorer)
    demo.queue().launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
