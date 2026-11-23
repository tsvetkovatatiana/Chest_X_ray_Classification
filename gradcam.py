"""
Grad-CAM for the chest X-ray classifier (TransferModel, ResNet18/34/50).

Shows WHERE the model looks when it predicts Normal or Pneumonia.
  - Good sign: heat is on the lungs.
  - Shortcut warning: heat on corners, text/markers, borders, or outside the body.

Outputs (in --out_dir):
  - gradcam_grid.png  : original image + heatmap overlay for sampled images
  - gradcam_mean.png  : average heatmap over all sampled images
                        (if this is bright in corners/edges, the model may use shortcuts)

Usage:
  python gradcam.py --model_path session/fold_1_best.pth --csv data/CSVs/test_data.csv
  python gradcam.py --model_path session/fold_1_best.pth --csv data/CSVs/test_data.csv --mode errors
"""

import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import read_x_ray
from model import TransferModel
from transforms import val_transform

LABEL_MAP = {"normal": 0, "pneumonia": 1}
CLASS_NAMES = ["normal", "pneumonia"]


class GradCAM:
    """Minimal Grad-CAM using forward/backward hooks on one conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
        """
        x: tensor [1, 3, H, W]
        Returns (cam [H, W] in 0..1, probability of pneumonia).
        CAM explains the PREDICTED class: pneumonia evidence if prob > 0.5,
        normal evidence otherwise.
        """
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)              # [1, 1]
        prob = torch.sigmoid(logit).item()

        score = logit if prob > 0.5 else -logit
        score.sum().backward()

        # channel weights = global-average-pooled gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)       # [1, C, 1, 1]
        cam = torch.relu((weights * self.activations).sum(dim=1))     # [1, h, w]
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, prob


def get_target_layer(model):
    # last residual block of ResNet = last conv feature map (7x7 for 224 input)
    return model.model.layer4[-1]


def overlay(gray_224, cam):
    """Blend a JET heatmap over the grayscale image. Returns RGB uint8."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray_224, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base, 0.6, heat, 0.4, 0)


def sample_rows(df, model, device, mode, n_per_class, seed):
    """Pick images to explain: random per class, or misclassified ones."""
    if mode == "random":
        parts = [g.sample(min(n_per_class, len(g)), random_state=seed)
                 for _, g in df.groupby("Label")]
        return pd.concat(parts).reset_index(drop=True)

    # mode == "errors": run the model and keep wrong predictions
    wrong = []
    with torch.no_grad():
        for _, row in df.iterrows():
            x = load_tensor(row["Path"]).to(device)
            prob = torch.sigmoid(model(x)).item()
            pred = int(prob > 0.5)
            if pred != LABEL_MAP[row["Label"]]:
                wrong.append(row)
            if len(wrong) >= 2 * n_per_class:
                break
    if not wrong:
        print("No misclassified images found in this CSV.")
        return df.iloc[0:0]
    return pd.DataFrame(wrong).reset_index(drop=True)


def load_tensor(path):
    image = Image.fromarray(read_x_ray(path))
    return val_transform(image).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM for chest X-ray model")
    parser.add_argument("--model_path", required=True, help="e.g. session/fold_1_best.pth")
    parser.add_argument("--backbone", default="resnet34", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--csv", default="data/CSVs/test_data.csv")
    parser.add_argument("--mode", default="random", choices=["random", "errors"],
                        help="random: N images per class; errors: misclassified images")
    parser.add_argument("--n", type=int, default=4, help="images per class (random) / half the total (errors)")
    parser.add_argument("--out_dir", default="gradcam_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = TransferModel(args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()  # eval mode, but gradients still flow for Grad-CAM

    df = pd.read_csv(args.csv)
    rows = sample_rows(df, model, device, args.mode, args.n, args.seed)
    if rows.empty:
        return

    cam_tool = GradCAM(model, get_target_layer(model))

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    axes = np.atleast_2d(axes)
    cams = []

    for i, row in rows.iterrows():
        x = load_tensor(row["Path"]).to(device)
        cam, prob = cam_tool(x)
        cams.append(cam)

        gray = cv2.resize(read_x_ray(row["Path"])[:, :, 0], (224, 224))
        pred = CLASS_NAMES[int(prob > 0.5)]
        ok = "OK" if pred == row["Label"] else "WRONG"

        axes[i, 0].imshow(gray, cmap="gray")
        axes[i, 0].set_title(f"true: {row['Label']}", fontsize=9)
        axes[i, 1].imshow(overlay(gray, cam))
        axes[i, 1].set_title(f"pred: {pred} (p_pneu={prob:.2f}) {ok}", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(args.out_dir, f"gradcam_grid_{args.mode}.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Saved {grid_path}")

    # Average heatmap: where does the model look in general?
    mean_cam = np.mean(cams, axis=0)
    plt.figure(figsize=(4, 4))
    plt.imshow(mean_cam, cmap="jet")
    plt.colorbar(fraction=0.046)
    plt.title(f"Mean Grad-CAM ({n} images)")
    plt.axis("off")
    mean_path = os.path.join(args.out_dir, f"gradcam_mean_{args.mode}.png")
    plt.savefig(mean_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {mean_path}")


if __name__ == "__main__":
    main()