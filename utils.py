import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
from args import get_args
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    return device


def scheduler_setup(optimizer):
    args = get_args()

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs), eta_min=args.min_lr)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.gamma, patience=3, min_lr=args.min_lr)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    return scheduler


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {"balanced_acc": balanced_accuracy_score(y_true, y_pred),
               'f1_score': f1_score(y_true, y_pred, average='macro'),
               'recall_score': recall_score(y_true, y_pred),
               'roc_auc': roc_auc_score(y_true, y_pred)}

    return metrics


def save_checkpoint(cur_fold, epoch, model, val_metric,
                    best_val_metric=None, prev_model_path=None,
                    comparator="gt", save_dir="session"):
    """
    Save the best model checkpoint for each fold based on validation metric.
    Also saves metrics.json with metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    improved = False
    if best_val_metric is None:
        improved = True
    elif comparator == "gt" and val_metric > best_val_metric:
        improved = True
    elif comparator == "lt" and val_metric < best_val_metric:
        improved = True

    if improved:
        if prev_model_path and os.path.exists(prev_model_path):
            try:
                os.remove(prev_model_path)
            except OSError:
                pass

        # Define new file paths
        ckpt_path = os.path.join(save_dir, f"fold_{cur_fold}_best.pth")
        metrics_path = os.path.join(save_dir, f"fold_{cur_fold}_metrics.json")

        torch.save(model.state_dict(), ckpt_path)

        # Save metadata
        metrics_data = {
            "fold": cur_fold,
            "best_epoch": epoch + 1,
            "best_val_metric": float(val_metric),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)

        print(f"Best model updated — fold {cur_fold}, epoch {epoch+1}, metric={val_metric:.4f}")
        print(f"Saved model to: {ckpt_path}")
        return val_metric, ckpt_path

    return best_val_metric, prev_model_path


def plot_loss_curve(train_losses, val_losses, save_dir, fold):
    """
        Plot the training and validation loss curves.

        Args:
            train_losses (list): List of average training losses per epoch.
            val_losses (list): List of average validation losses per epoch.
            save_dir (str): Directory to save the plot image.
            fold (int): Current fold number for naming the output file.
        """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)

    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"loss_curve_fold_{fold}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"[INFO] Saved training curve to {save_path}")
    plt.close()


def plot_confusion_matrix(model, dataloader, out_dir, device, fold=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch['label'].float().unsqueeze(1).to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = ["normal", "pneumonia"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)

    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(out_dir, f"confusion_matrix_fold_{fold}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix to {save_path}")


def plot_confusion_matrix_test(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["normal", "pneumonia"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to: {save_path}")

