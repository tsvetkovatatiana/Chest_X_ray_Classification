import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import os
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


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


def calculate_metrics(y_true, y_pred):
    """
    Args:
        y_true: True labels (list or numpy array)
        y_pred: Predicted labels (list or numpy array)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {"balanced_acc": balanced_accuracy_score(y_true, y_pred),
               'roc_auc': roc_auc_score(y_true, y_pred, average='micro')}

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


def inspect_dataset(dataset, num_samples=5, show_images=True):
    """
    Inspect dataset for label sanity and image statistics.

    Args:
        dataset: your PyTorch Dataset
        num_samples: how many samples to print and plot
        show_images: whether to plot sample images
    """
    all_labels = []
    print("Inspecting dataset...\n")

    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample['image']
        label = sample['label']

        # Convert to tensor if needed
        if not torch.is_tensor(label):
            label = torch.tensor(label)

        all_labels.append(label)

        if i < num_samples and show_images:
            if torch.is_tensor(image):
                img_np = image.detach().cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # CxHxW
                    img_np = np.transpose(img_np, (1, 2, 0))
                plt.imshow(img_np.squeeze(), cmap='gray')
                plt.title(f"Label: {label}")
                plt.show()

    all_labels = torch.stack([l if torch.is_tensor(l) else torch.tensor(l) for l in all_labels])
    print("Label dtype:", all_labels.dtype)
    print("Label min:", all_labels.min().item())
    print("Label max:", all_labels.max().item())
    print("Unique labels:", torch.unique(all_labels))
    print("Label counts:")
    for val in torch.unique(all_labels):
        print(f"  {val.item()}: {(all_labels == val).sum().item()}")

    print("\nImage statistics:")
    # Compute mean/std over dataset
    all_images = [dataset[i]['image'] for i in range(len(dataset))]
    imgs_tensor = torch.stack([i if torch.is_tensor(i) else torch.tensor(i) for i in all_images])
    print("Images min:", imgs_tensor.min().item())
    print("Images max:", imgs_tensor.max().item())
    print("Images mean:", imgs_tensor.mean().item())
    print("Images std:", imgs_tensor.std().item())