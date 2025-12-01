import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ChestXrayDataset

from model import TransferModel
from utils import check_device, calculate_metrics, plot_confusion_matrix_test
from args import get_args


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    metrics = calculate_metrics(all_labels, all_preds)

    return metrics, all_labels, all_preds


def main():
    args = get_args()
    device = check_device()

    os.makedirs("evaluation_results", exist_ok=True)

    test_csv = os.path.join(args.csv_dir, f"test_data.csv")
    test_data = pd.read_csv(test_csv)
    test_dataset = ChestXrayDataset(test_data, cache=False)
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=1,
        persistent_workers=True,
    )

    model = TransferModel(args.backbone).to(device)
    model_path = args.best_model_pth
    model.load_state_dict(torch.load(model_path, map_location=device))


    metrics, y_true, y_pred = evaluate_model(model, val_loader, device)

    print(
        f"Balanced_accuracy: {metrics['balanced_acc']:.4f} |"
        f"F1-macro: {metrics['f1_score']:.4f} | "
        f"Recall: {metrics['recall_score']:.4f} | "
        f"ROC_AUC: {metrics['roc_auc']:.4f}"
    )
    print("=" * 60)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"evaluation_results/test_metrics.csv", index=False)

    plot_confusion_matrix_test(y_true, y_pred,
                          f"evaluation_results/confusion_matrix_fold_test.png")


if __name__ == "__main__":
    main()
