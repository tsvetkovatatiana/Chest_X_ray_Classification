import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
import numpy as np
from utils import scheduler_setup, calculate_metrics, save_checkpoint, plot_loss_curve, plot_confusion_matrix
from args import get_args


def get_amp_context(device):
    """Mixed precision only on CUDA; plain context on MPS/CPU."""
    if device.type == "cuda":
        return lambda: torch.amp.autocast("cuda")
    return nullcontext


def train_model(model, train_loader, val_loader, cur_fold, device):
    """
    Train the model with mixed precision (amp) and backbone support for binary classification.
    Class imbalance is handled ONLY by the WeightedRandomSampler in main.py,
    so the loss has no pos_weight (otherwise the imbalance is corrected twice).
    """
    args = get_args()

    use_amp = device.type == "cuda"
    amp_context = get_amp_context(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler = scheduler_setup(optimizer)

    best_val_metric = None
    best_model_path = None
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch['image'].float().to(device, non_blocking=True)
            labels = batch['label'].float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                outputs = model(images)
                loss = criterion(outputs, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {avg_train_loss:.4f} |"
              f" Time: {epoch_time:.2f}s")

        metrics, _, _, _, avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        best_val_metric, best_model_path = save_checkpoint(
            cur_fold, epoch, model,
            metrics['balanced_acc'], best_val_metric, best_model_path,
            comparator='gt', save_dir=args.output_dir
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                # scheduler_setup uses mode="max", so step on a "higher is better" metric
                scheduler.step(metrics['roc_auc'])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6f}")

        plot_loss_curve(train_losses, val_losses, args.output_dir, cur_fold)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Confusion matrix once, for the BEST checkpoint (not every epoch, not the last epoch)
    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        plot_confusion_matrix(model, val_loader, args.output_dir, device, cur_fold)

    return best_model_path


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    amp_context = get_amp_context(device)

    # no_grad: no gradients are stored during validation -> less memory, faster
    with torch.no_grad(), amp_context():
        for batch in val_loader:
            images = batch['image'].float().to(device, non_blocking=True)
            labels = batch['label'].float().unsqueeze(1).to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs.float())
            preds = (probs > 0.5).long()

            # flatten (N, 1) -> (N,) so sklearn gets 1-D arrays
            y_true.extend(labels.cpu().numpy().ravel().tolist())
            y_pred.extend(preds.cpu().numpy().ravel().tolist())
            y_prob.extend(probs.cpu().numpy().ravel().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    avg_val_loss = val_loss / len(val_loader)

    if len(y_true) > 0:
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        print(
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"Balanced_accuracy: {metrics['balanced_acc']:.4f} | "
            f"F1-macro: {metrics['f1_score']:.4f} | "
            f"Recall: {metrics['recall_score']:.4f} | "
            f"ROC_AUC: {metrics['roc_auc']:.4f}"
        )
        print("=" * 60)
    else:
        metrics = {}

    return metrics, y_true, y_pred, y_prob, avg_val_loss