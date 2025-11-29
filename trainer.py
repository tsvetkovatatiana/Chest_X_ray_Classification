import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
import numpy as np
from utils import calculate_metrics, save_checkpoint, plot_loss_curve
from args import get_args


def train_model(model, train_loader, val_loader, cur_fold, device):
    """
    Train the model with mixed precision and backbone support for binary classification.
    """
    args = get_args()

    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Scheduler setup
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

    best_val_metric = None
    best_model_path = None
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images = batch['image'].float().to(device, non_blocking=True)
            labels = batch['label'].float().unsqueeze(1).to(device, non_blocking=True)  # shape [B,1]

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

            del images, labels, outputs, loss

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {avg_train_loss:.4f} |"
              f" Time: {epoch_time:.2f}s")

        metrics, y_true, y_pred, avg_val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)

        best_val_metric, best_model_path = save_checkpoint(
            cur_fold, epoch, model,
            metrics['balanced_acc'], best_val_metric, best_model_path,
            comparator='gt', save_dir=args.output_dir
        )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(metrics['pr_auc'])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[LR] Current learning rate: {current_lr:.6f}")

        plot_loss_curve(train_losses, val_losses, save_dir=args.output_dir, fold=cur_fold)

        gc.collect()
        torch.cuda.empty_cache()

    return best_model_path


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    with amp_context():
        for batch in val_loader:
            images = batch['image'].float().to(device, non_blocking=True)
            labels = batch['label'].float().unsqueeze(1).to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            del images, labels, outputs, loss

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_val_loss = val_loss / len(val_loader)

    if len(y_true) > 0:
        metrics = calculate_metrics(y_true, y_pred)
        print(
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"Balanced_accuracy: {metrics['balanced_acc']:.4f} | "
            f"ROC_AUC: {metrics['roc_auc']:.4f}"
        )
        print("=" * 60)
    else:
        metrics = {}

    return metrics, y_true, y_pred, avg_val_loss
