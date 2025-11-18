import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler
from contextlib import nullcontext

from args import get_args
from utils import calculate_metrics, save_checkpoint, plot_loss_curve


def train_model(model, train_loader, val_loader, cur_fold, device):
    """
        # TODO: update this
        Train the model with mixed precision and backbone freezing.

        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            cur_fold: Current fold number
            device: torch.device passed from main
        """

    args = get_args()

    # Enable Automatic Mixed Precision (AMP) only when running on a CUDA-capable GPU.
    # AMP allows certain operations to run in float16 (half precision) instead of float32,
    # which speeds up training and reduces memory
    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if use_amp else None

    # Scheduler setup
    # TODO: create a help function in utils to check scheduler
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

    # Initialize tracking variables
    best_val_metric = None
    best_model_path = None
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        epoch_start = time.time()  # Start the timer

        model.train()
        epoch_training_loss = 0.0

        # Train loop
        for batch in train_loader:
            images = batch['image'].float().to(device, non_bloking=True)
            labels = batch['label'].long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # set None for params that did not receive a gradient

            # mixed precision
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

            epoch_training_loss += loss.item()

            # free batch level tensors
            del images, labels, outputs, loss

        avg_train_loss = epoch_training_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1}/{args.epochs} | Train loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")

        metrics, y_true, y_pred, avg_val_loss = validate_model(model, val_loader, criterion, device)
        pr_auc = metrics['pr_auc']
        val_losses.append(avg_val_loss)

        best_pr_auc, best_model_path = save_checkpoint(cur_fold,
                                                       epoch,
                                                       model,
                                                       pr_auc,
                                                       best_val_metric=best_val_metric,
                                                       prev_model_path=best_model_path,
                                                       comparator='gt',
                                                       save_dir=args.output_dir)
        best_val_metric = best_pr_auc

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(metrics["pr_auc"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[LR] Current learning rate: {current_lr:.6f}")

        plot_loss_curve(train_losses, val_losses, save_dir=args.output_dir, fold=cur_fold)

        # clean up and free memory
        gc.collect()
        torch.cuda.empty_cache()

    return best_model_path


def validate_model(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    use_amp = device.type == "cuda"
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    with amp_context():
        for batch in val_loader:
            images = batch['image'].float().to(device, non_bloking=True)
            labels = batch['label'].long().to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            del images, labels, outputs, loss

        if len(y_true) == 0:
            return 0.0, np.array([]), np.array([])

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        metrics = calculate_metrics(y_true, y_pred, y_prob)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Validation Loss: {avg_val_loss:.4f} | "
            f"PR AUC: {metrics['pr_auc']:.4f} | "
            f"MCC: {metrics['mcc']:.4f}"
        )
        print("=" * 60)

        return metrics, y_true, y_pred, y_prob, avg_val_loss



