from args import get_args
import torch
import os
import pandas as pd
from dataset import ChestXrayDataset
from torch.utils.data import DataLoader
from model import TransferModel
from trainer import train_model
import gc

from utils import check_device

device = check_device()


def main():
    args = get_args()

    for fold in range(1, 6):
        print("=" * 60)
        print('Training fold: ', fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(str(fold))))

        train_dataset = ChestXrayDataset(train_set, cache=False)
        val_dataset = ChestXrayDataset(val_set, cache=False)


        num_workers = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=1,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=1,
            persistent_workers=True,
        )

        model = TransferModel(args.backbone).to(device)


        train_model(model, train_loader, val_loader, fold, device)

        # cleanup per fold
        del train_loader, val_loader, train_dataset, val_dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    print("All folds finished.")

if __name__ == "__main__":
    main()

