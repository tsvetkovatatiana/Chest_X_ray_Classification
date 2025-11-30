from args import get_args
import torch
import os
import pandas as pd
import numpy as np
from dataset import ChestXrayDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
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

        # oversampling
        label_map = {
            "normal": 0,
            "pneumonia": 1
        }
        labels = train_dataset.data['Label'].map(label_map)

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts

        sample_weights = class_weights[labels]
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


        num_workers = 4
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
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

