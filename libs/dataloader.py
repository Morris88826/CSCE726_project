import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FeaturesDataset(Dataset):
    """Dataset for precomputed features.

    Expects features_path (N, D) and labels_path (N,) where labels are ints 0..C-1.
    """

    def __init__(self, features_path: str, labels_path: str):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)
        assert self.features.shape[0] == self.labels.shape[0], "features/labels length mismatch"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx

def build_dataloaders(root_data_dir, batch_size):
    dataloader_list = []
    for split, shuffle in zip(["train", "val", "test"], [True, False, False]):
        data_dir = os.path.join(root_data_dir, split)
        if not os.path.exists(data_dir):
            if split == "train" or split == "val":
                raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
            else:
                dataloader_list.append(None)
                continue
        features = os.path.join(data_dir, "features.pt")
        labels = os.path.join(data_dir, "labels.pt")
        ds = FeaturesDataset(features, labels)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
        dataloader_list.append(dataloader)
    train_loader, val_loader, test_loader = dataloader_list
    return train_loader, val_loader, test_loader
