import torch
from torch.utils.data import Dataset


class RumpDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X_, y_, indices_in_original_dataset):
        self.X_ = X_
        self.y_ = y_
        self.indicesInOriginalDataset = indices_in_original_dataset

        assert self.X_.shape[0] == self.y_.shape[0] and self.X_.shape[0] == indices_in_original_dataset.shape[0]

    def __len__(self):
        return self.X_.shape[0]

    def __getitem__(self, idx):
        sample_X = self.X_[idx]
        sample_y = self.y_[idx]
        return sample_X, sample_y
