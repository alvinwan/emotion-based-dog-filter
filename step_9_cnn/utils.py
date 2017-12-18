"""
Convolutional neural network for Face Emotion Recognition (FER) 2013 Dataset

Utility for loading FER into PyTorch. Dataset curated by Pierre-Luc Carrier
and Aaron Courville in 2013.
"""

from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import numpy as np


class Fer2013Dataset(Dataset):
    """Face Emotion Recognition dataset

    Each sample is 1 x 1 x 48 x 48, and each label is a scalar.
    """

    def __init__(self, sample_path: str, label_path: str):
        """
        Args:
            sample_path: Path to `.npy` file containing samples nxd.
            label_path: Path to `.npy` file containign labels nx1.
        """
        self._samples = np.load(sample_path)
        self._labels = np.load(label_path)
        self._samples = self._samples.reshape((-1, 1, 48, 48))

        self.X = Variable(torch.from_numpy(self._samples)).float()
        self.Y = Variable(torch.from_numpy(self._labels)).float()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {'image': self._samples[idx], 'label': self._labels[idx]}
