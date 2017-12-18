from torch.utils.data import Dataset
import numpy as np


class Fer2013Dataset(Dataset):
    """Face Emotion Recognition dataset

    Each sample is 1 x 1 x 48 x 48, and each label is a scalar. Note that the

    """

    def __init__(self, sample_path: str, label_path: str):
        """
        Args:
            sample_path: Path to `.npy` file containing samples nxd.
            label_path: Path to `.npy` file containign labels nx1.
        """
        self.samples = np.load(sample_path)
        self.samples = self.samples.reshape((self.samples.shape[0], 1, 48, 48))
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.samples[idx]
        return {'image': image, 'label': self.labels[idx]}
