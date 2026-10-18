import torch
import numpy as np
import pandas as pd

from pyrocast.config import CNNConfig, DataConfig, Inputs
from pyrocast.utils.data.dataload import read_cubes


def flatten(lst):
    """Flatten a nested list into a single list."""
    result = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


class HimawariDataset(torch.utils.data.Dataset):
    """Dataset object withs data, labels and transformations"""

    def __init__(self, dataset_cubes, flags, transform=None, target_transform=None):
        self.img_labels = flags
        self.img_cubes = dataset_cubes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = np.moveaxis(self.img_cubes[idx], 0, -1)
        label = int(self.img_labels[idx])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ZarrCubeDataset(torch.utils.data.Dataset):
    """Dataset reading one (event, hour) cube from the zarr stores per item."""

    def __init__(
        self,
        index: pd.DataFrame,
        cfg: DataConfig,
        inputs: Inputs,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        era5_channels: list[int] | None = None,
    ):
        self.index = index.reset_index(drop=True)
        self.cfg = cfg
        self.inputs = inputs
        self.era5_channels = era5_channels
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def read(self, idx: int) -> np.ndarray:
        """Read the unnormalised cube for sample idx, shape (channels, H, W)."""
        row = self.index.iloc[idx]
        return read_cubes(
            self.cfg,
            row.event_id,
            [row.date_idx],
            row.satellite,
            self.inputs,
            [row.era5_idx],
            self.era5_channels,
        )[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        image = self.read(idx)
        if self.mean is not None:
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(image), float(self.index.label.iloc[idx])


def channel_stats(
    dataset: ZarrCubeDataset, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-channel mean and std from a random subset of a dataset.

    Args:
        dataset: dataset to sample from
        n_samples: maximum number of samples to read
        seed: random seed for the subset

    Returns:
        mean and std, each of shape (channels,); zero std is replaced by one
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(dataset))
    idxs = rng.choice(len(dataset), size=n, replace=False)
    cubes = np.stack([dataset.read(int(i)) for i in idxs]).astype(np.float64)
    mean = cubes.mean(axis=(0, 2, 3))
    std = cubes.std(axis=(0, 2, 3))
    return mean, np.where(std > 0, std, 1.0)


def load_loaders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: DataConfig,
    cnn: CNNConfig,
    inputs: Inputs,
    era5_channels: list[int] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Return train and test loaders for the CNN, normalised with channel
    statistics of the training set.

    Args:
        train: training sample index
        test: test sample index
        cfg: data configuration
        cnn: CNN configuration (batch size, workers, normalisation samples, seed)
        inputs: which stores to use
        era5_channels: ERA5 channels to read; None reads all

    Returns:
        train and test DataLoaders yielding (images, labels)
    """
    mean, std = channel_stats(
        ZarrCubeDataset(train, cfg, inputs, era5_channels=era5_channels),
        cnn.norm_samples,
        cnn.seed,
    )
    generator = torch.Generator().manual_seed(cnn.seed)
    trainloader = torch.utils.data.DataLoader(
        ZarrCubeDataset(train, cfg, inputs, mean, std, era5_channels),
        batch_size=cnn.batch_size,
        shuffle=True,
        num_workers=cnn.num_workers,
        generator=generator,
    )
    testloader = torch.utils.data.DataLoader(
        ZarrCubeDataset(test, cfg, inputs, mean, std, era5_channels),
        batch_size=cnn.batch_size,
        shuffle=False,
        num_workers=cnn.num_workers,
    )
    return trainloader, testloader
