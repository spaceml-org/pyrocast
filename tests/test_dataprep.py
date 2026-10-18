"""Tests for the CNN datasets and loaders in pyrocast.utils.data.dataprep."""

import numpy as np
import pandas as pd
import pytest
import torch

from pyrocast.config import CNNConfig, DataConfig
from pyrocast.utils.data.dataload import build_sample_index
from pyrocast.utils.data.dataprep import ZarrCubeDataset, channel_stats, load_loaders


@pytest.fixture
def data_cfg(data_root):
    return DataConfig(root=data_root, n_jobs=1)


@pytest.fixture
def index(data_cfg):
    return build_sample_index(data_cfg, "forecast")


class TestZarrCubeDataset:
    def test_item(self, index, data_cfg):
        ds = ZarrCubeDataset(index, data_cfg, "geostationary")
        image, label = ds[1]
        assert image.shape == (6, 200, 200)
        assert image.dtype == torch.float32
        assert label == 1.0
        assert len(ds) == len(index)

    def test_normalisation(self, index, data_cfg):
        mean, std = np.arange(6.0), np.full(6, 2.0)
        ds = ZarrCubeDataset(index, data_cfg, "geostationary", mean=mean, std=std)
        raw = ZarrCubeDataset(index, data_cfg, "geostationary")
        torch.testing.assert_close(
            ds[0][0], (raw[0][0] - torch.tensor(mean)[:, None, None].float()) / 2.0
        )

    def test_channel_stats(self, index, data_cfg):
        ds = ZarrCubeDataset(index[index.event_id == "1_1"], data_cfg, "era5")
        mean, std = channel_stats(ds, n_samples=100, seed=0)
        assert mean.shape == std.shape == (19,)
        np.testing.assert_allclose(mean, np.arange(19) + 250)
        assert np.all(std > 0)

    def test_era5_channels(self, index, data_cfg):
        ds = ZarrCubeDataset(index, data_cfg, "both", era5_channels=[16, 3])
        image, _ = ds[0]
        assert image.shape == (8, 200, 200)
        assert image[6:, 0, 0].tolist() == [16.0, 3.0]


class TestLoadLoaders:
    def test_batches(self, index, data_cfg):
        cnn_cfg = CNNConfig(batch_size=4, num_workers=0, norm_samples=8)
        train_rows, test_rows = index.iloc[:20], index.iloc[20:]
        train, test = load_loaders(
            train_rows, test_rows, data_cfg, cnn_cfg, "both", era5_channels=[4]
        )
        images, labels = next(iter(train))
        assert images.shape == (4, 7, 200, 200)
        assert labels.shape == (4,)
        assert (len(train.dataset), len(test.dataset)) == (20, len(index) - 20)
