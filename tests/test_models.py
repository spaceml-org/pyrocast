"""Tests for pyrocast.models — import and basic instantiation."""

import numpy as np
import pytest
import torch


class TestCNN:
    def test_forward_shape(self):
        from pyrocast.models.cnn import CNN

        model = CNN(n_channels=6)
        x = torch.randn(2, 6, 200, 200)
        y = model(x)
        assert y.shape == (2, 1)

    def test_output_range(self):
        from pyrocast.models.cnn import CNN

        model = CNN(n_channels=6)
        x = torch.randn(1, 6, 200, 200)
        y = model(x)
        assert 0.0 <= y.item() <= 1.0


class TestRandomForest:
    def test_init_and_fit(self):
        from pyrocast.models.random_forest import RandomForest

        rf = RandomForest(n_estimators=10)
        rng = np.random.RandomState(0)
        X = rng.rand(50, 5)
        y = rng.randint(0, 2, size=50)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == 50

    def test_inherits_sklearn(self):
        from sklearn.ensemble import RandomForestClassifier
        from pyrocast.models.random_forest import RandomForest

        rf = RandomForest()
        assert isinstance(rf, RandomForestClassifier)


class TestICPCNN:
    def test_encoder_output_shape(self):
        from pyrocast.models.icp_cnn import ICPEncoder

        enc = ICPEncoder(num_channels_in=6).double()
        x = torch.randn(2, 6, 200, 200, dtype=torch.float64)
        out = enc(x)
        assert out.shape == (2, 2, 8)

    def test_head_output_shape(self):
        from pyrocast.models.icp_cnn import ICPHead

        head = ICPHead(a=8, b=2, beta=1.0).double()
        z = torch.randn(2, 2, 8, dtype=torch.float64)
        env = torch.randn(2, 2, dtype=torch.float64)
        out, xb, xa = head(z, env)
        assert out.shape == (2, 2)


class TestDataprep:
    def test_flatten(self):
        from pyrocast.utils.data.dataprep import flatten

        assert flatten([[1, 2], [3, [4, 5]]]) == [1, 2, 3, 4, 5]
        assert flatten([]) == []
        assert flatten([1, 2, 3]) == [1, 2, 3]
