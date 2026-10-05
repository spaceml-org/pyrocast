"""
CNN and auto-encoder-pretrained CNN (AE-CNN) of the Pyrocast paper.

The CNN has six convolutional layers, each followed by max pooling (except the
last, which reduces the 6x6 map to 1x1), then two fully connected layers down to
a 16-dimensional hidden layer with dropout, and a linear output layer. The
AE-CNN first trains the encoder with a mirror decoder that reconstructs the
inputs from the 16-dimensional layer, then trains the classifier.
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from pyrocast.config import CNNExperimentConfig, load_config
from pyrocast.utils import metrics
from pyrocast.utils.data import dataprep as dp
from pyrocast.utils.data.dataload import get_sample_index
from pyrocast.utils.experiment import cross_validate, save_run

N_GEO_CHANNELS = 6
IMAGE_SIZE = 200
HIDDEN = 16


def n_channels(cfg: CNNExperimentConfig) -> int:
    """Number of input channels of an experiment."""
    n = 0
    if cfg.inputs in ("geostationary", "both"):
        n += N_GEO_CHANNELS
    if cfg.inputs in ("era5", "both"):
        n += len(cfg.era5_channels)
    return n


class Encoder(nn.Module):
    """Six convolutional and two fully connected layers, 200x200 input to 16."""

    def __init__(self, n_channels: int, dropout: float = 0.25):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.Conv2d(32, 64, kernel_size=3, padding=0),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ]
        )
        self.conv6 = nn.Conv2d(256, 512, kernel_size=6)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, HIDDEN)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Encode a batch of images.

        Args:
            x: tensor of shape (batch, n_channels, 200, 200)

        Returns:
            the hidden layer of shape (batch, 16), and the pooling indices and
            pre-pooling sizes of each pooled layer (for the decoder)
        """
        indices, sizes = [], []
        for conv in self.convs:
            x = F.relu(conv(x))
            sizes.append(x.shape[-2:])
            x, idx = self.pool(x)
            indices.append(idx)
        x = F.relu(self.conv6(x)).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        return x, indices, sizes


class Decoder(nn.Module):
    """Mirror of Encoder, from the 16-dimensional layer back to the inputs."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.fc2 = nn.Linear(HIDDEN, 128)
        self.fc1 = nn.Linear(128, 512)
        self.deconv6 = nn.ConvTranspose2d(512, 256, kernel_size=6)
        self.deconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(16, n_channels, kernel_size=3, padding=1),
                nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
                nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0),
                nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
                nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            ]
        )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, z: torch.Tensor, indices, sizes) -> torch.Tensor:
        x = F.relu(self.fc1(F.relu(self.fc2(z))))
        x = F.relu(self.deconv6(x[:, :, None, None]))
        for i in reversed(range(len(self.deconvs))):
            x = self.unpool(x, indices[i], output_size=sizes[i])
            x = self.deconvs[i](x)
            if i > 0:
                x = F.relu(x)
        return x


class CNN(nn.Module):
    """Pyrocast CNN classifier: Encoder and a linear layer to two class logits."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.encoder = Encoder(n_channels)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)[0])


class AutoEncoder(nn.Module):
    """Encoder and mirror Decoder, reconstructing the input images."""

    def __init__(self, encoder: Encoder, n_channels: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = Decoder(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(*self.encoder(x))


def resolve_device(name: str) -> torch.device:
    """
    Resolve a device name from the config.

    Args:
        name: "auto", "cpu" or "cuda"

    Returns:
        the torch device; "auto" picks CUDA when available
    """
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return torch.device(name)


def _train(model, loader, n_epochs, lr, device, loss_fn, name) -> list[float]:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    losses = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, inputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
        losses.append(running_loss / len(loader.dataset))
        logging.info("%s epoch %d loss %.6f", name, epoch + 1, losses[-1])
    return losses


def train_cnn(
    model: CNN,
    loader: torch.utils.data.DataLoader,
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """
    Train the classifier with cross-entropy loss and Adam.

    Args:
        model: model to train in place
        loader: training DataLoader yielding (images, labels)
        n_epochs: number of epochs
        lr: learning rate
        device: device to train on

    Returns:
        mean training loss per epoch
    """

    def loss_fn(m, x, y):
        return F.cross_entropy(m(x), y.long())

    return _train(model, loader, n_epochs, lr, device, loss_fn, "classifier")


def pretrain_autoencoder(
    model: AutoEncoder,
    loader: torch.utils.data.DataLoader,
    n_epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """
    Train the auto-encoder to reconstruct its inputs (MSE loss, Adam).

    Args:
        model: auto-encoder to train in place
        loader: training DataLoader yielding (images, labels); labels are unused
        n_epochs: number of epochs
        lr: learning rate
        device: device to train on

    Returns:
        mean reconstruction loss per epoch
    """

    def loss_fn(m, x, _):
        return F.mse_loss(m(x), x)

    return _train(model, loader, n_epochs, lr, device, loss_fn, "auto-encoder")


@torch.no_grad()
def predict(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict pyroCb probabilities over a DataLoader.

    Args:
        model: trained classifier returning two class logits
        loader: DataLoader yielding (images, labels)
        device: device to run on

    Returns:
        true labels and predicted probabilities
    """
    model.to(device).eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        y_true.append(labels.numpy())
        logits = model(inputs.to(device))
        y_pred.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def run(cfg: CNNExperimentConfig) -> dict:
    """
    Train and evaluate the CNN (or AE-CNN when pretrain_epochs > 0) on every
    fold, saving the config, metrics, out-of-fold predictions and weights.

    Args:
        cfg: CNN experiment config

    Returns:
        metrics from pyrocast.utils.metrics.cv_report plus per-fold training
        losses and sample counts
    """
    device = resolve_device(cfg.model.device)
    index = get_sample_index(cfg.data, cfg.mode)
    logging.info(
        "Sample index (%s): %d samples, %d events, %d wildfires",
        cfg.mode,
        len(index),
        index.event_id.nunique(),
        index.wildfire_id.nunique(),
    )
    era5_channels = cfg.era5_channels if cfg.inputs != "geostationary" else None
    channels = n_channels(cfg)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    losses, ae_losses = [], []

    def fit_predict(train, test, fold):
        torch.manual_seed(cfg.model.seed)
        trainloader, testloader = dp.load_loaders(
            index.iloc[train],
            index.iloc[test],
            cfg.data,
            cfg.model,
            cfg.inputs,
            era5_channels,
        )
        model = CNN(channels)
        if cfg.model.pretrain_epochs:
            ae_losses.append(
                pretrain_autoencoder(
                    AutoEncoder(model.encoder, channels),
                    trainloader,
                    cfg.model.pretrain_epochs,
                    cfg.model.lr,
                    device,
                )
            )
        losses.append(
            train_cnn(model, trainloader, cfg.model.n_epochs, cfg.model.lr, device)
        )
        torch.save(model.state_dict(), cfg.run_dir / f"model_fold{fold}.pt")
        return predict(model, testloader, device)[1]

    predictions = cross_validate(index, cfg.split, fit_predict)
    results = metrics.cv_report(predictions)
    results.update(
        losses=losses,
        n_samples=len(index),
        n_positive=int(index.label.sum()),
        n_events=int(index.event_id.nunique()),
        n_wildfires=int(index.wildfire_id.nunique()),
    )
    if ae_losses:
        results["pretrain_losses"] = ae_losses
    save_run(cfg.run_dir, cfg, results)
    predictions.to_csv(cfg.run_dir / "predictions.csv", index=False)
    logging.info("AUC: %.3f +/- %.3f", results["auc"], results["auc_std"])
    return results


def main(argv: list[str] | None = None) -> None:
    """
    Train the CNN from a YAML config.

    Args:
        argv: command-line arguments, defaults to sys.argv
    """
    parser = argparse.ArgumentParser(description="Train the Pyrocast CNN")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    run(load_config(args.config, CNNExperimentConfig))


if __name__ == "__main__":
    main()
