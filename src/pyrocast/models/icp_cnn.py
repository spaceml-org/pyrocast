"""
ICP-regularized CNN models for pyroconvection forecasting.

Implements a two-stage architecture: a CNN encoder (ICPEncoder) maps satellite
imagery to a 2x8 latent space split into causal and non-causal channels,
and a two-branch MLP head (ICPHead) produces predictions while an HSIC-based
causal loss penalises dependence between the branches.

Extracted from old_code/cnn-main/icp_nets.py.
"""

import os
import pickle
import time

import numpy as np
import scipy.special as scp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from pyrocast.icp.hsic import centering, hsicRBF


class ICPDataset(torch.utils.data.Dataset):
    """Dataset that returns (image, environment_variables, label)."""

    def __init__(
        self, dataset_cubes, envs, flags, transform=None, target_transform=None
    ):
        self.img_labels = flags
        self.img_cubes = dataset_cubes
        self.envs = envs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = np.moveaxis(self.img_cubes[idx], 0, -1)
        label = int(self.img_labels[idx])
        envs = np.moveaxis(self.envs[idx], 0, -1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, envs, label


_default_transform = transforms.Compose([transforms.ToTensor()])


class ICPEncoder(nn.Module):
    """6-layer CNN encoder producing a (batch, 2, 8) latent representation."""

    def __init__(self, num_channels_in):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels_in, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=6, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x, _ = self.pool(F.relu(self.conv1(x)))
        x, _ = self.pool(F.relu(self.conv2(x)))
        x, _ = self.pool(F.relu(self.conv3(x)))
        x, _ = self.pool(F.relu(self.conv4(x)))
        x, _ = self.pool(F.relu(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x.reshape(x.size(0), 2, 8)


class ICPHead(nn.Module):
    """Two-branch MLP head separating causal and non-causal features."""

    def __init__(self, a, b, beta):
        super().__init__()
        self.fc1a = nn.Linear(a + b, a)
        self.fc2a = nn.Linear(a, a)
        self.fc1b = nn.Linear(a, a)
        self.fc2b = nn.Linear(a, a)
        self.fc3 = nn.Linear(2 * a, 2)
        self.beta = beta

    def forward(self, input1, env_variables):
        xa = torch.cat((input1[..., 0, :], self.beta * env_variables), dim=-1)
        xa = F.relu(self.fc1a(xa))
        xa = F.relu(self.fc2a(xa))

        xb = input1[..., 1, :]
        xb = F.relu(self.fc1b(xb))
        xb = F.relu(self.fc2b(xb))

        output = self.fc3(torch.cat((xb, xa), dim=-1))
        return output, xb, xa


def causal_loss(pred, nn1_out, gt, lam, env_weights, device):
    """Cross-entropy + HSIC penalty between the two latent channels."""
    hsic_val, norm_k_x, norm_k_z = hsicRBF(nn1_out[:, 0, :], nn1_out[:, 1, :], device)
    floor = torch.tensor(0.01, device=device)
    hsic_val = torch.max(torch.stack([floor, hsic_val]))
    return nn.CrossEntropyLoss()(pred, gt) + lam * torch.log(hsic_val)


def evaluate_auc(loader, encoder, head, device):
    """Compute AUC on a DataLoader."""
    y_pred, y_true = [], []
    encoder.eval()
    head.eval()
    with torch.no_grad():
        for inputs, envs, targets in loader:
            y_true.extend(targets.numpy())
            inputs = inputs.double().to(device)
            envs = envs.to(device)
            zs = encoder(inputs)
            yhat, _, _ = head(zs, envs)
            y_pred.extend(yhat[:, 1].cpu().numpy())
    encoder.train()
    head.train()
    return roc_auc_score(y_true, y_pred)


def train_icp_cnn(
    encoder,
    head,
    optimizer,
    criterion,
    lam,
    trainloader,
    testloader,
    n_epochs,
    save_dir,
    file_tag,
    version,
    fold,
    rep,
    device,
    seed,
):
    """
    Train the ICP CNN (encoder + head) with checkpointing.

    Args:
        encoder: ICPEncoder instance
        head: ICPHead instance
        optimizer: torch optimizer
        criterion: loss function (causal_loss)
        lam: HSIC penalty weight
        trainloader: training DataLoader
        testloader: test DataLoader
        n_epochs: number of epochs
        save_dir: root directory for saving weights/losses
        file_tag: identifier string for filenames
        version: experiment version
        fold: CV fold index
        rep: repetition index
        device: torch device
        seed: random seed for reproducibility

    Returns:
        tuple of (encoder, head, (losses, ces, hsics, auc_tr, auc_te))
    """
    os.makedirs(os.path.join(save_dir, "losses"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "nn_weights"), exist_ok=True)

    loss_path = os.path.join(
        save_dir, "losses", f"losses_{file_tag}_{version}_f{fold}_r{rep}.pkl"
    )
    weights1_path = os.path.join(
        save_dir, "nn_weights", f"net1_{file_tag}_{version}_f{fold}_r{rep}"
    )
    weights2_path = os.path.join(
        save_dir, "nn_weights", f"net2_{file_tag}_{version}_f{fold}_r{rep}"
    )

    if os.path.exists(loss_path):
        with open(loss_path, "rb") as f:
            losses, ces, hsics, auc_tr, auc_te = pickle.load(f)
    else:
        losses, ces, hsics, auc_tr, auc_te = [], [], [], [], []

    torch.manual_seed(seed)

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_ce = 0.0
        running_hsic = 0.0
        batch_count = 0

        for inputs, envs, labels in trainloader:
            inputs = inputs.double().to(device)
            envs = envs.double().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            zs = encoder(inputs)
            outputs, _, _ = head(zs, envs)
            loss = criterion(
                outputs, zs, labels, lam, head.fc1a.weight[:, 8:13], device
            )

            hsic_val, _, _ = hsicRBF(zs[:, 0, :], zs[:, 1, :], device)
            ce = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += ce.item()
            running_hsic += hsic_val.item()
            batch_count += 1

        if batch_count == 0:
            continue

        epoch_loss = running_loss / batch_count
        if np.isnan(epoch_loss):
            break

        auc_train = evaluate_auc(trainloader, encoder, head, device)
        auc_test = evaluate_auc(testloader, encoder, head, device)

        losses.append(epoch_loss)
        ces.append(running_ce / batch_count)
        hsics.append(running_hsic / batch_count)
        auc_tr.append(auc_train)
        auc_te.append(auc_test)

        with open(loss_path, "wb") as f:
            pickle.dump(
                (losses, ces, hsics, auc_tr, auc_te), f, pickle.HIGHEST_PROTOCOL
            )
        torch.save(encoder.state_dict(), weights1_path)
        torch.save(head.state_dict(), weights2_path)

    return encoder, head, (losses, ces, hsics, auc_tr, auc_te)
