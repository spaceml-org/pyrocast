
import torchvision.transforms as transforms
import torch
import numpy as np
import utils.cube_matching as cm


class HimawariDataset(torch.utils.data.Dataset):
    """ Dataset object withs data, labels and transformations """

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


def load_loaders(train_list: list, test_list: list, parent_dir: str, event_id: str) -> torch.utils.data.DataLoader:
    """
    Return train and test loaders for CNN and AE-CNN

    Args:
        train_list (list): _description_
        test_list (list): _description_
        parent_dir (str): _description_
        event_id (str): _description_

    Returns:
        torch.utils.data.DataLoader: _description_
    """

    _, train_dataset_cubes, train_flags = cm.forecast_match(
        train_list, parent_dir, event_id)
    _, test_dataset_cubes, test_flags = cm.forecast_match(
        test_list, parent_dir, event_id)

    D = train_dataset_cubes.shape[1]

    # Transform values generated from means and variances of training set
    mu = [np.mean(train_dataset_cubes[:, i, :, :])
          for i in range(D)]
    sigma = [np.std(train_dataset_cubes[:, i, :, :])
             for i in range(D)]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mu, sigma)])

    batch_size = 64

    trainset = HimawariDataset(dataset_cubes=np.array(
        train_dataset_cubes), flags=train_flags, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = HimawariDataset(
        dataset_cubes=test_dataset_cubes, flags=test_flags, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def load_rfdata(train_list, test_list, parent_dir: str, event_id: str):
    """ Load data and calculates statistics for RF model """
    # Load data

    # Normalise

    xtrain, ytrain, xtest, ytest = [1, 1, 1, 1, ]

    return xtrain, ytrain, xtest, ytest
