import utils.dataprep as dp
from utils import metrics
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np


class CNN():
    """ CNN pyroCb forecast model """

    def __init__(self):
        super().__init__(n_channels)
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


if __name__ in "__main__":

    # Load data
    trainloader, testloader = dp.load_loaders()
    n_channels = trainloader

    # Initialise model
    cnn = CNN(n_channels)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9))

    # Train

    # Load data onto GPU
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn.to(device)

    n_epochs=100
    losses=np.zeros((n_epochs,)).astype(float)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss=0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels=data
            print(input.shape)
            inputs=inputs.type(torch.DoubleTensor)
            labels=labels.type(torch.DoubleTensor)
            inputs=inputs.to(device)
            labels=labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs=cnn(inputs)

            outputs=outputs.squeeze(1)
            # outputs = outputs.type(torch.LongTensor)
            # labels = labels.type(torch.LongTensor)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.6f}')
                running_loss=0.0
        losses[epoch]=loss.item()

    # Test
    y_pred=[]
    y_true=[]

    for i, (inputs, targets) in enumerate(testloader, 0):
        y_true.extend(targets.cpu().detach().numpy())
        yhat=cnn(inputs.to(device))
        y_pred.extend(yhat.cpu().detach().numpy())

    auc=metrics.auc(y_true, y_pred)
    print('AUC', auc)
