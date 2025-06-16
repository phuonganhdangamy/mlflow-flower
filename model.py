import torch
import torch.nn as nn
import torch.nn.functional as F


class ElectricityModel(nn.Module):
    """Custom model for electricity consumption prediction."""

    def __init__(self, input_dim):
        super(ElectricityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)  # Output layer for regression

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.Sigmoid()(x)


def train(model, trainloader, epochs, criterion, optimizer, device):
    """Train the model on the training set."""
    model.train()
    for _ in range(epochs):
        for batch in trainloader:
            data, target = batch
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()


def evaluate(model, testloader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            data, target = batch
            output = model(data.to(device))
            loss += criterion(output, target.to(device)).item()
    loss /= len(testloader.dataset)
    return loss