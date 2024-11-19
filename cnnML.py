import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn_data_handling import MiniBatchSampler, MyDataset, train_data_path, valid_data_path, test_data_path


class CNN(nn.Module):
    def __init__(self, l3=True, l4=False, l5=False, dense2=False, dropout_p=0.2,
                kernels_per_layer = [16, 16, 16, 128, 128, 128, 128], kernel_size_per_layer = [1, 3, 3, 1, 1, 3, 5], dense_out_features=8):

        super(CNN, self).__init__()

        self.l3, self.l4, self.l5 = l3, l4, l5
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)
        self.has_dense2 = dense2

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=kernels_per_layer[0], kernel_size=kernel_size_per_layer[0], stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=kernels_per_layer[0], out_channels=kernels_per_layer[1], kernel_size=kernel_size_per_layer[1], stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=kernels_per_layer[1], out_channels=kernels_per_layer[2], kernel_size=kernel_size_per_layer[2], stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=kernels_per_layer[3], kernel_size=kernel_size_per_layer[3], stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=kernels_per_layer[3], out_channels=kernels_per_layer[4], kernel_size=kernel_size_per_layer[4], stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=kernels_per_layer[4], out_channels=kernels_per_layer[5], kernel_size=kernel_size_per_layer[5], stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=kernels_per_layer[5], out_channels=kernels_per_layer[6], kernel_size=kernel_size_per_layer[6], stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()
        self.flat_size = self.get_flat_size()

        self.dense1 = nn.Linear(in_features=self.flat_size, out_features=8)
        self.dense2 = nn.Linear(in_features=64, out_features=8)

    def get_flat_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 69)
            x = self.conv1(x)
            x = self.conv2(x)
            if self.l3:
                x = self.conv3(x)
                if self.l4:
                    x = self.conv4(x)
                    if self.l5:
                        x = self.conv5(x)
            return x.numel()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.l3:
            x = self.conv3(x)
            if self.l4:
                x = self.conv4(x)
                if self.l5:
                    x = self.conv5(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense1(x)
        if self.has_dense2:
            x = self.dense2(x)
        return x
    
    def predict_proba(self, x):
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        return np.array(x).tolist()[0]

    def __repr__(self):
        return "CNN"

def train(cnn, data_loader, optimizer, loss_f, device, epochs, print_loss=True):
    cnn.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in data_loader:
            output = cnn.forward(x)
            loss = loss_f(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if print_loss:
            print(f"Training epoch {epoch + 1}, loss: {epoch_loss / len(data_loader)}")

def validate(cnn : CNN, data_loader, loss_f, device, print_loss=True):
    cnn.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = cnn.forward(x)
            loss = loss_f(output, y)
            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == y).sum().item()

    accuracy = correct / len(data_loader.dataset)
    avg_loss = total_loss / len(data_loader)
    if print_loss:
        print(f"Validation loss: {avg_loss}, accuracy: {accuracy}")

    return accuracy