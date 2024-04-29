import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Constants
BATCH_SIZE = 16
CLASSES = 4
SEQ_LEN = 120


class GenomeNet:
    def __init__(self, seed, epochs):
        self.seed = seed
        self.epochs = epochs
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = Net().to(self.device)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.CrossEntropyLoss()
        print(self.device)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
        print("Doing ", self.epochs, " epochs")

    def info(self):
        for data, label in network.train_loader:
            print("Loaded data details:")
            print("\tdata:", data)
            print("\tlabel:", label)
            print("\tdata shape:", data.shape)
            print("\tlabel shape:", label.shape)
            print("\tdata length:", len(data))
            print("\tlabel length:", len(label))
            break

    def one_hot_encoder(self, location):
        data = pd.read_csv(location, sep='\t', header=None)
        labels = np.array(data[0].values)
        sequences = data[1].values
        one_hot_encoded = np.zeros((len(sequences), 4, SEQ_LEN))
        for i, sequence in enumerate(sequences):
            for j, base in enumerate(sequence):
                if base == 'A' or base == 'a':
                    one_hot_encoded[i][0][j] = 1.
                    one_hot_encoded[i][1][j] = 0.
                    one_hot_encoded[i][2][j] = 0.
                    one_hot_encoded[i][3][j] = 0.
                if base == 'C' or base == 'c':
                    one_hot_encoded[i][0][j] = 0.
                    one_hot_encoded[i][1][j] = 1.
                    one_hot_encoded[i][2][j] = 0.
                    one_hot_encoded[i][3][j] = 0.
                if base == 'G' or base == 'g':
                    one_hot_encoded[i][0][j] = 0.
                    one_hot_encoded[i][1][j] = 0.
                    one_hot_encoded[i][2][j] = 1.
                    one_hot_encoded[i][3][j] = 0.
                if base == 'T' or base == 't':
                    one_hot_encoded[i][0][j] = 0.
                    one_hot_encoded[i][1][j] = 0.
                    one_hot_encoded[i][2][j] = 0.
                    one_hot_encoded[i][3][j] = 1.
        return GenomeSet(one_hot_encoded, labels)

    def load(self, location):
        self.train_dataset = self.one_hot_encoder(location + "/train-light.dna")
        self.test_dataset = self.one_hot_encoder(location + "/test-light.dna")
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)

    def train(self, current_epoch):
        print("inside train")
        self.model.train()
        for batch_ids, (input, label) in enumerate(self.train_loader):
            label = label.type(torch.LongTensor)
            input, label = input.to(self.device), label.to(self.device)
            torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()
        if (batch_ids + 1) % 2 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                current_epoch,
                batch_ids * len(input),
                len(self.train_loader.dataset),
                100.*batch_ids / len(self.train_loader),
                loss.item()))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for input, label in self.test_loader:
                label = label.type(torch.LongTensor)
                input, label = input.to(self.device), label.to(self.device)
                y_hat = self.model(input)
                test_loss += F.nll_loss(y_hat, label, reduction='sum').item()
                _, y_pred = torch.max(y_hat, 1)
                correct += (y_pred == label).sum().item()
            test_loss /= len(self.test_dataset)
            print("\nTest set: Average loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n"
                  .format(test_loss, correct, len(self.test_dataset),
                          100.*correct/len(self.test_dataset)))
            print('='*30)


class GenomeSet(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        return data, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv01 = nn.Conv1d(in_channels=4, out_channels=96,
                                kernel_size=11, stride=4)
        self.pool01 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv02 = nn.Conv1d(in_channels=96, out_channels=256,
                                kernel_size=5, padding=2)
        self.pool02 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv03 = nn.Conv1d(in_channels=256, out_channels=384,
                                kernel_size=3, padding=1)
        self.conv04 = nn.Conv1d(in_channels=384, out_channels=384,
                                kernel_size=3, padding=1)
        self.conv05 = nn.Conv1d(in_channels=384, out_channels=BATCH_SIZE,
                                kernel_size=3, padding=1)
        self.pool05 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(32, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, CLASSES)

    def forward(self, x):
        x = self.pool01(F.relu(self.conv01(x)))
        x = self.pool02(F.relu(self.conv02(x)))
        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.pool05(F.relu(self.conv05(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x
