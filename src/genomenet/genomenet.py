import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Constants
BATCH_SIZE = 64
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

    def info(self):
        print(self.device)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
        print("Doing ", self.epochs, " epochs")

    def one_hot_encoder(self, location):
        # Encode here
        data = pd.read_csv(location, sep='\t', header=None)
        labels = data[0].values
        sequences = data[1].values
        one_hot_encoded = np.zeros((len(sequences)*SEQ_LEN, CLASSES))
        bases = ['A', 'C', 'G', 'T']
        for i, sequence in enumerate(sequences):
            for base in sequence:
                if base == 'A' or base == 'a':
                    one_hot_encoded[0, i] = True
                if base == 'C' or base == 'c':
                    one_hot_encoded[1, i] = True
                if base == 'G' or base == 'g':
                    one_hot_encoded[2, i] = True
                if base == 'T' or base == 't':
                    one_hot_encoded[3, i] = True
        return GenomeSet(one_hot_encoded, labels)

    def load(self, location):
        train_dataset = self.one_hot_encoder(location + "/train-light.dna")
        test_dataset = self.one_hot_encoder(location + "/test-light.dna")
        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                       shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                      shuffle=True)

    def train(self, current_epoch):
        print("inside train")
        self.model.train()
        for batch_ids, (input, label) in enumerate(self.train_loader):
            input = input.to(self.device).type(torch.float)
            label = label.to(self.device).type(torch.float)
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
                input = input.to(self.device)
                label = label.to(self.device)
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
        self.data = torch.stack([torch.from_numpy(data) for d in data])
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        return data, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Conv1d(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(6, CLASSES)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
