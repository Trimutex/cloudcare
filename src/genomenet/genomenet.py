import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Constants
BATCH_SIZE = 16
CLASSES = 1
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
        data = pd.read_csv(location, sep='\t', header=None)
        labels = data[0].values
        sequences = data[1].values
        labelsArray = np.zeros((len(sequences) * SEQ_LEN, CLASSES),
                               dtype=float)
        one_hot_encoded = np.zeros((SEQ_LEN*len(sequences), 4),
                                   dtype=float)
        for i, sequence in enumerate(sequences):
            for j, base in enumerate(sequence):
                labelsArray[i*SEQ_LEN + j] = int(labels[i])
                if base == 'A' or base == 'a':
                    one_hot_encoded[i*SEQ_LEN + j][0] = 1.
                    one_hot_encoded[i*SEQ_LEN + j][1] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][2] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][3] = 0.
                if base == 'C' or base == 'c':
                    one_hot_encoded[i*SEQ_LEN + j][0] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][1] = 1.
                    one_hot_encoded[i*SEQ_LEN + j][2] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][3] = 0.
                if base == 'G' or base == 'g':
                    one_hot_encoded[i*SEQ_LEN + j][0] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][1] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][2] = 1.
                    one_hot_encoded[i*SEQ_LEN + j][3] = 0.
                if base == 'T' or base == 't':
                    one_hot_encoded[i*SEQ_LEN + j][0] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][1] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][2] = 0.
                    one_hot_encoded[i*SEQ_LEN + j][3] = 1.
        return GenomeSet(one_hot_encoded, labelsArray)

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
        self.data = torch.stack([torch.from_numpy(data).float() for d in data])
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]
        return data, label


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, CLASSES)
        self.relu = F.relu

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
