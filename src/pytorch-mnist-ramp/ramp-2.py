#!/usr/bin/env python

import torch
# import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np

# Constants
BATCH_SIZE = 16
CLASSES = 10


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv01 = nn.Conv2d(in_channels=1, out_channels=96,
                                kernel_size=11, stride=4, padding=0)
        self.conv02 = nn.Conv2d(in_channels=96, out_channels=96,
                                kernel_size=1)
        self.conv03 = nn.Conv2d(in_channels=96, out_channels=96,
                                kernel_size=1)
        self.pool03 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv04 = nn.Conv2d(in_channels=96, out_channels=256,
                                kernel_size=11, stride=4, padding=2)
        self.conv05 = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=1)
        self.pool05 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv06 = nn.Conv2d(in_channels=256, out_channels=384,
                                kernel_size=3, stride=1, padding=1)
        self.conv07 = nn.Conv2d(in_channels=384, out_channels=384,
                                kernel_size=1)
        self.conv08 = nn.Conv2d(in_channels=384, out_channels=384,
                                kernel_size=1)
        self.fc1 = nn.Linear(in_features=(2*2*384), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=CLASSES)

    def forward(self, x):
        # Layer 01
        x = F.relu(self.conv01(x))
        # Layer 02
        x = F.relu(self.conv02(x))
        # Layer 03
        x = self.pool03(F.relu(self.conv03(x)))
        x = F.dropout(x, 0.5)
        # Layer 04
        x = F.relu(self.conv04(x))
        # Layer 05
        x = self.pool05(F.relu(self.conv05(x)))
        x = F.dropout(x, 0.5)
        # Layer 06
        x = F.relu(self.conv06(x))
        # Layer 07
        x = F.relu(self.conv07(x))
        # Layer 08
        x = F.relu(self.conv08(x))
        x = F.dropout(x, 0.5)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # Layer 09
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        # Layer 10
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        # Layer 11
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epochs):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_loader):
        classes = classes.type(torch.LongTensor)
        img, classes = img.to(device), classes.to(device)
        torch.autograd.set_detect_anomaly(True)
        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, classes)

        loss.backward()
        optimizer.step()
    if (batch_ids + 1) % 2 == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_ids * len(img), len(train_loader.dataset),
            100.*batch_ids / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, classes in test_loader:
            img, classes = img.to(device), classes.to(device)
            y_hat = model(img)
            test_loss += F.nll_loss(y_hat, classes, reduction='sum').item()
            _, y_pred = torch.max(y_hat, 1)
            correct += (y_pred == classes).sum().item()
        test_loss /= len(test_dataset)
        print("\n Test set: Avarage loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n"
              .format(test_loss, correct, len(test_dataset),
                      100.*correct/len(test_dataset)))
        print('='*30)


# Global model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

transform_conf = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train_dataset = datasets.MNIST('~/temp/data/', train=True, download=True,
                               transform=transform_conf,)
test_dataset = datasets.MNIST('~/temp/data/', train=False, download=True,
                              transform=transform_conf)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

model = Net().to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


if __name__ == '__main__':
    seed = 42
    EPOCHS = 2

    for epoch in range(1, EPOCHS+1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
