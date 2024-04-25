import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Constants
BATCH_SIZE = 16
CLASSES = 10


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
        self.transform_conf = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def info(self):
        print(self.device)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))

    def one_hot_encoder(self, location):
        # Encode here
        label, sequence = np.genfromtxt(location, delimiter='\t')
        for i in range(0, 5):
            print(label[i], sequence[i])
        return sequence

    def load(self, location):
        train_dataset = self.one_hot_encoder(location + "/train.dna")
        test_dataset = self.one_hot_encoder(location + "/test.dna")
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True)

    def train(self, current_epoch):
        print("inside train")
        self.model.train()
        for batch_ids, (img, classes) in enumerate(self.train_loader):
            classes = classes.type(torch.LongTensor)
            img, classes = img.to(self.device), self.classes.to(self.device)
            torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            output = self.model(img)
            loss = self.loss_fn(output, classes)
            loss.backward()
            self.optimizer.step()
        if (batch_ids + 1) % 2 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                current_epoch,
                batch_ids * len(img),
                len(self.train_loader.dataset),
                100.*batch_ids / len(self.train_loader),
                loss.item()))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, classes in self.test_loader:
                img, classes = img.to(self.device), classes.to(self.device)
                y_hat = self.model(img)
                test_loss += F.nll_loss(y_hat, classes, reduction='sum').item()
                _, y_pred = torch.max(y_hat, 1)
                correct += (y_pred == classes).sum().item()
            test_loss /= len(self.test_dataset)
            print("\nTest set: Average loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n"
                  .format(test_loss, correct, len(self.test_dataset),
                          100.*correct/len(self.test_dataset)))
            print('='*30)


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
        self.conv09 = nn.Conv2d(in_channels=384, out_channels=10,
                                kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=10, out_channels=10,
                                kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels=10, out_channels=10,
                                kernel_size=1)

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
        # Layer 09
        x = F.relu(self.conv09(x))
        # Layer 10
        x = F.relu(self.conv10(x))
        # Layer 11
        x = F.relu(self.conv11(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x
