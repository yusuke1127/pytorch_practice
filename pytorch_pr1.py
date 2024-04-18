# task1

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable 
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define custom dataset class
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=transforms.ToTensor())
        else:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                         transform=transforms.ToTensor())

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label

    def __len__(self):
        return len(self.dataset)


# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn1(self.conv1_1(x))
        x = self.pool(self.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.bn2(self.conv2_1(x))
        x = self.pool(self.relu(x))
        x = self.bn3(self.conv3(x))
        x = self.bn3(self.conv3_1(x))
        x = self.pool(self.relu(x))
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Function to train the model
def train_and_test_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, running_loss / len(train_loader)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('correct: %d ' % correct)
    print('total: %d ' % total)
    print('Accuracy on test set: %d %%' % accuracy)
    return accuracy


# Load data
train_dataset = CustomCIFAR10Dataset(train=True)
test_dataset = CustomCIFAR10Dataset(train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
device = 1
if device >= 0:
    device = "cuda:{}".format(device)
else:
    device = "cpu"
model = CNNModel()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Train the model
accuracy = train_and_test_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)

# Check if accuracy meets the desired threshold
if accuracy >= 80:
    print("Desired accuracy achieved!")
else:
    print("Desired accuracy not achieved. Further optimization may be needed.")