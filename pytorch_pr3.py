import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Define Dataset
class ArrayDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

# Create an instance of the Dataset
T = 50
thetalist = torch.linspace(0, 2 * torch.pi, T)
circledata = []

for theta in range(T):
    x = torch.cos(thetalist[theta])
    y = torch.sin(thetalist[theta])
    circledata.append([x, y])

circledata = torch.tensor(circledata)
circletargets = torch.roll(circledata, shifts=2)
circledata = circledata.unsqueeze(0)
circletargets = circletargets.unsqueeze(0)

    
circle_dataset = ArrayDataset(circledata, circletargets)


# Create DataLoaders
train_loader = DataLoader(circle_dataset)

# Define RNNModel
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = nn.RNNCell(input_size, hidden_size)
    
    def forward(self, x, hidden):
        hidden = self.rnncell(x, hidden)
        output = hidden
        return output, hidden
"""
class RNN_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = RNNModel(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        output, h = self.rnn(x, hidden)
        return output, h
"""

def train(epochs, model, criterion, optimizer):
    losses = []

    for epoch in range(epochs):
        print('epoch:', epoch)
        optimizer.zero_grad()
        hidden = torch.zeros(hidden_size)
        for x_train, y_train in train_loader:
            for timestep in range(len(x_train)):
                running_loss = 0.
                x_train = x_train[timestep]
                x_train = x_train[None]
                y_train = y_train[timestep]
                y_train = y_train[None]
                output, hidden = model(x_train, hidden)
                loss = criterion(output, y_train)
                running_loss += loss
            running_loss.backward()
            optimizer.step()

        print(f'loss: {running_loss.item() / len(x_train):.6f}')
        losses.append(running_loss.item() / len(train_loader[0]))

    return output, losses
        
# --------------coding...-------------------------        
        
# Set device
device = 1
if device >= 0:
    device = "cuda:{}".format(device)
else:
    device = "cpu"
    
# Define hyperparameters
input_size = 2  # Input size
hidden_size = 2  # Hidden layer size
output_size = 1  # Output size
learning_rate = 0.001  # Learning rate
num_epochs = 3  # Number of epochs

# Initialize model, loss function, optimizer
model = RNNModel(input_size, hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output, losses = train(num_epochs, model, criterion, optimizer)
