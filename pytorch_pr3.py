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
    
circle_dataset = ArrayDataset(circledata, circletargets)

# Create DataLoaders
train_loader = DataLoader(circle_dataset)
test_loader = DataLoader(circle_dataset)


# Define RNNModel
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = nn.RNNCell(input_size, hidden_size)
    
    def forward(self, x, hidden):
        count = len(x)
        output = torch.Tensor()
        for idx in range(count):
            hidden = self.rnncell(x[idx], hidden)
            output = torch.cat(output, hidden)
        return output, hidden

class RNN_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = RNNModel(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        output, h = self.rnn(x, hidden)
        return output, h

net = RNN_Net(1, 64, 1)

def train(epochs, model, x_train, y_train, criterion, optimizer):
    losses = []

    for epoch in range(epochs):
        print('epoch:', epoch)
        optimizer.zero_grad()
        hidden = torch.zeros(50, hidden_size)
        output, hidden = model(x_train, hidden)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        print(f'loss: {loss.item() / len(x_train):.6f}')
        losses.append(loss.item() / len(x_train))

    return output, losses
        
# --------------coding...-------------------------        
        
# Set device
device = 1
if device >= 0:
    device = "cuda:{}".format(device)
else:
    device = "cpu"
    
# Define hyperparameters
input_size = 1  # Input size
hidden_size = 64  # Hidden layer size
output_size = 1  # Output size
learning_rate = 0.001  # Learning rate
num_epochs = 10  # Number of epochs

# Initialize model, loss function, optimizer
model = RNN_Net(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output, losses = train(num_epochs, model, train_loader, test_loader, criterion, optimizer)
