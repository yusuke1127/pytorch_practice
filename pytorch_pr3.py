import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = 1
if device >= 0:
    device = "cuda:{}".format(device)
else:
    device = "cpu"

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
thetalist = torch.linspace(0, 2 * torch.pi, T + 1)
circledata = []

for theta in range(T):
    x = 2 + 2 * torch.cos(thetalist[theta])
    y = 2 + 2 * torch.sin(thetalist[theta])
    circledata.append([x, y])

circledata = torch.tensor(circledata)
circletargets = torch.roll(circledata, shifts=2)
circledata = circledata.unsqueeze(0)
circletargets = circletargets.unsqueeze(0)
    
circle_dataset = ArrayDataset(circledata, circletargets)

# Create DataLoaders
train_loader = DataLoader(circle_dataset)

# Change tensor to numpy (for output)
truedata = circledata.to("cpu").detach().numpy()[0]
outputdata = []

# Define RNNModel
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnncell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
         
    def forward(self, x, hidden):
        hidden = self.rnncell(x, hidden)
        output = hidden
        output = output.view(-1, 2)
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
        running_outputdata = []
        print('epoch:', epoch)
        optimizer.zero_grad()
        hidden = torch.zeros(hidden_size).to(device)
        running_loss = 0.
        for x_train, y_train in train_loader:
            for timestep in range(x_train.shape[1]):
                traindata = x_train[0, 0].to(device)
                targetdata = y_train[0, timestep].to(device)
                targetdata = targetdata.view(-1, 2)
                if timestep == 0:
                    output, hidden = model(traindata, hidden)
                else:
                    output, hidden = model(output[0], hidden)
                output = torch.atanh(output).to(device)
                output, hidden = output.to(device), hidden.to(device)
                loss = criterion(output, targetdata)
                running_loss += loss
                
                running_outputdata_cell = output.to("cpu").detach().numpy()
                running_outputdata.append(running_outputdata_cell)
                    
        running_loss.backward()
        optimizer.step()

        print(f'loss: {running_loss.item() / x_train.shape[1]:.6f}')
        losses.append(running_loss.item() / x_train.shape[1])
        
        
        if running_loss.item() / x_train.shape[1] < 0.003:
            for i in range(x_train.shape[1]):
                outputdata.append(running_outputdata[i])
            break
               
    return output, losses
        
   
# Define hyperparameters
input_size = 2  # Input size
hidden_size = 2  # Hidden layer size
output_size = 2  # Output size
learning_rate = 0.001  # Learning rate
num_epochs = 30000  # Number of epochs

# Initialize model, loss function, optimizer
model = RNNModel(input_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output, losses = train(num_epochs, model, criterion, optimizer)

outputdata = np.array([arr for arr in outputdata])
outputdata = np.array([inner[0] for inner in outputdata])

 
# Output the results
true_x = truedata[:, 0]
true_y = truedata[:, 1]
predicted_x = outputdata[:, 0]
predicted_y = outputdata[:, 1]

plt.figure(figsize=(10, 10))
plt.scatter(true_x, true_y, label="true")
plt.scatter(predicted_x, predicted_y, label="predicted")
plt.xlabel("x")
plt.ylabel("y")
plt.grid("true")
plt.legend()
plt.savefig("result_pytorch_pr3.png")