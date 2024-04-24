import torch
from torch.utils.data import Dataset, DataLoader
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
         
    def forward(self, x, hidden):
        hidden = self.rnncell(x, hidden)
        output = torch.atanh(hidden)
        output = output.view(-1, 2)
        return output, hidden

def train(epochs, model, criterion, optimizer):
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch + 1))
        optimizer.zero_grad()
        hidden = torch.zeros(hidden_size).to(device)
        running_loss = 0.
        for x_train, y_train in train_loader:
            for timestep in range(x_train.shape[1]):
                traindata = x_train[0, timestep].to(device)
                targetdata = y_train[0, timestep].to(device)
                targetdata = targetdata.view(-1, 2)
                
                output, hidden = model(traindata, hidden)
                output, hidden = output.to(device), hidden.to(device)
                loss = criterion(output, targetdata)
                running_loss += loss
            
        running_loss.backward()
        optimizer.step()
                
        
        print('loss: {:.8f}'.format(running_loss.item() / x_train.shape[1]))
        
        if running_loss.item() / x_train.shape[1] < 1e-6:
            break
            
    return output
        

def test(model):
    hidden = torch.zeros(hidden_size).to(device)
    
    with torch.no_grad():
        for x_test, y_test in train_loader:
            for timestep in range(x_test.shape[1]):
                if timestep == 0:
                    input_co = x_test[0, 0].to(device)
                else:
                    input_co = predicted_co.to(device)
                
                predicted_co, hidden = model(input_co, hidden)
                predicted_co = predicted_co[0]
                predicted_co, hidden = predicted_co.to(device), hidden.to(device)
                outputdata_cell = predicted_co.to("cpu").detach().numpy()
                outputdata.append(outputdata_cell)
             
    

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



output = train(num_epochs, model, criterion, optimizer)

test(model)

outputdata = np.array([arr for arr in outputdata])

 
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