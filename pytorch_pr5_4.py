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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        return x

# Create an instance of the Dataset
T = 50
thetalist = torch.linspace(0, 2 * torch.pi, T)
lissajousdata = []

for theta in range(T):
    x = torch.sin(thetalist[theta])
    y = torch.sin(2 * thetalist[theta]) * (-1)
    lissajousdata.append([x, y])

lissajousdata = torch.tensor(lissajousdata)
lissajousdata = lissajousdata.unsqueeze(0)
    
lissajous_dataset = ArrayDataset(lissajousdata)

# Create DataLoaders
train_loader = DataLoader(lissajous_dataset)

# Change tensor to numpy (for output)
truedata = lissajousdata.to("cpu").detach().numpy()[0]

# Define FNNModel
class FNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
         
    def forward(self, x):
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x) 
        return x

def train(epochs, model, criterion, optimizer):
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch + 1))
        optimizer.zero_grad()
        running_loss = 0.
        for x_train in train_loader:
            for timestep in range(x_train.shape[1] - 4):
                traindata = x_train[0, timestep]
                for i in range(1, 4):
                    traindata = torch.cat((traindata, x_train[0, timestep + i]))
                traindata = traindata.to(device)
                targetdata = x_train[0, timestep + 4].to(device)
                targetdata = targetdata.view(-1, 2)
                
                output = model(traindata)
                output = output.to(device)
                loss = criterion(output, targetdata[0])
                running_loss += loss

        running_loss.backward()
        optimizer.step()
        
        print('loss: {:.8f}'.format(running_loss.item() / x_train.shape[1]))
        
        if running_loss.item() / x_train.shape[1] < 3e-6:
            break
    
    return output, running_loss


def test(model):
    outputdata = []
    with torch.no_grad():
        for x_test in train_loader:
            for timestep in range(x_test.shape[1] - 4):
                if timestep == 0:
                    input_co = x_test[0, 0]
                    outputdata.append(input_co.to("cpu").detach().numpy())
                    for i in range(1, 4):
                        input_co = torch.cat((input_co, x_test[0, timestep + i]))
                        outputdata.append(x_test[0, timestep + i].to("cpu").detach().numpy())
                input_co = input_co.to(device)
                predicted_co = model(input_co)
                for i in range(6):
                    input_co[i] = input_co[i + 2]
                input_co[6], input_co[7] = predicted_co[0], predicted_co[1]
                predicted_co = predicted_co.to(device)
                
                outputdata_cell = predicted_co.to("cpu").detach().numpy()
                outputdata.append(outputdata_cell)
    
    return outputdata

# Define hyperparameters
input_size = 8  # Input size
hidden_size = 32  # Hidden layer size
output_size = 2  # Output size
learning_rate = 0.001  # Learning rate
num_epochs = 30000  # Number of epochs

# Initialize model, loss function, optimizer
model = FNNModel(input_size, hidden_size, output_size)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output, losses = train(num_epochs, model, criterion, optimizer)

outputdata = test(model)
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
plt.savefig("result_pytorch_pr5_4.png")