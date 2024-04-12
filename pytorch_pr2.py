# task2

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20)  # Reduce to 20-dimensional features
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()  # To output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

device = 0
if device >= 0:
    device = "cuda:{}".format(device)
else:
    device = "cpu"

# Initialize the model, loss function, and optimizer
model = Autoencoder()
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        img = img.view(img.size(0), -1)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the autoencoder on some images
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for data in test_loader:
        images, _ = data
        images = images.to(device)
        images = images.view(images.size(0), -1)
        reconstructed = model(images)
        original_images = images.view(-1, 28, 28)
        reconstructed_images = reconstructed.view(-1, 28, 28)
        plt.figure(figsize=(9, 2))
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(original_images[i].to("cpu"), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(2, 5, i + 6)
            plt.imshow(reconstructed_images[i].to("cpu"), cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')
        plt.savefig("figure1")
        break  # Only show the first batch of images