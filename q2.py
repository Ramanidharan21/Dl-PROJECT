import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
num_workers = 0
batch_size = 20
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
num_workers=num_workers)
class Autoencoder(nn.Module):
 def __init__(self, encoding_dim):
 super(Autoencoder, self).__init__()
 self.encoder = nn.Linear(784, encoding_dim)
 self.decoder = nn.Linear(encoding_dim, 784)
 def forward(self, x):
 out = F.relu(self.encoder(x))
 out = torch.sigmoid(self.decoder(out))
 return out
encoding_dim = 64
model = Autoencoder(encoding_dim).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 10
for epoch in range(1, n_epochs + 1):
 train_loss = 0.0
 model.train()
 for data in train_loader:
 images, _ = data
 images = images.view(images.size(0), -1).to(device)
 optimizer.zero_grad()
 outputs = model(images)
 loss = criterion(outputs, images)
 loss.backward()
 optimizer.step()
 train_loss += loss.item() * images.size(0)
 train_loss = train_loss / len(train_loader.dataset)
 print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
dataiter = iter(test_loader)
images, labels = next(dataiter)
images_flatten = images.view(images.size(0), -1).to(device)
output = model(images_flatten)
images = images.numpy()
output = output.view(batch_size, 1, 28, 28)
output = output.detach().cpu().numpy()
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(25,4))
for images, row in zip([images, output], axes):
 for img, ax in zip(images, row):
 ax.imshow(np.squeeze(img), cmap='gray')
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
plt.show()