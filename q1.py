import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
# Image transforms
low_res_size = 128
high_res_size = 256
transform_low = transforms.Compose([transforms.Resize((low_res_size, low_res_size)), 
transforms.ToTensor()])
transform_high = transforms.Compose([transforms.Resize((high_res_size, high_res_size)), 
transforms.ToTensor()])
# Dataset class
class ImageDataset(Dataset):
 def __init__(self, root_dir):
 self.data = list(zip(os.listdir(os.path.join(root_dir, "low_res")), 
 os.listdir(os.path.join(root_dir, "high_res"))))
 self.root_dir = root_dir
 def __len__(self):
 return len(self.data)
 def __getitem__(self, index):
 low_res = Image.open(os.path.join(self.root_dir, "low_res", 
self.data[index][0])).convert("RGB")
 high_res = Image.open(os.path.join(self.root_dir, "high_res", 
self.data[index][1])).convert("RGB")
 return transform_low(low_res), transform_high(high_res)
# VGG-based perceptual loss
class VGGPerceptualLoss(nn.Module):
 def __init__(self):
 super().__init__()
 self.vgg = vgg19(pretrained=True).features[:25].eval()
 self.loss = nn.MSELoss()
 def forward(self, x, y):
 return self.loss(self.vgg(x), self.vgg(y))
# Generator class
class Generator(nn.Module):
 def __init__(self, num_channels=64, num_blocks=8):
 super().__init__()
 self.initial = ConvBlock(3, num_channels, kernel_size=7, stride=1, padding=3, 
use_BatchNorm=False)
 self.res_blocks = nn.Sequential(*[ResidualBlock(num_channels) for _ in 
range(num_blocks)])
 self.conv = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, 
padding=1, use_activation=False)
 self.upsample = UpsampleBlock(num_channels, scale_factor=2)
 self.final = nn.Conv2d(num_channels, 3, kernel_size=9, stride=1, padding=4)
 def forward(self, x):
 initial = self.initial(x)
 res = self.res_blocks(initial)
 return torch.sigmoid(self.final(self.upsample(self.conv(res) + initial)))
# Discriminator class
class Discriminator(nn.Module):
 def __init__(self, num_channels=64, features=[64, 128, 256, 512]):
 super().__init__()
 layers = []
 for idx, feature in enumerate(features):
 layers.append(
 ConvBlock(3 if idx == 0 else features[idx-1], feature, kernel_size=3, stride=2, 
padding=1)
 )
 self.model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(), 
nn.Linear(512*8*8, 1))
 def forward(self, x):
 return self.model(x)
# ConvBlock used for both Generator and Discriminator
class ConvBlock(nn.Module):
 def __init__(self, in_channels, out_channels, **kwargs):
 super().__init__()
 self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
 self.bn = nn.BatchNorm2d(out_channels)
 self.activation = nn.LeakyReLU(0.2, inplace=True)
 def forward(self, x):
 return self.activation(self.bn(self.conv(x)))
# Residual block for Generator
class ResidualBlock(nn.Module):
 def __init__(self, channels):
 super().__init__()
 self.block = nn.Sequential(
 ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
 ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, 
use_activation=False)
 )
 def forward(self, x):
 return x + self.block(x)
# Upsample block for Generator
class UpsampleBlock(nn.Module):
 def __init__(self, in_channels, scale_factor):
 super().__init__()
 self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
 self.ps = nn.PixelShuffle(scale_factor)
 self.activation = nn.PReLU()
 def forward(self, x):
 return self.activation(self.ps(self.conv(x)))
# Training function
def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, device):
 loop = tqdm(loader)
 for low_res, high_res in loop:
 low_res, high_res = low_res.to(device), high_res.to(device)
 # Train Discriminator
 fake = gen(low_res)
 disc_real = disc(high_res)
 disc_fake = disc(fake.detach())
 disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
 disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
 disc_loss = disc_loss_real + disc_loss_fake
 opt_disc.zero_grad(); disc_loss.backward(); opt_disc.step()
 # Train Generator
 disc_fake = disc(fake)
 adv_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
 perceptual_loss = 0.006 * vgg_loss(fake, high_res)
 gen_loss = adv_loss + perceptual_loss
 opt_gen.zero_grad(); gen_loss.backward(); opt_gen.step()
# Plotting function
def test_and_plot(val_loader, gen, device):
 gen.eval()
 with torch.no_grad():
 for idx, (low_res, high_res) in enumerate(val_loader):
 if idx == 2: break
 low_res, high_res = low_res.to(device), high_res.to(device)
 fake_high_res = gen(low_res)
 low_res, high_res, fake_high_res = low_res.cpu(), high_res.cpu(), fake_high_res.cpu()
 fig, axs = plt.subplots(1, 3, figsize=(15, 5))
 axs[0].imshow(low_res[0].permute(1, 2, 0)); axs[0].set_title("Low Resolution")
 axs[1].imshow(fake_high_res[0].permute(1, 2, 0)); axs[1].set_title("Generated")
 axs[2].imshow(high_res[0].permute(1, 2, 0)); axs[2].set_title("High Resolution")
 plt.show()
# Main
if __name__ == "__main__":
 device = "cuda" if torch.cuda.is_available() else "cpu"
 gen, disc = Generator().to(device), Discriminator().to(device)
 opt_gen, opt_disc = optim.Adam(gen.parameters(), lr=3e-4), optim.Adam(disc.parameters(), 
lr=3e-4)
 vgg_loss = VGGPerceptualLoss().to(device)
 mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()
 train_data = ImageDataset(root_dir="dataset/train")
 val_data = ImageDataset(root_dir="dataset/val")
 train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
 val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
 for epoch in range(10):
 print(f"Epoch {epoch+1}/10")
 train_fn(train_loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss, device)
 test_and_plot(val_loader, gen, device)
 torch.save(gen.state_dict(), "generator.pth")
 torch.save(disc.state_dict(), "discriminator.pth")