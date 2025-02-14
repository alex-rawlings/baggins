import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt

# Parameters
map_size = 85  # Update to 85x85 pixel map
latent_dim = 100  # Dimensionality of noise input

# Generator Model using Convolutional Layers
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=6, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=6, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), latent_dim, 1, 1)
        return self.model(z).squeeze(1)

# Discriminator Model using Convolutional Layers
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 1),  # Updated for 85x85 image size
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.unsqueeze(1)  # Add channel dimension for grayscale image
        return self.model(img)

# Plot helper
def plot_generated_map(generator, epoch):
    noise = torch.randn(1, latent_dim)
    with torch.no_grad():
        generated_map = generator(noise).squeeze(0).numpy()
    plt.imshow(generated_map, cmap='gray')
    plt.colorbar()
    plt.savefig(f"gan_output_{epoch}.png", dpi=300)

# Training Function
def train_gan(generator, discriminator, epochs=10000, batch_size=64):
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Load training data from "data.txt"
    real_maps = np.loadtxt("mag_data.txt")
    real_maps = real_maps.reshape(-1, map_size, map_size)  # Ensure proper shape
    real_maps = torch.tensor(real_maps, dtype=torch.float32) * 2 - 1  # Normalize to [-1, 1]

    for epoch in range(epochs):
        # Training the Discriminator
        idx = np.random.randint(0, real_maps.size(0), batch_size)
        real_samples = real_maps[idx]

        # Labels for real and fake samples
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Generate a batch of fake samples
        noise = torch.randn(batch_size, latent_dim)
        fake_samples = generator(noise)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_samples), real_labels)
        fake_loss = adversarial_loss(discriminator(fake_samples.detach()), fake_labels)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = adversarial_loss(discriminator(fake_samples), real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            plot_generated_map(generator, epoch)

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()

    train_gan(generator, discriminator)
