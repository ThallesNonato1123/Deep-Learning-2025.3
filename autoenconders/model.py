import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        
        # Estimate mean and log variance
        self.fc1 = nn.Linear(64*7*7, 400)  # 7x7 feature maps
        self.fc2_mean = nn.Linear(400, latent_dim)
        self.fc2_logvar = nn.Linear(400, latent_dim)
    
    def forward(self, x: torch.Tensor):
        # Feature extraction
        x = self.feature_extractor(x)

        # Estimate mean and log variance
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar
    
class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        # Transform latent variables to a suitable shape for later upsampling
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64*7*7),
            nn.ReLU(),
        )
        
        # Upsampling with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Ensuring output is in [0,1]
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z)

        z = z.view(z.size(0), 64, 7, 7)

        x_recon = self.decoder(z)
        return x_recon
    
class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        # Pass the input through the encoder
        mu, logvar = self.encoder(x)
        
        # Reparameterization step
        z = self.reparameterize(mu, logvar)
        
        # Pass the latent vector through the decoder
        x_reconstructed = self.decoder(z)
        
        return x_reconstructed, mu, logvar
        
