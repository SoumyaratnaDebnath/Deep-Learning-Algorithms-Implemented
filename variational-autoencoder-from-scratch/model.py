import torch
import torch.nn.functional as F
from torch import nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # for encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_sigma = nn.Linear(h_dim, z_dim)

        # for decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()
    
    def encode(self, x):
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_sigma(h)
        return mu, sigma

    def decode(self, z):
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h)) 
        # sigmoid for making sure that the pixel values are between 0 and 1 (normalized mnist)

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma*epsilon
        x_reconstucted = self.decode(z_reparameterized)
        return x_reconstucted, mu, sigma 
        # x_reconstructed for the construction loss, and mu & sigma for KL divergence

if __name__ == "__main__":
    x = torch.randn(4, 28*28) # batch_size = 4
    vae = VariationalAutoencoder(input_dim=28*28)
    x_recons, mu, sigma = vae(x)
    print(x_recons.shape)
