import torch
from torch import nn 
import torch.nn.functional as F


# input img -> hidden dim -> mean ,std -> parametrization trick -> decoder -> output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, z_dim):
        super().__init__()
        # encoder
        self.img_2hid1 = nn.Linear(input_dim, h1_dim)
        self.hid1_2hid2 = nn.Linear(h1_dim, h2_dim)
        self.hid2_2mu = nn.Linear(h2_dim, z_dim)
        self.hid2_2sigma   = nn.Linear(h2_dim, z_dim)

        # decoder
        self.z_2hid2 = nn.Linear(z_dim, h2_dim)
        self.hid2_2hid1 = nn.Linear(h2_dim, h1_dim)
        self.hid1_2img = nn.Linear(h1_dim, input_dim)
        
        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z/x)
        h1 = self.relu(self.img_2hid1(x))
        h2 = self.relu(self.hid1_2hid2(h1))
        mu, sigma = self.hid2_2mu(h2), self.hid2_2sigma(h2)
        return mu, sigma


    def decoder(self, z):
        # p_theta(x/z)
        h2 = self.relu(self.z_2hid2(z))
        h1 = self.relu(self.hid2_2hid1(h2))
        return torch.sigmoid(self.hid1_2img(h1))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma
    

