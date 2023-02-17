import torch
from torch import nn 
import torch.nn.functional as F


# input img -> hidden dim -> mean ,std -> parametrization trick -> decoder -> output image
class VariationalAutoEncoder2(nn.Module):
    def __init__(self, input_dim, z_dim):
        super().__init__()
        # encoder
        self.img_mu = nn.Linear(input_dim, z_dim)
        self.img_sigma = nn.Linear(input_dim, z_dim)

        # decoder
        self.z_img = nn.Linear(z_dim, 784)

        
        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z/x)
        mu, sigma = self.img_mu(x), self.img_sigma(x)
        return mu, sigma


    def decoder(self, z):
        # p_theta(x/z)
        return torch.sigmoid(self.z_img(z))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(1, 28*28) 
    #vae = VariationalAutoEncoder(input_dim = 28*28,z_dim=6)
    vae = torch.load('weights6D')
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape )
    print(mu.shape )
    print(sigma.shape)
    epsilon = torch.randn_like(sigma)
    z = mu+sigma
    print(z)
    print(vae.decoder(z))



    
