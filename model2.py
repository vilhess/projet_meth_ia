import torch
from torch import nn 
import torch.nn.functional as F


# input img -> hidden dim -> mean ,std -> parametrization trick -> decoder -> output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, z_dim):
        super().__init__()
        # encoder
        self.img_2hid1 = nn.Linear(input_dim, h1_dim)
        self.hid1_2hid2 = nn.Linear(h1_dim, h2_dim)
        self.hid2_2hid3 = nn.Linear(h2_dim, h3_dim)
        self.hid3_2hid4 = nn.Linear(h3_dim, h4_dim)
        self.hid4_2hid5 = nn.Linear(h4_dim, h5_dim)
        self.hid2_2mu = nn.Linear(h5_dim, z_dim)
        self.hid2_2sigma   = nn.Linear(h5_dim, z_dim)

        # decoder
        self.z_2hid5 = nn.Linear(z_dim, h5_dim)
        self.hid5_2hid4 = nn.Linear(h5_dim, h4_dim)
        self.hid4_2hid3 = nn.Linear(h4_dim, h3_dim)
        self.hid3_2hid2 = nn.Linear(h3_dim, h2_dim)
        self.hid2_2hid1 = nn.Linear(h2_dim, h1_dim)
        self.hid1_2img = nn.Linear(h1_dim, input_dim)
        
        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        # q_phi(z/x)
        h1 = self.relu(self.img_2hid1(x))
        h2 = self.relu(self.hid1_2hid2(h1))
        h3 = self.relu(self.hid2_2hid3(h2))
        h4 = self.relu(self.hid3_2hid4(h3))
        h5 = self.relu(self.hid4_2hid5(h4))
        mu, sigma = self.hid2_2mu(h5), self.hid2_2sigma(h5)
        return mu, sigma


    def decoder(self, z):
        # p_theta(x/z)
        h5 = self.relu(self.z_2hid5(z))
        h4 = self.relu(self.hid5_2hid4(h5))
        h3 = self.relu(self.hid4_2hid3(h4))
        h2 = self.relu(self.hid3_2hid2(h3))
        h1 = self.relu(self.hid2_2hid1(h2))
        return torch.sigmoid(self.hid1_2img(h1))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma


if __name__ == "__main__":
    x = torch.randn(4, 28*28) 
    vae = VariationalAutoEncoder(input_dim = 28*28, h1_dim = 200, h2_dim = 100, h3_dim = 50, h4_dim = 25, h5_dim = 10, z_dim=2)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape )
    print(mu.shape )
    print(sigma.shape )



    
