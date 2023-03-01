import torch
from torch import nn 
import torch.nn.functional as F


# input img -> hidden dim -> mean ,std -> parametrization trick -> decoder -> output image

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()

        convFeatures = (20, 30)
        self.convFeatures = convFeatures
        hiddenNum = 50

        # ENCODER
        # convolute -> maxpool -> convolute -> maxpool -> fully connected x2
        self.encodeConv1 = nn.Conv2d(in_channels=1, 
                               out_channels=convFeatures[0], 
                               kernel_size=3, 
                               padding=True)

        self.encodeMaxPool = nn.MaxPool2d(kernel_size=2, 
                                    return_indices=True)

        self.encodeConv2 = nn.Conv2d(in_channels=convFeatures[0], 
                               out_channels=convFeatures[1], 
                               kernel_size=3, 
                               padding=True)

        self.totalPooledSize = convFeatures[1] * (20//4) * (28//4)

        self.encodeFC = nn.Linear(in_features=self.totalPooledSize, 
                                  out_features=hiddenNum)

        self.encodeMeanFC = nn.Linear(in_features=hiddenNum, 
                                      out_features=3)

        self.encodeVarianceFC = nn.Linear(in_features=hiddenNum, 
                                          out_features=3)

        # DECODER
        # fully connected x2 -> unpool -> convolute -> unpool -> convolute x2
        self.decodeFC1 = nn.Linear(in_features=3,
                                   out_features=hiddenNum)

        self.decodeFC2 = nn.Linear(in_features=hiddenNum, 
                                   out_features=self.totalPooledSize)

        self.decodeUnPool1 = nn.MaxUnpool2d(kernel_size=2)

        self.decodeUnConv1 = nn.ConvTranspose2d(in_channels=convFeatures[1], 
                                                out_channels=convFeatures[0], 
                                                kernel_size=3, 
                                                padding=True)

        self.decodeUnPool2 = nn.MaxUnpool2d(kernel_size=2)

        self.decodeUnConv2 = nn.ConvTranspose2d(in_channels=convFeatures[0], 
                                                out_channels=1, 
                                                kernel_size=3, 
                                                padding=True)

        self.decodeFinalConv = nn.Conv2d(in_channels=1, 
                                         out_channels=1, 
                                         kernel_size=3, 
                                         padding=True)

        self.normalDist = torch.distributions.Normal(torch.zeros(3), torch.ones(3))

    def encode(self, x):
        x = F.leaky_relu(self.encodeConv1(x))
        (x, self.indices1) = self.encodeMaxPool(x)
        x = F.leaky_relu(self.encodeConv2(x))
        (x, self.indices2) = self.encodeMaxPool(x)
        x = x.view(-1, self.totalPooledSize)
        x = F.leaky_relu(self.encodeFC(x))
        mean = self.encodeMeanFC(x)
        varianceLog = self.encodeVarianceFC(x)
        return (mean, varianceLog)

    def sampleFromNormal(self, mean, varianceLog):
        # samples from N(mean, variance)
        # but keeps differentiabilty
        # using reparametrization trick
        if self.training:
            std = varianceLog.exp().pow(0.5)  # Standard deviation
            eps = self.normalDist.sample((mean.shape[0],)) # Sample from normal distribution(0, 1)
            sample = eps.mul(std).add_(mean) # Use a trick to make this N(mean, varianceLog.exp())
            return sample
        else:
            return mean
    
    def decode(self, z):
        z = F.leaky_relu(self.decodeFC1(z))
        z = F.leaky_relu(self.decodeFC2(z))
        z = z.view(-1, self.convFeatures[1], 28//4, 20//4)
        z = self.decodeUnPool1(z, indices=self.indices2)
        z = F.leaky_relu(self.decodeUnConv1(z))
        z = self.decodeUnPool2(z, indices=self.indices1)
        z = F.leaky_relu(self.decodeUnConv2(z))
        z = F.leaky_relu(self.decodeFinalConv(z))
        return z

    def forward(self, x):
        # Find latent probablity for thix x, sample from its distribution and decode
        mean, varianceLog = self.encode(x)
        z = self.sampleFromNormal(mean, varianceLog)
        return (self.decode(z), mean, varianceLog)
    




if __name__ == "__main__":
    x = torch.randn(4, 28*28) 
    vae = VariationalAutoEncoder()
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape )
    print(mu.shape )
    print(sigma.shape )