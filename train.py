import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn 
from model2 import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import optim
from statistics import mean
import matplotlib.pyplot as plt


#configuration 

DEVICE = torch.device("mps")
INPUT_DIM = 28*28
H1_DIM = 400
H2_DIM = 180
H3_DIM = 90
H4_DIM = 30
H5_DIM = 15
Z_DIM = 2
NUM_EPOCHS = 20
BATCH_SIZE = 5
LR_RATE = 1e-4

#dataset loading
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM, H1_DIM, H2_DIM, H3_DIM, H4_DIM, H5_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

#Start training

if __name__ == "__main__":
    loss_list = []

    for epoch in range(NUM_EPOCHS):
        l = []
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:
            #forward pass
            x = x.view(x.shape[0], INPUT_DIM).to(DEVICE) #view comme reshape
            x_reconstructed, mu, sigma = model(x)

            #compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = - torch.sum( 1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            #Backprop
            loss = reconstruction_loss + kl_div
            l.append(float(-loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item())
        loss_list.append(mean(l))
    
    plt.plot(loss_list, '-o')
    plt.xlabel('epoch')
    plt.ylabel('ELBO')
    plt.legend()
    plt.show()
    
    


    torch.save(model, "./weights2.2D")

