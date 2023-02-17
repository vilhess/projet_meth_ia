import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn 
from model6 import VariationalAutoEncoder2
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import optim


#configuration 

#DEVICE = torch.device("mps")
INPUT_DIM = 28*28
Z_DIM = 6
NUM_EPOCHS = 10
BATCH_SIZE = 5
LR_RATE = 1e-4

#dataset loading
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
train_loader = DataLoader(dataset = dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VariationalAutoEncoder2(INPUT_DIM, Z_DIM)
optimizer = optim.Adam(model.parameters(), lr = LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

#Start training

if __name__ == "__main__":

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, (x, _) in loop:
            #forward pass
            x = x.view(x.shape[0], INPUT_DIM) #view comme reshape
            x_reconstructed, mu, sigma = model(x)

            #compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = - torch.sum( 1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            #Backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss = loss.item())


    torch.save(model, "./weights6D")

