import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn 
from model_ff import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import optim
import scipy.io
from torch.nn import functional as F

INPUT_DIM = 28*20
H1_DIM = 280
H2_DIM = 140
H3_DIM = 70
Z_DIM = 6
NUM_EPOCHS = 1000
BATCH_SIZE =64
LR_RATE = 1e-3

def get_FreyFace_data_loader(batchSize):
    ff = scipy.io.loadmat('dataset_ff/frey_rawface.mat')
    ff = ff["ff"].T.reshape((-1, 1, 28, 20))
    ff = ff.astype('float32')/255.
    size = len(ff)
    ff = ff[:int(size/batchSize)*batchSize]
    ff_torch = torch.from_numpy(ff)

    train_loader = torch.utils.data.DataLoader(ff_torch, batchSize, shuffle=True)

    
    return train_loader


train_loader = get_FreyFace_data_loader(BATCH_SIZE)
model = VariationalAutoEncoder(INPUT_DIM, H1_DIM,   H2_DIM, H3_DIM, Z_DIM)
optimizer = optim.Adam(model.parameters(), lr = LR_RATE)
loss_fn = nn.BCELoss(reduction="sum")

if __name__ == "__main__":

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))
        for i, x in loop:
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


    torch.save(model, "weights_ff")