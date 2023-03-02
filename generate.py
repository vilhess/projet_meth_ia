import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn 
from model2 import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

#DEVICE = torch.device("mps")
dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
model = torch.load("weights")
model.eval()



def inference(digit, num_examples = 1):
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
           images.append(x)
           idx +=1
           if idx ==10:
            break

    encoding_digit = []
    for d in range(10):
        with torch.no_grad():
             mu, sigma = model.encode(images[d].view(1,784))
        encoding_digit.append((mu, sigma))

    mu, sigma = encoding_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"some_images/generated_{digit}_ex{example}.png")


def inference_1_img(img):
    img = torch.tensor(img).view(1, 784)
    mu, sigma = model.encode(img)
    z = mu + sigma*torch.rand_like(sigma)
    output = model.decoder(z).view(-1,1,28,28)
    return output

if __name__ == "__main__":
    for idx in range(10):
        inference(idx, num_examples=1)