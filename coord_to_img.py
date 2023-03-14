import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import matplotlib.pyplot as plt


dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
model1 = torch.load("save_weights/weights")
model1.eval()


def convert_to_img_without_show_mnist(coord):
    x, y = float(coord[0]), float(coord[1])
    coord = (x, y)
    coord = torch.tensor(coord).view(1, 2)
    img = model1.decoder(coord).detach()
    img = img.view(-1, 1, 28, 28)
    return img


model2 = torch.load("save_weights/weights_ff")
model2.eval()


def convert_to_img_without_show_frey_face(coord):
    u, v, w, x = (
        float(coord[0]),
        float(coord[1]),
        float(coord[2]),
        float(coord[3]),
    )
    coord = (u, v, w, x)
    coord = torch.tensor(coord).view(1, 4)
    img = model2.decoder(coord).detach()
    img = img.view(-1, 1, 28, 20)
    return img



