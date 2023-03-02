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


dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
model = torch.load("weights")
model.eval()

def convert_to_img(coord):
    x, y = float(coord[0]), float(coord[1])
    coord = (x, y)
    coord = torch.tensor(coord).view(1,2)
    img = model.decoder(coord).detach()
    img = img.view(28,28,1)
    plt.imshow(img, cmap = 'gray')
    plt.show()




def convert_to_img_without_show(coord):
    x, y = float(coord[0]), float(coord[1])
    coord = (x, y)
    coord = torch.tensor(coord).view(1,2)
    img = model.decoder(coord).detach()
    img = img.view(-1, 1, 28,28)
    return img


def convert_to_img_without_show_6D(coord):
    u, v, w, x, y, z = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3]), float(coord[4]), float(coord[5]), float(coord[6])
    coord = (u, v, w, x, y, z)
    coord = torch.tensor(coord).view(1,6)
    img = model.decoder(coord).detach()
    img = img.view(-1, 1, 28,28)
    return img

