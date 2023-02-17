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


idx = 0

for X, y in dataset:
    if y==idx:
        img = X.view(-1, 1, 28,28)
        save_image(img, f"some_images/img{y}.png")
        idx += 1
        if idx == 10:
            break

 