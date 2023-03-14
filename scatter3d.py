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
from mpl_toolkits import mplot3d


dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True)
model = torch.load("weights3d")
model.eval()




torch.manual_seed(0)

def mean_digit(digit, n_ech):

    list1 = []
    for x, y in dataset:
        if y==digit:
           list1.append(x)

    x = np.zeros(n_ech)
    y = np.zeros(n_ech)
    w = np.zeros(n_ech)

    for d in range(n_ech):
       with torch.no_grad():
            mu, sigma = model.encode(list1[d].view(1, 28*28))
            z = mu + sigma*torch.randn_like(sigma)
            x[d] = z[0,0]
            y[d] = z[0,1]
            w[d] = z[0,2]
    return x, y, w


x_0, y_0, w_0 = mean_digit(0, 400)
x_1, y_1, w_1 = mean_digit(1, 400)
x_2, y_2, w_2  = mean_digit(2, 400)
x_3, y_3, w_3 = mean_digit(3, 400)
x_4, y_4, w_4 = mean_digit(4, 400)
x_5, y_5, w_5 = mean_digit(5, 400)
x_6, y_6, w_6 = mean_digit(6, 400)
x_7, y_7, w_7 = mean_digit(7, 400)
x_8, y_8, w_8 = mean_digit(8, 400)
x_9, y_9, w_9 = mean_digit(9, 400)


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(x_0, y_0, w_0 ,c = 'red', label = '0', s = 2)
    ax.scatter3D(x_1, y_1,w_1, c = 'blue', label = '1', s = 2)
    ax.scatter3D(x_2, y_2,w_2, c = 'violet', label = '2', s = 2)
    ax.scatter3D(x_3, y_3,w_3, c = 'green', label = '3', s = 2)
    ax.scatter3D(x_4, y_4,w_4, c = 'yellow', label = '4', s = 2)
    ax.scatter3D(x_5, y_5,w_5, c = 'darkgoldenrod', label = '5', s = 2)
    ax.scatter3D(x_6, y_6,w_6, c="khaki", label = '6', s = 2)
    ax.scatter3D(x_7, y_7, w_7,c = 'gainsboro', label = '7', s = 2)
    ax.scatter3D(x_8, y_8, w_8,c = 'dimgray', label = '8', s = 2)
    ax.scatter3D(x_9, y_9,w_9,c = 'darkorange', label = '9', s = 2)
    ax.legend(fontsize="xx-large")
    plt.show()