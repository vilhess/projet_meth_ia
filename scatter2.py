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




torch.manual_seed(0)

def mean_digit(digit, n_ech):

    list1 = []
    for x, y in dataset:
        if y==digit:
           list1.append(x)

    x = np.zeros(n_ech)
    y = np.zeros(n_ech)

    for d in range(n_ech):
       with torch.no_grad():
            mu, sigma = model.encode(list1[d].view(1, 28*28))
            x[d] = mu[0,0]
            y[d] = mu[0,1]
    return x, y


x_0, y_0 = mean_digit(0, 200)
x_1, y_1 = mean_digit(1, 200)
x_2, y_2 = mean_digit(2, 200)
x_3, y_3 = mean_digit(3, 200)
x_4, y_4 = mean_digit(4, 200)
x_5, y_5 = mean_digit(5, 200)
x_6, y_6 = mean_digit(6, 200)
x_7, y_7 = mean_digit(7, 200)
x_8, y_8 = mean_digit(8, 200)
x_9, y_9 = mean_digit(9, 200)


if __name__ == "__main__":

    plt.scatter(x_0, y_0, c = 'red', label = '0')
    plt.scatter(x_1, y_1, c = 'blue', label = '1')
    plt.scatter(x_2, y_2, c = 'violet', label = '2')
    plt.scatter(x_3, y_3, c = 'green', label = '3')
    plt.scatter(x_4, y_4, c = 'yellow', label = '4')
    plt.scatter(x_5, y_5, c = 'darkgoldenrod', label = '5')
    plt.scatter(x_6, y_6, c="khaki", label = '6')
    plt.scatter(x_7, y_7, c = 'gainsboro', label = '7')
    plt.scatter(x_8, y_8, c = 'dimgray', label = '8')
    plt.scatter(x_9, y_9, c = 'darkorange', label = '9')
    plt.legend()
    plt.show()

    