import scipy.io
import torch
from PIL import Image


def get_FreyFace_data():
    ff = scipy.io.loadmat('dataset/FREYFACE/frey_rawface.mat')
    ff = ff["ff"].T.reshape((-1, 1, 28, 20))
    ff = ff.astype('float32')/255.
    return ff


data = get_FreyFace_data()

for i in range(10):
    img = data[i]
    