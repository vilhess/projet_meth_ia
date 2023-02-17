import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img import convert_to_img_without_show

model = torch.load("weights")
model.eval()

image = Image.open("space.png")
st.image(image, caption = '2D dimension cluster', use_column_width=True)

coord1 = st.slider('coord1', float(-6), float(6), float(0), step = 0.1)
coord2 = st.slider('coord2', float(-6), float(6), float(0), step = 0.1)
img = convert_to_img_without_show((coord1, coord2))
save_image(img, f"some_images/made.png")
img = Image.open("some_images/made.png")
st.image(img, caption = 'generated', use_column_width=True)