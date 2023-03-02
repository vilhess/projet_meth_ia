import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img import convert_to_img_without_show

model = torch.load("weights")
model.eval()

st.subheader("Here you can generate new digits by seeing where they lived in our latent space ")


image = Image.open("space.png")
st.image(image, caption = '2D dimension cluster', use_column_width=True)

col1, col2 = st.columns(2)
with col1 :
    coord1 = st.slider('coord1', float(-6), float(6), float(0), step = 0.1)
    coord2 = st.slider('coord2', float(-6), float(6), float(0), step = 0.1)
with col2 : 
    img = convert_to_img_without_show((coord1, coord2))
    save_image(img, f"some_images/made.png")
    img = Image.open("some_images/made.png")
    st.image(img, caption = 'generated', use_column_width=True)