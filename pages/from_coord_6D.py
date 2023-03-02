import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img_6d import convert_to_img_without_show_6D

model = torch.load("weights6D")
model.eval()

col1, col2 = st.columns(2)
with col1 : 
    coord1 = st.slider('coord1', float(-6), float(6), float(0), step = 0.1)
    coord2 = st.slider('coord2', float(-6), float(6), float(0), step = 0.1)
    coord3 = st.slider('coord3', float(-6), float(6), float(0), step = 0.1)
    coord4 = st.slider('coord4', float(-6), float(6), float(0), step = 0.1)
    coord5 = st.slider('coord5', float(-6), float(6), float(0), step = 0.1)
    coord6 = st.slider('coord6', float(-6), float(6), float(0), step = 0.1)

with col2 :
    img = convert_to_img_without_show_6D((coord1, coord2, coord3, coord4, coord5, coord6))
    save_image(img, f"some_images/made6d.png")
    img = Image.open("some_images/made6d.png")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.image(img, use_column_width=True)