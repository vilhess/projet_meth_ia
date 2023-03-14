import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img import convert_to_img_without_show_frey_face

model = torch.load("save_weights/weights_ff")
model.eval()


st.header("FREY FACE")
st.subheader(" ")
st.subheader(
    "In this part, we will play with the Frey Face dataset. It contains 1960 images of a single man with differents facial expressions. The dimension of each pictures is 28x20 pixels. "
)
st.subheader(" ")
st.subheader("Below you can see some of these images")
col1, col2, col3 = st.columns(3)
with col1:
    img = Image.open("images/some_ff_images/ex0.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex3.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex6.png")
    st.image(img, use_column_width=True)
with col2:
    img = Image.open("images/some_ff_images/ex1.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex4.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex7.png")
    st.image(img, use_column_width=True)
with col3:
    img = Image.open("images/some_ff_images/ex3.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex5.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_ff_images/ex9.png")
    st.image(img, use_column_width=True)


st.subheader(
    "Here we consider a 6D latent space. You can play with the coordinates and plot the face generated"
)
st.subheader(" ")

colss, colsss = st.columns(2)
with colss:
    coord1 = st.slider("x1", float(-7), float(7), float(0), step=0.1)
    coord2 = st.slider("x2", float(-7), float(7), float(0), step=0.1)
    coord3 = st.slider("x3", float(-7), float(7), float(0), step=0.1)
    coord4 = st.slider(
        "x4", float(-7),
        float(7),
        float(0),
        step=0.1,
    )

with colsss:
    img = convert_to_img_without_show_frey_face(
        (coord1, coord2, coord3, coord4)
    )
    save_image(img, "images/some_ff_images/made4d.png")
    img = Image.open("images/some_ff_images/made4d.png")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.image(img, use_column_width=True)
