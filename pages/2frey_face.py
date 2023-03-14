import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img_6d import convert_to_img_without_show_6D

model = torch.load("weights_ff")
model.eval()


st.header("FREY FACE")
st.subheader(" ")
st.subheader("In this part, we will play with the Frey Face dataset. It contains 1960 images of a single man with differents facial expressions. The dimension of each pictures is 28x20 pixels. ")
st.subheader(" ")
st.subheader("Below you can see some of these images")
col1, col2, col3 = st.columns(3)
with col1:
    img = Image.open('some_ff_images/ex0.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex3.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex6.png')
    st.image(img,use_column_width=True )
with col2:
    img = Image.open('some_ff_images/ex1.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex4.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex7.png')
    st.image(img,use_column_width=True )
with col3:
    img = Image.open('some_ff_images/ex3.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex5.png')
    st.image(img,use_column_width=True )
    img = Image.open('some_ff_images/ex9.png')
    st.image(img,use_column_width=True )


st.subheader("Here we consider a 6D latent space. You can play with the coordinates and plot the face generated")
st.subheader(" ")

colss, colsss = st.columns(2)
with colss : 
    coord1 = st.slider('x1', float(-7), float(7), float(0), step = 0.1)
    coord2 = st.slider('x2', float(-7), float(7), float(0), step = 0.1)
    coord3 = st.slider('x3', float(-7), float(7), float(0), step = 0.1)
    coord4 = st.slider('x4 : the most interesting one in our opinion', float(-7), float(7), float(0), step = 0.1)
    coord5 = st.slider('x5', float(-7), float(7), float(0), step = 0.1)
    coord6 = st.slider('x6', float(-7), float(7), float(0), step = 0.1)

with colsss :
    img = convert_to_img_without_show_6D((coord1, coord2, coord3, coord4, coord5, coord6))
    save_image(img, f"some_images/made6d.png")
    img = Image.open("some_images/made6d.png")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.header(" ")
    st.image(img, use_column_width=True)