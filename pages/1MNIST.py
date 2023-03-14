import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from coord_to_img import convert_to_img_without_show_mnist

model = torch.load("save_weights/weights")
model.eval()


st.header("MNIST")
st.subheader(" ")
st.subheader(" Intro mnist")
st.subheader(" ")
st.subheader("Below you can see some of these images")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    img = Image.open("images/some_mnist_images/img0.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img5.png")
    st.image(img, use_column_width=True)
with col2:
    img = Image.open("images/some_mnist_images/img1.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img6.png")
    st.image(img, use_column_width=True)
with col3:
    img = Image.open("images/some_mnist_images/img2.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img7.png")
    st.image(img, use_column_width=True)
with col4:
    img = Image.open("images/some_mnist_images/img3.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img8.png")
    st.image(img, use_column_width=True)
with col5:
    img = Image.open("images/some_mnist_images/img4.png")
    st.image(img, use_column_width=True)
    img = Image.open("images/some_mnist_images/img9.png")
    st.image(img, use_column_width=True)

st.subheader(" ")
st.subheader(
    "A VAE can denoise a noisy image. In clicking on these buttons, you can admirate the VAE in action."
)
st.subheader(" ")

if st.button("digit 0"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_0.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    direction = "images/some_mnist_images/generated.png"
    save_image(image, direction)
    image = Image.open(direction)
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 1"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_1.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 2"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_2.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 3"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_3.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 4"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_4.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 5"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_5.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 6"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_6.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 7"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_7.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 8"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_8.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)

if st.button("digit 9"):
    col1, col2 = st.columns(2)
    image = Image.open("images/some_mnist_images_flous/img_flou_9.png")
    with col1:
        st.image(image, use_column_width=True)

    image = ImageOps.grayscale(image)
    image = np.array(image.getdata())
    image = torch.from_numpy(image).view((1, 784))
    image = image / 255
    image = image.type(torch.FloatTensor)

    mu, sigma = model.encode(image)
    z = mu + sigma * torch.rand_like(sigma)
    out = model.decoder(z).detach()
    image = out.view((-1, 1, 28, 28))
    save_image(image, "images/some_mnist_images/generated.png")
    image = Image.open("images/some_mnist_images/generated.png")
    with col2:
        st.image(image, caption="generate", use_column_width=True)


st.header(" ")

st.subheader(
    "A variational autoencoder also allows us to represent our data in a latent space of lower dimension than the base image. We can plot these clusters in order to see where our data lives"
)

image = Image.open("images/space.png")
st.image(image, caption="2D dimension cluster", use_column_width=True)
st.header(" ")
st.subheader(
    "You can now try to play with the coordinates of the latent space and see the image generated."
)
st.header(" ")
col1, col2 = st.columns(2)
with col1:
    coord1 = st.slider("coord1", float(-6), float(6), float(0), step=0.1)
    coord2 = st.slider("coord2", float(-6), float(6), float(0), step=0.1)
with col2:
    img = convert_to_img_without_show_mnist((coord1, coord2))
    save_image(img, "images/some_mnist_images/made.png")
    img = Image.open("images/some_mnist_images/made.png")
    st.image(img, caption="generated", use_column_width=True)
