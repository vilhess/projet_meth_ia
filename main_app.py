import streamlit as st
from PIL import Image

st.title("Variational Auto-Encoder")
st.text(" ")
st.subheader(
    "A variational Auto-Encoder (VAE) is a type of neural network used to learn a latent representation of unsupervised input data. In other words, the VAE can extract importatn features of a dataset and represent them in a lower dimension space."
)
st.subheader(
    "The architecture of a VAE is based on two deep neural networks : an encoder and a decoder. The encoder takes raw data as input and transforms it into a probability distribution in the latent space. The decoder then uses samples from this ditribution to generate output data similar to that of the original dataset."
)
st.subheader(
    "The VAE is used in a variety of tasks, including image generation and recognition, data compression and others."
)
st.text(" ")
st.text(" ")
st.text(" ")
st.text(" ")
image = Image.open("architecture.png")
st.subheader("below the architecture of the VAE")
st.image(image, caption="", use_column_width=True)
