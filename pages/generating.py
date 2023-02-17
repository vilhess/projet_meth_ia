import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

model = torch.load("weights")
model.eval()



file_up = st.file_uploader("Upload an image")

image = Image.open(file_up)
st.image(image, caption='Uploaded Base Image.', use_column_width=True)



#model = torch.load("weights")
#model.eval()
image = ImageOps.grayscale(image)
image = np.array(image.getdata())
image = torch.from_numpy(image).view((1,784))
image = image/255
image = image.type(torch.FloatTensor)

mu, sigma = model.encode(image)
z = mu + sigma*torch.rand_like(sigma)
out = model.decoder(z).detach()
image = out.view((-1,1,28,28))
save_image(image, f"some_images/generated.png")
image = Image.open("some_images/generated.png")
st.image(image, caption = 'generate', use_column_width=True)


## a mettre autre onglet 





#st.write(coord1)






