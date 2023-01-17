FROM tensorflow/tensorflow:2.0.0-py3

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sagemaker-containers

# Copies the training code inside the container
COPY * /opt/ml/code/

# Defines train.py as script entry point
ENV SAGEMAKER_PROGRAM train.py
