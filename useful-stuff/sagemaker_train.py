import os
import argparse

# import tensorflow as tf
# import pandas as pd
# import numpy as np

from utils import str2bool

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### reads input channels training and testing from the environment variables
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    ### model inputs/outputs parameters

    ### data preprocessing parameters

    ### train, val and test set sizes

    ### batch size
    parser.add_argument("--batch_size", type=int, default=32)

    ### model hyperparameters

    ### training hyperparameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    ### callbacks parameters

    ### model saving parameters

    args, _ = parser.parse_known_args()

    ### fetch and preprocess data

    ### create training, validation and test dataset

    ### separate input and target data

    ### create batchs

    ### create model and callbacks

    model = None

    ### train model

    # model.fit()

    ### evaluate model

    # model.evaluate()

    ### save model

    # model.save(file_path)
