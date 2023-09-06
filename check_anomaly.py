import numpy as np
import pandas as pd
import torch
import pickle
import argparse

from autoencoder import Encoder, Decoder, AutoEncoder
from utils import process_img, isAnomaly

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--pic", "-p", type=str, required=True, help="path of the picture")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="path of model checkpoint")
    parser.add_argument("--kde_checkpoint", "-k", type=str, required=True, help="path of KDE model checkpoint")
    parser.add_argument("--kde_nsigma", "-s", type=float, required=False, default=2, help="Number of standard deviations for KDE threshold")
    parser.add_argument("--error_nsigma", "-S", type=float, required=False, default=2, help="Number of standard deviations for Error threshold")
    
    args = parser.parse_args()

    img_path = args.pic
    checkpoint = args.checkpoint
    kde_checkpoint = args.kde_checkpoint
    error_nsigma = args.error_nsigma
    kde_nsigma = args.kde_nsigma

    info = pd.read_csv("./data/info.csv")
    info = info.loc[info["type"] == "normal"]
    
    img_tensor = process_img(img_path)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    mean_reconstr_error = info["mean_reconstruction_error"].values[0]
    std_reconstr_error = info["std_reconstruction_error"].values[0]

    mean_kde = info["mean_kdes"].values[0]
    std_kde = info["std_kdes"].values[0]

    model = AutoEncoder.load_from_checkpoint(checkpoint, encoder=Encoder(3, 128), decoder=Decoder(3, 128))

    kde_model = pickle.load(open(kde_checkpoint, "rb"))

    error_threshold = mean_reconstr_error + error_nsigma * std_reconstr_error
    kde_threshold = mean_kde - kde_nsigma * std_kde

    print(isAnomaly(img_tensor, model, kde_model, error_threshold, kde_threshold))





