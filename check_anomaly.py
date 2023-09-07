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
    parser.add_argument("--latent_dim", "-l", type=int, required=True, help="the latent dimension for compressed image")
    parser.add_argument("--kde_quant", "-q", type=float, required=False, default=.02, help="Quantile for the KDE Threshold")
    parser.add_argument("--error_quant", "-Q", type=float, required=False, default=.25, help="Quantile for the Error Threshold")
    
    args = parser.parse_args()

    img_path = args.pic
    checkpoint = args.checkpoint
    kde_checkpoint = args.kde_checkpoint
    latent_dim = args.latent_dim
    error_quant = args.error_quant
    kde_quant = args.kde_quant
    
    img_tensor = process_img(img_path)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    model = AutoEncoder.load_from_checkpoint(checkpoint, encoder=Encoder(3, latent_dim), decoder=Decoder(3, latent_dim))
    kde_model = pickle.load(open(kde_checkpoint, "rb"))

    info = pd.read_csv("./data/info.csv")
    error_threshold = np.quantile(info["normal_recon_error"].values, error_quant)
    kde_threshold = np.quantile(info["normal_kde"].values, kde_quant)

    print(isAnomaly(img_tensor, model, kde_model, error_threshold, kde_threshold))





