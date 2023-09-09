from typing import Dict, Tuple, Any, Optional, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import albumentations as A
import os 
import sys
import wandb
import argparse

from rich.progress import track
from collections import defaultdict
from utility.utils import print_table, isAnomaly
from create_torch_dataset import AnimalDataset
from autoencoder import AutoEncoder, Encoder, Decoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


def plot_metrics(quantiles: List, metrics: Dict):
    """ Function to generate a matplotlib/seaborn plot for metrics """


    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    markers = ["-bo", "-ro", "-go", "-mo"]


    for idx, (key, value) in enumerate(metrics.items()):
        axs[idx].set_title(key.capitalize())
        axs[idx].plot(quantiles, value, markers[idx])
        axs[idx].grid()
        axs[idx].set_xlabel("quantiles")
        axs[idx].set_ylabel(key)
    
    plt.tight_layout()
    plt.savefig("metrics_result.png")


def plot_confusion_mat(quantiles: List, confusion_matrices: List):
    """ Function to plot the confusion matrices """

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    for idx, conf_mat in enumerate(confusion_matrices):

        axs[idx].set_title(f"Quantile: {quantiles[idx]}")
        disp = ConfusionMatrixDisplay(conf_mat, display_labels=["Normal", "Anomaly"])
        disp.plot(include_values=True, cmap="Blues", ax=axs[idx], colorbar=False, values_format=".2f")

    plt.tight_layout()
    plt.savefig("confusion_matrics.png")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-d", required=True, type=str, help="the validation data directory of the images")
    parser.add_argument("--model_checkpoint", "-m", required=True, type=str, help="the model checkpoint full file path")
    parser.add_argument("--latent_dim", "-l", required=True, type=int, help="the latent dimension of the model.\n Use the same one used in training.")


    args = parser.parse_args()

    data_dir = args.data_dir
    model_checkpoint = args.model_checkpoint
    latent_dim = args.latent_dim

    LABEL_MAP = {"normal": 0, "anomaly": 1}

    dataset = defaultdict(lambda: list())


    for root, folders, files in os.walk(data_dir):
        n_files = len(files)

        if "val" in root.split("/")[-2]:
            for i in track(range(n_files)):
                img_file = files[i]
                label = root.split("/")[-1]

                if label in ["cat", "dog"]:
                    dataset["label"].append("normal")

                else:
                    dataset["label"].append("anomaly")

                dataset["img_path"].append(os.path.join(root, img_file))

    dataset = pd.DataFrame(dataset).sample(frac=1, random_state=32)
    print_table(dataset, title="Evaluation data")

    ds = AnimalDataset(dataset, resize=(224, 224), transformations=A.Compose([A.Normalize(mean=.5, std=.5, always_apply=True)]))
    dl = td.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    model = AutoEncoder.load_from_checkpoint(model_checkpoint, encoder=Encoder(input_channels=3, latent_dim=latent_dim),
                                             decoder=Decoder(output_channels=3, latent_dim=latent_dim))
    
    labels = dataset["label"].map(LABEL_MAP).values.tolist()
    metrics = defaultdict(lambda: list())


    quantiles = [.02, .25, .5, .75, .90, .95]

    normal_data_info = pd.read_csv("./data/info.csv")
    normal_recon_errors = normal_data_info["normal_recon_error"].values
    
    thresholds = [np.quantile(normal_recon_errors, q) for q in quantiles]
    confusion_matrices = list()


    for thresh in thresholds:
        predictions = list()
        
        print(f"Threshold: {thresh}")

        for batch in track(dl):
            img = batch["image"]
            anomaly = isAnomaly(img, model, thresh)
            predictions.append(1 if anomaly else 0)

        metrics["accuracy"].append(accuracy_score(np.asarray(labels), np.asarray(predictions)))
        metrics["precision"].append(precision_score(np.asarray(labels), np.asarray(predictions)))
        metrics["recall"].append(recall_score(np.asarray(labels), np.asarray(predictions)))
        metrics["f1Score"].append(f1_score(np.asarray(labels), np.asarray(predictions)))

        confusion_matrices.append(confusion_matrix(np.asarray(labels), np.asarray(predictions)))

    plot_metrics(quantiles, metrics)
    plot_confusion_mat(quantiles, confusion_matrices)
        





    








                
