"""
@author: Adityam Ghosh
Date: 10-29-2023

"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import torchvision
import os
import sys
import yaml

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import CLIPTokenizerFast

from trainer import LitMobileCLiP
from utility.datasets import TextVisualDataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="mobile_clip_training", description="To train the Mobile CLIP model")

    parser.add_argument("--data_path", "-P", required=True, type=str,
                        help="the csv file path for the data")

    parser.add_argument("--config_path", "-p", required=True,
                        type=str, help="the config file path")

    parser.add_argument("--max_epochs", "-E", required=False, type=int,
                        default=500, help="maximum number of epochs to train the model")

    parser.add_argument("--early_stopping_patience", "-e", required=False, type=int,
                        default=10, help="number of iterations to wait before early stopping")

    parser.add_argument("--checkpoint_dir", "-C", required=False, type=str,
                        default="./checkpoints", help="the directory where the checkpoints will be saved")

    parser.add_argument("--checkpoint_filename", "-c", required=False, type=str,
                        default="model_checkpoint", help="the name of the checkpoint file to be saved as")

    args = parser.parse_args()

    data_path = args.data_path
    config_path = args.config_path
    max_epochs = args.max_epochs
    early_stopping_patience = args.early_stopping_patience
    checkpoint_dir = args.checkpoint_dir
    checkpoint_filename = args.checkpoint_filename

    csv_data = pd.read_csv(data_path)

    train_csv_data = csv_data.loc[csv_data["data_use_for"] == "train"]
    val_csv_data = csv_data.loc[csv_data["data_use_for"] == "val"]

    config = dict()
    with open(config_path, "r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    tokenizer = CLIPTokenizerFast.from_pretrained(
        "openai/clip-vit-base-patch32")

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomResizedCrop((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    train_ds = TextVisualDataset(
        data=train_csv_data, text_tokenizer=tokenizer, transformations=train_transforms, config=config)

    val_ds = TextVisualDataset(data=val_csv_data, text_tokenizer=tokenizer,
                               transformations=val_transforms, config=config)

    train_dl = td.DataLoader(
        train_ds, batch_size=config["train_batch_size"], shuffle=True, num_workers=4)

    val_dl = td.DataLoader(
        val_ds, batch_size=config["val_batch_size"], shuffle=False, num_workers=4)

    model = LitMobileCLiP(config)

    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", patience=early_stopping_patience, verbose=True)
    model_chkpt = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=checkpoint_dir,
                                  filename=checkpoint_filename, save_on_train_epoch_end=False, verbose=True)
    rich_prog_bar = RichProgressBar()

    logger = WandbLogger(project="MobileCLIP",
                         name="mobilenetv1_litetransformer")

    trainer = pl.Trainer(accelerator="cuda",
                         strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
                         devices=torch.cuda.device_count(),
                         precision=16,
                         max_epochs=max_epochs,
                         callbacks=[early_stop, model_chkpt, rich_prog_bar],
                         logger=logger)

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
