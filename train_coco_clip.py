"""
@author: Adityam Ghosh
Date: 10-29-2023

"""

import argparse
import numpy as np
import pandas as pd
import polars as pol
import torch
import torch.utils.data as td
import pytorch_lightning as pl
import torchvision
import albumentations as alb
import os
import sys
import yaml
import zipfile

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import CLIPTokenizerFast, CLIPModel, CLIPConfig, CLIPImageProcessor

from trainer_clip import LitCLIP
from utility.datasets import CocoCLIPDataset
from utility.transform_data import (
    HorizontalFlip,
    CenterSquareCrop,
)

torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clip_training", description="To train the CLIP model"
    )

    parser.add_argument(
        "--train_data_path",
        "-P",
        required=True,
        type=str,
        help="the csv file path for the train data",
    )
    parser.add_argument(
        "--val_data_path",
        "-V",
        required=True,
        type=str,
        help="the csv file path for the val data",
    )

    parser.add_argument(
        "--max_epochs",
        "-E",
        required=False,
        type=int,
        default=500,
        help="maximum number of epochs to train the model",
    )

    parser.add_argument(
        "--early_stopping_patience",
        "-e",
        required=False,
        type=int,
        default=10,
        help="number of iterations to wait before early stopping",
    )

    parser.add_argument(
        "--checkpoint_dir",
        "-C",
        required=False,
        type=str,
        default="./checkpoints",
        help="the directory where the checkpoints will be saved",
    )

    parser.add_argument(
        "--checkpoint_filename",
        "-c",
        required=False,
        type=str,
        default="model_checkpoint",
        help="the name of the checkpoint file to be saved as",
    )

    parser.add_argument(
        "--data_size",
        "-D",
        required=False,
        type=float,
        default=1.0,
        help="the amount of data to train on.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        "-a",
        required=False,
        type=int,
        default=1,
        help="the number of batches to accumulate gradients",
    )

    parser.add_argument(
        "--use_swa",
        "-s",
        required=False,
        type=int,
        default=1,
        help="whether to use SWA or not.",
    )

    parser.add_argument(
        "--load_pretrained_checkpoint",
        "-l",
        required=False,
        type=str,
        default="",
        help="load any previous checkpoint of the model to restart training",
    )

    parser.add_argument(
        "--learning_rate",
        "-L",
        required=False,
        type=float,
        default=1e-4,
        help="the learning rate for the model",
    )

    parser.add_argument(
        "--gradient_clip_val",
        "-g",
        required=False,
        type=float,
        default=20.0,
        help="the gradient clipping value",
    )

    args = parser.parse_args()

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    max_epochs = args.max_epochs
    early_stopping_patience = args.early_stopping_patience
    checkpoint_dir = args.checkpoint_dir
    checkpoint_filename = args.checkpoint_filename
    data_size = args.data_size
    accumulate_grad_batches = args.accumulate_grad_batches
    use_swa = args.use_swa
    load_pretrained_checkpoint = args.load_pretrained_checkpoint
    learning_rate = args.learning_rate
    gradient_clip_val = args.gradient_clip_val

    assert 0 < data_size <= 1, "Expected data size to be within the range (0, 1]"
    assert use_swa in [0, 1], "Expected use_swa to be either 0/1"

    train_csv_data = pol.read_csv(train_data_path)
    val_csv_data = pol.read_csv(val_data_path)

    train_csv_data = train_csv_data.select(
        ["file_name", "image_path", "caption"]
    ).sample(fraction=data_size, seed=32)
    val_csv_data = val_csv_data.select(["file_name", "image_path", "caption"]).sample(
        n=2000, seed=32
    )
    train_csv_data = train_csv_data.to_pandas()
    val_csv_data = val_csv_data.to_pandas()

    train_transforms = alb.Compose(
        [
            alb.SmallestMaxSize(256, always_apply=True),
            CenterSquareCrop(224),
            alb.ColorJitter(),
            HorizontalFlip(),
            alb.Resize(224, 224, always_apply=True),
        ]
    )

    val_transforms = alb.Compose([alb.Resize(224, 224, always_apply=True)])

    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_ds = CocoCLIPDataset(
        data=train_csv_data,
        text_tokenizer=tokenizer,
        img_processor=img_processor,
        transformations=train_transforms,
    )

    val_ds = CocoCLIPDataset(
        data=val_csv_data,
        text_tokenizer=tokenizer,
        img_processor=img_processor,
        transformations=val_transforms,
    )

    train_dl = td.DataLoader(
        train_ds,
        batch_size=2048,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )

    val_dl = td.DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    clip_model = CLIPModel(CLIPConfig())
    model = LitCLIP(clip_model, lr=learning_rate)

    early_stop = EarlyStopping(
        monitor="val_loss", mode="min", patience=early_stopping_patience, verbose=True
    )
    model_chkpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dir,
        filename=checkpoint_filename,
        save_on_train_epoch_end=False,
        verbose=True,
    )
    rich_prog_bar = RichProgressBar()
    swa = StochasticWeightAveraging(swa_lrs=[learning_rate])

    callbacks = (
        [early_stop, model_chkpt, rich_prog_bar, swa]
        if use_swa
        else [early_stop, model_chkpt, rich_prog_bar]
    )

    logger = WandbLogger(
        project="MobileCLIP",
        name="clip_model_coco",
    )

    logger.watch(model, log="gradients", log_freq=50)

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        devices=torch.cuda.device_count(),
        precision="16-mixed",
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="value",
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=None
        if load_pretrained_checkpoint == ""
        else load_pretrained_checkpoint,
    )
