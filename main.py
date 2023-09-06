import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import argparse
import os

import torch
import lightning.pytorch as pl
import torch.utils.data as td
import albumentations as A

from collections import defaultdict
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from rich.progress import track
from rich import print as rprint


from autoencoder import Encoder, Decoder, AutoEncoder
from create_torch_dataset import AnimalDataset
from utils import print_table, get_val_images, GenerateCallback

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", "-o", type=str, required=True, help="whether to train/test")
    parser.add_argument("--data_dir", "-d", type=str, required=True, help="directory of the data")
    parser.add_argument("--latent_dim", "-l", type=int, required=False, default=128, help="the latent dimension for the autoencoder")
    parser.add_argument("--max_epochs", "-e", type=int, required=False, default=500, help="max epochs to train")
    parser.add_argument("--learning_rate", "-L", type=float, required=False, default=0.001, help="learning rate for training")
    parser.add_argument("--test_size", "-s", type=float, required=False, default=.2, help="size of the test dataset")
    parser.add_argument("--train_batch_size", "-t", type=int, required=False, default=64, help="training batch size")
    parser.add_argument("--val_batch_size", "-v", type=int, required=False, default=128, help="validation_batch_size")
    parser.add_argument("--model_checkpoint", "-m", type=str, required=False, default="autoencoder", help="name of the model checkpoint")
    parser.add_argument("--checkpoint_dir", "-c", type=str, required=False, default="checkpoints", help="directory to store the checkpoints")

    args = parser.parse_args()

    objective = args.objective
    data_dir = args.data_dir
    latent_dim = args.latent_dim
    max_epochs = args.max_epochs
    lr = args.learning_rate
    test_size = args.test_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    model_checkpoint = args.model_checkpoint
    checkpoint_dir = args.checkpoint_dir

    if objective == "train":
        wandb.login()

        n_gpus = torch.cuda.device_count()

        data = defaultdict(lambda: list())

        if not os.path.exists("./data/dataframe.csv"):
            for root, folders, files in os.walk(data_dir):
                n_files = len(files)

                for i in track(range(n_files)):
                    img_file = files[i]
                    label = root.split("/")[-1]

                    if label in ["cat", "dog"]:
                        data["label"].append("normal")
                    else:
                        data["label"].append("anomaly")
                    
                    data["img_path"].append(os.path.join(root, img_file))

            
            data = pd.DataFrame(data).sample(frac=1, random_state=32)
            data.to_csv("./data/dataframe.csv", index=False)
        
        else:
            data = pd.read_csv("./data/dataframe.csv")
        
        print_table(data, title="Original dataset")
        rprint(f"[bold #f2593a] Original size of the data: [bold #3a7df2]{data.shape[0]}")

        # Seperate the Anomaly Data from the normal ones
        normal_data = data.loc[data["label"] == "normal"]
        anomaly_data = data.loc[data["label"] == "anomaly"]

        rprint(f"[bold #71f23a] Size of the Normal data: [bold #3af2dd]{normal_data.shape[0]}")
        rprint(f"[bold #f23a3a] Size of the Anomaly data: [bold #f23ad3]{anomaly_data.shape[0]}")

        # Create a train/test partition using the normal data
        df_train, df_val = train_test_split(normal_data, test_size=test_size, shuffle=True, random_state=42)
        rprint(f"[bold #f2b83a] Size of the training data: [bold #813af2]{df_train.shape[0]}")
        rprint(f"[bold #3af26b] Size of the validation data: [bold #f23abe]{df_val.shape[0]}")

        # Define transformations for train and validation test
        train_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=.5, contrast_limit=.2),
            A.Rotate(limit=(0, 90)),
            A.RandomShadow(),
            A.RandomSunFlare(),
            A.RandomToneCurve(),
            A.RGBShift(r_shift_limit=50, g_shift_limit=30, b_shift_limit=80),
            A.Blur(blur_limit=9),
            A.Normalize(mean=.5, std=.5, always_apply=True)
        ])

        val_transforms = A.Compose([
            A.Normalize(mean=.5, std=.5, always_apply=True)
        ])


        # Create PyTorch datasets 
        train_ds = AnimalDataset(df_train, resize=(224, 224), transformations=train_transforms)
        val_ds = AnimalDataset(df_val, resize=(224, 224), transformations=val_transforms)
        anomaly_ds = AnimalDataset(anomaly_data, resize=(224, 224), transformations=val_transforms)

        # Create PyTorch dataloaders
        train_dl = td.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
        val_dl = td.DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=4)

        # Define the models
        encoder = Encoder(3, latent_dim=latent_dim)
        decoder = Decoder(3, latent_dim=latent_dim)
        autoenc = AutoEncoder(encoder, decoder, lr=lr)

        # Define callbacks for training 
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)
        model_chkpt = ModelCheckpoint(monitor="val_loss", mode="min",
                                      dirpath=checkpoint_dir, filename=model_checkpoint, 
                                      verbose=True, save_on_train_epoch_end=False)
        prog_bar = RichProgressBar()
        img_logger = GenerateCallback(get_val_images(val_ds))
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Define the trainer 
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=n_gpus,
            strategy="ddp" if n_gpus > 1 else "auto",
            precision=16,
            max_epochs=max_epochs,
            min_epochs=5,
            callbacks=[early_stop, model_chkpt, prog_bar, lr_monitor, img_logger],
            logger=WandbLogger(project="AutoEncoders", name="vgg_auto_enc")
        )

        # Fit the model
        trainer.fit(autoenc, train_dataloaders=train_dl, val_dataloaders=val_dl)










