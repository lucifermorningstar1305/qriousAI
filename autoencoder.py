from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
import wandb


class Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: Optional[int]=128):
        
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1), # (224 x 224 x 64)
            nn.LeakyReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # (224 x 224 x 64)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (112 x 112 x 64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # (112 x 112 x 128)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # (112 x 112 x 128)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (56 x 56 x 128)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (28 x 28 x 256)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (14 x 14 x 512)
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (7 x 7 x 512)
        )

        self.linear = nn.Sequential(
            nn.Flatten(), # (25088)
            nn.Linear(in_features=7 * 7 * 512, out_features=latent_dim) # (latent_dim)
        )

    def forward(self, x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.linear(x)

        return x
    

class Decoder(nn.Module): 
    def __init__(self, output_channels: int, latent_dim: Optional[int]=128):

        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=7 * 7 * 512)
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, output_padding=0, padding=0), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (14 x 14 x 512)
            nn.LeakyReLU()
        )

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, output_padding=0, padding=0), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # (28 x 28 x 512)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1), # (28 x 28 x 256)
            nn.LeakyReLU()
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, output_padding=0, padding=0), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # (56 x 56 x 256)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1), # (56 x 56 x 128)
            nn.LeakyReLU()
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, output_padding=0, padding=0), # (112 x 112 x 128)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # (112 x 112 x 128)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), # (112 x 112 x 64)
            nn.LeakyReLU()
        )

        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, output_padding=0, padding=0), # (224 x 224 x 64)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # (224 x 224 x 64)
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=3, stride=1, padding=1), # (224 x 224 x 3)
            nn.Tanh()
        )

    def forward(self, x):
        linear_out = self.linear(x)
        linear_out = linear_out.reshape(linear_out.shape[0], -1, 7, 7)
        conv_out = self.block1(linear_out)
        conv_out = self.block2(conv_out)
        conv_out = self.block3(conv_out)
        conv_out = self.block4(conv_out)
        conv_out = self.block5(conv_out)
        
        return conv_out
    

class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder: Encoder, decoder: Decoder, lr: Optional[float]=1e-3, min_lr: Optional[float]=1e-6, 
                 factor:Optional[float]=.2, patience: Optional[int]=5):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)

        return out
    

    def _compute_loss(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:

        x = batch["image"]
        x_hat = self(x)

        reconst_loss = F.mse_loss(x_hat, x, reduction="none").sum(dim=[1, 2, 3]).mean(dim=[0])
        return reconst_loss
    

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:

        train_loss = self._compute_loss(batch, batch_idx)
        self.log("train_loss", train_loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        return {
            "loss": train_loss
        }
    
    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> None:
        
        val_loss = self._compute_loss(batch, batch_idx)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False, logger=True, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:

        test_loss = self._compute_loss(batch, batch_idx)
        self.log("test_loss", test_loss, on_epoch=True, on_step=False, logger=True, prog_bar=False)

        return {
            "test_loss": test_loss
        }
    
    def configure_optimizers(self) -> Dict:
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", min_lr=self.min_lr, factor=self.factor,
                                                               patience=self.patience, verbose=True)
        
        return {
            "optimizer" : optimizer, 
            "scheduler": scheduler,
            "monitor": "val_loss"
        }

