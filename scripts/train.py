#!/usr/bin/env python3
import math
from typing import Tuple
from vaes.ssvae import SSVAE
import argparse

import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt

from vaes.vae import BernoulliVAE, NormalVAE, VAE


def load_data(data_dir: str = "~/data", **kwargs) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        shuffle=True,
        **kwargs,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        shuffle=False,
        **kwargs,
    )

    return train_loader, test_loader


MODELS = {
    "normal": lambda: NormalVAE(
        lr=1e-3,
        x_size=(1, 28, 28),
        z_size=(50,),
        h_dim=1000,
    ),
    "bernoulli": lambda: BernoulliVAE(
        lr=1e-3,
        x_size=(1, 28, 28),
        z_size=(200,),
        h_dim=1000,
    ),
    "categorical": lambda: SSVAE(
        lr=1e-3,
        x_size=(1, 28, 28),
        z_size=(200,),
        y_size=(10,),
        h_dim=1000,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Which model to fit",
        required=True,
    )
    return parser.parse_args()


class PlotReconstruction(pl.Callback):
    def __init__(self, x: Tensor):
        self.x = x

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: VAE):
        model = model.eval()
        x_hat = model.reconstruct(self.x.to(model.device)).cpu()

        x = self.x.numpy().squeeze()
        x_hat = x_hat.numpy().squeeze()
        fig, (x_ax, x_hat_ax, res_ax) = plt.subplots(1, 3)
        fig.suptitle(f"Epoch {trainer.current_epoch}, z ~ {model.z_dist_name}")
        x_ax.imshow(x)
        x_ax.set_title("Original image")
        x_hat_ax.imshow(x_hat)
        x_hat_ax.set_title(f"Reconstruction")
        res_ax.imshow(x - x_hat)
        res_ax.set_title("Residuals")
        fig.savefig(f"plots/reconstruction-{model.z_dist_name.lower()}.png")


class AnnealTemperature(pl.Callback):
    def __init__(self, interval: int, rate: float, min_temp: float):
        self.interval = interval
        self.rate = rate
        self.min_temp = min_temp

    def on_train_epoch_end(
        self, trainer: pl.Trainer, model: pl.LightningModule
    ) -> None:
        i = trainer.current_epoch
        if i % self.interval == 0:
            new_temp = max((model.temp * math.exp(-self.rate * i), self.min_temp))
            model.set_temp(new_temp)


def has_discrete_latents(model) -> bool:
    return any(
        (
            isinstance(model, BernoulliVAE),
            isinstance(model, SSVAE),
        )
    )


def callbacks(model, train_loader):
    callbacks = [
        ModelCheckpoint(
            dirpath=f"models/",
            filename=f"vae-{model.z_dist_name.lower()}",
        ),
        PlotReconstruction(train_loader.dataset[0][0].unsqueeze(0)),
    ]

    if has_discrete_latents(model):
        callbacks.append(AnnealTemperature(interval=2, rate=0.003, min_temp=0.1))

    return callbacks


args = parse_args()
device = torch.device("cuda" if torch.has_cuda else "cpu")
train_loader, test_loader = load_data(batch_size=128, num_workers=16)
model = MODELS[args.model]()

trainer = pl.Trainer(
    gpus=1,
    callbacks=callbacks(model, train_loader),
    check_val_every_n_epoch=5,
    max_epochs=50,
)
trainer.fit(model, train_loader, test_loader)
