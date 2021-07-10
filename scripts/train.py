#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torch import distributions as D
from torchvision import datasets, transforms
import pytorch_lightning as pl

from matplotlib import pyplot as plt

from vaes.modules import Bernoulli, Normal
from vaes import VAE


def load_data(data_dir: str = "~/data", **kwargs) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        shuffle=True,
        **kwargs,
    )

    test_loader = DataLoader(
        datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        shuffle=False,
        **kwargs,
    )

    return train_loader, test_loader


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
        fig.savefig("reconstruction.png")


device = torch.device("cuda" if torch.has_cuda else "cpu")
train_loader, test_loader = load_data(batch_size=128, num_workers=16)
h_dim = 1000
model = VAE(
    lr=1e-3,
    x_dist=Bernoulli(h_dim, torch.Size((1, 28, 28))),
    z_dist=Normal(h_dim, torch.Size((50,))),
    h_dim=h_dim,
    z_prior=lambda pz: D.Normal(loc=torch.zeros_like(pz.mean), scale=1),
)

trainer = pl.Trainer(
    gpus=1,
    callbacks=[PlotReconstruction(train_loader.dataset[0][0])],
    check_val_every_n_epoch=5,
)
trainer.fit(model, train_loader, test_loader)
