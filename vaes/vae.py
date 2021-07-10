#!/usr/bin/env python3
from abc import abstractmethod
from typing import Callable, Tuple, Union
from math import prod
import torch
from torch import Tensor, nn
from torch import distributions as D
import pytorch_lightning as pl

from .modules import NNDistribution, Normal, Bernoulli


class VAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        x_dist: NNDistribution,
        z_dist: NNDistribution,
        z_prior: Callable[[D.Distribution], D.Distribution],
        h_dim: int = 1000,
    ):
        super().__init__()
        self.lr = lr
        self.z_prior = z_prior
        self.pz = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_dist.size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            z_dist,
        )

        self.px = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(z_dist.size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            x_dist,
        )

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pz = self.pz(x)
        z = pz.rsample()
        px = self.px(z)
        return (x, self._elbo(x, pz, px)) if return_elbo else x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        _, elbo = self(x, return_elbo=True)
        loss = -elbo.mean()
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, elbo = self(x, return_elbo=True)
        loss = -elbo.mean()
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def z_dist_name(self) -> str:
        return self.pz[-1].name

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self.pz(x).sample()
        px = self.px(z)
        return px.mean

    def _elbo(self, x: Tensor, pz: D.Distribution, px: D.Distribution) -> Tensor:
        z_prior = self.z_prior(pz)
        kl = D.kl_divergence(pz, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl


class NormalVAE(VAE):
    def __init__(self, lr: float, x_size: torch.Size, h_dim: int, z_dim: int):
        super(NormalVAE).__init(
            lr=lr,
            pz=nn.Sequential(
                nn.Flatten(),
                nn.Linear(prod(x_size), h_dim),
                nn.Softplus(),
                nn.Linear(h_dim, h_dim),
                nn.Softplus(),
                Normal(h_dim, z_dim),
            ),
            px=nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.Softplus(),
                nn.Linear(h_dim, h_dim),
                nn.Softplus(),
                Bernoulli(h_dim, x_size),
            ),
        )

    def prior(self):
        return super().prior()

    def z_dist_name(self) -> str:
        return self._class_name(self.pz[-1])
