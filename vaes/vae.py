#!/usr/bin/env python3
from typing import Tuple, Union
from math import prod
import torch
from torch import Tensor, nn
from torch import distributions as D
import pytorch_lightning as pl

from .modules import Bernoulli, Binary, Normal, RelaxedBernoulli


class NormalVAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        x_size: Union[Tuple, torch.Size],
        z_size: Union[Tuple, torch.Size],
        h_dim: int = 1000,
    ):
        super().__init__()
        self.lr = lr
        self.pz = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Normal(h_dim, z_size),
        )

        self.px = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(z_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Bernoulli(h_dim, x_size),
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
        return "Normal"

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self.pz(x).sample()
        px = self.px(z)
        return px.mean

    def _elbo(self, x: Tensor, pz: D.Distribution, px: D.Distribution) -> Tensor:
        z_prior = D.Normal(torch.ones_like(pz.mean), 1)
        kl = D.kl_divergence(pz, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl


class BernoulliVAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        x_size: Union[Tuple, torch.Size],
        z_size: Union[Tuple, torch.Size],
        h_dim: int = 1000,
        temp: float = 1.0,
    ):
        super().__init__()
        self.lr = lr
        self.temp = 1.0
        self.pz = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            RelaxedBernoulli(h_dim, z_size, temp=temp),
        )

        self.px = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(z_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Bernoulli(h_dim, x_size),
        )

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pz = self.pz(x)
        z = pz.rsample()
        px = self.px(z)
        return (
            (x, self._elbo(x, D.Bernoulli(logits=pz.logits), px)) if return_elbo else x
        )

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
        return "Bernoulli"

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self.pz(x).sample()
        px = self.px(z)
        return px.mean

    def _elbo(self, x: Tensor, pz: D.Bernoulli, px: D.Distribution) -> Tensor:
        z_prior = D.Bernoulli(logits=torch.ones_like(pz.probs))
        kl = D.kl_divergence(pz, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl

    def set_temp(self, new_temp: float) -> None:
        self.pz.temp = new_temp


VAE = Union[NormalVAE, BernoulliVAE]
