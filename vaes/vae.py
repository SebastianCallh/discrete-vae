#!/usr/bin/env python3
from abc import abstractmethod
from typing import Callable, Tuple, Union
from math import prod
import torch
from torch import Tensor, nn
from torch import distributions as D
import pytorch_lightning as pl

from .modules import LocScale, Probabilities


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
        self.pz_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            LocScale(h_dim, z_size),
        )

        self.px_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(z_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Probabilities(h_dim, x_size),
        )

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pz = D.Normal(*self.pz_params(x))
        z = pz.rsample()
        px = D.Bernoulli(*self.px_params(z), validate_args=False)
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

        z = D.Normal(*self.pz_params(x)).sample()
        px = D.Bernoulli(*self.px_params(z))
        return px.mean

    def _elbo(self, x: Tensor, pz: D.Distribution, px: D.Distribution) -> Tensor:
        z_prior = D.Normal(torch.ones_like(pz.mean), 1)
        kl = D.kl_divergence(pz, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl


class BinaryVAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        x_size: Union[Tuple, torch.Size],
        z_size: Union[Tuple, torch.Size],
        h_dim: int = 1000,
    ):
        super().__init__()
        self.lr = lr
        self.temp = 1.0
        self.pz_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Probabilities(h_dim, z_size),
        )

        self.px_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(z_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Probabilities(h_dim, x_size),
        )

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pz = D.RelaxedBernoulli(self.temp, *self.pz_params(x))
        z = pz.rsample()
        px = D.Bernoulli(*self.px_params(z), validate_args=False)
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
        return "Bernoulli"

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = D.Bernoulli(*self.pz_params(x)).sample()
        px = D.Bernoulli(*self.px_params(z))
        return px.mean

    def _elbo(self, x: Tensor, pz: D.RelaxedBernoulli, px: D.Distribution) -> Tensor:
        z_prior = D.Bernoulli(logits=torch.ones_like(pz.probs))
        z_posterior = D.Bernoulli(logits=pz.logits)
        kl = D.kl_divergence(z_posterior, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl


VAE = Union[NormalVAE, BinaryVAE]
