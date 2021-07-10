#!/usr/bin/env python3
from typing import Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch import distributions as D

from .modules import Bernoulli, SimpleCNN, SimpleTCNN


class DiscreteVAE(nn.Module):
    def __init__(
        self,
        x_shape: torch.Size,
        z_dim: int,
        h_dim: int = 1000,
        num_dists: int = 100,
        initial_temp: float = 1.0,
    ):
        super().__init__()

        self.temp = initial_temp
        self.num_dists = num_dists
        self.z_dim = z_dim
        # self.encode = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(x_dim, h_dim),
        #     nn.Softplus(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.Softplus(),
        #     nn.Linear(h_dim, self.num_dists * z_dim),
        # )
        C, H, W = x_shape
        self.encode = SimpleCNN(num_channels=C, out_dim=num_dists * z_dim)

        self.decode = nn.Sequential(
            SimpleTCNN(num_channels=C, out_dim=num_dists * z_dim),
            Bernoulli(h_dim, x_shape),
        )
        # self.decode = nn.Sequential(
        #     nn.Linear(self.num_dists * z_dim, h_dim),
        #     nn.Softplus(),
        #     nn.Linear(h_dim, h_dim),
        #     nn.Softplus(),
        #     Bernoulli(h_dim, torch.Size((1, 28, 28))),
        # )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        z = self._encode(x)
        px = self.decode(z)
        return px.mean

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        z, pz = self._encode(x, return_probs=True)
        px = self.decode(z)
        return (x, self._elbo(x, pz, px)) if return_elbo else x

    def _elbo(self, x: Tensor, pz: D.Distribution, px: D.Distribution) -> Tensor:
        z_prior = D.Categorical(logits=torch.ones_like(pz.probs))
        z_posterior = D.Categorical(probs=pz.probs)
        kl = D.kl_divergence(z_posterior, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl

    def _encode(
        self, x: Tensor, return_probs: bool = False
    ) -> Union[Tuple[Tensor, D.Distribution], Tensor]:
        z_logits = self.encode(x).view(-1, self.num_dists, self.z_dim)
        z_probs = F.softmax(z_logits, dim=-1)
        pz = D.RelaxedOneHotCategorical(self.temp, probs=z_probs)
        z = pz.rsample().reshape(-1, self.num_dists * self.z_dim)
        return (z, pz) if return_probs else z
