#!/usr/bin/env python3
from typing import Tuple, Union
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch import distributions as D


class ConcreteVAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        h_dim: int = 1000,
        num_dists: int = 100,
        initial_temp: float = 1.0,
    ):
        super().__init__()
        self.temp = initial_temp
        self.num_dists = num_dists
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, self.num_dists * z_dim),
        )

        self.decode = nn.Sequential(
            nn.Linear(self.num_dists * z_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        x_shape = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self._encode(x)
        x_logits = self.decode(z).reshape(x_shape)
        return x_logits.exp()

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        z, z_probs = self._encode(x, return_probs=True)
        x_probs = self.decode(z).reshape(x.shape)
        return (x, -self._elbo(x, z_probs, x_probs).mean()) if return_elbo else x

    def _elbo(self, x: Tensor, z_probs: Tensor, x_probs: Tensor) -> Tensor:
        px = D.Bernoulli(probs=x_probs, validate_args=False)
        z_prior = D.Categorical(logits=torch.ones_like(z_probs))
        z_posterior = D.Categorical(probs=z_probs)
        kl = D.kl_divergence(z_posterior, z_prior).sum(-1)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - kl

    def _encode(
        self, x: Tensor, return_probs: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        z_logits = self.encode(x).view(-1, self.num_dists, self.z_dim)
        z_probs = F.softmax(z_logits, dim=-1)
        pz = D.RelaxedOneHotCategorical(self.temp, probs=z_probs)
        z = pz.rsample().reshape(-1, self.num_dists * self.z_dim)
        return (z, z_probs) if return_probs else z
