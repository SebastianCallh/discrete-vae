from typing import Tuple, Union, Optional
from math import prod
import torch
from torch import Tensor, nn
from torch import distributions as D
import pytorch_lightning as pl

from .modules import Bernoulli, Categorical, Normal


class SSVAE(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        x_size: Union[Tuple, torch.Size],
        z_size: Union[Tuple, torch.Size],
        y_size: Union[Tuple, torch.Size],
        temp: float = 1.0,
        h_dim: int = 1000,
        y_decode_dim: int = 100,
        z_decode_dim: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.temp = 1.0
        self.z_size = z_size
        self.y_size = y_size
        self.pz = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Normal(h_dim, z_size),
        )

        self.py = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(x_size), h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Categorical(h_dim, y_size, temp),
        )

        self.px = Decoder(
            x_size=x_size,
            z_size=z_size,
            y_size=y_size,
            h_dim=h_dim,
            z_dim=z_decode_dim,
            y_dim=y_decode_dim,
        )

    def set_temp(self, new_temp: float) -> None:
        self.py.temp = new_temp

    def forward(
        self, x: Tensor, return_elbo: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        pz = self.pz(x)
        z = pz.rsample()
        py = self.py(x)
        y = py.rsample()

        px = self.px(z, y)
        return (x, self._elbo(x, pz, D.Categorical(py.probs), px)) if return_elbo else x

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
        return "Normal-Categorical"

    @torch.no_grad()
    def reconstruct(self, x: Tensor) -> Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        z = self.pz(x).sample()
        y = self.py(x).sample()
        px = self.px(z, y)
        return px.mean

    @torch.no_grad()
    def sample(self, num_samples: int = 1, y: Optional[int] = None) -> Tensor:
        if y is not None:
            y_one_hot = torch.zeros(num_samples, self.y_size[-1])
            y_one_hot[:, y] = 1
        else:
            py = D.OneHotCategorical(logits=torch.ones(self.y_size[-1]))
            y_one_hot = py.sample((num_samples,))

        pz = D.Normal(torch.ones(self.z_size[-1]), 1)
        z = pz.sample((num_samples,))

        return self.px(z, y_one_hot)

    def _elbo(
        self,
        x: Tensor,
        pz: D.Normal,
        py: D.Categorical,
        px: D.Distribution,
    ) -> Tensor:
        z_prior = D.Normal(torch.ones_like(pz.mean), 1)
        z_kl = D.kl_divergence(pz, z_prior).sum(-1)
        y_prior = D.Categorical(logits=torch.ones_like(py.logits))
        y_kl = D.kl_divergence(py, y_prior)
        ll = px.log_prob(x).sum(dim=[-1, -2, -3])
        return ll - z_kl - y_kl


class Decoder(nn.Module):
    def __init__(
        self,
        x_size: Union[Tuple, torch.Size],
        z_size: Union[Tuple, torch.Size],
        y_size: Union[Tuple, torch.Size],
        h_dim: int = 1000,
        y_dim: int = 100,
        z_dim: int = 500,
    ):
        super().__init__()

        self.embed_y = nn.Linear(y_size[-1], y_dim)
        self.embed_z = nn.Sequential(nn.Flatten(), nn.Linear(prod(z_size), z_dim))
        self.decode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(y_dim + z_dim, h_dim),
            nn.Softplus(),
            nn.Linear(h_dim, h_dim),
            nn.Softplus(),
            Bernoulli(h_dim, x_size),
        )

    def forward(self, z: Tensor, y: Tensor) -> D.Distribution:
        z_emb = self.embed_z(z)
        y_emb = self.embed_y(y)
        h = torch.cat((z_emb, y_emb), dim=-1)
        return self.decode(h)
