#!/usr/bin/env python3

from typing import Tuple
import math
import torch
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from tqdm import tqdm

from concrete_vae import DiscreteVAE


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
        shuffle=True,
        **kwargs,
    )

    return train_loader, test_loader


def plot_reconstruction(model: DiscreteVAE, x: torch.Tensor, epoch: int) -> plt.Figure:
    model = model.eval()
    x_hat = model.reconstruct(x.to(model.device)).cpu()

    x = x.numpy().squeeze()
    x_hat = x_hat.numpy().squeeze()
    fig, (x_ax, x_hat_ax, res_ax) = plt.subplots(1, 3)
    fig.suptitle(f"Epoch {epoch}")
    x_ax.imshow(x)
    x_ax.set_title("Original image")
    x_hat_ax.imshow(x_hat)
    x_hat_ax.set_title("Reconstruction")
    res_ax.imshow(x - x_hat)
    res_ax.set_title("Residuals")
    return fig


def train(
    model: DiscreteVAE,
    optimizer: Optimizer,
    epochs: int,
    data_loader: DataLoader,
    anneal_rate: float = 0.00005,
    min_temp: float = 0.1,
) -> None:
    model = model.train()
    epoch_iter = tqdm(range(epochs))
    for epoch in epoch_iter:
        for i, (x, _) in enumerate(data_loader):
            _, elbo = model(x.to(device), return_elbo=True)
            loss = -elbo.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                model.temp = max((model.temp * math.exp(-anneal_rate * i), min_temp))

        fig = plot_reconstruction(model, x=data_loader.dataset[0][0], epoch=epoch)
        fig.savefig("reconstruction.png")
        epoch_iter.set_postfix_str(f"{loss.item():.4f}")


device = torch.device("cuda" if torch.has_cuda else "cpu")
device = "cpu"
train_loader, test_loader = load_data(batch_size=128, num_workers=8)
model = DiscreteVAE(x_shape=torch.Size((1, 28, 28)), z_dim=10).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
train(
    model=model,
    optimizer=optimizer,
    epochs=30,
    data_loader=train_loader,
)

torch.save(model.state_dict(), "model.pt")
