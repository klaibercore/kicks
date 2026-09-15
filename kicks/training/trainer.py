"""VAE training loop.

Two checkpoints are kept, because the two things worth optimizing disagree:

* ``vae_best.pth`` — lowest validation loss. Best *reconstruction*.
* ``vae_best_eval.pth`` — best generative eval proxy. Latents sampled from the
  validation posterior are decoded and their perceptual descriptors compared
  against the corpus distribution. That tracks what `kicks eval` measures, which
  is what a listener notices, at a fraction of the cost (no vocoder needed).

A model can reconstruct beautifully and still generate mush, so which checkpoint
to serve is a real choice rather than a formality.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split

from ..analysis.descriptors import descriptor_matrix, descriptor_vector
from ..instruments import InstrumentProfile
from ..nn import VAE
from .loss import vae_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _corpus_descriptor_stats(
    dataset, profile: InstrumentProfile,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-descriptor mean/std over the training corpus (the proxy's reference)."""
    X = descriptor_matrix((dataset[i] for i in range(len(dataset))), profile)
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def _eval_proxy_score(
    model: VAE,
    mu_all: torch.Tensor,
    desc_mean: np.ndarray,
    desc_std: np.ndarray,
    profile: InstrumentProfile,
    device: torch.device,
    n_samples: int = 32,
) -> float:
    """Spectrogram-domain generative realism proxy (no vocoder needed).

    Samples latents from a Gaussian fit to the validation posterior means,
    decodes them, and measures how far the decoded hits' perceptual descriptors
    sit from the corpus distribution — mean |z-score|, lower is better.
    """
    mean = mu_all.mean(dim=0)
    cov = torch.cov(mu_all.T) + 1e-4 * torch.eye(mu_all.shape[1], device=mu_all.device)
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn(n_samples, mu_all.shape[1], device=mu_all.device)
    z = mean + eps @ chol.T
    with torch.no_grad():
        specs = model.decode(z.to(device)).cpu()
    scores = [
        np.abs((descriptor_vector(specs[i], profile) - desc_mean) / desc_std).mean()
        for i in range(n_samples)
    ]
    return float(np.mean(scores))


def _cyclical_beta(epoch: int, beta: float, anneal_epochs: int, cycles: int) -> float:
    """Beta ramps 0 -> beta over the first half of each cycle, then holds.

    Repeated ramps let the model re-learn structure the KL term flattened on the
    previous cycle, which is what keeps latent dimensions alive.
    """
    if anneal_epochs <= 0 or epoch >= anneal_epochs:
        return beta
    cycle_len = anneal_epochs / max(1, cycles)
    return beta * min(1.0, ((epoch % cycle_len) / cycle_len) * 2)


def train(
    model: VAE,
    dloader: DataLoader,
    optimizer: Optimizer,
    profile: InstrumentProfile,
    epochs: int = 500,
    device: torch.device | None = None,
    beta: float = 0.01,
    free_bits: float = 0.5,
    beta_anneal_epochs: int = 0,
    beta_cycles: int = 4,
    val_split: float = 0.1,
    scheduler: LRScheduler | None = None,
    transient_weight: float | None = None,
    eval_every: int = 5,
) -> dict[str, list[float]]:
    """Train the VAE. Returns per-epoch average loss, recon and kl."""
    paths = profile.paths
    os.makedirs(paths.model_dir or ".", exist_ok=True)

    epoch_loss: list[float] = []
    epoch_recon: list[float] = []
    epoch_kl: list[float] = []
    model.to(device)

    dataset = dloader.dataset
    n_val = int(len(dataset) * val_split)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=dloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=dloader.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_proxy = float("inf")
    desc_mean, desc_std = _corpus_descriptor_stats(dataset, profile)

    def _loss(recon, data, mu, logvar, b):
        return vae_loss(
            recon, data, mu, logvar, beta=b, free_bits=free_bits,
            transient=profile.transient_loss, transient_weight=transient_weight,
        )

    def _save(path: str, **extra) -> None:
        torch.save({"model": model.state_dict(), "instrument": profile.name,
                    **model.checkpoint_meta(), **extra}, path)

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}  Recon: {task.fields[recon]:.4f}  "
                   "KL: {task.fields[kl]:.4f}"),
    ) as progress:
        task = progress.add_task(
            f"Training {profile.name}", total=epochs,
            epoch=0, loss=0.0, recon=0.0, kl=0.0,
        )

        for epoch in range(epochs):
            current_beta = _cyclical_beta(epoch, beta, beta_anneal_epochs, beta_cycles)

            model.train()
            batch_loss, batch_recon, batch_kl = [], [], []
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                l, recon_l, kl = _loss(recon, data, mu, logvar, current_beta)
                batch_loss.append(l.item())
                batch_recon.append(recon_l.item())
                batch_kl.append(kl.item())
                l.backward()
                optimizer.step()

            avg = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
            avg_loss, avg_recon, avg_kl = avg(batch_loss), avg(batch_recon), avg(batch_kl)
            epoch_loss.append(avg_loss)
            epoch_recon.append(avg_recon)
            epoch_kl.append(avg_kl)
            if scheduler is not None:
                scheduler.step()
            progress.update(task, advance=1, epoch=epoch + 1,
                            loss=avg_loss, recon=avg_recon, kl=avg_kl)

            model.eval()
            val_losses, mus, logvars = [], [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    recon, mu, logvar = model(data)
                    vl, _, _ = _loss(recon, data, mu, logvar, current_beta)
                    val_losses.append(vl.item())
                    mus.append(mu)
                    logvars.append(logvar)
            val_loss = avg(val_losses)

            # Latent diagnostics: per-dim KL over the validation set says how many
            # dimensions actually carry information versus collapsing to the
            # prior — and whether free_bits is masking a collapse.
            mu_all = None
            if mus:
                mu_all = torch.cat(mus, dim=0)
                logvar_all = torch.cat(logvars, dim=0)
                kl_per_dim = -0.5 * (
                    1 + logvar_all - mu_all.pow(2) - logvar_all.exp()
                ).mean(0)
                progress.console.log(
                    f"epoch {epoch + 1}: val_loss={val_loss:.4f}  beta={current_beta:.4f}  "
                    f"active_dims={int((kl_per_dim > 0.01).sum().item())}/{model.latent_dim}  "
                    f"raw_kl={float(kl_per_dim.sum()):.3f}  "
                    f"mean_kl/dim={float(kl_per_dim.mean()):.3f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(paths.checkpoint, epoch=epoch + 1, val_loss=val_loss)

            if mu_all is not None and eval_every > 0 and (epoch + 1) % eval_every == 0:
                proxy = _eval_proxy_score(
                    model, mu_all.cpu(), desc_mean, desc_std, profile, device,
                )
                marker = ""
                if proxy < best_proxy:
                    best_proxy = proxy
                    _save(paths.eval_checkpoint, epoch=epoch + 1,
                          val_loss=val_loss, eval_proxy=proxy)
                    marker = f"  (new best -> {os.path.basename(paths.eval_checkpoint)})"
                progress.console.log(f"epoch {epoch + 1}: eval_proxy={proxy:.3f}{marker}")

    _save(paths.final_checkpoint, epoch=epochs, loss_history=epoch_loss)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, series, title in zip(
        axes,
        (epoch_loss, epoch_recon, epoch_kl),
        ("Total Loss", "Reconstruction (SC + L1)", "KL Divergence"),
    ):
        ax.plot(series)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid()
    plt.tight_layout()
    plt.savefig(paths.loss_curves)
    plt.close(fig)

    return {"loss": epoch_loss, "recon": epoch_recon, "kl": epoch_kl}
