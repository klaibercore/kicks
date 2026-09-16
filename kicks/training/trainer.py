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
import math
import time

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

from ..analysis.descriptors import descriptor_matrix
from ..instruments import InstrumentProfile
from ..nn import VAE
from .loss import vae_loss
from .metrics import DetailMetrics
from .tracking import TrainingRun, track_training

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
    generator = torch.Generator(device=mu_all.device).manual_seed(0)
    eps = torch.randn(n_samples, mu_all.shape[1], device=mu_all.device, generator=generator)
    z = mean + eps @ chol.T
    with torch.no_grad():
        specs = model.decode(z.to(device)).cpu()
    standardized = (descriptor_matrix(specs, profile) - desc_mean) / desc_std
    # Match spread as well as location. Mean absolute sample z-score rewards
    # emitting the same average hit every time (complete loss of diversity).
    return float(np.mean(standardized.mean(0) ** 2 + (standardized.std(0) - 1) ** 2))


def _cyclical_beta(epoch: int, beta: float, anneal_epochs: int, cycles: int,
                   floor: float = 0.0) -> float:
    """Beta ramps 0 -> beta over the first half of each cycle, then holds.

    Repeated ramps let the model re-learn structure the KL term flattened on the
    previous cycle, which is what keeps latent dimensions alive. ``floor`` (a
    fraction of ``beta``) keeps the KL pull from ever switching off: the snare's
    KL plateau forms in the near-zero-beta phase, when the encoder is free to
    scale |µ| up for a higher-SNR code and never comes back.
    """
    if anneal_epochs <= 0 or epoch >= anneal_epochs:
        return beta
    if floor > 0:
        return max(floor * beta, _cyclical_beta(epoch, beta, anneal_epochs, cycles))
    cycle_len = anneal_epochs / max(1, cycles)
    return beta * min(1.0, ((epoch % cycle_len) / cycle_len) * 2)


@track_training
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
    seed: int = 42,
    source_checkpoint: str | None = None,
    hf_detail_weight: float = 0.0,
    attack_change_weight: float = 0.0,
    beta_floor: float = 0.0,
    *,
    run_name: str | None = None,
    intent: str = "",
    hypothesis: str = "",
    success_criteria: str = "",
    runs_dir: str | None = None,
    tracker: TrainingRun | None = None,
) -> dict[str, list[float]]:
    """Train the VAE. Returns per-epoch average loss, recon and kl."""
    paths = profile.paths
    os.makedirs(paths.model_dir or ".", exist_ok=True)

    epoch_loss: list[float] = []
    epoch_recon: list[float] = []
    epoch_kl: list[float] = []
    model.to(device)

    dataset = dloader.dataset
    if epochs < 1 or eval_every < 0:
        raise ValueError("epochs must be positive and eval_every non-negative")
    if len(dataset) < 2 or not 0 < val_split < 1:
        raise ValueError("training needs at least two samples and a validation split in (0, 1)")
    n_val = max(1, min(len(dataset) - 1, int(len(dataset) * val_split)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=dloader.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=dloader.batch_size, shuffle=False)
    tracker.record["config"].update(train_samples=len(train_set), val_samples=len(val_set))
    tracker.progress(phase="reference statistics", force=True)

    best_val_loss = float("inf")
    best_proxy = float("inf")
    desc_mean, desc_std = _corpus_descriptor_stats(train_set, profile)

    def _loss(recon, data, mu, logvar, b, terms=None):
        return vae_loss(
            recon, data, mu, logvar, beta=b, free_bits=free_bits,
            transient=profile.transient_loss, transient_weight=transient_weight,
            hf_detail_weight=hf_detail_weight, attack_change_weight=attack_change_weight,
            terms=terms,
        )

    def _accumulate(sums: dict[str, float], terms: dict[str, float], n: int):
        for name, value in terms.items():
            sums[name] = sums.get(name, 0.0) + value * n

    def _save(path: str, **extra) -> None:
        torch.save({"model": model.state_dict(), "instrument": profile.name,
                    "training": {"seed": seed, "beta": beta, "free_bits": free_bits,
                                 "source_checkpoint": source_checkpoint,
                                 "hf_detail_weight": hf_detail_weight,
                                 "attack_change_weight": attack_change_weight,
                                 "beta_floor": beta_floor,
                                 "run_id": tracker.id,
                                 "preprocessing": "peak_safe_lufs_v1",
                                 "validation": "posterior_mean_fixed_beta_v1"},
                    **model.checkpoint_meta(), **extra}, path)

    def _validate():
        model.eval()
        total, count, mus, logvars = 0.0, 0, [], []
        reconstruction, kl_sum = 0.0, 0.0
        term_sums: dict[str, float] = {}
        detail = DetailMetrics(model.n_mels, profile.transient_loss)
        with torch.no_grad():
            for batch_index, data in enumerate(val_loader, 1):
                data = data.to(device)
                mu, logvar = model.encode(data)
                # Compare every epoch with the same objective and the same
                # posterior means. Annealed beta and random validation draws
                # previously made "best" checkpoints incomparable.
                decoded = model.decode(mu)
                terms: dict[str, float] = {}
                vl, vr, vk = _loss(decoded, data, mu, logvar, beta, terms)
                if not math.isfinite(vl.item()):
                    raise FloatingPointError("Validation loss became non-finite")
                _accumulate(term_sums, terms, len(data))
                total += vl.item() * len(data)
                reconstruction += vr.item() * len(data)
                kl_sum += vk.item() * len(data)
                detail.update(decoded, data)
                count += len(data)
                mus.append(mu)
                logvars.append(logvar)
                tracker.progress(batch=batch_index, batches=len(val_loader))
        mu_all, logvar_all = torch.cat(mus), torch.cat(logvars)
        kl_per_dim = -0.5 * (1 + logvar_all - mu_all.pow(2) - logvar_all.exp()).mean(0)
        metrics = {**detail.compute(), "val_reconstruction": reconstruction / count,
                   "val_kl": kl_sum / count, "active_dims": int((kl_per_dim > 0.01).sum().item()),
                   "raw_kl": float(kl_per_dim.sum()), "mean_kl": float(kl_per_dim.mean()),
                   **{f"val_term_{name}": value / count for name, value in term_sums.items()}}
        return total / count, mus, logvars, metrics

    tracker.progress(phase="baseline validation", force=True)
    baseline, _, _, baseline_metrics = _validate()
    tracker.epoch({"epoch": 0, "val_loss": baseline, **baseline_metrics,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "beta": _cyclical_beta(0, beta, beta_anneal_epochs, beta_cycles, beta_floor),
                   "elapsed_seconds": time.monotonic() - tracker.started})
    if source_checkpoint is not None:
        best_val_loss = baseline
        _save(paths.checkpoint, epoch=0, val_loss=best_val_loss)
        print(f"Fine-tune baseline validation: {best_val_loss:.6f}", flush=True)

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
            epoch_start = time.monotonic()
            current_beta = _cyclical_beta(epoch, beta, beta_anneal_epochs, beta_cycles, beta_floor)
            learning_rate = optimizer.param_groups[0]["lr"]
            tracker.progress(phase="training", epoch=epoch + 1, batch=0,
                             batches=len(train_loader), force=True)

            model.train()
            batch_loss, batch_recon, batch_kl = [], [], []
            batch_sizes = []
            train_terms: dict[str, float] = {}
            for batch_index, data in enumerate(train_loader, 1):
                data = data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                terms = {}
                l, recon_l, kl = _loss(recon, data, mu, logvar, current_beta, terms)
                if not math.isfinite(l.item()):
                    raise FloatingPointError("Training loss became non-finite")
                _accumulate(train_terms, terms, len(data))
                batch_loss.append(l.item())
                batch_recon.append(recon_l.item())
                batch_kl.append(kl.item())
                batch_sizes.append(len(data))
                l.backward()
                optimizer.step()
                tracker.progress(batch=batch_index, train_loss=batch_loss[-1],
                                 learning_rate=learning_rate, beta=current_beta)

            avg = lambda xs: float(np.average(xs, weights=batch_sizes)) if xs else 0.0  # noqa: E731
            avg_loss, avg_recon, avg_kl = avg(batch_loss), avg(batch_recon), avg(batch_kl)
            epoch_loss.append(avg_loss)
            epoch_recon.append(avg_recon)
            epoch_kl.append(avg_kl)
            if scheduler is not None:
                scheduler.step()
            progress.update(task, advance=1, epoch=epoch + 1,
                            loss=avg_loss, recon=avg_recon, kl=avg_kl)

            tracker.progress(phase="validation", batch=0, batches=len(val_loader), force=True)
            val_loss, mus, logvars, validation_metrics = _validate()

            # Latent diagnostics: per-dim KL over the validation set says how many
            # dimensions actually carry information versus collapsing to the
            # prior — and whether free_bits is masking a collapse.
            mu_all = None
            if mus:
                mu_all = torch.cat(mus, dim=0)
                progress.console.log(
                    f"epoch {epoch + 1}: val_loss={val_loss:.4f}  beta={current_beta:.4f}  "
                    f"active_dims={validation_metrics['active_dims']}/{model.latent_dim}  "
                    f"raw_kl={validation_metrics['raw_kl']:.3f}  "
                    f"mean_kl/dim={validation_metrics['mean_kl']:.3f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(paths.checkpoint, epoch=epoch + 1, val_loss=val_loss)

            proxy = None
            if mu_all is not None and len(mu_all) > 1 and eval_every > 0 and (epoch + 1) % eval_every == 0:
                tracker.progress(phase="generative evaluation", force=True)
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

            n_train = sum(batch_sizes)
            tracker.epoch({"epoch": epoch + 1, "train_loss": avg_loss,
                           "train_reconstruction": avg_recon, "train_kl": avg_kl,
                           **{f"train_term_{name}": value / n_train for name, value in train_terms.items()},
                           "val_loss": val_loss, **validation_metrics,
                           "eval_proxy": proxy, "beta": current_beta,
                           "learning_rate": learning_rate,
                           "epoch_seconds": time.monotonic() - epoch_start,
                           "elapsed_seconds": time.monotonic() - tracker.started})

    tracker.progress(phase="saving artifacts", force=True)
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
