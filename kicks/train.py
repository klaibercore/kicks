"""Training loop for the kick drum VAE."""

import matplotlib.pyplot as plt
import torch
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

from .loss import loss as loss_fn
from .model import VAE


def train(
    model: VAE,
    dloader: DataLoader,
    optimizer: Optimizer,
    epochs: int = 500,
    device: torch.device | None = None,
    save_dir: str = "models/",
    beta: float = 0.01,
    beta_anneal_epochs: int = 0,
    beta_cycles: int = 4,
    val_split: float = 0.1,
) -> dict[str, list[float]]:
    """Train the VAE. Returns per-epoch average losses for loss, recon, kl.

    Beta annealing uses a cyclical schedule: beta ramps linearly from 0 to the
    target value over (beta_anneal_epochs / beta_cycles) epochs, then repeats.
    This prevents posterior collapse while maintaining reconstruction quality.
    """
    epoch_loss: list[float] = []
    epoch_recon: list[float] = []
    epoch_kl: list[float] = []
    model.to(device)

    # Train/val split
    dataset = dloader.dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=dloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=dloader.batch_size, shuffle=False)

    best_val_loss = float("inf")

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: {task.fields[loss]:.4f}  Recon: {task.fields[recon]:.4f}  KL: {task.fields[kl]:.4f}"),
    ) as progress:
        task = progress.add_task("Training", total=epochs, epoch=0, loss=0.0, recon=0.0, kl=0.0)

        for epoch in range(epochs):
            if beta_anneal_epochs > 0 and epoch < beta_anneal_epochs:
                cycle_len = beta_anneal_epochs / beta_cycles
                cycle_pos = (epoch % cycle_len) / cycle_len
                current_beta = beta * min(1.0, cycle_pos * 2)  # ramp up in first half, hold in second
            else:
                current_beta = beta

            # Training
            model.train()
            batch_loss: list[float] = []
            batch_recon: list[float] = []
            batch_kl: list[float] = []

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                l, recon_l, kl = loss_fn(recon, data, mu, logvar, beta=current_beta)
                batch_loss.append(l.item())
                batch_recon.append(recon_l.item())
                batch_kl.append(kl.item())
                l.backward()
                optimizer.step()

            avg_loss = sum(batch_loss) / len(batch_loss)
            avg_recon = sum(batch_recon) / len(batch_recon)
            avg_kl = sum(batch_kl) / len(batch_kl)
            epoch_loss.append(avg_loss)
            epoch_recon.append(avg_recon)
            epoch_kl.append(avg_kl)
            progress.update(task, advance=1, epoch=epoch + 1, loss=avg_loss, recon=avg_recon, kl=avg_kl)

            # Validation
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    recon, mu, logvar = model(data)
                    vl, _, _ = loss_fn(recon, data, mu, logvar, beta=current_beta)
                    val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                }, save_dir + "vae_best.pth")

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "epoch": epochs,
        "loss_history": epoch_loss,
    }, save_dir + "vae_checkpoint.pth")

    # Plot loss components
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.plot(epoch_loss)
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid()
    ax2.plot(epoch_recon)
    ax2.set_title("Reconstruction (SC + L1)")
    ax2.set_xlabel("Epoch")
    ax2.grid()
    ax3.plot(epoch_kl)
    ax3.set_title("KL Divergence")
    ax3.set_xlabel("Epoch")
    ax3.grid()
    plt.tight_layout()
    plt.show()

    return {"loss": epoch_loss, "recon": epoch_recon, "kl": epoch_kl}
