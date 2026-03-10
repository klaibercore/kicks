"""Kicks CLI -- train, serve, and analyze kick drum models."""

import typer

app = typer.Typer(
    name="kicks",
    help="VAE-powered kick drum synthesizer.",
    no_args_is_help=True,
)


@app.command()
def train(
    data: str = typer.Option("data/kicks", "--data", "-d", help="Path to training data directory"),
    epochs: int = typer.Option(200, "--epochs", "-e", help="Number of training epochs"),
    latent_dim: int = typer.Option(128, "--latent-dim", help="VAE latent dimension"),
    beta: float = typer.Option(1.0, "--beta", help="KL beta weight"),
) -> None:
    """Train the kick drum VAE."""
    import os

    import soundfile as sf
    import torch
    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from kicks import KickDataset, KickDataloader, VAE
    from kicks.config import get_device
    from kicks.model import SAMPLE_RATE
    from kicks.train import train as train_loop
    from kicks.vocoder import load_vocoder, spec_to_audio

    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    dataset = KickDataset(data)
    dataloader = KickDataloader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset: {len(dataset)} samples from {data}")

    device = get_device()
    model = VAE(latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters, latent_dim={latent_dim}, device={device}")

    train_loop(
        model, dataloader, optimizer,
        epochs=epochs, device=device,
        beta=beta, beta_anneal_epochs=epochs, beta_cycles=4,
        scheduler=scheduler,
    )

    vocoder = load_vocoder(device)

    with torch.no_grad():
        model.eval()

        for i in range(min(20, len(dataset))):
            original = dataset[i].unsqueeze(0).to(device)
            recon, _, _ = model(original)
            audio = spec_to_audio(recon, dataset, vocoder, device)
            sf.write(f"output/recon_{i + 1}.wav", audio.squeeze(0).numpy(), SAMPLE_RATE)
            print(f"Saved output/recon_{i + 1}.wav")

        for i in range(10):
            z = torch.randn(1, latent_dim).to(device)
            spec = model.decode(z)
            audio = spec_to_audio(spec, dataset, vocoder, device)
            sf.write(f"output/gen_{i + 1}.wav", audio.squeeze(0).numpy(), SAMPLE_RATE)
            print(f"Saved output/gen_{i + 1}.wav")

    print("Done!")


@app.command()
def serve(
    data: str = typer.Option("data/kicks", "--data", "-d", help="Path to dataset directory"),
    port: int = typer.Option(8080, "--port", "-p", help="API port"),
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
) -> None:
    """Start the FastAPI synthesis server."""
    import os

    import uvicorn

    os.environ.setdefault("KICKS_DATA_DIR", data)
    uvicorn.run("kicks.server:app", host=host, port=port, reload=False)


@app.command()
def tui(
    data: str = typer.Option("data/kicks", "--data", "-d", help="Path to dataset directory"),
) -> None:
    """Launch the interactive TUI synthesizer."""
    from kicks.tui import run_tui

    run_tui(data_dir=data)


@app.command()
def cluster(
    data: str = typer.Option("data/kicks", "--data", "-d", help="Path to dataset directory"),
    samples: int = typer.Option(0, "--samples", "-n", help="Number of samples to cluster (0 = all)"),
) -> None:
    """Run GMM clustering and PCA analysis on the latent space."""
    from kicks._cluster_cmd import run_cluster

    run_cluster(data=data, n_samples=samples if samples > 0 else None)


@app.command("fine-tune")
def fine_tune(
    data: str = typer.Option("data/kicks", "--data", "-d", help="Path to training data directory"),
    epochs: int = typer.Option(200, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    grad_accum: int = typer.Option(4, "--grad-accum", help="Gradient accumulation steps"),
    save_every: int = typer.Option(50, "--save-every", help="Save checkpoint every N epochs"),
    save_dir: str = typer.Option("models/vocoder", "--save-dir", help="Directory for vocoder checkpoints"),
) -> None:
    """Fine-tune the BigVGAN vocoder on kick drum samples."""
    from kicks.finetune import finetune

    finetune(
        data_dir=data,
        save_dir=save_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_accum=grad_accum,
        save_every=save_every,
    )
