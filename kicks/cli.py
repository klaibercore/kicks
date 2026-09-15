"""Command-line entry point.

Every command takes ``--instrument`` and derives its paths from that
instrument's profile, so `kicks train -i snare` reads ``data/snares`` and writes
``models/snare/`` without any further arguments. Explicit paths still win when
given. Implementations are imported lazily — `kicks eval` should not wait on
torch and BigVGAN to load.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="kicks",
    help="VAE-powered drum synthesizer (kicks, snares, hi-hats).",
    no_args_is_help=True,
)

INSTRUMENT = typer.Option(
    None, "--instrument", "-i",
    help="Drum type: kick, snare or hihat. Defaults to KICKS_INSTRUMENT, then kick.",
)


@app.command()
def instruments() -> None:
    """List the registered instruments and whether each has a trained model."""
    import os

    from rich.console import Console
    from rich.table import Table

    from kicks.instruments import DEFAULT_INSTRUMENT, available, get_profile

    table = Table(title="Instruments")
    for col in ("Name", "Sliders", "Corpus", "Checkpoint", "Trained"):
        table.add_column(col)
    for name in available():
        p = get_profile(name)
        trained = os.path.exists(p.paths.checkpoint)
        label = f"{p.name}{'  (default)' if name == DEFAULT_INSTRUMENT else ''}"
        table.add_row(
            label,
            ", ".join(p.descriptor_labels),
            p.paths.data_dir,
            p.paths.checkpoint,
            "[green]yes[/green]" if trained else "[yellow]no[/yellow]",
        )
    Console().print(table)


@app.command()
def train(
    instrument: str = INSTRUMENT,
    data: str = typer.Option(None, "--data", "-d", help="Corpus directory"),
    epochs: int = typer.Option(200, "--epochs", "-e", help="Number of training epochs"),
    latent_dim: int = typer.Option(None, "--latent-dim", help="VAE latent dimension"),
    beta: float = typer.Option(0.02, "--beta", help="KL beta weight (capped in cyclical annealing)"),
    free_bits: float = typer.Option(0.2, "--free-bits", help="Per-dim KL floor in nats (prevents posterior collapse)"),
    beta_cycles: int = typer.Option(4, "--beta-cycles", help="Number of cyclical beta annealing cycles"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    transient_weight: float = typer.Option(None, "--transient-weight", help="Override the profile's transient-fidelity loss weight (0 = off)"),
    preview: int = typer.Option(10, "--preview", help="Reconstructions and samples to render after training (0 = skip)"),
    resume: str = typer.Option(None, "--resume", help="Fine-tune this checkpoint with a fresh optimizer"),
    learning_rate: float = typer.Option(None, "--learning-rate", help="Defaults to 1e-4 for fine-tuning, 1e-3 for a new model"),
    model_dir: str = typer.Option(None, "--model-dir", help="Output model root (use a separate directory for candidate weights)"),
    seed: int = typer.Option(42, "--seed", help="Reproducible training and validation split"),
) -> None:
    """Train the VAE for one instrument."""
    import os

    import soundfile as sf
    import torch
    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from kicks.audio.constants import SAMPLE_RATE
    from kicks.audio.vocoder import load_vocoder, spec_to_audio
    from kicks.config import get_device, load_vae_from_checkpoint
    from kicks.data import DrumDataset
    from kicks.instruments import get_profile
    from kicks.nn import VAE
    from kicks.training import train as train_loop

    if model_dir:
        os.environ["KICKS_MODEL_DIR"] = model_dir
    if epochs < 1 or batch_size < 1:
        raise typer.BadParameter("epochs and batch size must be positive")
    torch.manual_seed(seed)
    profile = get_profile(instrument)
    data = data or profile.paths.data_dir
    latent_dim = latent_dim or profile.latent_dim
    os.makedirs(profile.paths.model_dir or ".", exist_ok=True)
    os.makedirs(profile.paths.output_dir or ".", exist_ok=True)

    device = get_device()
    model = (load_vae_from_checkpoint(resume, device)[0] if resume
             else VAE(latent_dim=latent_dim).to(device))
    dataset = DrumDataset(data, profile, n_frames=model.n_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lr = learning_rate if learning_rate is not None else (1e-4 if resume else 1e-3)
    if lr <= 0:
        raise typer.BadParameter("learning rate must be positive")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, latent_dim={model.latent_dim}, device={device}")

    train_loop(
        model, dataloader, optimizer, profile,
        epochs=epochs, device=device,
        beta=beta, free_bits=free_bits,
        beta_anneal_epochs=0 if resume else epochs, beta_cycles=beta_cycles,
        scheduler=scheduler, transient_weight=transient_weight,
        seed=seed, source_checkpoint=resume,
    )

    if preview > 0:
        import numpy as np
        from kicks.synthesis.generator import fit_latent_prior

        out_dir = profile.paths.output_dir
        vocoder = load_vocoder(device, weights_dir=profile.paths.vocoder_dir)
        model, _ = load_vae_from_checkpoint(profile.paths.checkpoint, device)
        prior = fit_latent_prior(model, device, data, profile.paths.latent_prior)
        prior.random_state = np.random.RandomState(seed)
        preview_latents, _ = prior.sample(preview)
        with torch.no_grad():
            for i in range(min(preview * 2, len(dataset))):
                mu, _ = model.encode(dataset[i].unsqueeze(0).to(device))
                recon = model.decode(mu)
                audio = spec_to_audio(recon, vocoder, device)
                path = os.path.join(out_dir, f"recon_{i + 1}.wav")
                sf.write(path, audio.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_24")
            for i in range(preview):
                spec = model.decode(torch.tensor(preview_latents[i:i+1], dtype=torch.float32, device=device))
                audio = spec_to_audio(spec, vocoder, device)
                path = os.path.join(out_dir, f"gen_{i + 1}.wav")
                sf.write(path, audio.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_24")
        print(f"Wrote reconstructions and samples to {out_dir}/")

    print("Done!")


@app.command()
def serve(
    instrument: str = INSTRUMENT,
    port: int = typer.Option(8080, "--port", "-p", help="API port"),
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    data: str = typer.Option(None, "--data", "-d", help="Override the corpus root directory"),
    griffin_lim: bool = typer.Option(False, "--griffin-lim", help="Use Griffin-Lim instead of BigVGAN (lower quality, no GPU needed)"),
    control: str = typer.Option(None, "--control", help="Slider basis: 'descriptor' (default) or 'pca'"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (development)"),
) -> None:
    """Start the REST API (the website in web/ is its client)."""
    import os

    import uvicorn

    from kicks.instruments import get_profile

    profile = get_profile(instrument)
    os.environ["KICKS_INSTRUMENT"] = profile.name
    control = control or os.environ.get("KICKS_CONTROL", "descriptor")
    if control not in ("pca", "descriptor"):
        raise typer.BadParameter("control must be 'descriptor' or 'pca'")
    os.environ["KICKS_CONTROL"] = control
    if data:
        os.environ["KICKS_DATA_DIR"] = data
    if griffin_lim:
        os.environ["KICKS_VOCODER"] = "griffinlim"

    print(f"Serving {profile.display_name} on http://{host}:{port}")
    uvicorn.run("kicks.api:app", host=host, port=port, reload=reload)


@app.command()
def generate(
    instrument: str = INSTRUMENT,
    count: int = typer.Option(10, "--count", "-n", help="Number of samples to generate"),
    best_of: int = typer.Option(4, "--best-of", "-k", help="Candidates decoded per output slot; best eval score wins"),
    data: str = typer.Option(None, "--data", "-d", help="Corpus directory (latent prior + eval reference)"),
    out: str = typer.Option(None, "--out", "-o", help="Output directory"),
    seed: int = typer.Option(-1, "--seed", help="Random seed (-1 = random)"),
    griffin_lim: bool = typer.Option(False, "--griffin-lim", help="Use Griffin-Lim instead of BigVGAN"),
    refresh_prior: bool = typer.Option(False, "--refresh-prior", help="Re-fit the cached latent GMM prior"),
) -> None:
    """Generate one-shots by sampling the corpus latent prior (best-of-k selection)."""
    from kicks.synthesis import generate as generate_samples

    generate_samples(
        count=count,
        best_of=best_of,
        instrument=instrument,
        data_dir=data,
        out_dir=out,
        seed=None if seed < 0 else seed,
        vocoder_type="griffinlim" if griffin_lim else "bigvgan",
        refresh_prior=refresh_prior,
    )


@app.command("eval")
def evaluate(
    instrument: str = INSTRUMENT,
    generated: str = typer.Option("output", "--generated", "-g", help="Directory with generated .wav files"),
    pattern: str = typer.Option("gen_*.wav", "--pattern", help="Glob pattern for generated files"),
    reference: str = typer.Option(None, "--reference", "-r", help="Reference corpus directory"),
    ref_samples: int = typer.Option(400, "--ref-samples", help="Corpus subsample size for reference stats"),
    refresh_ref: bool = typer.Option(False, "--refresh-ref", help="Recompute cached reference statistics"),
    json_out: str = typer.Option("", "--json", help="Write full results to this JSON path"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print scores, not per-metric verdicts"),
) -> None:
    """Score generated samples against the corpus and explain how they sound."""
    from kicks.analysis.evaluation import run_eval

    run_eval(
        generated=generated,
        reference=reference,
        instrument=instrument,
        pattern=pattern,
        ref_samples=ref_samples,
        refresh_ref=refresh_ref,
        json_out=json_out or None,
        verbose=not quiet,
    )


@app.command()
def cluster(
    instrument: str = INSTRUMENT,
    data: str = typer.Option(None, "--data", "-d", help="Corpus directory"),
    samples: int = typer.Option(0, "--samples", "-n", help="Number of samples to cluster (0 = all)"),
) -> None:
    """Run GMM clustering and descriptor PCA over the corpus."""
    from kicks.analysis.clustering import run_cluster

    run_cluster(
        data=data, instrument=instrument,
        n_samples=samples if samples > 0 else None,
    )


@app.command("publish-analysis")
def publish_analysis(
    instrument: str = INSTRUMENT,
    out: str = typer.Option("web/public/analysis", "--out", "-o", help="Directory the website serves analysis data from"),
    all_instruments: bool = typer.Option(True, "--all/--only", help="Publish every analysed instrument (default) or only --instrument"),
) -> None:
    """Export cluster reports and cluster audio as static data for the website."""
    from kicks.analysis.publish import publish_analysis as publish

    names = None if all_instruments and instrument is None else [instrument]
    published = publish(out_dir=out, instruments=names)
    print(f"Published {len(published)} instrument(s) to {out}/")


@app.command()
def strip(
    instrument: str = INSTRUMENT,
    data: str = typer.Option(None, "--data", "-d", help="Sample directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze without modifying files"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Copy originals to a backup dir before modifying"),
    exclude_loops: bool = typer.Option(True, "--exclude-loops/--keep-loops", help="Detect and exclude full loops"),
    move_loops: bool = typer.Option(False, "--move-loops", help="Move detected loops out of the source dir (default: copy only)"),
) -> None:
    """Isolate single hits, trimming loop bleed and other instruments."""
    from kicks.corpus import run_strip

    run_strip(
        data=data,
        instrument=instrument,
        dry_run=dry_run,
        backup=backup,
        exclude_loops=exclude_loops,
        move_loops=move_loops,
    )


@app.command()
def clean(
    instrument: str = INSTRUMENT,
    data: str = typer.Option(None, "--data", "-d", help="Corpus directory to clean"),
    quarantine: str = typer.Option(None, "--quarantine", help="Where quarantined files are moved"),
    outlier_pct: float = typer.Option(2.0, "--outlier-pct", help="Percent of most atypical samples to quarantine"),
    apply: bool = typer.Option(False, "--apply", help="Actually move files (default: dry run)"),
) -> None:
    """Quarantine loops, double-hits and perceptual outliers from the corpus."""
    from kicks.corpus import run_clean

    run_clean(
        data=data, instrument=instrument, quarantine_dir=quarantine,
        outlier_pct=outlier_pct, apply=apply,
    )


@app.command()
def listen(
    instrument: str = INSTRUMENT,
    corpus: bool = typer.Option(False, "--corpus", help="Play corpus source files instead of generated output"),
    data: str = typer.Option(None, "--data", "-d", help="Directory to play (overrides --corpus and the default output dir)"),
    pattern: str = typer.Option(None, "--pattern", help="Glob to match (default: gen_*/recon_* for output, *.wav for corpus)"),
    count: int = typer.Option(0, "--count", "-n", help="Max files to play (0 = all)"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Play in random order"),
) -> None:
    """Play WAVs through the system audio player (afplay on macOS, else ffplay)."""
    import random
    import shutil
    import subprocess
    from pathlib import Path

    from kicks.instruments import get_profile

    profile = get_profile(instrument)
    directory = Path(data or (profile.paths.data_dir if corpus else profile.paths.output_dir))

    files = sorted(directory.glob(pattern)) if pattern else sorted(
        p for pat in (["*.wav"] if corpus or data else ["gen_*.wav", "recon_*.wav"])
        for p in directory.glob(pat)
    )
    deduped = list(dict.fromkeys(files))
    if shuffle:
        random.shuffle(deduped)
    if count > 0:
        deduped = deduped[:count]
    if not deduped:
        print(f"No matching .wav files in {directory}")
        raise typer.Exit(1)

    player = shutil.which("afplay")
    args = ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet") if not player else ()
    player = player or shutil.which("ffplay")
    if not player:
        print("No audio player found — install ffmpeg (ffplay) or play the files manually:")
        print("\n".join(str(p) for p in deduped[:20]))
        raise typer.Exit(1)

    print(f"Playing {len(deduped)} file(s) from {directory}/")
    for path in deduped:
        print(f"  {path.name}")
        subprocess.run([player, *args, str(path)], check=False)


@app.command()
def sweep(
    instrument: str = INSTRUMENT,
    server: str = typer.Option("http://localhost:8080", "--server", "-s", help="Base URL of a running `kicks serve`"),
    count: int = typer.Option(20, "--count", "-n", help="Number of slider combinations to sample"),
    out: str = typer.Option("output/sweep", "--out", "-o", help="Directory for generated WAVs"),
    seed: int = typer.Option(42, "--seed", help="Random seed for slider sampling"),
    json_out: str = typer.Option("output/sweep_report.json", "--json", help="Report JSON path"),
) -> None:
    """Sweep the REST API slider space and evaluate every result."""
    from kicks.sweep import run_sweep

    run_sweep(
        server=server, instrument=instrument, count=count,
        out_dir=out, seed=seed, json_out=json_out,
    )
