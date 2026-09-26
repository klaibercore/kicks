"""Training loop for the descriptor-conditioned waveform diffusion backend.

The objective is plain v-prediction: noise a hit to a random level, ask the
network for the velocity, and take the mean squared error against the true one.
Two things sit on top, both of them decisions from issue #3:

* **Conditioning dropout.** A fraction of each batch is trained against the
  learned null embedding instead of its descriptors, which is what makes
  classifier-free guidance available at sampling time. Retrofitting it would
  mean retraining, so it is on by default and turned off with ``0``.
* **EMA weights.** Sampling quality tracks the exponential moving average, not
  the live weights, so validation and every saved checkpoint use the average.

Validation is deliberately boring: each hit keeps the same noise level and the
same noise vector in every epoch, so the curve moves only when the model does.
The denoising loss says nothing about whether the sliders work, so an optional
control proxy samples a few hits at their validation descriptor targets and
measures how far the rendered audio lands from them. It costs a full sampling
pass per evaluation and is off by default.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager

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
from torch.utils.data import DataLoader, random_split

from ..data.waveforms import PREPROCESSING, label_waveform, peak_normalize
from ..instruments import InstrumentProfile
from ..nn.diffusion import diffuse, v_sample, v_target
from .tracking import TrainingRun, runs_root, split_fingerprint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Lowest and highest validation noise level. The ends are trimmed because a
#: sigma of exactly 0 or 1 carries no gradient signal worth tracking.
VALIDATION_RANGE = (0.02, 0.98)


class EMA:
    """Exponential moving average of the parameters, with buffers copied through.

    Buffers (the descriptor statistics) are constants, not learned quantities;
    averaging them would only make them lag their own value.
    """

    #: Warm-up horizon. Early on, ``(1 + step) / (WARMUP + step)`` is well below
    #: the target decay, so the average tracks the weights instead of staying
    #: pinned to the initialization — otherwise a short screening run would
    #: validate and checkpoint a model that has barely moved.
    WARMUP = 10

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.steps = 0
        self.parameters = {name for name, _ in model.named_parameters()}
        self.shadow = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
        }

    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        decay = min(self.decay, (1 + self.steps) / (self.WARMUP + self.steps))
        for name, value in model.state_dict().items():
            shadow = self.shadow[name]
            if name in self.parameters:
                shadow.mul_(decay).add_(value.detach().float(), alpha=1 - decay)
            else:
                shadow.copy_(value.detach().float())

    def state_dict(self, reference: torch.nn.Module) -> dict:
        """The averaged weights, cast back to the live model's dtypes."""
        return {
            name: self.shadow[name].to(dtype=value.dtype, device=value.device)
            for name, value in reference.state_dict().items()
        }

    @contextmanager
    def applied(self, model: torch.nn.Module):
        """Temporarily swap the averaged weights into ``model``."""
        backup = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(self.state_dict(model))
        try:
            yield model
        finally:
            model.load_state_dict(backup)


def diffusion_loss(
    model,
    audio: torch.Tensor,
    labels: torch.Tensor | None,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    cond_mask: torch.Tensor | None = None,
    reduce: bool = True,
):
    """Mean squared velocity error at the given noise levels.

    With ``reduce=False`` the per-sample losses come back instead, which is how
    validation splits its curve by noise level.
    """
    noisy = diffuse(audio, noise, sigmas)
    predicted = model(noisy, sigmas, labels=labels, cond_mask=cond_mask)
    squared = (predicted - v_target(audio, noise, sigmas)).pow(2)
    return squared.mean() if reduce else squared.mean(dim=tuple(range(1, squared.dim())))


def _accumulation_group(index: int, batches: int, grad_accum: int) -> int:
    """Micro-batches in the accumulation group that 1-based ``index`` belongs to.

    Every group holds ``grad_accum`` micro-batches except the last one of an
    epoch, which holds whatever remains. Dividing each loss by its own group's
    size makes every optimizer step a mean, not a fraction of one.
    """
    full = batches - batches % grad_accum
    return grad_accum if index <= full else batches - full


def _validation_sigmas(count: int, offset: int, total: int, device, dtype) -> torch.Tensor:
    """Fixed noise levels, assigned by position in the split rather than at random.

    The levels sweep the whole schedule across the split, so the curve keeps
    covering hard, half-noised and nearly clean audio however small the
    validation set is. Position is stable because the loader does not shuffle,
    so a given hit is measured at the same level in every epoch.
    """
    positions = torch.arange(offset, offset + count, device=device).to(dtype)
    low, high = VALIDATION_RANGE
    return low + (high - low) * positions / max(1, total - 1)


def train_diffusion(
    model,
    dataset,
    optimizer,
    profile: InstrumentProfile,
    *,
    epochs: int = 200,
    device: torch.device | None = None,
    batch_size: int = 8,
    val_split: float = 0.1,
    seed: int = 42,
    scheduler=None,
    cond_dropout: float = 0.1,
    ema_decay: float = 0.999,
    grad_accum: int = 1,
    eval_every: int = 0,
    eval_steps: int = 20,
    eval_samples: int = 8,
    eval_guidance: float = 1.0,
    source_checkpoint: str | None = None,
    run_name: str | None = None,
    intent: str = "",
    hypothesis: str = "",
    success_criteria: str = "",
    runs_dir: str | None = None,
    tracker: TrainingRun | None = None,
) -> dict[str, list[float]]:
    """Train the denoiser. Returns the per-epoch train, validation and proxy curves."""
    paths = profile.paths
    os.makedirs(paths.model_dir or ".", exist_ok=True)
    if epochs < 1 or batch_size < 1 or grad_accum < 1:
        raise ValueError("epochs, batch size and gradient accumulation must be positive")
    if eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if not 0.0 <= cond_dropout < 1.0:
        raise ValueError("conditioning dropout must be in [0, 1)")
    if len(dataset) < 2 or not 0 < val_split < 1:
        raise ValueError("training needs at least two samples and a validation split in (0, 1)")

    device = device or next(model.parameters()).device
    model.to(device)

    n_val = max(1, min(len(dataset) - 1, int(len(dataset) * val_split)))
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # A short trailing micro-batch would otherwise count as a full one inside
    # its accumulation group; dropping it keeps every optimizer step an
    # unweighted mean over equally sized micro-batches. Tiny corpora that do not
    # fill one micro-batch keep theirs, or there would be nothing to train on.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=len(train_set) >= batch_size,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Conditioning statistics come from the training split only; fitting them on
    # the whole corpus would leak the validation hits into every label the model
    # ever sees. A resumed model keeps the statistics it was trained with. The
    # label bank — the rows generation draws its targets from — is rebuilt from
    # the current split either way, so targets follow the corpus being trained
    # on rather than the one the weights started from.
    if hasattr(dataset, "label_stats"):
        if source_checkpoint is None:
            mean, std = dataset.label_stats(train_set.indices)
            model.set_label_stats(mean, std)
        model.set_label_bank(dataset.label_matrix(train_set.indices))

    execution_context = None
    if encoded_context := os.environ.get("KICKS_TRAINING_CONTEXT"):
        try:
            execution_context = json.loads(encoded_context)
        except json.JSONDecodeError as error:
            raise ValueError("KICKS_TRAINING_CONTEXT must be valid JSON") from error
        if not isinstance(execution_context, dict):
            raise ValueError("KICKS_TRAINING_CONTEXT must contain a JSON object")

    config = {
        "instrument": profile.name,
        "backend": "waveform_diffusion",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "val_split": val_split,
        "seed": seed,
        "cond_dropout": cond_dropout,
        "ema_decay": ema_decay,
        "eval_every": eval_every,
        "eval_steps": eval_steps,
        "eval_samples": eval_samples,
        "eval_guidance": eval_guidance,
        "source_checkpoint": source_checkpoint,
        "length": model.length,
        "descriptors": [d.key for d in profile.descriptors],
        "architecture": model.architecture,
        "parameters": sum(p.numel() for p in model.parameters()),
        "corpus_samples": len(dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "optimizer": type(optimizer).__name__,
        "checkpoint": str(os.path.abspath(paths.diffusion_checkpoint)),
        "objective": "v_prediction_angular_v1",
        "validation_objective": "fixed_sigma_grid_fixed_noise_v1",
        "preprocessing": PREPROCESSING,
        "split_fingerprint": split_fingerprint(dataset, seed, val_split),
        "runtime": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "accelerator": (
                {
                    "name": torch.cuda.get_device_name(device),
                    "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                }
                if device.type == "cuda" else {"name": str(device)}
            ),
        },
    }
    if execution_context is not None:
        config["execution"] = execution_context
    run = tracker or TrainingRun(
        runs_root(runs_dir, paths.output_root), config, name=run_name,
        notes={"objective": intent, "hypothesis": hypothesis,
               "success_criteria": success_criteria},
    )
    print(f"Training dashboard: http://127.0.0.1:6060/?run={run.id}", flush=True)
    print(f"  Standalone HTML: {os.path.join(os.path.abspath(str(run.directory)), 'index.html')}",
          flush=True)

    ema = EMA(model, ema_decay)
    noise_generator = torch.Generator().manual_seed(seed + 1)

    def save(path: str, **extra) -> None:
        # Write beside the target and rename, so a kill or a volume snapshot
        # mid-write never leaves a truncated best checkpoint behind.
        temporary = f"{path}.tmp"
        torch.save({
            "model": ema.state_dict(model),
            "label_bank": model.label_bank,
            "instrument": profile.name,
            "descriptors": [d.key for d in profile.descriptors],
            "training": {
                "seed": seed, "cond_dropout": cond_dropout, "ema_decay": ema_decay,
                "source_checkpoint": source_checkpoint, "run_id": run.id,
                "preprocessing": PREPROCESSING,
                "objective": config["objective"],
                "validation": config["validation_objective"],
            },
            **model.checkpoint_meta(), **extra,
        }, temporary)
        os.replace(temporary, path)

    def validate() -> dict[str, float]:
        totals = {"": [0.0, 0], "low": [0.0, 0], "mid": [0.0, 0], "high": [0.0, 0]}
        was_training = model.training
        with ema.applied(model), torch.no_grad():
            model.eval()
            offset = 0
            for index, (audio, labels) in enumerate(val_loader, 1):
                audio, labels = audio.to(device), labels.to(device)
                sigmas = _validation_sigmas(len(audio), offset, len(val_set), device, audio.dtype)
                noise = torch.randn(
                    audio.shape, generator=torch.Generator().manual_seed(seed + offset),
                ).to(device)
                per_sample = diffusion_loss(model, audio, labels, sigmas, noise, reduce=False)
                if not torch.isfinite(per_sample).all():
                    raise FloatingPointError("Validation loss became non-finite")
                for value, sigma in zip(per_sample.tolist(), sigmas.tolist()):
                    band = "low" if sigma < 1 / 3 else ("mid" if sigma < 2 / 3 else "high")
                    for key in ("", band):
                        totals[key][0] += value
                        totals[key][1] += 1
                offset += len(audio)
                run.progress(batch=index, batches=len(val_loader))
        model.train(was_training)
        return {
            f"val_loss{'_' + key + '_sigma' if key else ''}": total / count
            for key, (total, count) in totals.items() if count
        }

    def control_proxy() -> float | None:
        """Mean descriptor-target error of freshly sampled hits, in std units.

        A controllability proxy, not a perceptual rating and not a substitute
        for the control audit in ``scripts/validate_controls.py``.
        """
        targets = torch.stack([
            dataset[val_set.indices[i]][1]
            for i in range(min(eval_samples, len(val_set)))
        ])
        with ema.applied(model):
            audio = v_sample(
                model, labels=targets, steps=eval_steps, guidance=eval_guidance,
                generator=torch.Generator().manual_seed(seed + 7), device=device,
            ).cpu()
        errors = []
        scale = model.label_std.detach().cpu().numpy()
        for row, target in zip(audio, targets.numpy()):
            # Training labels were measured at the dataset's fixed peak, and the
            # mel path expects audio inside [-1, 1]; measure the sample the same
            # way rather than at whatever level the sampler happened to land on.
            measured = label_waveform(peak_normalize(row), profile)
            errors.append(np.abs(measured - target) / np.maximum(scale, 1e-6))
        value = float(np.mean(errors))
        return value if math.isfinite(value) else None

    train_curve: list[float] = []
    val_curve: list[float] = []
    proxy_curve: list[float | None] = []
    best_val = float("inf")
    best_control = float("inf")

    try:
        run.progress(phase="baseline validation", force=True)
        baseline = validate()
        run.epoch({"epoch": 0, **baseline,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_seconds": time.monotonic() - run.started})
        if source_checkpoint is not None:
            best_val = baseline["val_loss"]
            save(paths.diffusion_checkpoint, epoch=0, val_loss=best_val)
            print(f"Fine-tune baseline validation: {best_val:.6f}", flush=True)

        with Progress(
            TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("Train: {task.fields[loss]:.4f}  Val: {task.fields[val]:.4f}"),
        ) as progress:
            task = progress.add_task(
                f"Diffusion {profile.name}", total=epochs, epoch=0, loss=0.0, val=0.0,
            )
            for epoch in range(epochs):
                epoch_start = time.monotonic()
                if device.type == "cuda":
                    # Reset per epoch so CUDA memory figures can be compared with
                    # epoch timing instead of reporting one process-lifetime high.
                    torch.cuda.reset_peak_memory_stats(device)
                learning_rate = optimizer.param_groups[0]["lr"]
                run.progress(phase="training", epoch=epoch + 1, batch=0,
                             batches=len(train_loader), force=True)
                model.train()
                losses, sizes = [], []
                optimizer.zero_grad(set_to_none=True)
                batches = len(train_loader)
                for index, (audio, labels) in enumerate(train_loader, 1):
                    audio, labels = audio.to(device), labels.to(device)
                    count = len(audio)
                    sigmas = torch.rand(count, generator=noise_generator).to(device)
                    noise = torch.randn(audio.shape, generator=noise_generator).to(device)
                    keep = torch.rand(count, generator=noise_generator).to(device) >= cond_dropout
                    loss = diffusion_loss(model, audio, labels, sigmas, noise, cond_mask=keep)
                    if not math.isfinite(loss.item()):
                        raise FloatingPointError("Training loss became non-finite")
                    (loss / _accumulation_group(index, batches, grad_accum)).backward()
                    if index % grad_accum == 0 or index == batches:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        ema.update(model)
                    losses.append(loss.item())
                    sizes.append(count)
                    run.progress(batch=index, train_loss=losses[-1],
                                 learning_rate=learning_rate)

                average = float(np.average(losses, weights=sizes))
                train_curve.append(average)
                if scheduler is not None:
                    scheduler.step()

                run.progress(phase="validation", batch=0, batches=len(val_loader), force=True)
                metrics = validate()
                val_curve.append(metrics["val_loss"])
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    save(paths.diffusion_checkpoint, epoch=epoch + 1, val_loss=best_val)

                proxy = None
                if eval_every > 0 and (epoch + 1) % eval_every == 0:
                    run.progress(phase="control proxy", force=True)
                    proxy = control_proxy()
                    if proxy is not None and proxy < best_control:
                        best_control = proxy
                        save(paths.diffusion_control_checkpoint, epoch=epoch + 1,
                             val_loss=metrics["val_loss"], control_mae=proxy)
                    progress.console.log(
                        f"epoch {epoch + 1}: control_mae="
                        f"{'n/a' if proxy is None else f'{proxy:.3f}'} sd",
                    )
                proxy_curve.append(proxy)

                cuda_peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else None
                )

                progress.update(task, advance=1, epoch=epoch + 1,
                                loss=average, val=metrics["val_loss"])
                run.epoch({"epoch": epoch + 1, "train_loss": average, **metrics,
                           "control_mae": proxy, "learning_rate": learning_rate,
                           "cuda_peak_memory_mb": cuda_peak_memory_mb,
                           "epoch_seconds": time.monotonic() - epoch_start,
                           "elapsed_seconds": time.monotonic() - run.started})

        run.progress(phase="saving artifacts", force=True)
        save(paths.diffusion_final_checkpoint, epoch=epochs, loss_history=train_curve)
        _plot(paths.diffusion_loss_curves, train_curve, val_curve, proxy_curve)
    except KeyboardInterrupt:
        run.finish("interrupted", "Training interrupted; completed epochs are preserved.")
        raise
    except BaseException as error:
        run.finish("failed", f"{type(error).__name__}: {error}")
        raise
    else:
        run.finish()
    return {"train": train_curve, "val": val_curve, "control_mae": proxy_curve}


def _plot(path: str, train: list[float], val: list[float], proxy: list) -> None:
    measured = [(i + 1, v) for i, v in enumerate(proxy) if v is not None]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(train) + 1), train, label="train")
    axes[0].plot(range(1, len(val) + 1), val, label="validation")
    axes[0].set_title("v-prediction loss")
    axes[0].legend()
    if measured:
        axes[1].plot(*zip(*measured), marker="o")
    axes[1].set_title("Descriptor target error (sd)")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid()
    plt.tight_layout()
    plt.savefig(path)
    plt.close(figure)


__all__ = ["EMA", "diffusion_loss", "train_diffusion"]
