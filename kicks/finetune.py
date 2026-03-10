"""BigVGAN vocoder fine-tuning on kick drum audio."""

import glob
import itertools
import os
import warnings

import bigvgan
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
from bigvgan import discriminators, loss as bigvgan_loss, mel_spectrogram
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset

from .dataset import TARGET_LUFS
from .model import AUDIO_LENGTH, SAMPLE_RATE, N_FFT, N_MELS, HOP_LENGTH, WIN_SIZE, FMIN, FMAX
from .vocoder import BIGVGAN_MODEL, _patch_bigvgan_from_pretrained

_patch_bigvgan_from_pretrained()


class _KickAudioDataset(Dataset):
    """Loads .wav kick samples as LUFS-normalized waveforms for vocoder training.

    Uses the same audio pre-processing as KickDataset (LUFS normalization to
    -14 dB) so the vocoder is fine-tuned on the same amplitude distribution
    the VAE was trained on.
    """

    def __init__(self, dir: str) -> None:
        self.waveforms: list[torch.Tensor] = []
        lufs_meter = pyln.Meter(SAMPLE_RATE)

        for file in sorted(os.listdir(dir)):
            if not file.endswith(".wav"):
                continue
            data, sr = sf.read(os.path.join(dir, file), dtype="float32")
            if data.ndim == 1:
                audio = torch.from_numpy(data).unsqueeze(0)
            else:
                audio = torch.from_numpy(data.T)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
            if audio.shape[-1] > AUDIO_LENGTH:
                audio = audio[:, :AUDIO_LENGTH]
            elif audio.shape[-1] < AUDIO_LENGTH:
                audio = torch.nn.functional.pad(audio, (0, AUDIO_LENGTH - audio.shape[-1]))

            # LUFS loudness normalization (matches KickDataset in dataset.py)
            audio_np = audio.squeeze(0).numpy()
            loudness = lufs_meter.integrated_loudness(audio_np)
            if np.isfinite(loudness):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Possible clipped samples", module="pyloudnorm")
                    audio_np = pyln.normalize.loudness(audio_np, loudness, TARGET_LUFS)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio = torch.from_numpy(audio_np).unsqueeze(0).float()

            self.waveforms.append(audio.squeeze(0))
        if not self.waveforms:
            raise RuntimeError(f"No .wav files found in {dir}")
        print(f"Loaded {len(self.waveforms)} kick samples")

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.waveforms[idx]


def _find_latest_checkpoint(save_dir: str) -> str | None:
    """Find the most recent checkpoint_*.pth in save_dir."""
    files = glob.glob(os.path.join(save_dir, "checkpoint_*.pth"))
    if not files:
        return None
    return max(files, key=lambda f: int(f.split("_")[-1].split(".")[0]))


def finetune(
    data_dir: str = "data/kicks",
    save_dir: str = "models/vocoder",
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 1e-4,
    grad_accum: int = 4,
    save_every: int = 50,
) -> None:
    """Fine-tune BigVGAN vocoder on kick drum samples using GAN training."""
    from .config import get_device

    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    # Load generator
    generator = bigvgan.BigVGAN.from_pretrained(BIGVGAN_MODEL, use_cuda_kernel=False)
    h = generator.h

    # Verify BigVGAN's config matches our centralized audio constants.
    # A mismatch here means the vocoder would be fine-tuned on a different
    # mel representation than the VAE was trained on.
    assert h.sampling_rate == SAMPLE_RATE, f"sample rate: BigVGAN {h.sampling_rate} != {SAMPLE_RATE}"
    assert h.n_fft == N_FFT, f"n_fft: BigVGAN {h.n_fft} != {N_FFT}"
    assert h.num_mels == N_MELS, f"num_mels: BigVGAN {h.num_mels} != {N_MELS}"
    assert h.hop_size == HOP_LENGTH, f"hop_size: BigVGAN {h.hop_size} != {HOP_LENGTH}"
    assert h.win_size == WIN_SIZE, f"win_size: BigVGAN {h.win_size} != {WIN_SIZE}"

    generator = generator.train().to(device)

    # Freeze all but last 2 upsampling blocks + conv_post
    for param in generator.parameters():
        param.requires_grad = False
    num_ups = len(generator.ups)
    for i in range(max(0, num_ups - 2), num_ups):
        for param in generator.ups[i].parameters():
            param.requires_grad = True
        for j in range(generator.num_kernels):
            for param in generator.resblocks[i * generator.num_kernels + j].parameters():
                param.requires_grad = True
    for param in generator.conv_post.parameters():
        param.requires_grad = True
    if hasattr(generator, "activation_post"):
        for param in generator.activation_post.parameters():
            param.requires_grad = True

    trainable_g = [p for p in generator.parameters() if p.requires_grad]

    # Discriminators
    mpd = discriminators.MultiPeriodDiscriminator(h).to(device)
    cqtd = discriminators.MultiScaleSubbandCQTDiscriminator(h).to(device)
    mel_loss_fn = bigvgan_loss.MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate)

    # Optimizers & schedulers
    optim_g = torch.optim.AdamW(trainable_g, lr=lr, betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), cqtd.parameters()),
        lr=lr, betas=(h.adam_b1, h.adam_b2),
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Resume from checkpoint
    start_epoch = 1
    best_mel_loss = float("inf")
    ckpt_path = _find_latest_checkpoint(save_dir)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        mpd.load_state_dict(ckpt["mpd"])
        cqtd.load_state_dict(ckpt["cqtd"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if "mel_loss" in ckpt:
            best_mel_loss = ckpt["mel_loss"]
        print(f"Resumed from {ckpt_path} (epoch {ckpt['epoch']})")

    # Dataset
    dataset = _KickAudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    gen_total = sum(p.numel() for p in generator.parameters())
    gen_train = sum(p.numel() for p in trainable_g)
    print(f"Generator: {gen_total:,} params ({gen_train:,} trainable)")
    print(f"MPD: {sum(p.numel() for p in mpd.parameters()):,} params")
    print(f"CQT-D: {sum(p.numel() for p in cqtd.parameters()):,} params")

    with Progress(
        TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("G={task.fields[g_loss]:.3f}  D={task.fields[d_loss]:.3f}  Mel={task.fields[mel]:.3f}"),
    ) as progress:
        task = progress.add_task(
            "Fine-tuning", total=epochs - start_epoch + 1,
            epoch=start_epoch, g_loss=0.0, d_loss=0.0, mel=0.0,
        )

        for epoch in range(start_epoch, epochs + 1):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_mel = 0.0
            n_batches = 0

            for step, wav in enumerate(dataloader):
                wav = wav.to(device).unsqueeze(1)
                mel = mel_spectrogram(
                    wav.squeeze(1), N_FFT, N_MELS,
                    SAMPLE_RATE, HOP_LENGTH, WIN_SIZE,
                    FMIN, FMAX, center=False,
                ).to(device)

                # --- Discriminator step ---
                with torch.amp.autocast(device.type, enabled=use_amp):
                    with torch.no_grad():
                        wav_gen_d = generator(mel)
                    y_df_hat_r, y_df_hat_g, _, _ = mpd(wav, wav_gen_d)
                    loss_disc_f, _, _ = bigvgan_loss.discriminator_loss(y_df_hat_r, y_df_hat_g)
                    y_ds_hat_r, y_ds_hat_g, _, _ = cqtd(wav, wav_gen_d)
                    loss_disc_s, _, _ = bigvgan_loss.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_d_val = loss_disc_f.item() + loss_disc_s.item()
                scaler.scale((loss_disc_f + loss_disc_s) / grad_accum).backward()
                del wav_gen_d, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
                del loss_disc_f, loss_disc_s

                # --- Generator step ---
                with torch.amp.autocast(device.type, enabled=use_amp):
                    wav_gen = generator(mel)
                    loss_mel = mel_loss_fn(wav, wav_gen) * h.lambda_melloss
                    mel_val = loss_mel.item()
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav, wav_gen)
                    loss_gen_f, _ = bigvgan_loss.generator_loss(y_df_hat_g)
                    loss_fm_f = bigvgan_loss.feature_loss(fmap_f_r, fmap_f_g)
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = cqtd(wav, wav_gen)
                    loss_gen_s, _ = bigvgan_loss.generator_loss(y_ds_hat_g)
                    loss_fm_s = bigvgan_loss.feature_loss(fmap_s_r, fmap_s_g)

                loss_g_val = loss_gen_f.item() + loss_gen_s.item() + loss_fm_f.item() + loss_fm_s.item() + mel_val
                scaler.scale((loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel) / grad_accum).backward()
                del wav_gen, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
                del fmap_f_r, fmap_f_g, fmap_s_r, fmap_s_g
                del loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s, loss_mel

                if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                    scaler.unscale_(optim_d)
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(mpd.parameters(), cqtd.parameters()), h.clip_grad_norm,
                    )
                    scaler.step(optim_d)
                    optim_d.zero_grad()

                    scaler.unscale_(optim_g)
                    torch.nn.utils.clip_grad_norm_(trainable_g, h.clip_grad_norm)
                    scaler.step(optim_g)
                    optim_g.zero_grad()

                    scaler.update()

                epoch_g_loss += loss_g_val
                epoch_d_loss += loss_d_val
                epoch_mel += mel_val
                n_batches += 1

            scheduler_g.step()
            scheduler_d.step()

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            avg_mel = epoch_mel / max(n_batches, 1)

            progress.update(task, advance=1, epoch=epoch, g_loss=avg_g, d_loss=avg_d, mel=avg_mel)

            # Save best by mel loss
            if avg_mel < best_mel_loss:
                best_mel_loss = avg_mel
                torch.save(
                    {"generator": generator.state_dict(), "epoch": epoch, "mel_loss": avg_mel},
                    os.path.join(save_dir, "best.pth"),
                )

            # Periodic full checkpoint (for resuming)
            if epoch % save_every == 0:
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "mpd": mpd.state_dict(),
                        "cqtd": cqtd.state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "mel_loss": best_mel_loss,
                    },
                    os.path.join(save_dir, f"checkpoint_{epoch}.pth"),
                )

    # Save final weights
    torch.save(
        {"generator": generator.state_dict(), "epoch": epochs, "mel_loss": best_mel_loss},
        os.path.join(save_dir, "final.pth"),
    )
    print(f"Done! Best mel loss: {best_mel_loss:.4f}")
    print(f"Weights saved to {save_dir}/")
