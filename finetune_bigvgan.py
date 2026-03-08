#!/usr/bin/env python3
"""BigVGAN Fine-tuning Script for Kick Drum Vocoder."""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages with version numbers."""
    packages = [
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "bigvgan==2.4.1",
        "huggingface_hub==0.36.2",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


install_requirements()

import glob
import itertools

import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import bigvgan
from bigvgan import discriminators
from bigvgan import loss

MultiPeriodDiscriminator = discriminators.MultiPeriodDiscriminator
MultiScaleSubbandCQTDiscriminator = discriminators.MultiScaleSubbandCQTDiscriminator
generator_loss = loss.generator_loss
discriminator_loss = loss.discriminator_loss
feature_loss = loss.feature_loss
MultiScaleMelSpectrogramLoss = loss.MultiScaleMelSpectrogramLoss
mel_spectrogram = bigvgan.mel_spectrogram

SAMPLE_RATE = 44100
AUDIO_LENGTH = 65536
VOCODER_MODEL = "nvidia/bigvgan_v2_44khz_128band_256x"
print(f"Loading BigVGAN model: {VOCODER_MODEL}...")


class KickAudioDataset(Dataset):
    def __init__(self, dir: str) -> None:
        self.waveforms: list[torch.Tensor] = []
        for file in sorted(os.listdir(dir)):
            if not file.endswith('.wav'):
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
            audio = audio / (audio.abs().max() + 1e-8)
            self.waveforms.append(audio.squeeze(0))
        print(f'Loaded {len(self.waveforms)} kick samples')

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.waveforms[idx]


def find_latest_checkpoint(save_dir: str):
    """Find the most recent checkpoint_*.pth in save_dir."""
    pattern = os.path.join(save_dir, 'checkpoint_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))


def finetune(
    data_dir: str = "data/kicks",
    save_dir: str = "models",
    epochs: int = 200,
    batch_size: int = 2,
    lr: float = 1e-4,
    grad_accum: int = 4,
    save_every: int = 50,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')

    generator = bigvgan.BigVGAN.from_pretrained(VOCODER_MODEL, use_cuda_kernel=False)
    h = generator.h
    generator = generator.train().to(device)

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
    if hasattr(generator, 'activation_post'):
        for param in generator.activation_post.parameters():
            param.requires_grad = True

    trainable_g = [p for p in generator.parameters() if p.requires_grad]

    mpd = MultiPeriodDiscriminator(h).to(device)
    cqtd = MultiScaleSubbandCQTDiscriminator(h).to(device)

    mel_loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate)

    optim_g = torch.optim.AdamW(trainable_g, lr=lr, betas=(h.adam_b1, h.adam_b2))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), cqtd.parameters()),
        lr=lr, betas=(h.adam_b1, h.adam_b2),
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay)
    scaler = GradScaler()

    start_epoch = 1
    best_mel_loss = float('inf')
    ckpt_path = find_latest_checkpoint(save_dir)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        mpd.load_state_dict(ckpt['mpd'])
        cqtd.load_state_dict(ckpt['cqtd'])
        optim_g.load_state_dict(ckpt['optim_g'])
        optim_d.load_state_dict(ckpt['optim_d'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        if 'mel_loss' in ckpt:
            best_mel_loss = ckpt['mel_loss']
        print(f'Resumed from {ckpt_path} (epoch {ckpt["epoch"]})')
    else:
        print('Starting from scratch')

    dataset = KickAudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    gen_total = sum(p.numel() for p in generator.parameters())
    gen_train = sum(p.numel() for p in trainable_g)
    print(f'Generator: {gen_total:,} params ({gen_train:,} trainable)')
    print(f'MPD: {sum(p.numel() for p in mpd.parameters()):,} params')
    print(f'CQT-D: {sum(p.numel() for p in cqtd.parameters()):,} params')
    print(f'Training epochs {start_epoch}-{epochs}, batch_size={batch_size}, grad_accum={grad_accum}')

    for epoch in range(start_epoch, epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_mel = 0.0
        n_batches = 0

        for step, wav in enumerate(dataloader):
            wav = wav.to(device).unsqueeze(1)
            mel = mel_spectrogram(
                wav.squeeze(1), h.n_fft, h.num_mels,
                h.sampling_rate, h.hop_size, h.win_size,
                h.fmin, h.fmax, center=False,
            ).to(device)

            with autocast():
                with torch.no_grad():
                    wav_gen_d = generator(mel)
                y_df_hat_r, y_df_hat_g, _, _ = mpd(wav, wav_gen_d)
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, _, _ = cqtd(wav, wav_gen_d)
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_d_val = loss_disc_f.item() + loss_disc_s.item()
            scaler.scale((loss_disc_f + loss_disc_s) / grad_accum).backward()
            del wav_gen_d, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
            del loss_disc_f, loss_disc_s
            torch.cuda.empty_cache()

            with autocast():
                wav_gen = generator(mel)
                loss_mel = mel_loss_fn(wav, wav_gen) * h.lambda_melloss
                mel_val = loss_mel.item()
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav, wav_gen)
                loss_gen_f, _ = generator_loss(y_df_hat_g)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = cqtd(wav, wav_gen)
                loss_gen_s, _ = generator_loss(y_ds_hat_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)

            loss_g_val = loss_gen_f.item() + loss_gen_s.item() + loss_fm_f.item() + loss_fm_s.item() + mel_val
            scaler.scale((loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + loss_mel) / grad_accum).backward()
            del wav_gen, y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g
            del fmap_f_r, fmap_f_g, fmap_s_r, fmap_s_g
            del loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s, loss_mel
            torch.cuda.empty_cache()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optim_d)
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(mpd.parameters(), cqtd.parameters()), h.clip_grad_norm)
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

        if epoch % 10 == 0 or epoch == start_epoch:
            print(f'Epoch {epoch:>3d}/{epochs}  G={avg_g:.3f}  D={avg_d:.3f}  Mel={avg_mel:.3f}')

        if avg_mel < best_mel_loss:
            best_mel_loss = avg_mel
            torch.save({
                'generator': generator.state_dict(),
                'epoch': epoch,
                'mel_loss': avg_mel,
            }, os.path.join(save_dir, 'best.pth'))

        if epoch % save_every == 0:
            torch.save({
                'generator': generator.state_dict(),
                'mpd': mpd.state_dict(),
                'cqtd': cqtd.state_dict(),
                'optim_g': optim_g.state_dict(),
                'optim_d': optim_d.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'mel_loss': best_mel_loss,
            }, os.path.join(save_dir, f'checkpoint_{epoch}.pth'))
            print(f'  Saved checkpoint at epoch {epoch}')

    torch.save({
        'generator': generator.state_dict(),
        'epoch': epochs,
        'mel_loss': best_mel_loss,
    }, os.path.join(save_dir, 'final.pth'))

    print(f'\nDone! Best mel loss: {best_mel_loss:.4f}')
    print(f'Weights saved to {save_dir}/')


if __name__ == "__main__":
    finetune()
