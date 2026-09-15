"""Inference for ETH DISCO's DisCoder Z checkpoint.

Adapted from ETH-DISCO/discoder, commit
8aee1ee82008d4ac48cce6df7a85c65f7d157581 (MIT; see DISCODER_LICENSE).
DAC and alias-free activations come from their upstream packages.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from bigvgan.alias_free_activation.torch import Activation1d
from dac.model.dac import Decoder as DACDecoder
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ..audio.constants import HOP_LENGTH, LOG_MEL_MIN


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).square()


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size,
                                  dilation=d, padding=(kernel_size - 1) * d // 2))
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2))
            for _ in dilations
        ])
        self.acts = nn.ModuleList([Activation1d(Snake1d(channels)) for _ in range(2 * len(dilations))])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts[::2], self.acts[1::2]):
            x = x + c2(a2(c1(a1(x))))
        return x


class DisCoderEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        initial = cfg["initial_out_channels"]
        intermediate = cfg["intermediate_dim"]
        latent = cfg["codebook_size"]
        def conv(cin, cout, stride=1):
            return weight_norm(nn.Conv1d(cin, cout, 7, padding=3, stride=stride))
        self.init_conv = conv(config["mel"]["n_mels"], initial)
        self.sec_conv = conv(initial, intermediate)
        self.up_conv = conv(intermediate, intermediate)
        self.acts = nn.ModuleList([Activation1d(Snake1d(c))
                                   for c in (initial, intermediate, intermediate, initial, latent)])
        self.resblocks = nn.ModuleList([
            ResBlock(intermediate, kernel, dilations)
            for kernel, dilations in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilations"])
        ])
        self.strided_conv = conv(intermediate, initial, stride=2)
        self.down_conv = conv(initial, latent)
        self.output_conv = nn.Conv1d(latent, latent, 7, padding=3)

    def forward(self, mel):
        skip = self.init_conv(mel)
        x = self.acts[0](skip)
        x = self.acts[1](self.sec_conv(x))
        x = self.acts[2](self.up_conv(x))
        residual = x
        for block in self.resblocks:
            x = block(x)
        x = self.acts[3](self.strided_conv(x + residual))
        x = self.acts[4](self.down_conv(x))
        return self.output_conv(x), F.avg_pool1d(skip, 2, stride=2)


class DisCoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # All weights of this DAC 44 kHz decoder are in the DisCoder checkpoint.
        self.dac_decoder = DACDecoder(input_channel=1024, channels=1536, rates=[8, 8, 4, 2])

    def forward(self, latent, skip):
        return self.dac_decoder(latent + skip)


class DisCoder(nn.Module):
    # The released model has 430M parameters. Bound activation memory during
    # CLI best-of-k generation as well as the single-hit interactive path.
    inference_batch_size = 1

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        if cfg["predict_type"] != "z" or cfg["activation"] != "snake" or cfg["resblock_type"] != "AMP":
            raise ValueError("Only the official DisCoder Z/Snake/AMP checkpoint is supported")
        self.config = config
        self.encoder = DisCoderEncoder(config)
        self.decoder = DisCoderDecoder()

    def forward(self, mel):
        if mel.ndim != 3 or mel.shape[1] != self.config["mel"]["n_mels"] or mel.shape[-1] < 1:
            raise ValueError("DisCoder expects (batch, 128, frames) log-mel input")
        frames = mel.shape[-1]
        # Pad to a training segment: stride-2 convolution and average pooling
        # otherwise disagree for odd frame counts. Crop back to the exact hit.
        multiple = self.config["segment_size"] // HOP_LENGTH
        pad = (-frames) % multiple
        if pad:
            mel = F.pad(mel, (0, pad), value=LOG_MEL_MIN)
        latent, skip = self.encoder(mel)
        return self.decoder(latent, skip)[..., :frames * HOP_LENGTH]

    def remove_weight_norm(self):
        for module in self.modules():
            if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            elif hasattr(module, "weight_g"):
                torch.nn.utils.remove_weight_norm(module)
