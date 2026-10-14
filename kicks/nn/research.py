"""Descriptor-conditioned parameter, noise/DSP, GAN and codec-sequence models.

Original compact research implementations, not replicas of upstream named
checkpoints. Every family is explicitly identified in checkpoint metadata.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from ..audio.constants import AUDIO_LENGTH
from ..instruments import get_profile
from .ddsp import NoiseBands, finish_audio, modal_body

KINDS = ('parameters', 'ddsp', 'noiseband', 'hybrid', 'drumgan', 'codec', 'wavevae')


class Conditioned(nn.Module):
    def __init__(self, instrument):
        super().__init__()
        self.instrument = instrument
        self.n_descriptors = len(get_profile(instrument).descriptors)
        self.register_buffer('label_mean', torch.zeros(self.n_descriptors))
        self.register_buffer('label_std', torch.ones(self.n_descriptors))
        self.label_bank = None

    def set_labels(self, labels):
        if labels.ndim != 2 or labels.shape[1] != self.n_descriptors or not len(labels) or not torch.isfinite(labels).all():
            raise ValueError('invalid training label bank')
        self.label_bank = labels.detach().float().cpu()
        self.label_mean.copy_(labels.mean(0))
        self.label_std.copy_(labels.std(0, unbiased=False).clamp_min(1e-3))

    def condition(self, labels):
        return ((labels-self.label_mean)/self.label_std).clamp(-8, 8)


class ParameterVAE(Conditioned):
    def __init__(self, instrument='kick', hidden=64, latent=8):
        super().__init__(instrument)
        self.dim = len(get_profile(instrument).procedural.parameters)
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(self.dim+self.n_descriptors, hidden), nn.SiLU(), nn.Linear(hidden, latent*2))
        self.decoder = nn.Sequential(nn.Linear(latent+self.n_descriptors, hidden), nn.SiLU(), nn.Linear(hidden, self.dim))

    def decode(self, z, labels):
        return self.decoder(torch.cat((z, self.condition(labels)), 1)).sigmoid()

    def forward(self, positions, labels):
        mu, lv = self.encoder(torch.cat((positions, self.condition(labels)), 1)).chunk(2, 1)
        lv = 6*torch.tanh(lv/6)
        z = mu+torch.exp(.5*lv)*torch.randn_like(mu)
        return self.decode(z, labels), -.5*(1+lv-mu.square()-lv.exp()).mean()

    def sample(self, labels, seed=0):
        z = torch.randn(len(labels), self.latent, generator=torch.Generator().manual_seed(seed)).to(labels)
        return self.decode(z, labels)


class TextureVAE(Conditioned):
    def __init__(self, instrument='kick', hidden=64, latent=8, bands=16, frames=64, length=AUDIO_LENGTH, kind='hybrid'):
        super().__init__(instrument)
        self.latent, self.frames, self.kind, self.length = latent, frames, kind, length
        self.dim = len(get_profile(instrument).procedural.parameters)
        self.encoder = nn.Sequential(nn.Linear(128+self.n_descriptors, hidden), nn.SiLU(), nn.Linear(hidden, latent*2))
        self.decoder = nn.Sequential(nn.Linear(latent+self.n_descriptors, hidden), nn.SiLU(), nn.Linear(hidden, self.dim+bands*frames))
        self.noise = NoiseBands(bands, length, low_hz=40 if kind == 'noiseband' else (1500 if instrument == 'kick' else 800))
        # Quiet late tails before fitting; full-waveform spectral objectives teach
        # the attack/body/noise decomposition jointly, without subtraction.
        nn.init.constant_(self.decoder[-1].bias[self.dim:], -4)

    def decode(self, z, labels, seed=0):
        out = self.decoder(torch.cat((z, self.condition(labels)), 1))
        positions = out[:, :self.dim].sigmoid()
        gains = F.softplus(out[:, self.dim:]).reshape(len(z), self.noise.bands, self.frames)
        # A smooth tail boundary is separate from the learned envelope.
        gains = gains*torch.linspace(1, 0, self.frames, device=gains.device)[None, None]
        texture = self.noise(gains, seed=seed)
        body = modal_body(self.instrument, positions, self.length)
        if self.kind == 'noiseband':
            audio = texture
        elif self.kind == 'ddsp':
            audio = body+texture*.5
        else:
            audio = body+texture
        return finish_audio(audio), positions

    def forward(self, audio, labels, seed=0):
        # Encode energy, rather than arbitrary reference noise phase.
        summary = F.adaptive_avg_pool1d(audio.square(), 128).clamp_min(1e-8).log().flatten(1)
        mu, lv = self.encoder(torch.cat((summary, self.condition(labels)), 1)).chunk(2, 1)
        lv = 6*torch.tanh(lv/6)
        z = mu+torch.exp(.5*lv)*torch.randn_like(mu)
        rendered, positions = self.decode(z, labels, seed)
        return rendered, -.5*(1+lv-mu.square()-lv.exp()).mean(), positions

    def sample(self, labels, seed=0):
        z = torch.randn(len(labels), self.latent, generator=torch.Generator().manual_seed(seed)).to(labels)
        return self.decode(z, labels, seed)[0]


class DrumGAN(Conditioned):
    """Conditional waveform GAN borrowing DrumGAN's descriptor-control idea.

    Full-band local architecture, not Sony's historical 16 kHz implementation.
    """
    def __init__(self, instrument='kick', width=32, latent=32, length=AUDIO_LENGTH):
        super().__init__(instrument)
        if length % 1024:
            raise ValueError('GAN length must be divisible by 1024')
        self.latent, self.length, self.start = latent, length, length//1024
        self.input = nn.Linear(latent+self.n_descriptors, width*4*self.start)
        layers = []
        previous = width*4
        for channels in (width*4, width*2, width, width//2, 1):
            layers += [nn.ConvTranspose1d(previous, channels, 8, 4, 2)]
            layers += [nn.Tanh() if channels == 1 else nn.LeakyReLU(.2)]
            previous = channels
        self.generator = nn.Sequential(*layers)
        self.width = width

    def sample(self, labels, seed=0):
        z = torch.randn(len(labels), self.latent, generator=torch.Generator().manual_seed(seed)).to(labels)
        x = self.input(torch.cat((z, self.condition(labels)), 1)).reshape(len(labels), self.width*4, self.start)
        return finish_audio(self.generator(x))


class WaveCritic(nn.Module):
    def __init__(self, descriptors, width=16):
        super().__init__()
        layers, previous = [], 1
        for channels in (width, width*2, width*4, width*4, width*4):
            layers += [nn.Conv1d(previous, channels, 8, 4, 2), nn.LeakyReLU(.2)]
            previous = channels
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(previous, 1)
        self.condition = nn.Linear(descriptors, previous)

    def forward(self, audio, labels):
        x = self.layers(audio).mean(-1)
        return self.output(x).squeeze(1)+(x*self.condition(labels)).sum(1)/math.sqrt(x.shape[1])


class WaveVAE(Conditioned):
    """Compact conditional waveform VAE, a RAVE-inspired research baseline.

    Uses an explicit normal latent prior and full-waveform reconstruction loss;
    this is not the upstream RAVE architecture or its streaming export format.
    """
    def __init__(self, instrument='kick', width=16, latent=16, length=AUDIO_LENGTH):
        super().__init__(instrument)
        if length % 1024: raise ValueError('wave VAE length must be divisible by 1024')
        self.latent,self.frames=latent,length//1024
        layers,previous=[],1
        for channels in (width,width*2,width*4,width*4,width*4):
            layers += [nn.Conv1d(previous,channels,8,4,2),nn.SiLU()];previous=channels
        self.encoder=nn.Sequential(*layers,nn.Conv1d(previous,latent*2,1))
        self.conditioning=nn.Linear(self.n_descriptors,latent)
        layers,previous=[],latent
        for channels in (width*4,width*4,width*2,width,1):
            layers += [nn.ConvTranspose1d(previous,channels,8,4,2),nn.Tanh() if channels==1 else nn.SiLU()]
            previous=channels
        self.decoder=nn.Sequential(*layers)

    def decode(self,z,labels):
        return finish_audio(self.decoder(z+self.conditioning(self.condition(labels))[:,:,None]))

    def forward(self,audio,labels):
        mu,lv=self.encoder(audio).chunk(2,1);lv=6*torch.tanh(lv/6)
        z=mu+(.5*lv).exp()*torch.randn_like(mu)
        return self.decode(z,labels),-.5*(1+lv-mu.square()-lv.exp()).mean()

    def sample(self,labels,seed=0):
        z=torch.randn(len(labels),self.latent,self.frames,generator=torch.Generator().manual_seed(seed)).to(labels)
        return self.decode(z,labels)


class CodecDenoiser(Conditioned):
    """Continuous quantized DAC-latent sequence v-diffusion with temporal attention."""
    def __init__(self, instrument='kick', channels=1024, frames=128, hidden=64, layers=2):
        super().__init__(instrument)
        self.channels, self.frames = channels, frames
        self.input = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, channels)
        self.conditioning = nn.Linear(self.n_descriptors+3, hidden)
        self.position = nn.Parameter(torch.randn(1, frames, hidden)*.01)
        block = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout=0, batch_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.register_buffer('latent_mean', torch.zeros(1, channels, 1))
        self.register_buffer('latent_std', torch.ones(1, channels, 1))

    def forward(self, x, sigmas, labels):
        t = torch.stack((sigmas, torch.sin(sigmas*math.pi), torch.cos(sigmas*math.pi)), 1)
        c = self.conditioning(torch.cat((self.condition(labels), t), 1))[:, None]
        return self.output(self.transformer(self.input(x.transpose(1, 2))+self.position+c)).transpose(1, 2)

    def sample(self, labels, seed=0, steps=25):
        from .diffusion import alpha_beta
        if not 1 <= steps <= 200:
            raise ValueError('steps must be in [1, 200]')
        x = torch.randn(len(labels), self.channels, self.frames, generator=torch.Generator().manual_seed(seed)).to(labels)
        levels = torch.linspace(1, 0, steps+1, device=x.device)
        for i in range(steps):
            a, b = alpha_beta(levels[i]); na, nb = alpha_beta(levels[i+1])
            v = self(x, levels[i].expand(len(x)), labels)
            x = na*(a*x-b*v)+nb*(b*x+a*v)
        return x*self.latent_std+self.latent_mean


def create_model(kind, instrument, architecture=None):
    if kind not in KINDS:
        raise ValueError(f'unknown research model: {kind}')
    architecture = dict(architecture or {})
    if kind == 'parameters': return ParameterVAE(instrument, **architecture)
    if kind == 'drumgan': return DrumGAN(instrument, **architecture)
    if kind == 'codec': return CodecDenoiser(instrument, **architecture)
    if kind == 'wavevae': return WaveVAE(instrument, **architecture)
    return TextureVAE(instrument, kind=kind, **architecture)
