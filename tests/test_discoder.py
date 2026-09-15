"""Check the checkpoint's mel contract and inference boundary without downloads."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from kicks.audio.constants import HOP_LENGTH, LOG_MEL_MIN
from kicks.audio.vocoder import load_vocoder, validate_discoder_config


CONFIG = {
    "sample_rate": 44100,
    "segment_size": 16384,
    "mel": {"n_fft": 1024, "win_length": 1024, "hop_length": 256,
            "n_mels": 128, "f_min": 0, "f_max": None},
}


def test_checkpoint_mel_contract():
    validate_discoder_config(CONFIG)
    for section, key, value in (("mel", "hop_length", 512), ("mel", "f_max", 16000),
                                ("mel", "n_fft", 2048), ("mel", "n_mels", 80)):
        config = deepcopy(CONFIG)
        config[section][key] = value
        with pytest.raises(ValueError, match="mel representation"):
            validate_discoder_config(config)
    with pytest.raises(ValueError, match="mel representation"):
        validate_discoder_config({**CONFIG, "sample_rate": 48000})


@pytest.mark.parametrize("frames", [1, 63, 64, 65, 255, 256])
def test_short_and_odd_inputs_are_padded_with_silence_then_cropped(frames):
    from kicks.nn.discoder import DisCoder

    class Encoder(nn.Module):
        def forward(self, mel):
            self.input = mel
            return mel[:, :1, ::2], mel[:, :1, 1::2]

    class Decoder(nn.Module):
        def forward(self, latent, skip):
            return (latent + skip).repeat_interleave(2 * HOP_LENGTH, dim=-1)

    # Exercise the production boundary without constructing the 430M weights.
    model = DisCoder.__new__(DisCoder)
    nn.Module.__init__(model)
    model.config, model.encoder, model.decoder = CONFIG, Encoder(), Decoder()
    mel = torch.zeros(2, 128, frames)
    output = model(mel)
    assert output.shape == (2, 1, frames * HOP_LENGTH)
    assert model.encoder.input.shape[-1] % 64 == 0
    torch.testing.assert_close(model.encoder.input[..., :frames], mel)
    assert torch.all(model.encoder.input[..., frames:] == LOG_MEL_MIN)
    with pytest.raises(ValueError, match="log-mel input"):
        model(torch.zeros(1, 80, frames))


def test_unknown_backend_fails_instead_of_loading_another_model():
    with pytest.raises(ValueError, match="Unknown vocoder"):
        load_vocoder(torch.device("cpu"), "discode")


def test_default_and_explicit_backend_selection(monkeypatch):
    from kicks.audio import vocoder

    monkeypatch.setattr(vocoder, "load_discoder", lambda device: "discoder")
    monkeypatch.setattr(vocoder, "load_bigvgan", lambda device, weights_dir: "bigvgan")
    monkeypatch.delenv("KICKS_VOCODER", raising=False)
    assert load_vocoder(torch.device("cpu")) == "discoder"
    monkeypatch.setenv("KICKS_VOCODER", "bigvgan")
    assert load_vocoder(torch.device("cpu")) == "bigvgan"
    assert load_vocoder(torch.device("cpu"), "discoder") == "discoder"


def test_profile_chooses_its_vocoder_unless_forced(monkeypatch):
    from kicks.audio.vocoder import resolve_vocoder_type
    from kicks.instruments import get_profile

    monkeypatch.delenv("KICKS_VOCODER", raising=False)
    # Measured per instrument: DisCoder for the pitched drums, BigVGAN for hats.
    assert resolve_vocoder_type(get_profile("kick")) == "discoder"
    assert resolve_vocoder_type(get_profile("snare")) == "discoder"
    assert resolve_vocoder_type(get_profile("hihat")) == "bigvgan"
    monkeypatch.setenv("KICKS_VOCODER", "griffinlim")
    assert resolve_vocoder_type(get_profile("hihat")) == "griffinlim"
    assert resolve_vocoder_type(get_profile("hihat"), "discoder") == "discoder"
    with pytest.raises(ValueError, match="Unknown vocoder"):
        resolve_vocoder_type(get_profile("kick"), "discode")


def test_server_shares_one_vocoder_per_backend(monkeypatch):
    from kicks.api import state as state_module
    from kicks.instruments import get_profile

    loads = []
    monkeypatch.setattr(state_module, "load_vocoder",
                        lambda device, kind, weights_dir: loads.append((kind, weights_dir)) or object())
    monkeypatch.delenv("KICKS_VOCODER", raising=False)
    server = state_module.ServerState()
    kick, snare, hihat = (get_profile(n) for n in ("kick", "snare", "hihat"))
    assert server.vocoder_for(kick) is server.vocoder_for(snare)      # both DisCoder
    assert server.vocoder_for(hihat) is not server.vocoder_for(kick)  # BigVGAN
    assert server.vocoder_for(hihat) is server.vocoder_for(hihat)
    assert [kind for kind, _ in loads] == ["discoder", "bigvgan"]
    assert server.vocoder_type == "profile"
    server.vocoder_override = "bigvgan"
    # Forced BigVGAN keeps the kick's own fine-tuned weights separate from the hat's.
    assert server.vocoder_for(kick) is not server.vocoder_for(hihat)
    assert server.vocoder_type == "bigvgan"


def test_vocoder_batch_limit_preserves_every_hit(monkeypatch):
    from kicks.audio import vocoder

    class LimitedVocoder:
        inference_batch_size = 1

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, mel):
            self.batch_sizes.append(len(mel))
            return mel[:, :1, :].repeat_interleave(HOP_LENGTH, dim=-1)

    backend = LimitedVocoder()
    # Isolate batching from the post-chain, whose gain invariance has its own test.
    monkeypatch.setattr(vocoder, "_post_process", lambda waveform: waveform)
    specs = torch.stack([torch.full((1, 128, 64), value) for value in (0.2, 0.5, 0.8)])
    audio = vocoder.spec_to_audio(specs, backend, torch.device("cpu"))
    assert backend.batch_sizes == [1, 1, 1]
    assert audio.shape == (3, 64 * HOP_LENGTH)
    torch.testing.assert_close(audio[:, 0], vocoder.denormalize(specs)[:, 0, 0, 0])
