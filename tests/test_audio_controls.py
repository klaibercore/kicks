"""Audible regressions and the numerical contract behind calibrated sliders."""

import numpy as np
import pytest
import torch

from kicks.analysis.basis import DescriptorBasis
from kicks.analysis.descriptors import descriptor_matrix, descriptor_tensor
from kicks.audio.constants import LOG_MEL_MAX, LOG_MEL_MIN, SAMPLE_RATE
from kicks.audio.effects import apply_drive, apply_envelope
from kicks.audio.io import lufs_normalize
from kicks.audio.vocoder import _post_process
from kicks.instruments import get_profile


@pytest.mark.parametrize("instrument", ["kick", "snare", "hihat"])
def test_differentiable_descriptors_match_numpy(instrument):
    profile = get_profile(instrument)
    spec = torch.rand(3, 1, 128, 256, generator=torch.Generator().manual_seed(17), requires_grad=True)
    actual = descriptor_tensor(spec, profile)
    np.testing.assert_allclose(actual.detach().numpy(), descriptor_matrix(spec, profile), rtol=1e-5, atol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(spec.grad).all()


def test_kick_controls_are_invariant_to_gain():
    rng = np.random.default_rng(0)
    magnitude = rng.uniform(0.05, 1, (1, 128, 256))
    # The mel floor is a representation detail, not audible energy.
    def spec(gain):
        return (np.log(gain * magnitude + np.exp(LOG_MEL_MIN)) - LOG_MEL_MIN) / (LOG_MEL_MAX - LOG_MEL_MIN)
    profile = get_profile("kick")
    np.testing.assert_allclose(descriptor_matrix(spec(1), profile), descriptor_matrix(spec(0.1), profile), atol=1e-8)


def test_decay_increases_with_sustain():
    profile = get_profile("kick")
    frames = np.arange(256)
    def hit(decay_frames):
        magnitude = np.ones((128, 1)) * np.exp(-frames / decay_frames)
        return (np.log(magnitude + np.exp(LOG_MEL_MIN)) - LOG_MEL_MIN) / (LOG_MEL_MAX - LOG_MEL_MIN)
    desc = profile.descriptors[-1]
    assert desc.compute(hit(40)) > 3 * desc.compute(hit(10))


def test_loudness_normalization_preserves_transient_shape():
    class Meter:
        def integrated_loudness(self, _):
            return -40.0
    signal = torch.tensor([[0.0, 0.1, -0.25, 0.8, -0.4, 0.0]])
    normalized = lufs_normalize(signal, Meter(), target_lufs=-14)
    torch.testing.assert_close(normalized, signal / 0.8)
    assert normalized.abs().max() <= 1


def test_decay_shapes_the_hit_before_padding():
    hit = torch.zeros(SAMPLE_RATE)
    hit[:SAMPLE_RATE // 4] = 1
    shaped = apply_envelope(hit, decay_ms=100)
    assert shaped[SAMPLE_RATE // 10] == pytest.approx(0.001, rel=1e-5)
    assert shaped[0] == 1
    assert torch.count_nonzero(shaped[SAMPLE_RATE // 4:]) == 0


def test_zero_drive_is_identity():
    x = torch.linspace(-1, 1, 1000)
    torch.testing.assert_close(apply_drive(x, 0), x, rtol=0, atol=0)
    assert apply_drive(x, 1).abs().max() <= 1


def test_batch_processing_does_not_link_sample_loudness():
    t = torch.arange(8000) / SAMPLE_RATE
    hit = torch.sin(2 * torch.pi * 60 * t) * torch.exp(-20 * t)
    alone = _post_process(hit[None])[0]
    batch = _post_process(torch.stack([hit, hit * 0.1]))
    torch.testing.assert_close(batch[0], alone)
    torch.testing.assert_close(batch[1], alone, atol=3e-5, rtol=3e-5)


def test_nonlinear_solver_cancels_coupling_in_differently_scaled_latents():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(600, 3)) * [10, 0.1, 2]
    def response(latents):
        a, b, c = (latents / [10, 0.1, 2]).T
        return np.column_stack([a + 0.3 * b * b, b + 0.3 * np.sin(a), c + 0.1 * a * b])
    basis = DescriptorBasis(z, response(z))
    for target in ([0.7, -0.6, 0.3], [-0.5, 0.5, -0.8], [0, 0, 0]):
        measured = response(basis.solve(target, response))[0]
        np.testing.assert_allclose(measured, target, atol=2e-4)


def test_collapsed_latents_fail_clearly():
    with pytest.raises(ValueError, match="collapsed"):
        DescriptorBasis(np.ones((10, 3)), np.ones((10, 2)))


def test_waveform_feedback_meets_independent_targets():
    from kicks.audio.controls import correct_waveform
    from kicks.audio.mel import spectrogram
    profile = get_profile("kick")
    t = torch.arange(65536) / SAMPLE_RATE
    hit = (torch.sin(2 * torch.pi * 70 * t) * torch.exp(-t / 0.08)
           + 0.06 * torch.sin(2 * torch.pi * 500 * t) * torch.exp(-t / 0.06)
           + 0.01 * torch.sin(2 * torch.pi * 4000 * t) * torch.exp(-t / 0.02)
           + 0.003 * torch.sin(2 * torch.pi * 6000 * t) * torch.exp(-t / 0.07))
    original = descriptor_tensor(spectrogram(hit[None]), profile)[0].numpy()
    spans = np.array([15, 6, 24, 33, 60])
    target = original + [1, 0.25, -0.5, 0.5, 1]
    result = correct_waveform(hit, target, spans, profile)
    measured = descriptor_tensor(spectrogram(result[None]), profile)[0].numpy()
    assert np.max(np.abs((measured - target) / spans)) < 0.002
    assert result.shape == hit.shape and torch.isfinite(result).all()
    assert result.abs().max() <= 1


def test_log_mel_padding_is_silence(monkeypatch):
    from kicks.audio import mel
    monkeypatch.setattr(mel, "_bigvgan_mel", lambda *a, **kw: torch.ones(1, 128, 2))
    padded = mel.spectrogram(torch.zeros(1, 1024), n_frames=5)
    assert torch.count_nonzero(padded[..., 2:]) == 0


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_api_rejects_nonfinite_parameters(value):
    from fastapi import HTTPException
    from starlette.requests import Request
    from kicks.api.app import _float_param
    request = Request({"type": "http", "query_string": f"sub={value}".encode()})
    with pytest.raises(HTTPException) as exc:
        _float_param(request, "sub")
    assert exc.value.status_code == 422
