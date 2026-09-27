"""Serving the waveform diffusion backend beside the VAE.

A tiny untrained denoiser stands in for a checkpoint: these cover how the API
maps sliders onto the corpus, selects and validates the backend, caches renders
and reports control fit. They establish nothing about how the backend sounds.
"""

from __future__ import annotations

import importlib
import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from kicks.api.state import parse_diffusion_checkpoints
from kicks.audio.constants import AUDIO_LENGTH, N_FRAMES, N_MELS
from kicks.instruments import get_profile
from kicks.nn import WaveformUNet
from kicks.synthesis.diffusion import SLIDER_QUANTILES, slider_range, slider_targets

#: The module, not the FastAPI instance ``kicks.api`` re-exports under the same name.
api = importlib.import_module("kicks.api.app")
HIHAT = get_profile("hihat")


def denoiser(length: int = AUDIO_LENGTH) -> WaveformUNet:
    model = WaveformUNet(
        len(HIHAT.descriptors), channels=(8, 16, 16), factors=(4, 4, 4), cond_dim=32,
        blocks_per_scale=1, attention_scales=1, attention_heads=2, length=length,
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    # A skewed, heavy-tailed bank, like a real decay column.
    bank = torch.rand(400, len(HIHAT.descriptors), generator=generator) ** 3 * 200
    model.set_label_stats(bank.mean(0), bank.std(0))
    model.set_label_bank(bank)
    return model.eval()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The API with a hi-hat diffusion checkpoint and no VAE loaded."""
    model = denoiser()
    path = tmp_path / "diffusion_best.pth"
    torch.save({"model": model.state_dict(), "label_bank": model.label_bank,
                "descriptors": HIHAT.descriptor_keys, "epoch": 7,
                "training": {"run_id": "test-run"}, **model.checkpoint_meta()}, path)
    monkeypatch.setattr(api.state, "device", torch.device("cpu"))
    monkeypatch.setattr(api.state, "diffusion_checkpoints", {"hihat": str(path)})
    monkeypatch.setattr(api.state, "_diffusion", {})
    api.render_cache.clear()
    api.audio_cache.clear()
    api.app.dependency_overrides[api.rate_limiter] = lambda: None
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()
    api.render_cache.clear()
    api.audio_cache.clear()


def test_sliders_walk_the_corpus_by_quantile():
    model = denoiser(length=1024)
    bank = model.label_bank
    low, high = slider_range(model)
    np.testing.assert_allclose(low, torch.quantile(bank, SLIDER_QUANTILES[0], dim=0).numpy())
    np.testing.assert_allclose(high, torch.quantile(bank, SLIDER_QUANTILES[1], dim=0).numpy())
    centre = slider_targets(model, [0.5] * 5)
    np.testing.assert_allclose(centre.numpy(), torch.quantile(bank, 0.5, dim=0).numpy(), rtol=1e-5)
    np.testing.assert_allclose(slider_targets(model, [0.0] * 5).numpy(), low, rtol=1e-5)
    np.testing.assert_allclose(slider_targets(model, [9.0] * 5).numpy(), high, rtol=1e-5)

    sweep = [slider_targets(model, [p, .5, .5, .5, .5])[0].item() for p in np.linspace(0, 1, 11)]
    assert sweep == sorted(sweep)
    # A skewed column puts the median well below the midpoint of its range.
    assert centre[0] < (low[0] + high[0]) / 2

    with pytest.raises(ValueError, match="expected 5"):
        slider_targets(model, [0.5] * 4)
    model.set_label_bank(None)
    with pytest.raises(ValueError, match="no label bank"):
        slider_targets(model, [0.5] * 5)


def test_checkpoint_spec_parses_and_refuses_malformed_items():
    assert parse_diffusion_checkpoints("hihat=a.pth, snares=b.pth") == {"hihat": "a.pth", "snare": "b.pth"}
    assert parse_diffusion_checkpoints(None) == {}
    with pytest.raises(ValueError, match="INSTRUMENT=PATH"):
        parse_diffusion_checkpoints("hihat")
    with pytest.raises(KeyError):
        parse_diffusion_checkpoints("cowbell=a.pth")


def test_instruments_and_config_advertise_the_diffusion_backend(served):
    rows = {row["name"]: row for row in served.get("/instruments").json()["instruments"]}
    assert "diffusion" in rows["hihat"]["backends"]
    assert "diffusion" not in rows["kick"]["backends"]
    assert served.get("/health").json()["diffusion"] == ["hihat"]

    config = served.get("/config?instrument=hihat&backend=diffusion").json()
    assert config["backend"] == "diffusion"
    assert config["variation"] is True
    assert config["steps"] == {"default": 50, "min": 10, "max": 200}
    assert config["checkpoint"] == {"name": "diffusion_best.pth", "epoch": 7, "run_id": "test-run"}
    assert [s["name"] for s in config["sliders"]] == [d.label for d in HIHAT.descriptors]
    for slider in config["sliders"]:
        assert slider["target_min"] < slider["target_max"]


def test_backend_and_steps_are_validated(served):
    assert served.get("/generate?instrument=hihat&backend=gan").status_code == 422
    assert served.get("/generate?instrument=hihat&backend=diffusion&steps=5").status_code == 422
    assert served.get("/generate?instrument=hihat&backend=diffusion&steps=ten").status_code == 422
    missing = served.get("/config?instrument=snare&backend=diffusion")
    assert missing.status_code == 503
    assert "--diffusion snare=PATH" in missing.json()["detail"]


def test_renders_are_seeded_cached_and_reshaped_without_resampling(served, monkeypatch):
    calls = []
    render = api.render_hit
    monkeypatch.setattr(api, "render_hit", lambda *a, **k: calls.append(k) or render(*a, **k))

    def wav(query: str) -> np.ndarray:
        response = served.get(f"/generate?instrument=hihat&backend=diffusion&steps=10&{query}")
        assert response.status_code == 200, response.text
        audio, rate = sf.read(io.BytesIO(response.content))
        assert rate == 44100 and audio.shape == (AUDIO_LENGTH,)
        return audio

    first = wav("seed=3&s1=0.2")
    assert np.abs(first).max() == pytest.approx(0.9, abs=1e-3)
    assert calls[-1] == {"seed": 3, "steps": 10, "device": torch.device("cpu")}
    api.audio_cache.clear()
    np.testing.assert_allclose(wav("seed=3&s1=0.2"), first, atol=1e-6)
    assert len(calls) == 1  # the render cache served the repeat

    wav("seed=3&s1=0.2&drive=0.5")
    assert len(calls) == 1  # an effect reshapes the cached hit
    assert not np.allclose(wav("seed=4&s1=0.2"), first)
    wav("seed=3&attack=0.8")  # a slider addressed by its label
    assert len(calls) == 3


def test_evaluation_reports_fit_against_the_slider_targets(served, monkeypatch):
    monkeypatch.setattr(api, "analyze_hit", lambda audio, profile: {"peak": float(np.abs(audio).max())})

    class Report:
        score, grade, likeness_pct, verdicts = 50.0, "C", 50.0, []

    monkeypatch.setattr(api, "score_sample", lambda *a, **k: Report())
    monkeypatch.setattr(api.state.get_diffusion("hihat"), "eval_ref", object())

    body = served.get("/evaluate?instrument=hihat&backend=diffusion&steps=10&s5=0.9").json()
    assert body["backend"] == "diffusion"
    assert set(body["descriptors"]) == set(HIHAT.descriptor_keys)
    model = api.state.get_diffusion("hihat").model
    expected = slider_targets(model, [.5, .5, .5, .5, .9]).tolist()
    assert list(body["control_fit"]["targets"].values()) == pytest.approx(expected)
    assert body["control_fit"]["includes_effects"] is False


def test_spectrogram_of_a_diffusion_render_is_measured_from_the_audio(served):
    body = served.get("/spectrogram?instrument=hihat&backend=diffusion&steps=10").json()
    assert body["source"] == "render"
    assert body["shape"] == [N_MELS, N_FRAMES]
    data = np.array(body["data"])
    assert data.shape == (N_MELS, N_FRAMES) and 0 <= data.min() and data.max() <= 1


def test_export_filenames_tag_diffusion_renders(served):
    diff = api.state.get_diffusion("hihat")
    assert api._export_filename(diff, [.5, .5, .5, .5, 1.0], seed=9) == \
        "hihat_diffusion_attack50_body50_sizzle50_bright50_decay100_v9.wav"
