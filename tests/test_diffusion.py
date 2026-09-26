"""The descriptor-conditioned waveform diffusion backend (issue #3).

Nothing here trains a usable model or claims anything about how the backend
sounds; these cover the contracts the experiment rests on — that the schedule
is what it says it is, that the sliders reach the network, that a texture seed
and a slider value are separable, and that a run is recorded and reloadable.
"""

from __future__ import annotations

import math
import os
import pathlib
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from kicks.analysis.descriptors import descriptor_vector
from kicks.audio.constants import SAMPLE_RATE
from kicks.audio.mel import spectrogram
from kicks.config import load_diffusion_from_checkpoint, load_vae_from_checkpoint
from kicks.data.waveforms import TARGET_PEAK, WaveformDataset, peak_normalize
from kicks.instruments import get_profile
from kicks.nn import VAE, WaveformUNet
from kicks.nn.diffusion import (
    alpha_beta,
    diffuse,
    guided_v,
    v_sample,
    v_target,
)
from kicks.synthesis.diffusion import draw_labels, generate_diffusion
from kicks.training.diffusion import (
    EMA,
    VALIDATION_RANGE,
    _accumulation_group,
    _validation_sigmas,
    diffusion_loss,
    train_diffusion,
)
from kicks.training.tracking import read_run

# A LUFS meter needs at least 400 ms, so test hits are real-length, not toy.
LENGTH = 32768
KICK = get_profile("kick")


def tiny(length: int = LENGTH, descriptors: int = len(KICK.descriptors)) -> WaveformUNet:
    return WaveformUNet(
        descriptors, channels=(8, 16, 16), factors=(4, 4, 4), cond_dim=32,
        blocks_per_scale=1, attention_scales=1, attention_heads=2, length=length,
    )


def wake(model: WaveformUNet, seed: int = 0) -> WaveformUNet:
    """Perturb the zero-initialised output paths so the network is not the zero map."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    return model


def corpus(directory: pathlib.Path, count: int = 12) -> pathlib.Path:
    """Synthetic kicks: a decaying sine with a noise transient, all different."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    time = np.arange(LENGTH) / SAMPLE_RATE
    for index in range(count):
        hit = np.sin(2 * np.pi * (45 + 10 * rng.random()) * time)
        hit *= np.exp(-(15 + 40 * rng.random()) * time)
        hit[:80] += 0.4 * rng.standard_normal(80)
        sf.write(directory / f"k{index}.wav", (0.8 * hit).astype(np.float32), SAMPLE_RATE)
    return directory


def scoped(tmp_path: pathlib.Path):
    """A kick profile whose corpus, weights and output all live under tmp_path."""
    return replace(KICK, paths=replace(
        KICK.paths, data_root=str(tmp_path), model_root=str(tmp_path / "models"),
        output_root=str(tmp_path / "output"),
    ))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def test_angular_schedule_keeps_unit_norm_and_reaches_both_ends():
    sigmas = torch.linspace(0, 1, 9)
    alpha, beta = alpha_beta(sigmas)
    torch.testing.assert_close(alpha**2 + beta**2, torch.ones(9))

    x = torch.randn(4, 1, 64)
    noise = torch.randn(4, 1, 64)
    clean, pure = torch.zeros(4), torch.ones(4)
    # sigma 0 is the corpus, sigma 1 is noise, and the velocity swaps with them.
    torch.testing.assert_close(diffuse(x, noise, clean), x)
    torch.testing.assert_close(diffuse(x, noise, pure), noise, atol=1e-6, rtol=0)
    torch.testing.assert_close(v_target(x, noise, clean), noise)
    torch.testing.assert_close(v_target(x, noise, pure), -x, atol=1e-6, rtol=0)


def test_velocity_recovers_the_signal_and_the_noise_at_any_level():
    x, noise = torch.randn(3, 1, 64), torch.randn(3, 1, 64)
    sigmas = torch.tensor([0.1, 0.5, 0.9])
    noisy, velocity = diffuse(x, noise, sigmas), v_target(x, noise, sigmas)
    alpha, beta = alpha_beta(sigmas.reshape(-1, 1, 1))
    torch.testing.assert_close(alpha * noisy - beta * velocity, x)
    torch.testing.assert_close(beta * noisy + alpha * velocity, noise)


# ---------------------------------------------------------------------------
# Network and conditioning
# ---------------------------------------------------------------------------

def test_denoiser_returns_the_input_shape_and_starts_as_the_zero_map():
    model = tiny(length=1024)
    x = torch.randn(3, 1, 1024)
    out = model(x, torch.rand(3), labels=torch.randn(3, 5))
    assert out.shape == x.shape
    # Zero-initialised FiLM and output convolutions: depth costs nothing at init.
    assert float(out.detach().abs().max()) == 0.0


@pytest.mark.parametrize("factors,length", [((3,), 1024), ((2, 2, 2), 100)])
def test_denoiser_rejects_shapes_it_cannot_resample_exactly(factors, length):
    with pytest.raises(ValueError):
        WaveformUNet(5, channels=(8,) * len(factors), factors=factors, length=length)


def test_sliders_reach_the_network_and_dropping_them_is_the_null_branch():
    model = wake(tiny(length=1024))
    x, sigmas = torch.randn(2, 1, 1024), torch.full((2,), 0.4)
    low = torch.zeros(2, 5)
    high = torch.ones(2, 5) * 3

    assert not torch.allclose(model(x, sigmas, labels=low), model(x, sigmas, labels=high))

    kept = model(x, sigmas, labels=high, cond_mask=torch.tensor([True, False]))
    unconditional = model(x, sigmas, labels=None)
    conditional = model(x, sigmas, labels=high)
    # Row 1 keeps its descriptors, row 2 falls back to the learned null embedding.
    torch.testing.assert_close(kept[1], unconditional[1])
    torch.testing.assert_close(kept[0], conditional[0])
    assert not torch.allclose(conditional[0], unconditional[0])


def test_label_statistics_standardize_raw_units_and_survive_a_flat_descriptor():
    model = wake(tiny(length=1024))
    mean, std = torch.tensor([1.0, -2, 3, 0, 5]), torch.tensor([2.0, 4, 1, 8, 0.5])
    model.set_label_stats(mean, std)
    raw = torch.tensor([[3.0, 2, 4, 8, 5.5]])
    x, sigmas = torch.randn(1, 1, 1024), torch.full((1,), 0.3)
    with_stats = model(x, sigmas, labels=raw)

    plain = wake(tiny(length=1024))
    plain.load_state_dict(model.state_dict())
    plain.set_label_stats(torch.zeros(5), torch.ones(5))
    torch.testing.assert_close(plain(x, sigmas, labels=(raw - mean) / std), with_stats)

    # A descriptor that never varies must not divide the conditioning by zero.
    model.set_label_stats(mean, torch.zeros(5))
    assert float(model.label_std.min()) > 0
    assert torch.isfinite(model(x, sigmas, labels=raw)).all()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_is_deterministic_and_separates_texture_from_sliders():
    model = wake(tiny(length=1024))
    quiet, loud = torch.zeros(1, 5), torch.ones(1, 5) * 2

    def draw(seed, labels):
        return v_sample(model, labels=labels, steps=4,
                        generator=torch.Generator().manual_seed(seed))

    torch.testing.assert_close(draw(1, quiet), draw(1, quiet))
    assert not torch.allclose(draw(1, quiet), draw(2, quiet))   # texture seed moved
    assert not torch.allclose(draw(1, quiet), draw(1, loud))    # only the sliders moved


def test_guidance_blends_the_two_branches_and_costs_nothing_at_one():
    model = wake(tiny(length=1024))
    x, sigmas, labels = torch.randn(2, 1, 1024), torch.full((2,), 0.6), torch.ones(2, 5)
    conditional = model(x, sigmas, labels=labels)
    unconditional = model(x, sigmas, labels=None)

    torch.testing.assert_close(guided_v(model, x, sigmas, labels, 1.0), conditional)
    torch.testing.assert_close(guided_v(model, x, sigmas, labels, 0.0), unconditional)
    torch.testing.assert_close(
        guided_v(model, x, sigmas, labels, 3.0),
        unconditional + 3.0 * (conditional - unconditional),
    )


def test_sampler_refuses_a_label_batch_that_does_not_match_the_noise():
    model = tiny(length=1024)
    with pytest.raises(ValueError):
        v_sample(model, labels=torch.zeros(3, 5), noise=torch.randn(2, 1, 1024), steps=2)
    with pytest.raises(ValueError):
        v_sample(model, batch=1, steps=0)


def correlated_bank(rows: int = 400, seed: int = 0) -> torch.Tensor:
    """A bank whose second column follows its first: what real sliders do."""
    generator = torch.Generator().manual_seed(seed)
    first = 10 + 3 * torch.randn(rows, generator=generator)
    return torch.stack([
        first, 2 * first + 0.2 * torch.randn(rows, generator=generator),
        30 + torch.randn(rows, generator=generator), torch.full((rows,), 40.0),
        (5 * torch.randn(rows, generator=generator)).abs(),           # a decay: never negative
    ], dim=1)


def test_targets_are_resampled_from_the_bank_not_its_moments():
    model = tiny(length=1024)
    bank = correlated_bank()
    model.set_label_stats(bank.mean(dim=0), bank.std(dim=0))
    model.set_label_bank(bank)

    labels = draw_labels(model, 200, None, KICK, seed=3)
    assert labels.shape == (200, 5)
    # Every target is a row a real hit had — in particular no negative decay,
    # which the Gaussian around these moments would produce for ~30 % of draws.
    rows = {tuple(row.tolist()) for row in bank}
    assert all(tuple(row.tolist()) in rows for row in labels)
    assert float(labels[:, 4].min()) >= 0
    torch.testing.assert_close(labels, draw_labels(model, 200, None, KICK, seed=3))
    assert not torch.equal(labels, draw_labels(model, 200, None, KICK, seed=4))
    with pytest.raises(ValueError, match="no descriptor"):
        draw_labels(model, 2, {"sizzle": 1.0}, KICK, seed=3)


def test_pinning_a_descriptor_keeps_the_others_consistent_with_it():
    model = tiny(length=1024)
    bank = correlated_bank()
    model.set_label_stats(bank.mean(dim=0), bank.std(dim=0))
    model.set_label_bank(bank)

    high = draw_labels(model, 64, {"sub": 16.0}, KICK, seed=1)
    low = draw_labels(model, 64, {"sub": 4.0}, KICK, seed=1)
    assert torch.allclose(high[:, 0], torch.full((64,), 16.0))
    assert torch.allclose(low[:, 0], torch.full((64,), 4.0))
    # "punch" follows "sub" in this corpus, so the free column moved with the
    # pinned one instead of sitting at the corpus mean for both.
    assert float(high[:, 1].mean()) > 2 * 14 and float(low[:, 1].mean()) < 2 * 6
    # A value the corpus never had is honoured, and flagged.
    with pytest.warns(UserWarning, match="outside the training range"):
        extreme = draw_labels(model, 4, {"sub": 100.0}, KICK, seed=1)
    assert torch.allclose(extreme[:, 0], torch.full((4,), 100.0))


def test_without_a_bank_targets_fall_back_to_gaussians_with_a_warning():
    model = tiny(length=1024)
    model.set_label_stats([10.0, 20, 30, 40, 50], [1.0, 1, 1, 1, 1])
    with pytest.warns(UserWarning, match="no label bank"):
        labels = draw_labels(model, 64, {"punch": 7.5}, KICK, seed=3)
    assert labels.shape == (64, 5)
    assert torch.allclose(labels[:, 1], torch.full((64,), 7.5))
    assert abs(float(labels[:, 0].mean()) - 10.0) < 0.5
    with pytest.raises(ValueError, match="at least one row"):
        model.set_label_bank(torch.zeros(0, 5))


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def test_peak_normalization_is_gain_only_and_leaves_silence_alone():
    audio = torch.randn(1, 128) * 0.03
    scaled = peak_normalize(audio)
    assert float(scaled.abs().max()) == pytest.approx(TARGET_PEAK)
    torch.testing.assert_close(scaled / (TARGET_PEAK / float(audio.abs().max())), audio)
    torch.testing.assert_close(peak_normalize(torch.zeros(1, 128)), torch.zeros(1, 128))


def test_corpus_labels_describe_the_waveform_it_actually_returns(tmp_path):
    dataset = WaveformDataset(str(corpus(tmp_path / "kicks", 6)), KICK,
                              length=LENGTH, verbose=False)
    audio, labels = dataset[0]

    assert len(dataset) == 6 and audio.shape == (1, LENGTH)
    assert float(audio.abs().max()) == pytest.approx(TARGET_PEAK, abs=1e-5)
    # Label and target agree by construction, whatever the descriptor kinds are.
    np.testing.assert_allclose(
        labels.numpy(), descriptor_vector(spectrogram(audio), KICK), rtol=1e-4, atol=1e-4,
    )

    mean, std = dataset.label_stats([0, 1, 2])
    expected = torch.stack(dataset.labels[:3]).numpy()
    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-5)
    np.testing.assert_allclose(std, expected.std(axis=0), rtol=1e-5)
    # Statistics must follow the indices they were given, not the whole corpus.
    assert not np.allclose(mean, dataset.label_stats()[0])


# ---------------------------------------------------------------------------
# Training machinery
# ---------------------------------------------------------------------------

def test_validation_levels_sweep_the_whole_schedule_however_small_the_split():
    for total in (3, 40):
        levels = _validation_sigmas(total, 0, total, torch.device("cpu"), torch.float32)
        assert float(levels[0]) == pytest.approx(VALIDATION_RANGE[0])
        assert float(levels[-1]) == pytest.approx(VALIDATION_RANGE[1])
    # Batching must not change which level a hit is measured at.
    whole = _validation_sigmas(8, 0, 8, torch.device("cpu"), torch.float32)
    tail = _validation_sigmas(3, 5, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(whole[5:], tail)


def test_every_optimizer_step_averages_the_micro_batches_it_actually_saw():
    # Nine micro-batches in groups of four: two full groups, then one of one.
    assert [_accumulation_group(i, 9, 4) for i in range(1, 10)] == [4] * 8 + [1]
    assert [_accumulation_group(i, 8, 4) for i in range(1, 9)] == [4] * 8
    assert [_accumulation_group(i, 3, 4) for i in range(1, 4)] == [3] * 3
    assert [_accumulation_group(i, 5, 1) for i in range(1, 6)] == [1] * 5


def test_ema_warms_up_towards_the_weights_then_averages_them():
    model = tiny(length=1024)
    ema = EMA(model, 0.999)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)

    ema.update(model)
    first = ema.state_dict(model)["stem.weight"].mean().item()
    # Without warm-up a decay of 0.999 would leave this within 0.001 of zero.
    assert first > 0.1
    for _ in range(200):
        ema.update(model)
    assert ema.state_dict(model)["stem.weight"].mean().item() == pytest.approx(1.0, abs=1e-2)

    # Buffers are constants, not averages: they must copy through immediately.
    model.set_label_stats([1.0, 2, 3, 4, 5], [1.0, 1, 1, 1, 1])
    ema.update(model)
    torch.testing.assert_close(ema.state_dict(model)["label_mean"], model.label_mean)


def test_loss_is_zero_for_a_perfect_denoiser_and_reports_per_sample_values():
    class Oracle(torch.nn.Module):
        length = 64

        def forward(self, noisy, sigmas, labels=None, cond_mask=None):
            return v_target(x, noise, sigmas)

    x, noise = torch.randn(4, 1, 64), torch.randn(4, 1, 64)
    sigmas = torch.tensor([0.1, 0.3, 0.6, 0.9])
    assert diffusion_loss(Oracle(), x, None, sigmas, noise).item() == pytest.approx(0.0)
    assert diffusion_loss(Oracle(), x, None, sigmas, noise, reduce=False).shape == (4,)


def test_training_records_a_run_writes_checkpoints_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("KICKS_TRAINING_CONTEXT", '{"provider":"test","job_id":"job-1"}')
    profile = scoped(tmp_path)
    dataset = WaveformDataset(str(corpus(tmp_path / "kicks", 12)), profile,
                              length=LENGTH, verbose=False)
    model = tiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = []
    optimizer.register_step_post_hook(lambda *_: steps.append(1))

    curves = train_diffusion(
        model, dataset, optimizer, profile, epochs=2, batch_size=4, val_split=0.25,
        device=torch.device("cpu"), grad_accum=2, eval_every=2, eval_steps=2,
        eval_samples=2, runs_dir=str(tmp_path / "runs"), run_name="test",
        intent="cover the loop", hypothesis="it runs", success_criteria="none",
    )
    assert len(curves["train"]) == 2 and len(curves["val"]) == 2
    assert all(math.isfinite(value) for value in curves["train"] + curves["val"])

    run = read_run(next(iter((tmp_path / "runs").iterdir())))
    assert run["status"] == "completed"
    assert run["config"]["backend"] == "waveform_diffusion"
    assert run["config"]["split_fingerprint"]
    assert run["config"]["effective_batch"] == 8
    assert run["config"]["runtime"]["torch"] == torch.__version__
    assert run["config"]["runtime"]["accelerator"]["name"] == "cpu"
    assert run["config"]["execution"] == {"provider": "test", "job_id": "job-1"}
    # Nine training hits at batch 4 accumulate in pairs: the short third
    # micro-batch is dropped, so each epoch is exactly one optimizer step.
    assert run["config"]["train_samples"] == 9 and len(steps) == 2
    assert run["notes"]["objective"] == "cover the loop"
    assert len(run["history"]) == 3                       # epoch 0 baseline plus two
    assert run["history"][-1]["control_mae"] is not None
    assert run["history"][-1]["cuda_peak_memory_mb"] is None
    assert {"val_loss_low_sigma", "val_loss_high_sigma"} <= set(run["history"][-1])

    for path in (profile.paths.diffusion_checkpoint, profile.paths.diffusion_final_checkpoint,
                 profile.paths.diffusion_control_checkpoint, profile.paths.diffusion_loss_curves):
        assert os.path.exists(path), path
    # Checkpoints are written beside the target and renamed into place.
    assert not [name for name in os.listdir(profile.paths.model_dir) if name.endswith(".tmp")]

    # The saved weights are the EMA, so reloading must not resurrect the live ones.
    reloaded, meta = load_diffusion_from_checkpoint(
        profile.paths.diffusion_checkpoint, torch.device("cpu"), profile,
    )
    assert meta["training"]["run_id"] == run["id"]
    assert meta["descriptors"] == [d.key for d in profile.descriptors]
    assert reloaded.architecture == model.architecture
    # The bank is the training split's rows — nine of twelve — and comes back.
    assert reloaded.label_bank.shape == (9, len(profile.descriptors))
    torch.testing.assert_close(reloaded.label_bank, model.label_bank)
    labels = draw_labels(reloaded, 2, None, profile, seed=1)
    torch.testing.assert_close(
        v_sample(reloaded, labels=labels, steps=2,
                 generator=torch.Generator().manual_seed(5)),
        v_sample(reloaded, labels=labels, steps=2,
                 generator=torch.Generator().manual_seed(5)),
    )


def test_training_marks_a_failed_run_instead_of_leaving_it_running(tmp_path):
    profile = scoped(tmp_path)
    dataset = WaveformDataset(str(corpus(tmp_path / "kicks", 8)), profile,
                              length=LENGTH, verbose=False)
    model = tiny()
    with pytest.raises(ValueError):
        train_diffusion(model, dataset, torch.optim.Adam(model.parameters()), profile,
                        epochs=1, batch_size=4, val_split=0.25, cond_dropout=1.5,
                        device=torch.device("cpu"), runs_dir=str(tmp_path / "runs"))
    assert not (tmp_path / "runs").exists()   # rejected before a record was opened

    with pytest.raises(FloatingPointError):
        train_diffusion(_Exploding(), dataset, torch.optim.Adam(model.parameters()), profile,
                        epochs=1, batch_size=4, val_split=0.25,
                        device=torch.device("cpu"), runs_dir=str(tmp_path / "runs"))
    run = read_run(next(iter((tmp_path / "runs").iterdir())))
    assert run["status"] == "failed" and "FloatingPointError" in run["error"]


class _Exploding(torch.nn.Module):
    """A denoiser that returns NaN, standing in for a diverged run."""

    length = LENGTH
    architecture = {"channels": [], "factors": []}
    n_descriptors = len(KICK.descriptors)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("label_std", torch.ones(self.n_descriptors))

    def forward(self, noisy, sigmas, labels=None, cond_mask=None):
        return torch.full_like(noisy, float("nan")) + self.weight

    label_bank = None

    def set_label_stats(self, mean, std):
        pass

    def set_label_bank(self, bank):
        pass

    def checkpoint_meta(self):
        return {"model_kind": "waveform_diffusion"}


# ---------------------------------------------------------------------------
# Checkpoints and the command line
# ---------------------------------------------------------------------------

def test_the_loader_refuses_a_vae_and_a_foreign_instruments_sliders(tmp_path):
    torch.save({"model": VAE(latent_dim=4).state_dict(), **VAE(latent_dim=4).checkpoint_meta()},
               tmp_path / "vae.pth")
    with pytest.raises(ValueError, match="not a waveform_diffusion checkpoint"):
        load_diffusion_from_checkpoint(str(tmp_path / "vae.pth"), torch.device("cpu"))

    model = tiny(length=1024)
    torch.save({"model": model.state_dict(), "descriptors": ["attack", "body", "sizzle",
                "bright", "decay"], **model.checkpoint_meta()}, tmp_path / "hihat.pth")
    with pytest.raises(ValueError, match="conditioned on"):
        load_diffusion_from_checkpoint(str(tmp_path / "hihat.pth"), torch.device("cpu"), KICK)
    # The same file loads fine for the instrument it was actually trained on.
    loaded, _ = load_diffusion_from_checkpoint(
        str(tmp_path / "hihat.pth"), torch.device("cpu"), get_profile("hihat"),
    )
    assert loaded.length == 1024

    # A VAE checkpoint is not accidentally readable as a denoiser, and vice versa.
    with pytest.raises(Exception):
        load_vae_from_checkpoint(str(tmp_path / "hihat.pth"), torch.device("cpu"))


def test_generation_writes_audio_and_honours_explicit_targets(tmp_path):
    profile = scoped(tmp_path)
    model = wake(tiny())
    dataset = WaveformDataset(str(corpus(tmp_path / "kicks", 4)), profile,
                              length=LENGTH, verbose=False)
    model.set_label_stats(*dataset.label_stats())
    model.set_label_bank(dataset.label_matrix())
    os.makedirs(profile.paths.model_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "label_bank": model.label_bank,
                "instrument": "kick", "descriptors": [d.key for d in profile.descriptors],
                "epoch": 1, **model.checkpoint_meta()}, profile.paths.diffusion_checkpoint)

    paths = generate_diffusion(
        count=2, instrument="kick", checkpoint=profile.paths.diffusion_checkpoint,
        out_dir=str(tmp_path / "out"), steps=2, seed=1, label_seed=2, prefix="diff",
    )
    assert [os.path.basename(p) for p in paths] == ["diff_1.wav", "diff_2.wav"]
    for path in paths:
        audio, rate = sf.read(path)
        assert rate == SAMPLE_RATE and audio.shape == (LENGTH,)
        assert np.abs(audio).max() <= 1.0     # never written clipped

    with pytest.raises(ValueError, match="unconditional"):
        generate_diffusion(count=1, instrument="kick", unconditional=True,
                           targets={"sub": 1.0},
                           checkpoint=profile.paths.diffusion_checkpoint,
                           out_dir=str(tmp_path / "out"), steps=1)


def test_the_dashboard_template_switches_on_the_backend():
    """The viewer is shared with the VAE, so it must read a diffusion run too.

    A string check only; the rendered charts and mobile layout are inspected by
    hand, as the repository instructions require for dashboard changes.
    """
    template = (pathlib.Path("kicks/training/tracking.py").parent / "dashboard.html").read_text()

    assert "waveform_diffusion" in template
    # Every element the backend switch retargets must still exist to be retargeted.
    for element in ("airLabel", "latentLabel", "detailTitle", "detailCaption",
                    "latentTitle", "latentCaption", "lossCaption", "epochHead", "scope"):
        assert f'id="{element}"' in template, element
    for metric in ("val_loss_low_sigma", "val_loss_mid_sigma", "val_loss_high_sigma",
                   "control_mae"):
        assert metric in template, metric
    # A VAE run and a diffusion run must not be offered as a like-for-like compare.
    assert "'instrument','backend','objective'" in template


def test_the_cli_exposes_both_diffusion_commands():
    from typer.testing import CliRunner

    from kicks.cli import app

    runner = CliRunner()
    listing = runner.invoke(app, ["--help"])
    assert listing.exit_code == 0
    assert "diffusion-train" in listing.output and "diffusion-generate" in listing.output
    for command, flag in (("diffusion-train", "--cond-dropout"),
                          ("diffusion-generate", "--guidance")):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0 and flag in result.output
