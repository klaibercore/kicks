"""Stage 2–5 of docs/high-fidelity-generation.md: loss terms, architecture options,
waveform fidelity metrics, evidence attachment and promotion."""
import json
import os
import threading
from dataclasses import replace
from urllib.request import urlopen

import numpy as np
import pytest
import scipy.signal
import torch
from torch.utils.data import DataLoader

from kicks.analysis.fidelity import (
    compare_waveforms, late_energy_db, level_match, onset_ms, spectral_flatness, summarise,
)
from kicks.audio.constants import SAMPLE_RATE
from kicks.config import load_vae_from_checkpoint
from kicks.instruments import get_profile
from kicks.nn import VAE
from kicks.training.loss import attack_change_loss, hf_detail_loss, transient_loss, vae_loss
from kicks.training.promotion import PromotionRefused, promote
from kicks.training.tracking import TrainingRun, attach_report, dashboard_server, find_run, read_run
from kicks.training.trainer import train

SPEC = get_profile("kick").transient_loss


def _hit(frames_on=40, level=0.8):
    """A bright synthetic hit: full-band energy decaying ~40 dB over ``frames_on`` frames.

    Normalised-mel units span ~126 dB, so the decay is kept shallow enough that
    the whole tail stays inside the 60 dB activity floor the HF term weights by.
    """
    target = torch.zeros(1, 1, 128, 256)
    decay = 1 - 0.4 * torch.arange(frames_on) / frames_on
    target[..., :frames_on] = level * decay
    return target


# ---------------------------------------------------------------------------
# Stage 2: loss terms
# ---------------------------------------------------------------------------

def test_hf_detail_is_symmetric_where_the_transient_term_is_not():
    target = _hit()
    missing_tail = target.clone()
    missing_tail[..., SPEC.band:, SPEC.tail_start:] = 0            # drop the real HF tail
    excess_tail = target.clone()
    excess_tail[..., SPEC.band:, SPEC.tail_start:] *= 2            # add an equal error
    # The plan's probe: the transient term charges nothing for a removed tail...
    assert transient_loss(missing_tail, target, SPEC) == 0
    assert transient_loss(excess_tail, target, SPEC) > 0
    # ...while the HF detail term charges both, and equally.
    assert hf_detail_loss(missing_tail, target, SPEC) > 0
    assert hf_detail_loss(missing_tail, target, SPEC) == pytest.approx(
        float(hf_detail_loss(excess_tail, target, SPEC)), rel=1e-5)
    assert hf_detail_loss(target, target, SPEC) == 0


def test_hf_detail_does_not_reward_or_punish_a_raised_floor_where_the_reference_is_silent():
    target = _hit(frames_on=20)
    noisy_silence = target.clone()
    noisy_silence[..., SPEC.band:, 120:] = 0.05                    # HF hiss long after the hit
    assert hf_detail_loss(noisy_silence, target, SPEC) == 0          # weight 0: the tail term's job
    assert transient_loss(noisy_silence, target, SPEC) > 0           # which still catches it


def test_attack_change_sees_a_smeared_onset_the_level_loss_averages_away():
    target = torch.zeros(1, 1, 128, 256)
    target[..., 2:] = 0.6                                             # sharp step at frame 2
    smeared = torch.zeros(1, 1, 128, 256)
    smeared[..., :SPEC.click_frames] = torch.linspace(0, 0.6, SPEC.click_frames)   # ramp, same-ish mass
    smeared[..., SPEC.click_frames:] = 0.6
    window_mean = lambda x: x[..., :SPEC.click_frames].mean().item()  # noqa: E731
    assert abs(window_mean(smeared) - window_mean(target)) < 0.1
    assert attack_change_loss(smeared, target, SPEC) > 0.05
    assert attack_change_loss(target, target, SPEC) == 0


def test_vae_loss_is_unchanged_with_the_new_weights_off_and_reports_terms_when_on():
    torch.manual_seed(3)
    target, recon = _hit(), _hit(level=0.7)
    mu, logvar = torch.randn(1, 32), torch.zeros(1, 32)
    total, rec, kl = vae_loss(recon, target, mu, logvar, beta=0.02, transient=SPEC)
    terms = {}
    on_total, on_rec, on_kl = vae_loss(recon, target, mu, logvar, beta=0.02, transient=SPEC,
                                       hf_detail_weight=0.5, attack_change_weight=0.5, terms=terms)
    assert set(terms) == {"multi_resolution", "transient", "hf_detail", "attack_change"}
    assert on_kl == kl
    assert on_rec.item() == pytest.approx(rec.item() + 0.5 * terms["hf_detail"] + 0.5 * terms["attack_change"], rel=1e-5)
    assert on_total.item() == pytest.approx(on_rec.item() + 0.02 * kl.item(), rel=1e-5)
    with pytest.raises(ValueError):
        vae_loss(recon, target, mu, logvar, transient=None, hf_detail_weight=1.0)


# ---------------------------------------------------------------------------
# Stage 3: architecture options
# ---------------------------------------------------------------------------

def test_default_architecture_keeps_the_shipped_state_dict_layout():
    keys = list(VAE(32).state_dict())
    assert len(keys) == 57
    assert keys[0] == "encoder.0.weight" and keys[-1] == "decoder.9.bias"
    assert not any("film" in k or "body" in k for k in keys)
    assert VAE(32).architecture == {"residual": False, "latent_skips": False, "channels": [32, 64, 128, 256]}


def _copy_shared_weights(plain: VAE, option: VAE):
    kinds = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.Linear)
    source = [m for m in plain.modules() if isinstance(m, kinds)]
    target = [m for name, m in option.named_modules()
              if isinstance(m, kinds) and "body" not in name and "film" not in name]
    assert len(source) == len(target)
    for s, t in zip(source, target):
        t.load_state_dict(s.state_dict())


@pytest.mark.parametrize("options", [{"residual": True}, {"latent_skips": True},
                                     {"residual": True, "latent_skips": True}])
def test_options_start_as_the_identity_of_the_plain_network(options):
    torch.manual_seed(0)
    plain, option = VAE(32).eval(), VAE(32, **options).eval()
    _copy_shared_weights(plain, option)
    x, z = torch.rand(2, 1, 128, 256), torch.randn(2, 32)
    with torch.no_grad():
        assert torch.equal(plain.encode(x)[0], option.encode(x)[0])
        assert torch.equal(plain.decode(z), option.decode(z))
    assert sum(p.numel() for p in option.parameters()) > sum(p.numel() for p in plain.parameters())
    assert option.architecture["residual"] == options.get("residual", False)
    assert option.architecture["latent_skips"] == options.get("latent_skips", False)


def test_architecture_round_trips_through_checkpoints_and_legacy_files_load_as_default(tmp_path):
    model = VAE(64, residual=True, latent_skips=True)
    torch.save({"model": model.state_dict(), **model.checkpoint_meta()}, tmp_path / "new.pth")
    loaded, meta = load_vae_from_checkpoint(str(tmp_path / "new.pth"), torch.device("cpu"))
    assert loaded.latent_dim == 64 and loaded.residual and loaded.latent_skips
    assert meta["architecture"]["latent_skips"] is True
    legacy = VAE(32)
    torch.save({"model": legacy.state_dict(), "latent_dim": 32, "n_mels": 128, "n_frames": 256}, tmp_path / "old.pth")
    loaded, _ = load_vae_from_checkpoint(str(tmp_path / "old.pth"), torch.device("cpu"))
    assert loaded.architecture == legacy.architecture
    with torch.no_grad():
        z = torch.randn(1, 32)
        assert torch.equal(loaded.decode(z), legacy.eval().decode(z))


# ---------------------------------------------------------------------------
# Stage 2+3 through the trainer: components and architecture reach the record
# ---------------------------------------------------------------------------

def test_training_record_carries_loss_components_and_architecture(tmp_path):
    torch.manual_seed(15)
    model = VAE(4, n_frames=32, latent_skips=True)
    profile = get_profile("kick")
    profile = replace(profile, paths=replace(profile.paths, model_root=str(tmp_path / "models"),
                                            output_root=str(tmp_path / "output")))
    dataset = list(torch.rand(8, 1, 128, 32))
    train(model=model, dloader=DataLoader(dataset, batch_size=4),
          optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), profile=profile, epochs=1,
          device=torch.device("cpu"), val_split=0.25, eval_every=0, runs_dir=str(tmp_path / "runs"),
          hf_detail_weight=0.3, attack_change_weight=0.2)
    run = read_run(next((tmp_path / "runs").iterdir()))
    assert run["config"]["hf_detail_weight"] == 0.3 and run["config"]["attack_change_weight"] == 0.2
    assert run["config"]["architecture"]["latent_skips"] is True
    epoch = run["history"][1]
    for key in ("val_term_multi_resolution", "val_term_transient", "val_term_hf_detail", "val_term_attack_change",
                "train_term_hf_detail", "train_term_attack_change"):
        assert epoch[key] is not None and epoch[key] >= 0
    saved = torch.load(profile.paths.checkpoint, map_location="cpu", weights_only=True)
    assert saved["architecture"]["latent_skips"] is True
    assert saved["training"]["hf_detail_weight"] == 0.3


# ---------------------------------------------------------------------------
# Stage 4: waveform fidelity metrics
# ---------------------------------------------------------------------------

def _synthetic_hit(seed=0, sr=SAMPLE_RATE):
    t = np.arange(sr) / sr
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 8) + 0.3 * rng.standard_normal(sr) * np.exp(-t * 40)
    x[:100] = 0
    return x


def test_identical_waveforms_measure_zero_everywhere():
    profile = get_profile("kick")
    ref = _synthetic_hit()
    row = compare_waveforms(*level_match(ref, ref.copy()), profile)
    assert all(value is not None and abs(value) < 1e-6 for value in row.values())


def test_metrics_isolate_the_failure_they_name():
    profile, ref = get_profile("kick"), _synthetic_hit()
    dull = scipy.signal.sosfilt(scipy.signal.butter(4, 4000 / (SAMPLE_RATE / 2), output="sos"), ref)
    row = compare_waveforms(*level_match(ref, dull), profile)
    assert row["8_16k_body_mae_db"] > 20 and row["8_16k_body_bias_db"] < -20   # missing air, signed
    assert abs(row["onset_error_ms"]) < 1                                        # timing untouched
    shifted = np.roll(ref, int(0.010 * SAMPLE_RATE))
    assert compare_waveforms(*level_match(ref, shifted), profile)["onset_error_ms"] == pytest.approx(10, abs=0.5)
    late = ref.copy()
    late[SAMPLE_RATE // 2:] += 0.01 * np.random.default_rng(1).standard_normal(SAMPLE_RATE - SAMPLE_RATE // 2)
    assert compare_waveforms(*level_match(ref, late), profile)["late_excess_db"] > 10
    assert spectral_flatness(np.random.default_rng(2).standard_normal(SAMPLE_RATE)) > 0.5
    assert spectral_flatness(np.sin(2 * np.pi * 5000 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)) < 0.1


def test_level_match_equalises_rms_and_never_clips():
    ref = _synthetic_hit()
    loud, quiet = ref * 3, ref * 0.01
    matched_ref, a, b = level_match(ref, loud, quiet)
    rms = lambda x: np.sqrt(np.mean(x ** 2))  # noqa: E731
    assert rms(a) == pytest.approx(rms(matched_ref), rel=1e-6) and rms(b) == pytest.approx(rms(matched_ref), rel=1e-6)
    assert max(np.abs(matched_ref).max(), np.abs(a).max(), np.abs(b).max()) <= 0.99 + 1e-9


def test_summary_and_helpers_skip_unmeasurable_values():
    assert summarise([{"a": 1.0}, {"a": None}, {"a": 3.0}], ("a", "b")) == {"a": 2.0, "b": None}
    assert onset_ms(np.zeros(1000)) is None
    assert late_energy_db(np.ones(1000), 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Evidence and promotion
# ---------------------------------------------------------------------------

def test_reports_attach_to_a_run_survive_flushes_and_reach_the_api(tmp_path):
    run = TrainingRun(tmp_path, {"instrument": "kick", "epochs": 1})
    attach_report(run.directory, "fidelity", tmp_path / "f.json", {"vae_8_16k_body_mae_db": 6.1, "bad": float("nan")}, "n=4")
    run.epoch({"epoch": 1, "val_loss": 1.0})
    record = read_run(run.directory)
    assert record["reports"][0]["summary"] == {"vae_8_16k_body_mae_db": 6.1, "bad": None}
    assert '"reports"' in (run.directory / "data.js").read_text()
    assert find_run(tmp_path, run.id[:12]) == run.directory
    with pytest.raises(ValueError):
        attach_report(run.directory, "Not A Slug", None, {})
    server = dashboard_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/runs/{run.id}") as response:
            assert json.load(response)["reports"][0]["kind"] == "fidelity"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_promotion_needs_a_decision_and_evidence_then_copies_and_records(tmp_path):
    profile = get_profile("kick")
    profile = replace(profile, paths=replace(profile.paths, model_root=str(tmp_path / "models"),
                                            output_root=str(tmp_path / "output")))
    run = TrainingRun(tmp_path / "runs", {"instrument": "kick", "epochs": 1})
    candidate = tmp_path / "candidate.pth"
    torch.save({"model": {}, "instrument": "kick", "epoch": 7, "val_loss": 0.5}, candidate)
    (tmp_path / "candidate.controls.npz").write_bytes(b"sidecar")
    os.makedirs(os.path.dirname(profile.paths.checkpoint), exist_ok=True)
    torch.save({"model": {}, "instrument": "kick", "epoch": 1}, profile.paths.checkpoint)

    with pytest.raises(PromotionRefused, match="no decision"):
        promote(profile, str(candidate), run.id, runs_dir=str(tmp_path / "runs"))
    with pytest.raises(PromotionRefused, match="fidelity or controls"):
        promote(profile, str(candidate), run.id, decision="Promote", runs_dir=str(tmp_path / "runs"))
    attach_report(run.directory, "fidelity", None, {"vae_8_16k_body_mae_db": 5.0})
    attach_report(run.directory, "controls", None, {"pass_rate": 1.0})
    torch.save({"model": {}, "instrument": "snare"}, tmp_path / "wrong.pth")
    with pytest.raises(PromotionRefused, match="trained for snare"):
        promote(profile, str(tmp_path / "wrong.pth"), run.id, decision="Promote", runs_dir=str(tmp_path / "runs"))

    summary = promote(profile, str(candidate), run.id, decision="Promote: HF error down, listening 1/16 rejected",
                      runs_dir=str(tmp_path / "runs"))
    assert torch.load(profile.paths.checkpoint, weights_only=True)["epoch"] == 7
    assert torch.load(tmp_path / "models" / "vae_best_prev.pth", weights_only=True)["epoch"] == 1
    assert (tmp_path / "models" / "vae_best.controls.npz").read_bytes() == b"sidecar"
    record = read_run(run.directory)
    assert record["reports"][-1]["kind"] == "promotion"
    assert record["reports"][-1]["summary"]["checkpoint_sha256"] == summary["checkpoint_sha256"]
    assert "Promoted" in record["notes"]["decision"] and record["notes"]["decision"].startswith("Promote: HF")
