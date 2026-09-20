"""`scripts/diffusion_listening.py`: blind generation A/B pairs for the Listening lab.

A tiny denoiser with a label bank, a four-hit corpus and a hand-made cluster
report stand in for the real thing; the script must pair each sample with the
corpus hit its target came from, keep the key blind in the report, and attach a
generation A/B — not a fidelity report — to the run.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from dataclasses import replace

import soundfile as sf
import torch

from kicks.data.waveforms import WaveformDataset
from kicks.instruments import get_profile
from kicks.training.tracking import TrainingRun, read_run

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load("diffusion_listening", ROOT / "scripts" / "diffusion_listening.py")
helpers = load("diffusion_helpers", ROOT / "tests" / "test_diffusion.py")


def test_pairs_are_blind_level_matched_and_attached_as_a_generation_ab(tmp_path, monkeypatch):
    kick = get_profile("kick")
    profile = replace(kick, paths=replace(
        kick.paths, data_root=str(tmp_path), model_root=str(tmp_path / "models"),
        output_root=str(tmp_path / "output")))
    corpus = helpers.corpus(tmp_path / "kicks", 4)
    dataset = WaveformDataset(str(corpus), profile, length=helpers.LENGTH, verbose=False)

    model = helpers.wake(helpers.tiny())
    model.set_label_stats(*dataset.label_stats())
    model.set_label_bank(dataset.label_matrix())
    (tmp_path / "models").mkdir()
    checkpoint = tmp_path / "models" / "diffusion_best.pth"
    torch.save({"model": model.state_dict(), "label_bank": model.label_bank, "instrument": "kick",
                "descriptors": [d.key for d in profile.descriptors], "epoch": 3, "val_loss": 0.5,
                **model.checkpoint_meta()}, checkpoint)

    # A cluster report whose per-file descriptors are the dataset's own labels.
    keys = [d.key for d in profile.descriptors]
    report = {"samples": [{"original_path": path, "cluster": 0,
                           "descriptors": dict(zip(keys, dataset.labels[i].tolist()))}
                          for i, path in enumerate(dataset.paths)]}
    (tmp_path / "cluster.json").write_text(json.dumps(report))

    run = TrainingRun(tmp_path / "output" / "training", {"instrument": "kick", "epochs": 1})
    monkeypatch.setattr(script, "get_profile", lambda name=None: profile)
    monkeypatch.setattr(script, "get_device", lambda: torch.device("cpu"))
    out = tmp_path / "output" / "fidelity" / "demo-ab"
    assert script.main(["--instrument", "kick", "--checkpoint", str(checkpoint), "--run", run.id,
                        "--out", str(out), "--count", "3", "--steps", "2",
                        "--cluster-report", str(tmp_path / "cluster.json")]) == 0

    key = json.loads((out / "listening" / "key.json").read_text())
    assert [pair["pair"] for pair in key] == [1, 2, 3]
    assert all({pair["a_is"], pair["b_is"]} == {"reference", "diffusion"} for pair in key)
    for pair in key:
        for name in (pair["a"], pair["b"], pair["a_hf"], pair["b_hf"]):
            audio, rate = sf.read(out / "listening" / name)
            assert rate == 44100 and len(audio) == helpers.LENGTH
            assert abs(audio).max() <= 0.99 + 1e-6              # shared anti-clip gain
    assert sorted(p.name for p in (out / "generation").glob("*.wav")) == ["diff_01.wav", "diff_02.wav", "diff_03.wav"]

    written = json.loads((out / "report.json").read_text())
    assert written["method"] == "diffusion_generation_ab_v1"
    assert written["held_out"] is False and written["vocoder"] is None and written["samples"] == []
    assert written["run_id"] == run.id and written["epoch"] == 3
    # Targets are bank rows, so every reference resolves to its own source file.
    assert all(entry["reference_distance_sd"] < 1e-3 for entry in written["generation"])
    assert set(written["hits"]) <= {p.name for p in corpus.glob("*.wav")}

    attached = read_run(run.directory)["reports"]
    assert len(attached) == 1 and attached[0]["kind"] == "fidelity"
    assert "generation A/B" in attached[0]["note"] and attached[0]["summary"]["held_out"] is False
