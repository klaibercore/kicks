"""Training telemetry, audio-detail metrics, lifecycle and local note editing."""
import json
import subprocess
import sys
import threading
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from kicks.instruments import get_profile
from kicks.training.metrics import DetailMetrics
from kicks.training.tracking import TrainingRun, dashboard_server, read_run, validation_trend
from kicks.training.trainer import train


def test_dashboard_import_does_not_load_the_model_stack():
    subprocess.run([sys.executable, "-c", "import sys; from kicks.training.tracking import dashboard_server; "
                    "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"], check=True)


class TinyVAE(nn.Module):
    latent_dim, n_mels, n_frames = 3, 128, 16

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(128 * 16, 3)
        self.decoder = nn.Linear(3, 128 * 16)

    def encode(self, data):
        mean = self.encoder(data.flatten(1))
        return mean, torch.zeros_like(mean)

    def decode(self, z):
        return self.decoder(z).sigmoid().reshape(-1, 1, 128, 16)

    def forward(self, data):
        mean, logvar = self.encode(data)
        return self.decode(mean + torch.randn_like(mean)), mean, logvar

    def checkpoint_meta(self):
        return {"latent_dim": 3, "n_mels": 128, "n_frames": 16}


def training_args(tmp_path):
    torch.manual_seed(15)
    model = TinyVAE()
    profile = get_profile("kick")
    profile = replace(profile, paths=replace(profile.paths, model_root=str(tmp_path / "models"),
                                            output_root=str(tmp_path / "output")))
    dataset = list(torch.rand(12, 1, 128, 16))
    return dict(model=model, dloader=DataLoader(dataset, batch_size=3),
                optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), profile=profile,
                epochs=2, device=torch.device("cpu"), val_split=.25, eval_every=1,
                runs_dir=str(tmp_path / "runs"), run_name="HF <experiment>",
                intent="Keep the attack", hypothesis="More detail survives", success_criteria="Listen and measure")


def test_each_training_run_records_baseline_epochs_configuration_and_completion(tmp_path):
    result = train(**training_args(tmp_path))
    directories = list((tmp_path / "runs").iterdir())
    assert len(directories) == 1
    run = read_run(directories[0])
    assert run["status"] == "completed"
    assert [r["epoch"] for r in run["history"]] == [0, 1, 2]
    assert len(result["loss"]) == 2
    assert run["config"]["train_samples"] == 9 and run["config"]["val_samples"] == 3
    assert run["config"]["split_fingerprint"] is None
    assert run["notes"]["objective"] == "Keep the attack"
    assert run["history"][0].get("train_loss") is None
    assert run["history"][2]["eval_proxy"] is not None
    assert run["history"][2]["hf_mae_db"] >= 0
    assert run["best"]["val_loss"]["value"] == min(r["val_loss"] for r in run["history"])
    assert (directories[0] / "index.html").exists()
    assert "<experiment>" not in (directories[0] / "data.js").read_text()


@pytest.mark.parametrize("exception,status", [(RuntimeError("test failure"), "failed"),
                                             (KeyboardInterrupt(), "interrupted")])
def test_failure_or_interrupt_preserves_baseline_and_records_status(tmp_path, monkeypatch, exception, status):
    kwargs = training_args(tmp_path)
    def fail():
        raise exception
    monkeypatch.setattr(kwargs["optimizer"], "step", fail)
    with pytest.raises(type(exception)):
        train(**kwargs)
    run = read_run(next((tmp_path / "runs").iterdir()))
    assert run["status"] == status and run["error"]
    assert [r["epoch"] for r in run["history"]] == [0]


def test_high_frequency_metrics_detect_detail_loss_without_changing_for_low_band_error():
    transient = get_profile("kick").transient_loss
    target = torch.full((2, 1, 128, 16), .5)
    identical = DetailMetrics(128, transient)
    identical.update(target, target)
    assert all(value == 0 for value in identical.compute().values())
    low_only = target.clone()
    low_only[..., :10, :] -= .1
    metric = DetailMetrics(128, transient)
    metric.update(low_only, target)
    assert all(value == 0 for value in metric.compute().values())
    missing_air = target.clone()
    missing_air[..., 100:115, :] -= .1
    full, batched = DetailMetrics(128, transient), DetailMetrics(128, transient)
    full.update(missing_air, target)
    for r, t in zip(missing_air, target):
        batched.update(r[None], t[None])
    assert full.compute()["air_mae_db"] > 0
    assert full.compute() == pytest.approx(batched.compute(), rel=1e-5)
    silent = DetailMetrics(128, transient)
    silent.update(torch.zeros_like(target), torch.zeros_like(target))
    assert all(value is None for value in silent.compute().values())


def test_dashboard_notes_are_safe_and_survive_concurrent_metric_updates(tmp_path):
    run = TrainingRun(tmp_path, {"instrument": "kick", "epochs": 2}, notes={"objective": "Before"})
    second = TrainingRun(tmp_path, {"instrument": "kick", "epochs": 2})
    assert second.id != run.id
    server = dashboard_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/api/runs") as response:
            assert len(json.load(response)["runs"]) == 2
        request = Request(base + f"/api/runs/{run.id}/notes",
                          data=json.dumps({"observations": "</script><script>alert(1)</script>"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert response.status == 200
        run.progress(force=True, epoch=1)
        run.epoch({"epoch": 1, "val_loss": 0.4, "eval_proxy": float("nan")})
        record = read_run(run.directory)
        assert record["notes"]["observations"].startswith("</script>")
        assert record["history"][0]["eval_proxy"] is None
        assert "</script>" not in (run.directory / "notes.js").read_text()
        assert "</script>" not in (run.directory / "data.js").read_text()
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/runs/%2e%2e")
        assert error.value.code == 404
        invalid = Request(base + f"/api/runs/{run.id}/notes", data=b'{"objective":42}',
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(invalid)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _wav(path, samples: int = 64):
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(struct.pack(f"<{samples}h", *([0] * samples)))
    return path


def _fidelity_report(directory, instrument="kick", pairs=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps({
        "instrument": instrument, "checkpoint": "/models/vae_best.pth", "epoch": 3,
        "held_out": True, "summary": {"mel_mae_db": 1.5},
        "samples": [{"path": "data/kicks/secret_name.wav"}],
    }))
    key = []
    for index in range(1, pairs + 1):
        names = {side: f"pair_{index:02d}_{side}.wav" for side in "AB"}
        for name in names.values():
            _wav(directory / "listening" / name)
        key.append({"pair": index, "a_is": "vae", "b_is": "reference",
                    "a": names["A"], "b": names["B"]})
    (directory / "listening" / "key.json").write_text(json.dumps(key))
    return directory


def test_listening_lab_serves_reports_allow_listed_audio_and_records_verdicts(tmp_path):
    from kicks.training.tracking import attach_report

    runs = tmp_path / "training"
    sibling = _fidelity_report(tmp_path / "fidelity" / "demo", pairs=2)
    elsewhere = _fidelity_report(tmp_path / "audits" / "custom-out", instrument="snare")
    _wav(tmp_path / "outside.wav")
    (tmp_path / "fidelity" / "broken").mkdir(parents=True)
    (tmp_path / "fidelity" / "broken" / "report.json").write_text("not json")

    run = TrainingRun(runs, {"instrument": "snare", "epochs": 1})
    attach_report(run.directory, "fidelity", elsewhere / "report.json", {"mel_mae_db": 2.0})

    server = dashboard_server(runs, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(base + path) as response:
            return json.load(response)

    def post(path, payload, content_type="application/json"):
        request = Request(base + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": content_type})
        with urlopen(request) as response:
            return json.load(response)

    try:
        assert get("/api/capabilities")["listening"] is True
        assert get("/api/capabilities")["trend"] is True
        assert [row["trend"] for row in get("/api/runs")["runs"]] == [[]]  # no epochs yet
        # Both discovery sources appear; the unparsable directory does not.
        reports = {entry["slug"]: entry for entry in get("/api/fidelity")["reports"]}
        assert set(reports) == {"demo", "custom-out"}
        assert reports["demo"]["pairs"] == 2 and reports["demo"]["judged"] == 0
        assert reports["custom-out"]["instrument"] == "snare"
        assert reports["demo"]["checkpoint"] == "vae_best.pth"

        # The blind key stays blind until asked for explicitly.
        detail = get("/api/fidelity/demo")
        assert [pair["a"] for pair in detail["pairs"]] == ["pair_01_A.wav", "pair_02_A.wav"]
        assert all("a_is" not in pair for pair in detail["pairs"])
        assert get("/api/fidelity/demo/key")["key"][0]["a_is"] == "vae"

        # Audio: only names inside the report's listening/reconstruction/generation dirs.
        with urlopen(base + "/api/fidelity/demo/audio/pair_01_A.wav") as response:
            assert response.headers["Content-Type"] == "audio/wav"
            assert len(response.read()) == (sibling / "listening" / "pair_01_A.wav").stat().st_size
        for bad in ("/api/fidelity/demo/audio/..%2F..%2Foutside.wav",
                    "/api/fidelity/demo/audio/missing.wav",
                    "/api/fidelity/demo/audio/report.json",
                    "/api/fidelity/nope/audio/pair_01_A.wav",
                    "/api/fidelity/%2e%2e"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + bad)
            assert error.value.code == 404, bad

        # Verdicts persist next to the key, merge per pair and band, and clear on null.
        assert post("/api/fidelity/demo/verdicts", {"pair": 1, "band": "full", "choice": "a"})[
            "verdicts"]["1"]["full"]["choice"] == "a"
        post("/api/fidelity/demo/verdicts", {"pair": 1, "band": "hf", "choice": "tie"})
        saved = json.loads((sibling / "listening" / "verdicts.json").read_text())
        assert set(saved["1"]) == {"full", "hf"}
        assert {e["slug"]: e["judged"] for e in get("/api/fidelity")["reports"]}["demo"] == 1
        post("/api/fidelity/demo/verdicts", {"pair": 1, "band": "full", "choice": None})
        post("/api/fidelity/demo/verdicts", {"pair": 1, "band": "hf", "choice": None})
        assert json.loads((sibling / "listening" / "verdicts.json").read_text()) == {}

        for payload, code in (({"pair": 1, "band": "mid", "choice": "a"}, 400),
                              ({"pair": 1, "band": "full", "choice": "maybe"}, 400),
                              ({"pair": "x", "band": "full", "choice": "a"}, 400)):
            with pytest.raises(HTTPError) as error:
                post("/api/fidelity/demo/verdicts", payload)
            assert error.value.code == code, payload
        with pytest.raises(HTTPError) as error:
            post("/api/fidelity/demo/verdicts", {"pair": 1, "band": "full", "choice": "a"},
                 content_type="text/plain")
        assert error.value.code == 415
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_validation_trend_keeps_the_ends_and_thins_evenly():
    history = [{"epoch": e, "val_loss": 1.0 / (e + 1)} for e in range(201)]
    history.insert(5, {"epoch": 4.5, "val_loss": float("nan")})
    history.insert(9, {"epoch": 8.5})
    trend = validation_trend(history, points=48)
    assert len(trend) == 48
    assert trend[0] == 1.0 and trend[-1] == 1.0 / 201
    assert trend == sorted(trend, reverse=True)
    assert validation_trend(history[:4]) == [1.0, 0.5, 1.0 / 3, 0.25]
