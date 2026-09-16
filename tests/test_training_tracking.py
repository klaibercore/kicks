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
from kicks.training.tracking import TrainingRun, dashboard_server, read_run
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
