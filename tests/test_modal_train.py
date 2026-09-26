"""Local contracts for the standalone Modal training wrapper."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/modal_train.py"
spec = importlib.util.spec_from_file_location("modal_train", SCRIPT)
modal_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modal_train)


def _fixture(tmp_path: Path):
    data = tmp_path / "data"
    entries = []
    next_id = 1
    for corpus, count in (("kicks", 2), ("snares", 1), ("hihats", 2)):
        directory = data / corpus
        directory.mkdir(parents=True)
        (directory / "SOURCES.md").write_text("license\n")
        for _ in range(count):
            name = f"#{next_id}.wav"
            payload = f"audio-{next_id}".encode()
            (directory / name).write_bytes(payload)
            entries.append({
                "id": next_id, "corpus": corpus, "old": f"old-{next_id}.wav",
                "new": name, "size": len(payload), "sha256": sha256(payload).hexdigest(),
            })
            next_id += 1
    mapping = data / "corpus_ids.json"
    mapping.write_text(json.dumps({
        "version": 1, "applied": True, "next": next_id, "entries": entries,
    }))
    return data, mapping


def test_verify_corpus_tree_checks_names_sizes_hashes_and_sources(tmp_path, monkeypatch):
    data, mapping = _fixture(tmp_path)
    monkeypatch.setattr(modal_train, "EXPECTED_COUNTS", {"kicks": 2, "snares": 1, "hihats": 2})
    result = modal_train.verify_corpus_tree(data, mapping)
    assert result["corpora"]["kicks"]["wav_files"] == 2
    (data / "hihats/#5.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        modal_train.verify_corpus_tree(data, mapping)


def test_remote_style_verification_allows_missing_sources(tmp_path, monkeypatch):
    data, mapping = _fixture(tmp_path)
    monkeypatch.setattr(modal_train, "EXPECTED_COUNTS", {"kicks": 2, "snares": 1, "hihats": 2})
    for corpus in modal_train.EXPECTED_COUNTS:
        (data / corpus / "SOURCES.md").unlink()

    with pytest.raises(FileNotFoundError, match="missing license record"):
        modal_train.verify_corpus_tree(data, mapping)

    result = modal_train.verify_corpus_tree(
        data,
        mapping,
        require_sources=False,
        hash_workers=3,
    )
    assert all(not corpus["sources"] for corpus in result["corpora"].values())

    with pytest.raises(ValueError, match="hash_workers"):
        modal_train.verify_corpus_tree(data, mapping, hash_workers=0)


def test_safe_paths_and_wrapper_owned_training_flags_are_refused():
    for value in ("/absolute", "../escape", "models/../escape"):
        with pytest.raises(ValueError):
            modal_train.safe_relative_path(value, field="test")
    modal_train.validate_extra_args(["--epochs", "2", "--batch-size=4"])
    with pytest.raises(ValueError, match="owned"):
        modal_train.validate_extra_args(["--data=/wrong"])


def test_build_spec_rejects_empty_tracking_and_unsafe_job_id(monkeypatch):
    monkeypatch.setattr(modal_train, "mapping_digest", lambda: "mapping")
    monkeypatch.setattr(modal_train, "source_digest", lambda: "source")
    base = dict(
        instrument="hihat", gpu="L4", run_name="Smoke", intent="Intent",
        hypothesis="Hypothesis", success_criteria="Criteria",
        model_dir="models/experiments/smoke", resume=None, subset_manifest=None,
        job_id="safe-job", training_args=[], memory_gib=16, cpu=4, commit_interval_seconds=300,
    )
    spec = modal_train.build_spec(Namespace(**base))
    assert spec["job_id"] == "safe-job"
    assert (spec["memory_mib"], spec["cpu"], spec["commit_interval_seconds"]) == (16 * 1024, 4.0, 300)
    with pytest.raises(ValueError, match="commit-interval"):
        modal_train.build_spec(Namespace(**{**base, "commit_interval_seconds": 5}))
    with pytest.raises(ValueError, match="cannot be empty"):
        modal_train.build_spec(Namespace(**{**base, "hypothesis": " "}))
    for job_id in ("../escape", None, ""):
        with pytest.raises(ValueError, match="job-id"):
            modal_train.build_spec(Namespace(**{**base, "job_id": job_id}))
    with pytest.raises(ValueError, match="memory-gib"):
        modal_train.build_spec(Namespace(**{**base, "memory_gib": 256}))


def test_plan_prints_the_data_directory_the_remote_trainer_reads():
    full = {"instrument": "hihat", "corpus": "hihats", "job_id": "j1", "subset": None}
    subset = {**full, "subset": {"files": ["#18104.wav"]}}
    assert modal_train.remote_data_dir(full) == Path("/corpus/hihats")
    assert modal_train.remote_data_dir(subset) == Path("/tmp/kicks-subsets/hihat-j1")


def test_a_restarted_or_reused_job_is_refused(tmp_path):
    spec = {"job_id": "j1", "model_dir": "models/experiments/smoke"}
    modal_train.check_fresh_start(spec, tmp_path)                      # fresh volume

    record = tmp_path / "output/modal/jobs/j1.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"status": "running"}')
    with pytest.raises(RuntimeError, match="restarted attempt"):
        modal_train.check_fresh_start(spec, tmp_path)

    (tmp_path / "models/experiments/smoke/hihat").mkdir(parents=True)
    (tmp_path / "models/experiments/smoke/hihat/diffusion_best.pth").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="new --model-dir"):
        modal_train.check_fresh_start({**spec, "job_id": "j2"}, tmp_path)


def test_trainer_runs_with_periodic_commits_that_never_stop_it(tmp_path):
    commits = []

    def flaky_commit():
        commits.append(1)
        if len(commits) == 1:
            raise OSError("transient")

    sleeper = [sys.executable, "-c", "import time; time.sleep(0.6)"]
    code = modal_train.run_with_commits(sleeper, cwd=tmp_path, env=dict(os.environ),
                                        commit=flaky_commit, interval=0.1)
    assert code == 0 and len(commits) >= 3
    failing = [sys.executable, "-c", "raise SystemExit(3)"]
    assert modal_train.run_with_commits(failing, cwd=tmp_path, env=dict(os.environ),
                                        commit=lambda: None, interval=5) == 3


def test_sync_merge_skips_in_flight_temp_files(tmp_path):
    staged = tmp_path / "staged/run-a"
    staged.mkdir(parents=True)
    (staged / "run.json").write_text("{}")
    (staged / ".run.json.abc123.tmp").write_text("partial")
    (staged / "diffusion_best.pth.tmp").write_bytes(b"partial")
    assert modal_train.merge_downloaded_tree(tmp_path / "staged", tmp_path / "local",
                                             preserve_notes=True) == 1
    assert sorted(p.name for p in (tmp_path / "local/run-a").iterdir()) == ["run.json"]


def test_remote_command_contains_required_tracking_and_destinations():
    spec = {
        "instrument": "hihat", "run_name": "Smoke", "intent": "Test image",
        "hypothesis": "CUDA is faster", "success_criteria": "Two epochs complete",
        "model_dir": "models/experiments/modal-smoke", "resume": None,
        "training_args": ["--epochs", "2"],
    }
    command = modal_train.remote_training_command(spec, Path("/tmp/subset"))
    joined = " ".join(command)
    assert "diffusion-train" in command
    for value in ("Smoke", "Test image", "CUDA is faster", "Two epochs complete"):
        assert value in command
    assert "/results/models/experiments/modal-smoke" in joined
    assert "/results/output/training" in joined


def test_sync_merge_preserves_existing_local_notes(tmp_path):
    staged = tmp_path / "staged/run-a"
    local = tmp_path / "local/run-a"
    staged.mkdir(parents=True)
    local.mkdir(parents=True)
    (staged / "run.json").write_text('{"epoch": 2}')
    (staged / "notes.json").write_text('{"objective": "remote"}')
    (staged / "notes.js").write_text("remote")
    (local / "notes.json").write_text('{"objective": "local"}')
    (local / "notes.js").write_text("local")
    assert modal_train.merge_downloaded_tree(tmp_path / "staged", tmp_path / "local", preserve_notes=True) == 1
    assert json.loads((local / "run.json").read_text()) == {"epoch": 2}
    assert json.loads((local / "notes.json").read_text())["objective"] == "local"
    assert (local / "notes.js").read_text() == "local"


def test_an_interrupted_cloud_call_reports_that_nothing_was_recorded():
    assert modal_train.completed({"ok": True}, "x") == {"ok": True}
    with pytest.raises(RuntimeError, match="interrupted before Modal returned the image test"):
        modal_train.completed(None, "the image test")


def test_the_runtime_path_excludes_the_project_venv():
    assert str(modal_train.PROJECT_PYTHON.parent) not in modal_train.SYSTEM_PATH.split(":")
    assert modal_train.SYSTEM_PATH.split(":")[0] == "/usr/local/bin"


def test_cloud_commands_require_an_explicit_per_step_confirmation():
    with pytest.raises(ValueError, match="--confirm-cloud"):
        modal_train.require_cloud_confirmation(Namespace(confirm_cloud=False))
    modal_train.require_cloud_confirmation(Namespace(confirm_cloud=True))
