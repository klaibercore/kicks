#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["modal==1.5.5"]
# ///
"""Prepare, verify, launch, and synchronize Kicks training on Modal.

The script deliberately runs outside the project's environment.  Modal needs a
modern protobuf, while the locked Kicks environment contains protobuf 3.19.6
for ``descript-audiotools``.  The remote image keeps the same split: training
runs in the uv-created ``/.uv/.venv`` subprocess, and the image resets ``PATH``
after ``uv_sync`` so Modal's runtime starts with the base image's interpreter
rather than the venv's (it would otherwise import protobuf 3.19.6 and crash).

Cloud-touching commands require ``--confirm-cloud``.  This is a guardrail, not
an approval mechanism: review each printed plan before invoking that command.
Run ``plan`` and ``check-local`` freely; neither contacts Modal.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

try:  # Tests import the pure helpers from the project environment (without Modal).
    import modal
except ImportError:  # pragma: no cover - exercised only outside the standalone script
    modal = None


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "data" / "corpus_ids.json"
CORPUS_VOLUME_NAME = "kicks-corpus"
RESULTS_VOLUME_NAME = "kicks-training"
CORPUS_MOUNT = Path("/corpus")
RESULTS_MOUNT = Path("/results")
REMOTE_WORKSPACE = Path("/workspace")
PROJECT_PYTHON = Path("/.uv/.venv/bin/python")
#: PATH of the official python:*-slim base image, without the project venv.
SYSTEM_PATH = "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
LOCAL_STATE = ROOT / "output" / "modal"
RECEIPT = LOCAL_STATE / "corpus-verification.json"
LAUNCHES = LOCAL_STATE / "launches"

CORPORA = {"kick": "kicks", "snare": "snares", "hihat": "hihats"}
EXPECTED_COUNTS = {"kicks": 10_109, "snares": 7_994, "hihats": 5_824}
GPU_RE = re.compile(
    r"^(?:any|T4|L4|A10|L40S|A100|A100-40GB|A100-80GB|RTX-PRO-6000|"
    r"H100!?|H200|B200\+?|B300)(?::[1-8])?$",
    re.IGNORECASE,
)
JOB_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
RESERVED_TRAIN_FLAGS = {
    "--data", "-d", "--model-dir", "--runs-dir", "--run-name", "--intent",
    "--hypothesis", "--success-criteria", "--resume", "--instrument", "-i",
}
SYNC_MODEL_FILES = {
    "diffusion_best.pth", "diffusion_best_control.pth", "diffusion_checkpoint.pth",
    "diffusion_loss_curves.png",
}
#: Modal commits a volume by itself only when the container exits.
COMMIT_INTERVAL_SECONDS = 300
#: Reservations are billed even when unused. The largest corpus (kicks) needs
#: about 2.5 GiB as float32 waveforms; 16 GiB leaves room for Torch and CUDA.
DEFAULT_MEMORY_GIB = 16
DEFAULT_CPU = 4.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_mapping(path: Path = MAPPING) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("version") != 1 or not value.get("applied"):
        raise ValueError(f"{path} is not an applied version-1 corpus mapping")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} has no corpus entries")
    return value


def mapping_digest(path: Path = MAPPING) -> str:
    return sha256_file(path)


def entries_by_corpus(mapping: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = {name: [] for name in EXPECTED_COUNTS}
    for entry in mapping["entries"]:
        corpus = entry.get("corpus")
        if corpus not in grouped:
            raise ValueError(f"unknown corpus in mapping: {corpus!r}")
        grouped[corpus].append(entry)
    return grouped


def verify_corpus_tree(
    data_root: Path,
    mapping_path: Path = MAPPING,
    *,
    hash_files: bool = True,
    require_sources: bool = True,
    hash_workers: int = 1,
) -> dict[str, Any]:
    """Verify names, sizes, and hashes against the permanent license mapping.

    Local verification also requires each corpus's license record.  The Modal
    volume intentionally contains only training WAVs, so remote verification
    records whether ``SOURCES.md`` is present without requiring it.
    """
    mapping = load_mapping(mapping_path)
    if hash_workers < 1:
        raise ValueError("hash_workers must be at least 1")
    grouped = entries_by_corpus(mapping)
    result: dict[str, Any] = {
        "mapping_sha256": mapping_digest(mapping_path),
        "hashes_checked": hash_files,
        "corpora": {},
    }
    for corpus, expected_count in EXPECTED_COUNTS.items():
        directory = data_root / corpus
        if not directory.is_dir():
            raise FileNotFoundError(f"missing corpus directory: {directory}")
        sources_present = (directory / "SOURCES.md").is_file()
        if require_sources and not sources_present:
            raise FileNotFoundError(f"missing license record: {directory / 'SOURCES.md'}")
        expected = {entry["new"]: entry for entry in grouped[corpus]}
        live = {path.name: path for path in directory.glob("*.wav") if path.is_file()}
        missing, extra = sorted(expected.keys() - live.keys()), sorted(live.keys() - expected.keys())
        if missing or extra:
            raise ValueError(
                f"{corpus}: filename drift (missing={missing[:3]}, extra={extra[:3]})",
            )
        if len(live) != expected_count or len(expected) != expected_count:
            raise ValueError(
                f"{corpus}: expected {expected_count} mapped WAVs, got "
                f"{len(expected)} mapped and {len(live)} live",
            )
        ordered_names = sorted(expected, key=lambda name: int(name[1:-4]))
        for name in ordered_names:
            entry = expected[name]
            path = live[name]
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"{path}: size differs from corpus_ids.json")
        if hash_files:
            paths = [live[name] for name in ordered_names]
            if hash_workers == 1:
                digests = map(sha256_file, paths)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=hash_workers) as executor:
                    digests = list(executor.map(sha256_file, paths))
            for name, digest in zip(ordered_names, digests, strict=True):
                if digest != expected[name]["sha256"]:
                    raise ValueError(f"{live[name]}: SHA-256 differs from corpus_ids.json")
        result["corpora"][corpus] = {
            "wav_files": len(live),
            "sources": sources_present,
            "first": min(live),
            "last": max(live, key=lambda name: int(name[1:-4])),
        }
    result["verified_at"] = utc_now()
    return result


def safe_relative_path(text: str, *, field: str) -> Path:
    value = Path(text)
    if value.is_absolute() or not value.parts or any(part in ("", ".", "..") for part in value.parts):
        raise ValueError(f"{field} must be a normalized path relative to the repository")
    return value


def load_subset(path: Path, instrument: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    corpus = CORPORA[instrument]
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) != manifest.get("size"):
        raise ValueError(f"{path}: invalid size/files")
    if len(files) != len(set(files)):
        raise ValueError(f"{path}: duplicate filenames")
    expected_names = {entry["new"] for entry in entries_by_corpus(load_mapping())[corpus]}
    unknown = sorted(set(files) - expected_names)
    if unknown:
        raise ValueError(f"{path}: files are not in {corpus}: {unknown[:3]}")
    local_dir = ROOT / "data" / corpus
    missing = [name for name in files if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: local subset sources are missing: {missing[:3]}")
    resolved = path.resolve()
    try:
        manifest_name = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        manifest_name = resolved.name
    return {
        "size": len(files),
        "seed": int(manifest["seed"]),
        "files": sorted(files, key=lambda name: int(name[1:-4])),
        "manifest": manifest_name,
        "manifest_sha256": sha256_file(path),
    }


def source_digest() -> str:
    digest = hashlib.sha256()
    paths: Iterable[Path] = [ROOT / "pyproject.toml", ROOT / "uv.lock", Path(__file__)]
    paths = [*paths, *sorted((ROOT / "kicks").rglob("*.py"))]
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_extra_args(arguments: list[str]) -> None:
    for item in arguments:
        flag = item.split("=", 1)[0]
        if flag in RESERVED_TRAIN_FLAGS:
            raise ValueError(f"{flag} is owned by the Modal wrapper; use its dedicated option")


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    if args.instrument not in CORPORA:
        raise ValueError(f"unknown instrument: {args.instrument}")
    if not GPU_RE.fullmatch(args.gpu):
        raise ValueError(f"unsupported Modal GPU request: {args.gpu!r}")
    for field in ("run_name", "intent", "hypothesis", "success_criteria"):
        if not getattr(args, field).strip():
            raise ValueError(f"--{field.replace('_', '-')} cannot be empty")
    model_dir = safe_relative_path(args.model_dir, field="--model-dir")
    if not str(model_dir).startswith("models/experiments/"):
        raise ValueError("--model-dir must live under models/experiments/")
    resume = safe_relative_path(args.resume, field="--resume") if args.resume else None
    extra = list(args.training_args or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    validate_extra_args(extra)
    subset = load_subset(Path(args.subset_manifest), args.instrument) if args.subset_manifest else None
    job_id = args.job_id
    if not job_id or not JOB_RE.fullmatch(job_id):
        raise ValueError("--job-id must be 1-80 letters, numbers, dots, underscores, or hyphens")
    if not 4 <= args.memory_gib <= 128:
        raise ValueError("--memory-gib must be between 4 and 128")
    if not 1 <= args.cpu <= 16:
        raise ValueError("--cpu must be between 1 and 16")
    if not 30 <= args.commit_interval_seconds <= 3600:
        raise ValueError("--commit-interval-seconds must be between 30 and 3600")
    return {
        "job_id": job_id,
        "created_at": utc_now(),
        "instrument": args.instrument,
        "corpus": CORPORA[args.instrument],
        "gpu": args.gpu,
        "memory_mib": int(args.memory_gib * 1024),
        "cpu": float(args.cpu),
        "commit_interval_seconds": int(args.commit_interval_seconds),
        "run_name": args.run_name.strip(),
        "intent": args.intent.strip(),
        "hypothesis": args.hypothesis.strip(),
        "success_criteria": args.success_criteria.strip(),
        "model_dir": model_dir.as_posix(),
        "resume": resume.as_posix() if resume else None,
        "subset": subset,
        "training_args": extra,
        "mapping_sha256": mapping_digest(),
        "source_sha256": source_digest(),
    }


def remote_training_command(spec: dict[str, Any], data_dir: Path) -> list[str]:
    model_dir = RESULTS_MOUNT / spec["model_dir"]
    command = [
        str(PROJECT_PYTHON), "-c", "from kicks.cli import app; app()",
        "diffusion-train", "--instrument", spec["instrument"], "--data", str(data_dir),
        "--model-dir", str(model_dir), "--runs-dir", str(RESULTS_MOUNT / "output/training"),
        "--run-name", spec["run_name"], "--intent", spec["intent"],
        "--hypothesis", spec["hypothesis"], "--success-criteria", spec["success_criteria"],
    ]
    if spec.get("resume"):
        command.extend(["--resume", str(RESULTS_MOUNT / spec["resume"])])
    command.extend(spec.get("training_args") or [])
    return command


def remote_data_dir(spec: dict[str, Any]) -> Path:
    """The ``--data`` directory the remote trainer reads; ``plan`` prints the same one."""
    if not spec.get("subset"):
        return CORPUS_MOUNT / spec["corpus"]
    return Path("/tmp/kicks-subsets") / f"{spec['instrument']}-{spec['job_id']}"


def rebuild_remote_subset(spec: dict[str, Any]) -> Path:
    subset = spec.get("subset")
    if not subset:
        return remote_data_dir(spec)
    destination = remote_data_dir(spec)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    source = CORPUS_MOUNT / spec["corpus"]
    for name in subset["files"]:
        target = source / name
        if not target.is_file():
            raise FileNotFoundError(f"uploaded corpus is missing subset member {target}")
        (destination / name).symlink_to(target)
    (destination / "manifest.json").write_text(json.dumps(subset, indent=2) + "\n")
    return destination


def _job_record_path(job_id: str, results_root: Path = RESULTS_MOUNT) -> Path:
    return results_root / "output/modal/jobs" / f"{job_id}.json"


def check_fresh_start(spec: dict[str, Any], results_root: Path = RESULTS_MOUNT) -> None:
    """Refuse to start over an earlier attempt's results.

    Modal re-runs an input after worker preemption. A second attempt would begin
    again at epoch 0 and overwrite the first attempt's checkpoints in the same
    model directory, so it stops instead: the partial run stays intact and
    resuming it (fresh optimizer, restarted cosine) is a reviewed decision.
    """
    record = _job_record_path(spec["job_id"], results_root)
    if record.exists():
        status = json.loads(record.read_text()).get("status")
        raise RuntimeError(
            f"job {spec['job_id']} already has a record (status {status!r}); "
            "a restarted attempt is refused to protect the first attempt's results",
        )
    model_dir = results_root / spec["model_dir"]
    if model_dir.is_dir() and any(path.is_file() for path in model_dir.rglob("*")):
        raise RuntimeError(
            f"{spec['model_dir']} already holds files on {RESULTS_VOLUME_NAME}; "
            "use a new --model-dir for every launch",
        )


def run_with_commits(
    command: list[str], *, cwd: Path, env: dict[str, str], commit,
    interval: float = COMMIT_INTERVAL_SECONDS,
) -> int:
    """Run the trainer, committing the results volume every ``interval`` seconds.

    Without this, a mid-run ``sync`` sees nothing and a hard kill (timeout, OOM,
    preemption) loses every checkpoint of the run. A failed commit is reported
    and retried at the next interval; it never stops training.
    """
    process = subprocess.Popen(command, cwd=cwd, env=env)
    try:
        while True:
            try:
                return process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                try:
                    commit()
                except Exception as error:  # noqa: BLE001 - keep training; the next commit retries
                    print(f"warning: periodic volume commit failed: {error}", file=sys.stderr, flush=True)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _write_remote_job(spec: dict[str, Any], status: str, **extra: Any) -> None:
    path = _job_record_path(spec["job_id"])
    previous = json.loads(path.read_text()) if path.exists() else {}
    value = {**previous, **spec, "status": status, "updated_at": utc_now(), **extra}
    atomic_json(path, value)


def merge_downloaded_tree(staged: Path, destination: Path, *, preserve_notes: bool) -> int:
    copied = 0
    for source in sorted(path for path in staged.rglob("*") if path.is_file()):
        # A periodic commit can snapshot a writer's in-flight temp file
        # (tracking's ".run.json.<id>.tmp", a checkpoint's "<name>.tmp").
        if source.name.startswith(".") or source.name.endswith(".tmp"):
            continue
        relative = source.relative_to(staged)
        target = destination / relative
        if preserve_notes and target.name in {"notes.json", "notes.js"} and target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.sync-{os.getpid()}")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        copied += 1
    return copied


def completed(value: Any, what: str) -> Any:
    """Modal's ``app.run()`` swallows Ctrl-C and leaves its block normally; say so plainly."""
    if value is None:
        raise RuntimeError(f"interrupted before Modal returned {what}; nothing was recorded")
    return value


def require_cloud_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_cloud:
        raise ValueError("cloud command refused: review this step, then add --confirm-cloud")


if modal is not None:
    verify_image = (
        modal.Image.debian_slim(python_version="3.11")
        .add_local_file(str(MAPPING), "/verification/corpus_ids.json", copy=True)
    )
    training_image = (
        # Same interpreter minor version as the local runs (.python-version).
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsndfile1")
        .uv_sync(str(ROOT), groups=[], uv_version="0.11.7")
        # uv_sync prepends /.uv/.venv/bin to PATH. Modal starts its own runtime
        # with the first `python` on PATH, which would then import the venv's
        # protobuf 3.19.6 and crash at container start. Restore the base image's
        # PATH; training calls the venv's interpreter by absolute path.
        .env({"PATH": SYSTEM_PATH})
        .add_local_dir(
            str(ROOT / "kicks"), str(REMOTE_WORKSPACE / "kicks"), copy=True,
            # Local bytecode changes on every test run and would force a rebuild.
            ignore=["**/__pycache__", "**/*.pyc"],
        )
    )
    verify_app = modal.App("kicks-corpus-verify")
    image_app = modal.App("kicks-image-test")
    train_app = modal.App("kicks-training")
    corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
    results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

    @verify_app.function(
        image=verify_image,
        volumes={str(CORPUS_MOUNT): corpus_volume.with_mount_options(read_only=True)},
        timeout=14_400,
    )
    def verify_uploaded_corpus(expected_mapping_sha256: str) -> dict[str, Any]:
        mapping_path = Path("/verification/corpus_ids.json")
        if sha256_file(mapping_path) != expected_mapping_sha256:
            raise ValueError("verification image contains a different corpus_ids.json")
        return verify_corpus_tree(
            CORPUS_MOUNT,
            mapping_path,
            hash_files=True,
            require_sources=False,
            hash_workers=32,
        )

    @image_app.function(image=training_image, timeout=1800)
    def test_training_image() -> dict[str, Any]:
        import google.protobuf

        if Path(sys.prefix).resolve() == PROJECT_PYTHON.parent.parent.resolve():
            raise RuntimeError(f"Modal's runtime is running inside the project venv ({sys.prefix})")

        program = (
            "import json, google.protobuf, torch; "
            "from kicks.nn import WaveformUNet; "
            "m=WaveformUNet(n_descriptors=3, channels=(8,16), factors=(2,2), "
            "cond_dim=16, blocks_per_scale=1, attention_scales=0); "
            "print(json.dumps({'protobuf': google.protobuf.__version__, "
            "'torch': torch.__version__, 'cuda': torch.cuda.is_available(), "
            "'parameters': sum(p.numel() for p in m.parameters())}))"
        )
        completed = subprocess.run(
            [str(PROJECT_PYTHON), "-c", program], cwd=REMOTE_WORKSPACE,
            text=True, capture_output=True, check=True,
        )
        project = json.loads(completed.stdout.strip().splitlines()[-1])
        if project["protobuf"] != "3.19.6":
            raise RuntimeError(f"locked project protobuf is {project['protobuf']}, expected 3.19.6")
        return {"modal_runtime_python": sys.executable,
                "modal_runtime_protobuf": google.protobuf.__version__, "project": project}

    @train_app.function(
        image=training_image,
        volumes={
            str(CORPUS_MOUNT): corpus_volume.with_mount_options(read_only=True),
            str(RESULTS_MOUNT): results_volume,
        },
        timeout=86_400,
        memory=DEFAULT_MEMORY_GIB * 1024,
        cpu=DEFAULT_CPU,
        # One job per container: without this a detached app keeps the GPU
        # container idle until scale-down after training ends (~2 min billed
        # on the first smoke run), and a later job could reuse its /tmp.
        single_use_containers=True,
    )
    def train_remote(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            check_fresh_start(spec)
        except RuntimeError as error:
            _write_remote_job(spec, "refused", finished_at=utc_now(), error=str(error))
            results_volume.commit()
            raise
        data_dir = rebuild_remote_subset(spec)
        command = remote_training_command(spec, data_dir)
        _write_remote_job(spec, "running", command=command, started_at=utc_now())
        results_volume.commit()
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": str(REMOTE_WORKSPACE),
            "KICKS_DATA_DIR": str(CORPUS_MOUNT),
            "KICKS_OUTPUT_DIR": str(RESULTS_MOUNT / "output"),
            "KICKS_TRAINING_CONTEXT": json.dumps({
                "provider": "modal",
                "job_id": spec["job_id"],
                "gpu_request": spec["gpu"],
                "corpus_volume": CORPUS_VOLUME_NAME,
                "results_volume": RESULTS_VOLUME_NAME,
                "mapping_sha256": spec["mapping_sha256"],
                "source_sha256": spec["source_sha256"],
                "subset": spec.get("subset"),
            }, sort_keys=True),
        })
        try:
            returncode = run_with_commits(
                command, cwd=REMOTE_WORKSPACE, env=environment, commit=results_volume.commit,
                interval=spec.get("commit_interval_seconds", COMMIT_INTERVAL_SECONDS),
            )
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        except BaseException as error:
            _write_remote_job(
                spec, "failed", finished_at=utc_now(),
                error=f"{type(error).__name__}: {error}",
            )
            results_volume.commit()
            raise
        _write_remote_job(spec, "completed", finished_at=utc_now(), returncode=0)
        results_volume.commit()
        return {"job_id": spec["job_id"], "status": "completed"}

else:  # pragma: no cover - the definitions exist only in the standalone environment
    verify_app = image_app = train_app = corpus_volume = results_volume = None


def download_volume_tree(volume: Any, remote_prefix: str, destination: Path) -> int:
    count = 0
    prefix = PurePosixPath(remote_prefix.lstrip("/"))
    for entry in volume.listdir(remote_prefix, recursive=True):
        if getattr(entry.type, "name", "") != "FILE":
            continue
        remote_path = PurePosixPath(entry.path.lstrip("/"))
        try:
            relative = remote_path.relative_to(prefix)
        except ValueError as error:
            raise ValueError(f"Modal returned path outside {remote_prefix}: {entry.path}") from error
        local_path = destination / Path(*relative.parts)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("wb") as handle:
            for chunk in volume.read_file(entry.path):
                handle.write(chunk)
        count += 1
    return count


TERMINAL_JOB_STATES = {"completed", "failed", "refused"}


def read_volume_json(volume: Any, path: str) -> dict[str, Any] | None:
    """A JSON file from a volume, or None until the job's first commit writes it."""
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except Exception:  # noqa: BLE001 - absent, or not yet committed
        return None


def find_job_run(volume: Any, job_id: str, known: dict[str, str | None]) -> str | None:
    """The run directory whose record names ``job_id``.

    The trainer picks the run ID, so the wrapper finds it by reading each new
    run's ``config.execution.job_id``. ``known`` caches decided directories; a
    run whose record is not committed yet is looked at again next time.
    """
    try:
        entries = volume.listdir("output/training")
    except Exception:  # noqa: BLE001 - the directory appears with the first run
        return None
    for entry in entries:
        name = PurePosixPath(entry.path).name
        if name in known:
            continue
        run = read_volume_json(volume, f"output/training/{name}/run.json")
        if run is not None:
            known[name] = run.get("config", {}).get("execution", {}).get("job_id")
    return next((name for name, job in known.items() if job == job_id), None)


def live_sync_once(volume: Any, job_id: str, state: dict[str, Any], destination: Path) -> dict[str, Any]:
    """One polling pass: the job's status and its run record, never checkpoints."""
    job = read_volume_json(volume, f"output/modal/jobs/{job_id}.json")
    run_dir = state.get("run_dir") or find_job_run(volume, job_id, state.setdefault("known", {}))
    state["run_dir"] = run_dir
    info = {"status": (job or {}).get("status", "waiting"), "run_dir": run_dir,
            "epoch": None, "val_loss": None}
    if run_dir:
        with tempfile.TemporaryDirectory(prefix="kicks-modal-live-") as temporary:
            staged = Path(temporary)
            download_volume_tree(volume, f"output/training/{run_dir}", staged / run_dir)
            merge_downloaded_tree(staged, destination, preserve_notes=True)
        try:
            history = json.loads((destination / run_dir / "run.json").read_text()).get("history") or []
        except (OSError, ValueError):
            history = []
        if history:
            info.update(epoch=history[-1].get("epoch"), val_loss=history[-1].get("val_loss"))
    return info


def command_check_local(args: argparse.Namespace) -> int:
    result = verify_corpus_tree(ROOT / "data", hash_files=not args.sizes_only)
    for manifest in sorted((ROOT / "data/_subsets").glob("*/manifest.json")):
        instrument = manifest.parent.name.split("-", 1)[0]
        if instrument in CORPORA:
            load_subset(manifest, instrument)
    receipt = {
        **result,
        "scope": "local",
        "source_sha256": source_digest(),
        "subset_manifests": [
            str(path.relative_to(ROOT)) for path in sorted((ROOT / "data/_subsets").glob("*/manifest.json"))
        ],
    }
    atomic_json(LOCAL_STATE / "local-corpus.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def command_upload(args: argparse.Namespace) -> int:
    require_cloud_confirmation(args)
    # Refuse to copy a locally altered corpus. The mapping is the only durable
    # source/license identity after the numeric rename.
    verify_corpus_tree(ROOT / "data", hash_files=True)
    volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=True)
    corpus = args.corpus
    with volume.batch_upload(force=True) as batch:
        for path in sorted((ROOT / "data" / corpus).glob("*.wav")):
            batch.put_file(str(path), f"/{corpus}/{path.name}")
    print(json.dumps({
        "uploaded_at": utc_now(), "volume": CORPUS_VOLUME_NAME,
        "corpus": corpus, "wav_files": EXPECTED_COUNTS[corpus], "sources": False,
    }, indent=2))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    require_cloud_confirmation(args)
    local = verify_corpus_tree(ROOT / "data", hash_files=True)
    remote = None
    with modal.enable_output(), verify_app.run():
        remote = verify_uploaded_corpus.remote(local["mapping_sha256"])
    remote = completed(remote, "the verification")
    receipt = {
        "verified_at": utc_now(), "volume": CORPUS_VOLUME_NAME,
        "mapping_sha256": local["mapping_sha256"], "local": local, "remote": remote,
    }
    atomic_json(RECEIPT, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def command_image_test(args: argparse.Namespace) -> int:
    require_cloud_confirmation(args)
    result = None
    with modal.enable_output(), image_app.run():
        result = test_training_image.remote()
    result = {"tested_at": utc_now(), **completed(result, "the image test")}
    atomic_json(LOCAL_STATE / "image-test.json", result)
    print(json.dumps(result, indent=2))
    return 0


def validate_receipt(spec: dict[str, Any]) -> dict[str, Any]:
    if not RECEIPT.is_file():
        raise FileNotFoundError(f"run the reviewed `verify` step first; missing {RECEIPT}")
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("mapping_sha256") != spec["mapping_sha256"]:
        raise ValueError("local corpus mapping changed since the uploaded corpus was verified")
    current = verify_corpus_tree(ROOT / "data", hash_files=True)
    if current["mapping_sha256"] != receipt["mapping_sha256"]:
        raise ValueError("local corpus changed since verification")
    return receipt


def command_plan(args: argparse.Namespace) -> int:
    spec = build_spec(args)
    print(json.dumps({
        "cloud_action": "detached GPU training",
        "corpus_volume": CORPUS_VOLUME_NAME,
        "results_volume": RESULTS_VOLUME_NAME,
        "resources": {
            "gpu": spec["gpu"], "memory_mib": spec["memory_mib"], "cpu": spec["cpu"],
            "timeout_hours": 24, "commit_interval_seconds": spec["commit_interval_seconds"],
            "single_use_container": True,
        },
        "spec": spec,
        "remote_command": remote_training_command(spec, remote_data_dir(spec)),
    }, indent=2))
    return 0


def command_launch(args: argparse.Namespace) -> int:
    require_cloud_confirmation(args)
    spec = build_spec(args)
    receipt = validate_receipt(spec)
    spec["corpus_verification"] = {
        "verified_at": receipt["verified_at"],
        "mapping_sha256": receipt["mapping_sha256"],
    }
    call_id = None
    with modal.enable_output(), train_app.run(name=f"kicks-{spec['job_id']}", detach=True):
        call = train_remote.with_options(
            gpu=spec["gpu"], memory=spec["memory_mib"], cpu=spec["cpu"],
        ).spawn(spec)
        call_id = call.object_id
    call_id = completed(call_id, "a function call ID (check `modal app list`)")
    launch = {**spec, "function_call_id": call_id, "launched_at": utc_now()}
    atomic_json(LAUNCHES / f"{spec['job_id']}.json", launch)
    print(json.dumps(launch, indent=2))
    return 0


def _launch_records(job_id: str | None, all_jobs: bool) -> list[dict[str, Any]]:
    if job_id:
        path = LAUNCHES / f"{job_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return [json.loads(path.read_text())]
    paths = sorted(LAUNCHES.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no launch records under {LAUNCHES}")
    chosen = paths if all_jobs else [paths[-1]]
    return [json.loads(path.read_text()) for path in chosen]


def command_sync(args: argparse.Namespace) -> int:
    require_cloud_confirmation(args)
    launches = _launch_records(args.job_id, args.all)
    volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
    with tempfile.TemporaryDirectory(prefix="kicks-modal-sync-") as temporary:
        staged = Path(temporary)
        runs = staged / "runs"
        downloaded = download_volume_tree(volume, "output/training", runs)
        copied = merge_downloaded_tree(
            runs, ROOT / "output/training", preserve_notes=True,
        )
        checkpoint_count = 0
        for launch in launches:
            relative = safe_relative_path(launch["model_dir"], field="launch model_dir")
            remote_model = relative.as_posix()
            model_stage = staged / "models" / launch["job_id"]
            download_volume_tree(volume, remote_model, model_stage)
            for path in list(model_stage.rglob("*")):
                if path.is_file() and path.name not in SYNC_MODEL_FILES:
                    path.unlink()
            checkpoint_count += merge_downloaded_tree(
                model_stage, ROOT / relative, preserve_notes=False,
            )
        jobs_stage = staged / "jobs"
        download_volume_tree(volume, "output/modal/jobs", jobs_stage)
        jobs_count = merge_downloaded_tree(
            jobs_stage, LOCAL_STATE / "jobs", preserve_notes=False,
        )
    print(json.dumps({
        "remote_run_files": downloaded, "local_run_files_updated": copied,
        "checkpoint_files_updated": checkpoint_count, "job_files_updated": jobs_count,
        "preserved": "existing local notes.json and notes.js",
    }, indent=2))
    return 0


def command_live(args: argparse.Namespace) -> int:
    """Mirror one job's run record into the local dashboard until the job ends."""
    require_cloud_confirmation(args)
    if args.interval < 10:
        raise ValueError("--interval must be at least 10 seconds")
    volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
    state: dict[str, Any] = {}
    deadline = time.monotonic() + args.max_minutes * 60
    last_line = announced = None
    try:
        while True:
            info = live_sync_once(volume, args.job_id, state, ROOT / "output/training")
            if info["run_dir"] and info["run_dir"] != announced:
                announced = info["run_dir"]
                print(f"dashboard: http://127.0.0.1:6060/?run={announced}  (start it with `uv run kicks dashboard`)",
                      flush=True)
            line = f"job {args.job_id}: {info['status']}"
            if info["epoch"] is not None:
                line += f", epoch {info['epoch']}"
                if isinstance(info["val_loss"], (int, float)):
                    line += f", val {info['val_loss']:.4f}"
            if line != last_line:
                print(f"[{utc_now()}] {line}", flush=True)
                last_line = line
            if info["status"] in TERMINAL_JOB_STATES:
                break
            if time.monotonic() > deadline:
                print("watch limit reached; the job continues on Modal", flush=True)
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped watching; the job continues on Modal", flush=True)
        return 0
    if args.final_sync:
        return command_sync(argparse.Namespace(confirm_cloud=True, job_id=args.job_id, all=False))
    print(f"job ended; pull its checkpoints with `sync --confirm-cloud --job-id {args.job_id}`", flush=True)
    return 0


def add_cloud_confirmation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confirm-cloud", action="store_true",
        help="required after this individual cloud step has been reviewed",
    )


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instrument", choices=sorted(CORPORA), required=True)
    parser.add_argument("--gpu", required=True, help="Modal GPU, for example L4 or A100-40GB")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--success-criteria", required=True)
    parser.add_argument("--model-dir", required=True, help="repository-relative, under models/experiments/")
    parser.add_argument("--subset-manifest", help="local manifest rebuilt as symlinks on Modal")
    parser.add_argument("--resume", help="repository-relative checkpoint already in the results volume")
    parser.add_argument(
        "--job-id", required=True,
        help="unique identifier; pass the same one to plan and launch so the plan is exact",
    )
    parser.add_argument("--memory-gib", type=float, default=DEFAULT_MEMORY_GIB,
                        help="container memory reservation (billed even when unused)")
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU, help="reserved CPU cores")
    parser.add_argument("--commit-interval-seconds", type=int, default=COMMIT_INTERVAL_SECONDS,
                        help="seconds between results-volume commits while training (30-3600)")
    parser.add_argument(
        "training_args", nargs=argparse.REMAINDER,
        help="additional diffusion-train flags after --; wrapper-owned flags are rejected",
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-local", help="hash the local corpus; no cloud access")
    check.add_argument("--sizes-only", action="store_true", help="skip WAV hashes (not valid for launch)")
    check.set_defaults(handler=command_check_local)
    upload = commands.add_parser("upload", help="upload or repair one reviewed corpus directory")
    add_cloud_confirmation(upload)
    upload.add_argument("--corpus", choices=sorted(EXPECTED_COUNTS), required=True)
    upload.set_defaults(handler=command_upload)
    verify = commands.add_parser("verify", help="hash the uploaded corpus on a CPU container")
    add_cloud_confirmation(verify)
    verify.set_defaults(handler=command_verify)
    image = commands.add_parser("image-test", help="build/test the locked training image on CPU")
    add_cloud_confirmation(image)
    image.set_defaults(handler=command_image_test)
    plan = commands.add_parser("plan", help="print an exact launch plan; no cloud access")
    add_training_arguments(plan)
    plan.set_defaults(handler=command_plan)
    launch = commands.add_parser("launch", help="launch reviewed training, detached")
    add_cloud_confirmation(launch)
    add_training_arguments(launch)
    launch.set_defaults(handler=command_launch)
    sync = commands.add_parser("sync", help="pull run records and checkpoints")
    add_cloud_confirmation(sync)
    sync.add_argument("--job-id")
    sync.add_argument("--all", action="store_true", help="sync model artifacts for every recorded launch")
    sync.set_defaults(handler=command_sync)
    live = commands.add_parser(
        "live", help="mirror a running job's run record into the local dashboard (read-only, no checkpoints)",
    )
    add_cloud_confirmation(live)
    live.add_argument("--job-id", required=True)
    live.add_argument("--interval", type=int, default=30, help="seconds between polls (at least 10)")
    live.add_argument("--max-minutes", type=int, default=25 * 60, help="stop watching after this long")
    live.add_argument("--final-sync", action="store_true", help="pull the checkpoints once the job ends")
    live.set_defaults(handler=command_live)
    return root


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if modal is None and args.command not in {"check-local", "plan"}:
            raise RuntimeError("run this as a standalone uv script so modal==1.5.5 is available")
        return args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
