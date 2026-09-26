"""Local training records and dashboard server; no tensor or web dependencies."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import socket
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

NOTE_FIELDS = ("objective", "hypothesis", "success_criteria", "observations", "decision")
TEMPLATE = Path(__file__).with_name("dashboard.html")

#: Fidelity report subdirectories the dashboard may stream audio from.
FIDELITY_AUDIO_DIRS = ("listening", "reconstruction", "generation")
REPORT_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
AUDIO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wav$")
VERDICT_CHOICES = ("a", "b", "tie", "reject")
VERDICT_BANDS = ("full", "hf")
MAX_AUDIO_BYTES = 256 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_root(value=None, output_root="output") -> Path:
    return Path(value or os.environ.get("KICKS_RUNS_DIR")
                or Path(os.environ.get("KICKS_OUTPUT_DIR", output_root)) / "training")


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def atomic_write(path: Path, content: str):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_notes(directory: Path) -> dict:
    try:
        return json.loads((directory / "notes.json").read_text())
    except (OSError, ValueError):
        return {key: "" for key in NOTE_FIELDS}


def write_notes(directory: Path, notes: dict):
    atomic_write(directory / "notes.json", json.dumps(notes, indent=2))
    encoded = json.dumps(notes).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    atomic_write(directory / "notes.js", f"window.receiveTrainingNotes({encoded});\n")


def read_reports(directory: Path) -> list:
    try:
        return json.loads((directory / "reports.json").read_text())
    except (OSError, ValueError):
        return []


def _write_sibling_script(directory: Path, name: str, function: str, payload) -> None:
    encoded = json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    atomic_write(directory / f"{name}.js", f"window.{function}({encoded});\n")


def attach_report(directory: Path, kind: str, path, summary: dict, note: str = "") -> dict:
    """Record a piece of evidence (fidelity report, control audit, promotion) against a run.

    Evidence is appended, never rewritten: the run keeps the history of what was
    measured. ``summary`` is the handful of numbers the dashboard shows inline;
    the full report stays at ``path``. Like notes, reports live in their own file
    so the training writer's ``run.json`` is never touched.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", kind):
        raise ValueError(f"report kind must be a short lowercase slug, got {kind!r}")
    if not (directory / "run.json").is_file():
        raise FileNotFoundError(f"{directory} is not a training run")
    entry = {"kind": kind, "created_at": now(), "path": str(Path(path).resolve()) if path else None,
             "summary": _clean(summary), "note": note}
    reports = [*read_reports(directory), entry]
    atomic_write(directory / "reports.json", json.dumps(reports, indent=2))
    _write_sibling_script(directory, "reports", "receiveTrainingReports", reports)
    return entry


def fidelity_dirs(root: Path) -> dict:
    """Discover fidelity report directories: ``slug -> dir`` for the listening lab.

    Two sources, both explicit: the conventional ``<output>/fidelity/*/report.json``
    next to the runs root, and every ``fidelity`` report attached to a run (even one
    rendered to a custom ``--out``). Missing directories are skipped; a directory only
    qualifies when its ``report.json`` parses and names an instrument.
    """
    root = Path(root).resolve()
    found: dict[str, Path] = {}

    def add(directory: Path):
        try:
            directory = directory.resolve()
            report = json.loads((directory / "report.json").read_text())
            if not isinstance(report, dict) or not report.get("instrument"):
                return
        except (OSError, ValueError):
            return
        slug = directory.name
        if not REPORT_SLUG.fullmatch(slug):
            return
        if slug in found and found[slug] != directory:
            slug = f"{directory.name}-{hashlib.sha256(str(directory).encode()).hexdigest()[:6]}"
        found[slug] = directory

    sibling = root.parent / "fidelity"
    if sibling.is_dir():
        for child in sorted(sibling.iterdir()):
            if child.is_dir():
                add(child)
    if root.is_dir():
        for run_dir in sorted(root.iterdir()):
            if not (run_dir / "run.json").is_file():
                continue
            for entry in read_reports(run_dir):
                if entry.get("kind") != "fidelity" or not entry.get("path"):
                    continue
                path = Path(entry["path"])
                add(path.parent if path.name == "report.json" else path)
    return found


def read_fidelity(directory: Path) -> dict:
    """Report metadata for the listening lab; audio stays on disk until requested."""
    report = json.loads((directory / "report.json").read_text())
    return {"slug": directory.name, "directory": str(directory), "report": report}


def read_verdicts(directory: Path) -> dict:
    try:
        verdicts = json.loads((directory / "listening" / "verdicts.json").read_text())
        return verdicts if isinstance(verdicts, dict) else {}
    except (OSError, ValueError):
        return {}


def write_verdicts(directory: Path, verdicts: dict):
    atomic_write(directory / "listening" / "verdicts.json", json.dumps(verdicts, indent=2))


def fidelity_summary(report: dict, directory: Path) -> dict:
    """Compact row for the report picker; no corpus filenames, no blind key."""
    summary = report.get("summary") or {}
    key_path = directory / "listening" / "key.json"
    try:
        pairs = len(json.loads(key_path.read_text())) if key_path.is_file() else 0
    except (OSError, ValueError):
        pairs = 0
    verdicts = read_verdicts(directory)
    judged = sum(1 for entry in verdicts.values()
                 if isinstance(entry, dict) and any(entry.get(band) for band in VERDICT_BANDS))
    return {
        "slug": directory.name,
        "instrument": report.get("instrument"),
        "created_at": report.get("created_at"),
        "run_id": report.get("run_id"),
        "held_out": report.get("held_out"),
        "hit_origin": report.get("hit_origin"),
        "vocoder": report.get("vocoder"),
        "checkpoint": Path(report["checkpoint"]).name if report.get("checkpoint") else None,
        "epoch": report.get("epoch"),
        "val_loss": report.get("val_loss"),
        "pairs": pairs,
        "generations": len(report.get("generation") or []),
        "reconstructions": len([s for s in report.get("samples") or [] if isinstance(s, dict)]),
        "judged": judged,
        "summary": {key: summary.get(key) for key in
                    ("samples", "mel_mae_db", "vae_8_16k_body_mae_db", "vae_2_8k_body_mae_db",
                     "vocoder_8_16k_body_mae_db", "vae_penalty_8_16k_body_db",
                     "vae_onset_error_ms", "vae_envelope_mae_db", "vae_late_excess_db",
                     "generation_mean_score", "generation_pass_rate", "seconds")
                    if summary.get(key) is not None},
    }


def find_run(root: Path, run_id: str) -> Path:
    """Resolve a run ID (or unique prefix) to its directory under ``root``."""
    root = Path(root)
    exact = root / run_id
    if (exact / "run.json").is_file():
        return exact
    matches = [d for d in root.iterdir() if d.name.startswith(run_id) and (d / "run.json").is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no training run matching {run_id!r} under {root}")
    raise ValueError(f"{run_id!r} matches several runs: {', '.join(sorted(d.name for d in matches))}")


def read_run(directory: Path) -> dict:
    record = json.loads((directory / "run.json").read_text())
    record["notes"] = read_notes(directory)
    record["reports"] = read_reports(directory)
    return record


class TrainingRun:
    """One writer owns metrics; notes live separately so browser edits survive."""

    def __init__(self, root: Path, config: dict, *, name=None, notes=None):
        root.mkdir(parents=True, exist_ok=True)
        instrument = config["instrument"]
        self.id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{instrument}-{uuid.uuid4().hex[:8]}"
        self.directory = root / self.id
        self.directory.mkdir()
        self.started = time.monotonic()
        self.last_write = 0.0
        self.record = {
            "schema_version": 1, "id": self.id, "name": name or f"{instrument.title()} experiment",
            "status": "running", "phase": "preparing", "started_at": now(), "updated_at": now(),
            "pid": os.getpid(), "hostname": socket.gethostname(), "config": config,
            "progress": {"epoch": 0, "batch": 0, "batches": 0}, "history": [],
            "best": {}, "error": None,
        }
        write_notes(self.directory, {key: (notes or {}).get(key, "") for key in NOTE_FIELDS})
        atomic_write(self.directory / "index.html", TEMPLATE.read_text())
        self.flush()

    def flush(self):
        self.record["updated_at"] = now()
        self.record["elapsed_seconds"] = time.monotonic() - self.started
        data = _clean(self.record)
        atomic_write(self.directory / "run.json", json.dumps(data, indent=2, allow_nan=False))
        data["notes"] = read_notes(self.directory)
        data["reports"] = read_reports(self.directory)
        # A local HTML file can load a sibling script without file:// fetch/CORS.
        encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        atomic_write(self.directory / "data.js", f"window.receiveTrainingRun({encoded});\n")
        self.last_write = time.monotonic()

    def progress(self, *, phase=None, force=False, **values):
        if phase:
            self.record["phase"] = phase
        self.record["progress"].update(values)
        if force or time.monotonic() - self.last_write >= 2:
            self.flush()

    def epoch(self, values: dict):
        self.record["history"].append(values)
        for metric in ("val_loss", "hf_mae_db", "air_mae_db", "eval_proxy"):
            value = values.get(metric)
            if value is not None and math.isfinite(value):
                previous = self.record["best"].get(metric)
                if previous is None or value < previous["value"]:
                    self.record["best"][metric] = {"value": value, "epoch": values["epoch"]}
        self.flush()

    def finish(self, status="completed", error=None):
        self.record.update(status=status, phase=status, error=error, ended_at=now())
        self.flush()


def split_fingerprint(dataset, seed, val_split):
    """Identity of the data and its split: seed, ratio and every file's path, size and mtime."""
    digest = hashlib.sha256(repr((seed, val_split, len(dataset))).encode())
    paths = getattr(dataset, "paths", None)
    if paths is None:
        return None  # An arbitrary Dataset has no reliable content identity.
    for path in paths:
        stat = os.stat(path)
        digest.update(repr((str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns)).encode())
    return digest.hexdigest()


def track_training(function):
    """Give every invocation a durable lifecycle, including interruption/failure."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        profile, model = p["profile"], p["model"]
        dataset = p["dloader"].dataset
        config = {key: p[key] for key in ("epochs", "beta", "free_bits", "beta_anneal_epochs",
                                         "beta_cycles", "val_split", "eval_every", "seed", "source_checkpoint",
                                         "hf_detail_weight", "attack_change_weight", "beta_floor")}
        config.update(
            instrument=profile.name, device=str(p["device"] or next(model.parameters()).device),
            latent_dim=model.latent_dim, n_mels=model.n_mels, n_frames=model.n_frames,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            batch_size=p["dloader"].batch_size, corpus_samples=len(dataset),
            learning_rate=p["optimizer"].param_groups[0]["lr"],
            optimizer=type(p["optimizer"]).__name__,
            transient_weight=p["transient_weight"] if p["transient_weight"] is not None else profile.transient_loss.weight,
            transient_windows=vars(profile.transient_loss),
            architecture=getattr(model, "architecture", None),
            checkpoint=str(Path(profile.paths.checkpoint).resolve()),
            validation_objective="posterior_mean_fixed_beta_v1",
            detail_metric="active_log_mel_db_v1", preprocessing="peak_safe_lufs_v1",
            split_fingerprint=split_fingerprint(dataset, p["seed"], p["val_split"]),
        )
        run = p["tracker"] or TrainingRun(
            runs_root(p["runs_dir"], profile.paths.output_root), config,
            name=p["run_name"], notes={"objective": p["intent"], "hypothesis": p["hypothesis"],
                                      "success_criteria": p["success_criteria"]},
        )
        bound.arguments["tracker"] = run
        print(f"Training dashboard: http://127.0.0.1:6060/?run={run.id}", flush=True)
        print(f"  Start viewer with: kicks dashboard --runs-dir {shlex.quote(str(run.directory.parent.resolve()))}", flush=True)
        print(f"  Standalone HTML: {run.directory.resolve() / 'index.html'}", flush=True)
        try:
            result = function(*bound.args, **bound.kwargs)
        except KeyboardInterrupt:
            run.finish("interrupted", "Training interrupted; completed epochs are preserved.")
            raise
        except BaseException as exc:
            run.finish("failed", f"{type(exc).__name__}: {exc}")
            raise
        else:
            run.finish()
            return result
    return wrapped


def validation_trend(history: list, points: int = 48) -> list:
    """Validation loss for the run list's sparkline, evenly thinned to ``points`` values.

    The first and last recorded values are always kept, so the sparkline starts at
    the baseline and ends at the latest epoch.
    """
    values = [row["val_loss"] for row in history
              if isinstance(row.get("val_loss"), (int, float)) and math.isfinite(row["val_loss"])]
    if len(values) <= points:
        return values
    step = (len(values) - 1) / (points - 1)
    return [values[round(i * step)] for i in range(points)]


def dashboard_server(root: Path, port=6060) -> ThreadingHTTPServer:
    """Serve records and prose edits on loopback; never expose arbitrary files."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def reports_dir(slug: str):
        """Resolve a listening-lab slug to a discovered, qualified report directory."""
        if not REPORT_SLUG.fullmatch(slug):
            raise ValueError("Invalid report id")
        found = fidelity_dirs(root)
        if slug not in found:
            raise FileNotFoundError(slug)
        return found[slug]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data, content_type="application/json"):
            body = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def route(self):
            return unquote(urlsplit(self.path).path).strip("/").split("/")

        def directory(self, name):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("Invalid run ID")
            path = (root / name).resolve()
            if path.parent != root or not (path / "run.json").is_file():
                raise FileNotFoundError(name)
            return path

        def do_GET(self):
            parts = self.route()
            try:
                if parts == [""] or parts == ["index.html"]:
                    return self.reply(200, TEMPLATE.read_text(), "text/html")
                if parts == ["api", "capabilities"]:
                    return self.reply(200, {"viewer": 3, "listening": True, "instruments": True,
                                            "trend": True})
                if parts == ["api", "fidelity"]:
                    rows = [fidelity_summary(read_fidelity(directory)["report"], directory)
                            for directory in fidelity_dirs(root).values()]
                    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
                    return self.reply(200, {"reports": rows})
                if len(parts) == 4 and parts[:2] == ["api", "fidelity"] and parts[3] == "key":
                    directory = reports_dir(parts[2])
                    key = json.loads((directory / "listening" / "key.json").read_text())
                    return self.reply(200, {"key": key})
                if len(parts) == 3 and parts[:2] == ["api", "fidelity"]:
                    directory = reports_dir(parts[2])
                    data = read_fidelity(directory)
                    report = data["report"]
                    key_path = directory / "listening" / "key.json"
                    try:
                        pairs = json.loads(key_path.read_text()) if key_path.is_file() else []
                    except (OSError, ValueError):
                        pairs = []
                    for pair in pairs:  # Blind until the listener reveals: strip the assignment.
                        pair.pop("a_is", None)
                        pair.pop("b_is", None)
                    return self.reply(200, {
                        "slug": data["slug"], "instrument": report.get("instrument"),
                        "created_at": report.get("created_at"), "run_id": report.get("run_id"),
                        "held_out": report.get("held_out"), "hit_origin": report.get("hit_origin"),
                        "vocoder": report.get("vocoder"), "epoch": report.get("epoch"),
                        "val_loss": report.get("val_loss"), "method": report.get("method"),
                        "windows": report.get("windows"), "summary": report.get("summary"),
                        "samples": report.get("samples"), "generation": report.get("generation"),
                        "pairs": pairs, "verdicts": read_verdicts(directory),
                    })
                if len(parts) == 5 and parts[:2] == ["api", "fidelity"] and parts[3] == "audio":
                    return self.audio(parts[2], parts[4])
                if parts == ["api", "runs"]:
                    records = []
                    for directory in root.iterdir():
                        try:
                            row = read_run(self.directory(directory.name))
                            row["trend"] = validation_trend(row.pop("history", None) or [])
                            records.append(row)
                        except (OSError, ValueError):
                            continue
                    return self.reply(200, {"runs": sorted(records, key=lambda r: r["started_at"], reverse=True)})
                if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                    return self.reply(200, read_run(self.directory(parts[2])))
                self.reply(404, {"error": "Not found"})
            except (OSError, ValueError):
                self.reply(404, {"error": "Run not found"})

        def audio(self, slug: str, name: str):
            """Stream one allow-listed WAV from a known report's audio directories."""
            if not AUDIO_NAME.fullmatch(name):
                return self.reply(404, {"error": "Not found"})
            try:
                directory = reports_dir(slug)
            except (OSError, ValueError):
                return self.reply(404, {"error": "Report not found"})
            for sub in FIDELITY_AUDIO_DIRS:
                candidate = (directory / sub / name).resolve()
                try:
                    candidate.relative_to((directory / sub).resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    size = candidate.stat().st_size
                    if size > MAX_AUDIO_BYTES:
                        return self.reply(413, {"error": "Audio too large"})
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(size))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    with candidate.open("rb") as handle:
                        while chunk := handle.read(1 << 20):
                            self.wfile.write(chunk)
                    return
            self.reply(404, {"error": "Audio not found"})

        def do_POST(self):
            parts = self.route()
            if len(parts) == 4 and parts[:2] == ["api", "fidelity"] and parts[3] == "verdicts":
                return self.verdicts(parts[2])
            if len(parts) != 4 or parts[:2] != ["api", "runs"] or parts[3] != "notes":
                return self.reply(404, {"error": "Not found"})
            if self.headers.get_content_type() != "application/json":
                return self.reply(415, {"error": "Use application/json"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536:
                    return self.reply(413, {"error": "Notes must be at most 64 KB"})
                directory = self.directory(parts[2])
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict) or any(k not in NOTE_FIELDS or not isinstance(v, str)
                                                       for k, v in values.items()):
                    raise ValueError("Expected prose fields containing text")
                notes = {**read_notes(directory), **values}
                write_notes(directory, notes)
                self.reply(200, {"notes": notes})
            except FileNotFoundError:
                self.reply(404, {"error": "Run not found"})
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

        def verdicts(self, slug: str):
            """Merge blind-listening verdicts into the report's ``verdicts.json``."""
            if self.headers.get_content_type() != "application/json":
                return self.reply(415, {"error": "Use application/json"})
            try:
                directory = reports_dir(slug)
            except (OSError, ValueError):
                return self.reply(404, {"error": "Report not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536:
                    return self.reply(413, {"error": "Verdicts must be at most 64 KB"})
                values = json.loads(self.rfile.read(length))
                if not isinstance(values, dict) or "pair" not in values or "band" not in values:
                    raise ValueError("Expected {pair, band, choice}")
                if values["band"] not in VERDICT_BANDS:
                    raise ValueError(f"band must be one of {VERDICT_BANDS}")
                choice = values.get("choice")
                if choice is not None and choice not in VERDICT_CHOICES:
                    raise ValueError(f"choice must be one of {VERDICT_CHOICES} or null")
                pair = str(int(values["pair"]))
                verdicts = read_verdicts(directory)
                entry = verdicts.setdefault(pair, {})
                if choice is None:
                    entry.pop(values["band"], None)
                    if not entry:
                        verdicts.pop(pair, None)
                else:
                    entry[values["band"]] = {"choice": choice, "at": now()}
                write_verdicts(directory, verdicts)
                self.reply(200, {"verdicts": verdicts})
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
