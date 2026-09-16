"""Stage 5: adopt a candidate checkpoint, on the record.

Promotion is a copy plus a paper trail. The copy is the easy part; the point of
this module is that it refuses to happen without a written decision and the
evidence the plan asks for (a waveform fidelity report and a slider-control
audit attached to the run), and that it leaves behind the checkpoint's identity
so a served model can always be traced back to the experiment that produced it.

Standard library only, apart from a lazy ``torch.load`` that reads the
candidate's metadata to confirm it belongs to the instrument being promoted.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..instruments import InstrumentProfile
from .tracking import attach_report, find_run, read_notes, read_run, runs_root, write_notes

#: Evidence kinds a promotion needs attached to its run.
REQUIRED_EVIDENCE = ("fidelity", "controls")


class PromotionRefused(RuntimeError):
    """The candidate cannot be promoted as things stand; the message says why."""


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(
    profile: InstrumentProfile,
    candidate: str,
    run_id: str,
    decision: str = "",
    runs_dir: str | None = None,
    allow_missing_evidence: bool = False,
) -> dict:
    """Copy ``candidate`` over the profile's served checkpoint and record why.

    The previous checkpoint is kept as ``<name>_prev.pth`` beside it (the
    convention already used by hand in ``models/``). A ``.controls.npz``
    calibration sidecar next to the candidate travels with it; a stale one at
    the destination is left for the server to detect and refit.
    """
    import torch

    source = Path(candidate)
    if not source.is_file():
        raise PromotionRefused(f"candidate checkpoint not found: {source}")
    run_dir = find_run(runs_root(runs_dir, profile.paths.output_root), run_id)
    run = read_run(run_dir)
    notes = read_notes(run_dir)

    decision = (decision or notes.get("decision", "")).strip()
    if not decision:
        raise PromotionRefused(
            "no decision written: pass --decision or fill in 'Decision & next action' in the dashboard first")
    if run["config"].get("instrument") != profile.name:
        raise PromotionRefused(f"run {run['id']} trained a {run['config'].get('instrument')}, not a {profile.name}")

    kinds = {report["kind"] for report in run.get("reports", [])}
    missing = [kind for kind in REQUIRED_EVIDENCE if kind not in kinds]
    if missing and not allow_missing_evidence:
        raise PromotionRefused(
            f"run {run['id']} has no {' or '.join(missing)} report attached; run `kicks fidelity --run` and "
            f"`scripts/validate_controls.py --run` first, or pass --allow-missing-evidence and say so in the decision")

    meta = torch.load(source, map_location="cpu", weights_only=True)
    if meta.get("instrument") not in (None, profile.name):
        raise PromotionRefused(f"{source} was trained for {meta.get('instrument')}, not {profile.name}")

    destination = Path(profile.paths.checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    if destination.exists():
        previous = destination.with_name(f"{destination.stem}_prev{destination.suffix}")
        shutil.copy2(destination, previous)
    shutil.copy2(source, destination)
    sidecar = source.with_name(f"{source.stem}.controls.npz")
    sidecar_copied = False
    if sidecar.is_file():
        shutil.copy2(sidecar, destination.with_name(f"{destination.stem}.controls.npz"))
        sidecar_copied = True

    sha = checkpoint_sha256(destination)
    summary = {"checkpoint_sha256": sha, "epoch": meta.get("epoch"), "val_loss": meta.get("val_loss"),
               "source": str(source.resolve()), "destination": str(destination.resolve()),
               "previous": str(previous.resolve()) if previous else None,
               "calibration_sidecar": sidecar_copied, "missing_evidence": missing}
    attach_report(run_dir, "promotion", destination, summary, note=decision)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    record = (f"\n\n[{stamp}] Promoted {source} → {destination} (sha256 {sha[:12]}, epoch {meta.get('epoch')})."
              + (f" Previous kept at {previous}." if previous else "")
              + (f" Missing evidence: {', '.join(missing)}." if missing else ""))
    write_notes(run_dir, {**notes, "decision": decision + record})
    return summary
