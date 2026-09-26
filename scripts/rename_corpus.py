"""Rename corpus hits to anonymous numbered names: ``#<n>.wav``.

Every hit gets one number, unique across all renamed corpora, so ``#14.wav``
names exactly one sound anywhere in the project. Numbers are issued corpus by
corpus in instrument-registry order (kicks, snares, hi-hats), sorted by the
current file name within each corpus. Files already called ``#<n>.wav`` keep
their number; hits added later continue from the highest number ever issued,
so a second run only numbers the newcomers.

The mapping lives in ``data/corpus_ids.json`` (untracked, like the rest of
``data/``): old name, new name, corpus, size and SHA-256 per hit. It is the
only per-hit provenance left once the names are gone: ``SOURCES.md`` attributes
sources, some CC BY 4.0, by the *old* name prefixes. It is written before the
first rename, so an interrupted run resumes with ``--apply`` or reverses with
``--undo``. A rename changes no bytes and no mtimes.

Artifacts that refer to hits by name are translated in the same pass:

* subset directories under ``data/_subsets`` (symlinks and ``manifest.json``);
* the cluster reports (``filename``, ``original_path``) that
  ``scripts/make_subset.py`` and ``scripts/diffusion_listening.py`` read.

Historical evidence is left as recorded: fidelity, calibration and listening
reports keep the old names in their ``hits`` lists — look them up in the
mapping. Split fingerprints cover file paths, so ``--run`` warns and falls
back for every run recorded before the rename, and fingerprinted caches (eval
references, calibration sidecars) rebuild on first use. The splits themselves
survive wherever the sort order does: snare and hi-hat numbers all have five
digits, so they sort exactly like the old names — the same seed gives the same
split, and new subsets nest with the old ones. Kick numbers span one to five
digits (``#1``, ``#10``, ``#100``…), so the kick corpus reorders.

Dry run by default; ``--apply`` renames, ``--undo`` reverses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kicks.audio.io import list_wavs
from kicks.instruments import available, get_profile

NUMBERED = re.compile(r"#(\d+)\.wav")
SCHEME = "#<n>.wav, global across corpora, registry order then sorted old name"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload, indent: int | None = 2) -> None:
    """Atomic replace, so a crash never leaves half a manifest or report."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=indent)
        handle.write("\n")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "scheme": SCHEME, "next": 1, "applied": False,
                "entries": [], "history": []}
    with open(path) as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise ValueError(f"{path}: unsupported manifest version {manifest.get('version')!r}")
    return manifest


def plan(corpora: list[tuple[str, Path]], manifest: dict) -> list[dict]:
    """New entries for every hit not yet numbered; existing entries are reused.

    A hit whose old name is already in the manifest (after an ``--undo``) keeps
    its number, provided its size still matches — otherwise the file changed
    under that name and numbering it again would silently merge two sounds.
    """
    by_old = {(e["corpus"], e["old"]): e for e in manifest["entries"]}
    live_numbers = [int(m.group(1)) for _, d in corpora for n in list_wavs(str(d))
                    if (m := NUMBERED.fullmatch(n))]
    next_id = max([manifest["next"], *(n + 1 for n in live_numbers)])
    planned = []
    for corpus, directory in corpora:
        for name in list_wavs(str(directory)):
            if NUMBERED.fullmatch(name):
                continue
            path = directory / name
            known = by_old.get((corpus, name))
            if known is not None:
                if path.stat().st_size != known["size"]:
                    raise ValueError(f"{path} changed size since it was numbered {known['new']}")
                continue
            planned.append({"id": next_id, "corpus": corpus, "old": name,
                            "new": f"#{next_id}.wav", "size": path.stat().st_size,
                            "sha256": sha256(path)})
            next_id += 1
    return planned


def move(entries: list[dict], dirs: dict[str, Path], forward: bool) -> int:
    """Make the filesystem match the entries; idempotent, refuses ambiguity."""
    moved = 0
    for entry in entries:
        if entry["corpus"] not in dirs:
            continue
        directory = dirs[entry["corpus"]]
        src, dst = (entry["old"], entry["new"]) if forward else (entry["new"], entry["old"])
        src_path, dst_path = directory / src, directory / dst
        if src_path.exists() and dst_path.exists():
            raise FileExistsError(f"both {src_path} and {dst_path} exist")
        if dst_path.exists():
            continue
        if not src_path.exists():
            raise FileNotFoundError(f"{src_path} is missing; the corpus changed since numbering")
        os.rename(src_path, dst_path)
        moved += 1
    return moved


def translate_subsets(root: Path, maps: dict[str, dict[str, str]]) -> list[dict]:
    """Re-point subset symlinks and manifests through ``maps[corpus][old] -> new``."""
    changed = []
    if not root.is_dir():
        return changed
    for directory in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        manifest_path = directory / "manifest.json"
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        corpus_dir = Path(manifest["corpus_dir"])
        mapping = maps.get(corpus_dir.name, {})
        count = 0
        for link in sorted(directory.glob("*.wav")):
            if not link.is_symlink() or link.name not in mapping:
                continue
            new = directory / mapping[link.name]
            if new.exists() or new.is_symlink():
                raise FileExistsError(f"{new} already exists")
            link.unlink()
            os.symlink(corpus_dir / mapping[link.name], new)
            count += 1
        files = [mapping.get(name, name) for name in manifest["files"]]
        if files != manifest["files"]:
            manifest["files"] = sorted(files)
            write_json(manifest_path, manifest)
        if count:
            changed.append({"path": str(directory), "links": count})
    return changed


def translate_report(path: Path, corpus: str, mapping: dict[str, str]) -> int:
    """Rename ``filename`` / ``original_path`` in one cluster report; returns rows changed."""
    if not path.is_file():
        return 0
    with open(path) as handle:
        report = json.load(handle)
    count = 0
    for row in report.get("samples", []):
        original = Path(row.get("original_path", ""))
        if original.parent.name == corpus and original.name in mapping:
            row["original_path"] = str(original.with_name(mapping[original.name]))
            count += 1
        if row.get("filename") in mapping:
            row["filename"] = mapping[row["filename"]]
    if count:
        write_json(path, report)
    return count


def translate_artifacts(data_root: Path, reports: dict[str, Path],
                        entries: list[dict], forward: bool) -> list[dict]:
    maps: dict[str, dict[str, str]] = {}
    for e in entries:
        old, new = (e["old"], e["new"]) if forward else (e["new"], e["old"])
        maps.setdefault(e["corpus"], {})[old] = new
    changed = translate_subsets(data_root / "_subsets", maps)
    for corpus, report in reports.items():
        rows = translate_report(report, corpus, maps.get(corpus, {}))
        if rows:
            changed.append({"path": str(report), "rows": rows})
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instrument", action="append", default=None,
                        help="limit to these instruments (repeatable); default: all registered")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data root>/corpus_ids.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="rename (default is a dry run)")
    action.add_argument("--undo", action="store_true", help="restore the old names")
    args = parser.parse_args(argv)

    profiles = [get_profile(name) for name in (args.instrument or available())]
    data_root = Path(profiles[0].paths.data_root)
    corpora = [(p.paths.corpus_name, Path(p.paths.data_dir)) for p in profiles]
    dirs = dict(corpora)
    reports = {p.paths.corpus_name: Path(p.paths.cluster_analysis) for p in profiles}
    manifest_path = args.manifest or data_root / "corpus_ids.json"
    manifest = load_manifest(manifest_path)

    def selected() -> list[dict]:
        return [e for e in manifest["entries"] if e["corpus"] in dirs]

    if args.undo:
        moved = move(selected(), dirs, forward=False)
        changed = translate_artifacts(data_root, reports, selected(), forward=False)
        manifest["applied"] = False
        manifest["history"].append({"at": now(), "action": "undo", "files": moved,
                                    "artifacts": changed})
        write_json(manifest_path, manifest)
        print(f"restored {moved} old names; translated {len(changed)} artifacts back")
        return 0

    planned = plan(corpora, manifest)
    pending = [e for e in selected() if (dirs[e["corpus"]] / e["old"]).exists()]
    for corpus, directory in corpora:
        count = sum(1 for e in planned if e["corpus"] == corpus)
        live = len(list_wavs(str(directory)))
        print(f"{corpus:>8}: {live} files, {count} to number"
              + (f" (#{min(e['id'] for e in planned if e['corpus'] == corpus)}"
                 f"–#{max(e['id'] for e in planned if e['corpus'] == corpus)})" if count else ""))
    for entry in planned[:3]:
        print(f"  {entry['corpus']}/{entry['old']} -> {entry['new']}")
    if pending:
        print(f"  plus {len(pending)} previously numbered files still under their old names")
    if not args.apply:
        print("dry run; pass --apply to rename")
        return 0
    if not planned and not pending:
        print("nothing to rename")
        return 0

    # Record the mapping before touching a file, so any interruption is recoverable.
    manifest["entries"].extend(planned)
    manifest["next"] = max([manifest["next"], *(e["id"] + 1 for e in planned)])
    write_json(manifest_path, manifest)
    moved = move(selected(), dirs, forward=True)
    changed = translate_artifacts(data_root, reports, selected(), forward=True)
    manifest["applied"] = True
    manifest["history"].append({"at": now(), "action": "apply", "numbered": len(planned),
                                "files": moved, "artifacts": changed})
    write_json(manifest_path, manifest)
    print(f"renamed {moved} files; mapping in {manifest_path}")
    for item in changed:
        print(f"  updated {item['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
