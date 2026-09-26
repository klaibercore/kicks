"""`scripts/rename_corpus.py`: numbered corpus names with a reversible mapping.

The script never decodes audio, so the corpora here are small byte files; the
roots point at ``tmp_path`` through the same environment overrides the CLI uses.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rename_corpus.py"
spec = importlib.util.spec_from_file_location("rename_corpus", SCRIPT)
rename_corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rename_corpus)

CORPORA = {"kicks": ["b.wav", "a.wav"], "snares": ["s1.wav"], "hihats": ["h2.wav", "h1.wav", "h3.wav"]}
REPORTS = {"kicks": "cluster_analysis.json", "snares": "snare/cluster_analysis.json",
           "hihats": "hihat/cluster_analysis.json"}


@pytest.fixture
def roots(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "output"
    for corpus, names in CORPORA.items():
        (data / corpus).mkdir(parents=True)
        for name in names:
            (data / corpus / name).write_bytes(f"{corpus}/{name}".encode())
        report = output / REPORTS[corpus]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"samples": [
            {"filename": n, "original_path": f"data/{corpus}/{n}", "cluster": 0} for n in sorted(names)
        ]}))
    subset = data / "_subsets" / "hihat-2"
    subset.mkdir(parents=True)
    for name in ("h1.wav", "h3.wav"):
        os.symlink((data / "hihats" / name).resolve(), subset / name)
    (subset / "manifest.json").write_text(json.dumps({
        "size": 2, "corpus_dir": str((data / "hihats").resolve()), "files": ["h1.wav", "h3.wav"],
    }))
    monkeypatch.setenv("KICKS_DATA_DIR", str(data))
    monkeypatch.setenv("KICKS_OUTPUT_DIR", str(output))
    return data, output


def listing(data):
    return {c: sorted(p.name for p in (data / c).iterdir()) for c in CORPORA}


def test_dry_run_changes_nothing(roots):
    data, _ = roots
    before = listing(data)
    assert rename_corpus.main([]) == 0
    assert listing(data) == before
    assert not (data / "corpus_ids.json").exists()


def test_numbers_are_global_in_registry_then_name_order(roots):
    data, output = roots
    assert rename_corpus.main(["--apply"]) == 0
    assert listing(data) == {"kicks": ["#1.wav", "#2.wav"], "snares": ["#3.wav"],
                             "hihats": ["#4.wav", "#5.wav", "#6.wav"]}
    # Bytes travel with the name: sorted old names get ascending numbers.
    assert (data / "kicks" / "#1.wav").read_bytes() == b"kicks/a.wav"
    assert (data / "hihats" / "#6.wav").read_bytes() == b"hihats/h3.wav"

    manifest = json.loads((data / "corpus_ids.json").read_text())
    assert manifest["applied"] and manifest["next"] == 7
    entry = next(e for e in manifest["entries"] if e["new"] == "#5.wav")
    assert entry == {**entry, "corpus": "hihats", "old": "h2.wav", "size": len(b"hihats/h2.wav")}
    assert len(entry["sha256"]) == 64

    subset = data / "_subsets" / "hihat-2"
    assert sorted(p.name for p in subset.glob("*.wav")) == ["#4.wav", "#6.wav"]
    assert (subset / "#6.wav").read_bytes() == b"hihats/h3.wav"
    assert json.loads((subset / "manifest.json").read_text())["files"] == ["#4.wav", "#6.wav"]

    rows = json.loads((output / REPORTS["hihats"]).read_text())["samples"]
    assert [(r["filename"], r["original_path"]) for r in rows] == [
        ("#4.wav", "data/hihats/#4.wav"), ("#5.wav", "data/hihats/#5.wav"),
        ("#6.wav", "data/hihats/#6.wav")]


def test_rerun_numbers_only_newcomers(roots):
    data, _ = roots
    rename_corpus.main(["--apply"])
    rename_corpus.main(["--apply"])
    assert listing(data)["snares"] == ["#3.wav"]
    (data / "snares" / "new.wav").write_bytes(b"new")
    rename_corpus.main(["--apply"])
    assert listing(data)["snares"] == ["#3.wav", "#7.wav"]


def test_undo_restores_names_and_artifacts_and_reapply_keeps_numbers(roots):
    data, output = roots
    before = listing(data)
    report_before = (output / REPORTS["kicks"]).read_text()
    rename_corpus.main(["--apply"])
    assert rename_corpus.main(["--undo"]) == 0
    assert listing(data) == before
    assert json.loads((output / REPORTS["kicks"]).read_text()) == json.loads(report_before)
    subset = data / "_subsets" / "hihat-2"
    assert sorted(p.name for p in subset.glob("*.wav")) == ["h1.wav", "h3.wav"]
    assert json.loads((subset / "manifest.json").read_text())["files"] == ["h1.wav", "h3.wav"]

    rename_corpus.main(["--apply"])
    assert (data / "kicks" / "#1.wav").read_bytes() == b"kicks/a.wav"
    assert len(json.loads((data / "corpus_ids.json").read_text())["entries"]) == 6


def test_refuses_a_changed_file_under_a_known_name(roots):
    data, _ = roots
    rename_corpus.main(["--apply"])
    rename_corpus.main(["--undo"])
    (data / "kicks" / "a.wav").write_bytes(b"a different, longer sound")
    with pytest.raises(ValueError, match="changed size"):
        rename_corpus.main(["--apply"])


def test_instrument_filter_leaves_other_corpora_alone(roots):
    data, _ = roots
    rename_corpus.main(["--apply", "--instrument", "snare"])
    assert listing(data)["snares"] == ["#1.wav"]
    assert listing(data)["hihats"] == ["h1.wav", "h2.wav", "h3.wav"]
    assert (data / "_subsets" / "hihat-2" / "h1.wav").is_symlink()
