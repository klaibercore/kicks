"""`scripts/make_subset.py`: stratified, seeded, nested corpus subsets.

The script samples file *names* from a cluster report; it never reads audio,
so the corpus here is a directory of empty .wav files and a hand-made report.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_subset.py"
spec = importlib.util.spec_from_file_location("make_subset", SCRIPT)
make_subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_subset)

FAMILY_SIZES = {0: 4, 1: 8, 2: 12, 3: 16}          # 40 files, shares 10/20/30/40 %


def fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, list[dict]]:
    corpus = tmp_path / "hihats"
    corpus.mkdir()
    samples = []
    for family, size in FAMILY_SIZES.items():
        for index in range(size):
            name = f"f{family}_{index:02d}.wav"
            (corpus / name).write_bytes(b"")
            samples.append({"original_path": f"data/hihats/{name}", "cluster": family})
    report = tmp_path / "cluster_analysis.json"
    report.write_text(json.dumps({"samples": samples}))
    return corpus, report, samples


def families_of(out_dir: pathlib.Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for path in out_dir.glob("*.wav"):
        family = int(path.name[1])
        counts[family] = counts.get(family, 0) + 1
    return counts


def test_subset_is_proportional_seeded_and_made_of_resolving_symlinks(tmp_path):
    corpus, report, samples = fixture(tmp_path)
    out = tmp_path / "sub-10"
    manifest = make_subset.build_subset(samples, 10, 42, out, corpus)

    assert families_of(out) == {0: 1, 1: 2, 2: 3, 3: 4}
    assert manifest["size"] == 10 and manifest["seed"] == 42
    assert [f["subset"] for f in manifest["families"]] == [1, 2, 3, 4]
    for path in out.glob("*.wav"):
        assert path.is_symlink() and path.resolve().parent == corpus.resolve()
    assert json.loads((out / "manifest.json").read_text())["files"] == manifest["files"]

    # Same seed, same files; a different seed draws differently.
    again = make_subset.build_subset(samples, 10, 42, tmp_path / "sub-10b", corpus)
    assert again["files"] == manifest["files"]
    other = make_subset.build_subset(samples, 10, 7, tmp_path / "sub-10c", corpus)
    assert other["files"] != manifest["files"]


def test_subsets_with_the_same_seed_nest(tmp_path):
    corpus, report, samples = fixture(tmp_path)
    ladder = [
        set(make_subset.build_subset(samples, size, 42, tmp_path / f"sub-{size}", corpus)["files"])
        for size in (5, 10, 20, 40)
    ]
    for smaller, larger in zip(ladder, ladder[1:]):
        assert smaller < larger
    assert len(ladder[-1]) == 40


def test_a_stale_report_is_refused_unless_forced(tmp_path, monkeypatch, capsys):
    corpus, report, samples = fixture(tmp_path)
    (corpus / "newcomer.wav").write_bytes(b"")           # another session moved a file in

    with pytest.raises(ValueError, match="rerun `kicks cluster`"):
        make_subset.check_current(samples, corpus)

    # Point the hi-hat profile's corpus at the fixture so main() sees the drift.
    from dataclasses import replace
    real = make_subset.get_profile("hihat")
    monkeypatch.setattr(
        make_subset, "get_profile",
        lambda name=None: replace(real, paths=replace(real.paths, data_root=str(tmp_path))),
    )
    argv = ["--instrument", "hihat", "--size", "10", "--seed", "1",
            "--report", str(report), "--out", str(tmp_path / "sub")]
    assert make_subset.main(argv) == 2
    assert "rerun `kicks cluster`" in capsys.readouterr().err
    assert not (tmp_path / "sub").exists()

    assert make_subset.main(argv + ["--force"]) == 0
    captured = capsys.readouterr()
    assert "stale report" in captured.err and "10 of 40 files" in captured.out
    manifest = json.loads((tmp_path / "sub" / "manifest.json").read_text())
    assert "newcomer.wav" not in manifest["files"] and len(manifest["files"]) == 10


def test_an_occupied_output_directory_is_refused_unless_forced(tmp_path):
    corpus, report, samples = fixture(tmp_path)
    out = tmp_path / "sub"
    make_subset.build_subset(samples, 4, 42, out, corpus)
    with pytest.raises(FileExistsError, match="--force"):
        make_subset.build_subset(samples, 4, 42, out, corpus)
    rebuilt = make_subset.build_subset(samples, 6, 42, out, corpus, force=True)
    assert len(list(out.glob("*.wav"))) == 6 and rebuilt["size"] == 6
    with pytest.raises(ValueError, match="between 1 and 40"):
        make_subset.build_subset(samples, 41, 42, tmp_path / "too-big", corpus)
