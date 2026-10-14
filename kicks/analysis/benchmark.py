"""Backend-neutral audio audits and content-addressed corpus split manifests.

Generation metrics are reference-free. A paired spectral error is only computed
when a caller explicitly supplies the source waveform being reconstructed.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np
import scipy.signal as signal
import soundfile as sf

from ..audio.constants import AUDIO_LENGTH, SAMPLE_RATE
from ..audio.waveform import bandpass, load_audio
from ..instruments import available, get_profile


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(payload)
    temporary.replace(path)


def _source(old: str, corpus: str) -> dict:
    name = old.lower()
    # Conservative, inspectable rules derived from the local source records.
    # Unknown files share one group, so unidentified related hits cannot leak.
    group, family, license_ = f"unresolved:{corpus}", "unknown", "unresolved"
    if name.startswith('808_'):
        group, family, license_ = 'tr808-fischer', 'electronic', 'CC0-1.0'
    elif name.startswith('mck_'):
        group, family, license_ = 'mck:' + '_'.join(name.split('_')[1:2]), 'electronic', 'CC0-1.0'
    elif name.startswith('spi_'):
        group, family, license_ = 'sonic-pi', 'electronic', 'CC0-1.0'
    elif name.startswith('hat_'):
        group, family, license_ = 'karoryfer-hat', 'acoustic', 'CC0-1.0'
    elif corpus == 'snares' and re.match(r'^\d+[a-z]+_', name):
        group, family, license_ = 'karoryfer-frankensnare', 'acoustic', 'CC0-1.0'
    articulation = 'unknown'
    if corpus == 'hihats':
        for token, label in [('xtight', 'closed'), ('closed', 'closed'), ('ch_', 'closed'),
                             ('open', 'open'), ('oh_', 'open'), ('chik', 'pedal'), ('pedal', 'pedal')]:
            if token in name:
                articulation = label
                break
    return {"group": group, "family": family, "license": license_, "articulation": articulation,
            "label_origin": "filename inference; review before a release benchmark"}


def freeze_corpus(data_root='data', out='output/benchmarks/corpus.json', *, registry='data/corpus_ids.json',
                  groups=None, seed=42, validation=.15, test=.15):
    if validation <= 0 or test <= 0 or validation + test >= 1:
        raise ValueError('validation and test fractions must be positive and sum to less than one')
    root = Path(data_root).resolve()
    registry_path = Path(registry)
    entries = json.loads(registry_path.read_text()).get('entries', []) if registry_path.exists() else []
    lookup = {(r['corpus'], r['new']): r for r in entries}
    overrides = json.loads(Path(groups).read_text()) if groups else {}
    rows, sources = [], []
    seen_override = set()
    for instrument in available():
        profile = get_profile(instrument)
        corpus = profile.paths.corpus_name
        directory = root / corpus
        source_file = directory / 'SOURCES.md'
        if source_file.exists():
            sources.append({"path": str(source_file), "sha256": file_hash(source_file)})
        for path in sorted(directory.glob('*.wav')):
            key = f'{corpus}/{path.name}'
            registered = lookup.get((corpus, path.name), {})
            digest = file_hash(path)
            if registered.get('sha256') and registered['sha256'] != digest:
                raise ValueError(f'corpus bytes changed: {key}')
            labels = _source(registered.get('old', path.name), corpus)
            if key in overrides:
                if not isinstance(overrides[key], dict) or set(overrides[key]) - {'group', 'family', 'license', 'articulation'}:
                    raise ValueError(f'invalid label override: {key}')
                labels.update(overrides[key])
                labels['label_origin'] = 'explicit override'
                seen_override.add(key)
            if not source_file.exists() and labels['label_origin'] != 'explicit override':
                labels['license'] = 'unresolved'
            if any(not isinstance(labels[k], str) or not labels[k] for k in ('group', 'family', 'license', 'articulation')):
                raise ValueError(f'invalid labels: {key}')
            info = sf.info(path)
            rows.append({"id": key, "path": str(path), "instrument": instrument, "sha256": digest,
                         "size": path.stat().st_size, "original_name": registered.get('old'),
                         "sample_rate": info.samplerate, "channels": info.channels, "frames": info.frames,
                         "duration_seconds": info.duration, "longer_than_window": info.duration > AUDIO_LENGTH / SAMPLE_RATE,
                         **labels})
    if not rows:
        raise ValueError('no corpus WAV files found')
    if set(overrides) - seen_override:
        raise ValueError('label overrides contain unknown corpus files')
    # Union exact duplicates with source groups, including copies across instruments.
    parent = {r['group']: r['group'] for r in rows}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    duplicate_groups = {}
    for row in rows:
        old = duplicate_groups.setdefault(row['sha256'], row['group'])
        a, b = sorted((find(old), find(row['group'])))
        parent[b] = a
    unresolved_roots = {find(r['group']) for r in rows if r['group'].startswith('unresolved:')}
    split_groups = {}
    for row in rows:
        group = find(row['group'])
        value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
        split_groups[group] = 'test' if value < test else 'validation' if value < test + validation else 'train'
        if group in unresolved_roots:
            split_groups[group] = 'train'
        row['split_group'] = group
        row['split'] = split_groups[group]
    # With few packs a hash split can accidentally put every known source in
    # training. Repair coverage by moving whole, resolved groups; never split
    # duplicates or manufacture a held-out fold from unidentified sources.
    members = {group: {r['instrument'] for r in rows if r['split_group'] == group}
               for group in split_groups}
    for split in ('test', 'validation'):
        missing = {r['instrument'] for r in rows} - {
            r['instrument'] for r in rows if split_groups[r['split_group']] == split}
        while missing:
            candidates = [g for g in split_groups if split_groups[g] == 'train'
                          and g not in unresolved_roots and members[g] & missing
                          and all(any(other != g and split_groups[other] == 'train' and inst in members[other]
                                      for other in members) for inst in members[g])]
            if not candidates:
                break
            chosen = min(candidates, key=lambda g: (-len(members[g] & missing),
                         hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest()))
            split_groups[chosen] = split
            missing -= members[chosen]
    for row in rows:
        row['split'] = split_groups[row['split_group']]
    warnings = []
    for instrument in available():
        for split in ('train', 'validation', 'test'):
            if not any(r['instrument'] == instrument and r['split'] == split for r in rows):
                warnings.append(f'{instrument}: no {split} source groups; cannot claim a complete held-out benchmark')
    spec = {"version": 1, "seed": seed, "fractions": {"validation": validation, "test": test},
            "preprocessing": {"sample_rate": SAMPLE_RATE, "length": AUDIO_LENGTH,
                              "normalization": "analysis_peak; original recordings retained"},
            "sources": sources, "rows": rows,
            "limitations": ['Exact duplicates are grouped; near-duplicate detection still requires review.',
                            'Source-level splits can leave a family or instrument absent from a fold.',
                            'Fractions are targets; coverage repair moves whole resolved source groups.',
                            'Unresolved source groups remain training-only; review inferred labels.'],
            "warnings": warnings}
    document = {**spec, "sha256": identity(spec), "created_at": datetime.now(timezone.utc).isoformat()}
    if Path(out).exists():
        raise FileExistsError(f'refusing to replace frozen manifest {out}')
    write_json(Path(out), document)
    return document


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    spec = {k: v for k, v in document.items() if k not in {'sha256', 'created_at'}}
    if document.get('version') != 1 or identity(spec) != document.get('sha256'):
        raise ValueError('manifest content or version does not match its identity')
    return document


def measure(audio: np.ndarray, profile, reference: np.ndarray | None = None) -> dict:
    from .evaluation import analyze_hit
    from .fidelity import texture_metrics, compare_waveforms, level_match

    x = np.asarray(audio)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected a finite nonempty mono waveform')
    peak = float(np.max(np.abs(x)))
    result = {"peak": peak, "clipping": bool(peak >= 1), "silent": bool(peak < 1e-6),
              "metrics": analyze_hit(x.astype(np.float32), profile),
              "texture": texture_metrics(x, profile)}
    if reference is not None:
        ref, cand = level_match(reference, x)
        result['paired_reconstruction'] = compare_waveforms(ref, cand, profile)
    return result


def descriptors(audio, profile):
    # Lazy model-stack import: corpus hashing and waveform metrics need no Torch.
    import torch
    from ..audio.mel import spectrogram
    from .descriptors import compute_descriptors
    return compute_descriptors(spectrogram(torch.from_numpy(np.asarray(audio, dtype=np.float32))[None]), profile)


def run_dsp_benchmark(instrument='kick', out='output/fidelity/dsp-kick', *, settings=24, seeds=3,
                       texture_path=None, texture_mix=.25):
    from ..synthesis.procedural import Preset, RENDERER_VERSION

    if settings < 1 or seeds < 1:
        raise ValueError('settings and seeds must be positive')
    directory = Path(out)
    if directory.exists():
        raise FileExistsError(f'refusing to overwrite benchmark {out}')
    profile = get_profile(instrument)
    voice = profile.procedural
    directory.mkdir(parents=True)
    (directory / 'generation').mkdir()
    rng = np.random.default_rng(42)
    parameter_sets = [voice.resolve(preset=p) for p in voice.presets]
    while len(parameter_sets) < settings:
        parameter_sets.append(voice.from_positions(rng.uniform(.08, .92, len(voice.parameters)).tolist()))
    texture = None if not texture_path else {"path": str(Path(texture_path).resolve()),
        "sha256": file_hash(texture_path), "mix": texture_mix}
    # Warm up once; cold render is retained separately.
    start = time.perf_counter()
    Preset(profile.name, parameter_sets[0], texture=texture).audio()
    cold = (time.perf_counter() - start) * 1000
    rows = []
    for i, values in enumerate(parameter_sets[:settings]):
        for seed in range(seeds):
            preset = Preset(profile.name, values, seed, texture=texture)
            start = time.perf_counter()
            audio = preset.audio()
            elapsed = (time.perf_counter() - start) * 1000
            path = directory / 'generation' / f'case_{i:03d}_seed_{seed}.wav'
            sf.write(path, audio, SAMPLE_RATE, subtype='PCM_24')
            preset.save(path.with_suffix('.json'))
            rows.append({"case_id": f'{i}:{seed}', "file": path.name, "seed": seed,
                         "condition": values, "preset_sha256": preset.identity, "audio_sha256": file_hash(path),
                         "render_ms": elapsed, "descriptors": descriptors(audio, profile), **measure(audio, profile)})
    times = [r['render_ms'] for r in rows]
    summary = {"samples": 0, "held_out": False, "pairs": 0, "generated": len(rows),
               "cold_render_ms": cold, "warm_render_p50_ms": float(np.percentile(times, 50)),
               "warm_render_p95_ms": float(np.percentile(times, 95)),
               "silent": sum(r['silent'] for r in rows), "clipping": sum(r['clipping'] for r in rows)}
    report = {"instrument": profile.name, "backend": 'hybrid' if texture else 'dsp',
              "renderer_version": RENDERER_VERSION, "control_space": 'physical',
              "method": 'generation_integrity_v1; physical controls, no descriptor-target guarantee',
              "held_out": False, "hits": [], "samples": [], "summary": summary, "generation": rows,
              "texture": texture, "design_sha256": identity([r['condition'] for r in rows]),
              "created_at": datetime.now(timezone.utc).isoformat(), "listening_judged": False}
    write_json(directory / 'report.json', report)
    return report


def audit_directory(generated, instrument='kick', *, out, pattern='*.wav', backend='external',
                    targets=None, reference_directory=None):
    """Audit final WAVs from any backend under the same signal/metric contract.

    Targets are raw descriptor values keyed by filename, never inferred from
    unrelated audio. An explicitly supplied same-name reference directory opts
    into paired reconstruction analysis; ordinary generation has no such error.
    """
    import shutil

    profile = get_profile(instrument)
    inputs = sorted(Path(generated).glob(pattern))
    if not inputs or Path(out).exists():
        raise ValueError('input WAVs are required and the output directory must be new')
    target_rows = json.loads(Path(targets).read_text()) if targets else {}
    if targets and set(target_rows) != {p.name for p in inputs}:
        raise ValueError('descriptor targets must name every input file exactly once')
    for row in target_rows.values():
        if set(row) != set(profile.descriptor_keys) or not all(np.isfinite(v) for v in row.values()):
            raise ValueError('targets must contain every profile descriptor with finite raw values')
    output = Path(out)
    (output/'generation').mkdir(parents=True)
    rows = []
    for path in inputs:
        audio, sr = sf.read(path, dtype='float32')
        if sr != SAMPLE_RATE or audio.shape != (AUDIO_LENGTH,):
            raise ValueError(f'{path.name}: expected mono 44.1 kHz, {AUDIO_LENGTH} samples; no implicit conversion')
        ref = None
        ref_identity = None
        if reference_directory is not None:
            ref_path = Path(reference_directory)/path.name
            ref, ref_sr = sf.read(ref_path, dtype='float32')
            if ref_sr != SAMPLE_RATE or ref.shape != audio.shape:
                raise ValueError('paired reconstruction references must have the same signal contract')
            ref_identity = file_hash(ref_path)
        measured = descriptors(audio, profile)
        target = target_rows.get(path.name)
        row = {"case_id": path.name, "file": path.name, "audio_sha256": file_hash(path),
               "condition": target, "descriptors": measured, "reference_sha256": ref_identity,
               "target_error_raw": None if target is None else {k: measured[k] - target[k] for k in target},
               **measure(audio, profile, reference=ref)}
        rows.append(row)
        shutil.copyfile(path, output/'generation'/path.name)
    report = {"instrument": profile.name, "backend": backend,
              "method": 'paired_reconstruction_audit_v1' if reference_directory else 'generation_audio_audit_v1',
              "control_space": 'descriptor' if targets else 'unspecified', "held_out": False,
              "samples": [], "hits": [], "generation": rows,
              "design_sha256": identity([r['condition'] for r in rows]) if targets else None,
              "summary": {"samples": 0, "pairs": 0, "held_out": False, "generated": len(rows),
                          "silent": sum(r['silent'] for r in rows), "clipping": sum(r['clipping'] for r in rows)},
              "listening_judged": False}
    write_json(output/'report.json', report)
    return report


def listening_comparison(left, right, out, *, seed=42):
    """Compare identical physical-condition designs, without reconstruction claims."""
    from .fidelity import level_match, _write_listening_page

    left, right, out = Path(left), Path(right), Path(out)
    a, b = json.loads((left/'report.json').read_text()), json.loads((right/'report.json').read_text())
    if a['instrument'] != b['instrument'] or not a.get('design_sha256') or a['design_sha256'] != b.get('design_sha256'):
        raise ValueError('comparison needs the same instrument and condition design')
    if out.exists():
        raise FileExistsError(f'refusing to overwrite comparison {out}')
    if len(a['generation']) != len(b['generation']):
        raise ValueError('comparison sets must have the same number of cases')
    (out/'listening').mkdir(parents=True)
    rng = random.Random(seed)
    pairs = []
    for i, (x, y) in enumerate(zip(a['generation'], b['generation']), 1):
        if x['case_id'] != y['case_id'] or x['condition'] != y['condition']:
            raise ValueError('comparison case identities differ')
        waves = []
        for directory, row in ((left, x), (right, y)):
            path = directory/'generation'/row['file']
            if path.parent.resolve() != (directory/'generation').resolve() or file_hash(path) != row['audio_sha256']:
                raise ValueError('comparison audio identity changed')
            audio, sr = sf.read(path, dtype='float32')
            if sr != SAMPLE_RATE:
                raise ValueError('comparison sample rates differ')
            waves.append(audio)
        matched = level_match(*waves)
        flip = rng.choice([False, True])
        names = ['left', 'right'][::(-1 if flip else 1)]
        chosen = dict(zip(['left', 'right'], matched))
        pair = {"pair": i, "a_is": names[0], "b_is": names[1]}
        for tag, side in zip(('a', 'b'), names):
            for suffix, wave in (('', chosen[side]), ('_hf', bandpass(chosen[side], (2000, None)))):
                name = f'pair_{i:02d}_{tag.upper()}{suffix}.wav'
                sf.write(out/'listening'/name, wave, SAMPLE_RATE, subtype='PCM_24')
                pair[tag + suffix] = name
        pairs.append(pair)
    write_json(out/'listening'/'key.json', pairs)
    _write_listening_page(out/'listening', pairs, generation=True)
    report = {"instrument": a['instrument'], "method": 'matched_condition_generation_ab_v1',
              "held_out": False, "samples": [], "generation": [], "hits": [],
              "left_report_sha256": file_hash(left/'report.json'), "right_report_sha256": file_hash(right/'report.json'),
              "comparison": {"left": {"directory": str(left.resolve()), "method": a.get('method')},
                             "right": {"directory": str(right.resolve()), "method": b.get('method')}},
              "summary": {"samples": 0, "held_out": False, "pairs": len(pairs)}, "listening_judged": False}
    write_json(out/'report.json', report)
    return report
