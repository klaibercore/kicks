#!/usr/bin/env python3
"""Download drum one-shots from the smpldsnds/drum-abuse GitHub packs.

Fetches only the snare / hi-hat categories (via each repo's instruments/*.json
index), dedupes by content hash against the existing corpus and within the
download set, and stages results under data/_staging/{snare,hihat}/abuse/.

Staging keeps the raw downloads out of the corpus until they are validated;
integration into data/<corpus> is a separate step.

Usage:
    uv run python scripts/fetch_drum_abuse.py [snare|hihat|all]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data" / "_staging"

VOLUMES = [1, 2, 3, 4, 5]
RAW = "https://raw.githubusercontent.com/smpldsnds/drum-abuse-vol{vol}/main"

# instrument-category json -> corpus folder / kept prefix
TARGETS = {
    "snare": ["snare", "rim"],
    "hihat": ["hihat-closed", "hihat-open"],
    "kick": ["kick"],
}

UA = {"User-Agent": "kicks-corpus-fetch (local research corpus)"}


def fetch(url: str, binary: bool = False):
    parts = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, urllib.parse.quote(parts.path), "", "")
    )
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read() if binary else r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                import time

                time.sleep(5 * (attempt + 1))
                continue
            raise


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def corpus_hashes(corpus_dir: Path) -> set[str]:
    hashes = set()
    for p in corpus_dir.glob("*.wav"):
        hashes.add(hashlib.sha1(p.read_bytes()).hexdigest())
    return hashes


def load_index(vol: int, category: str) -> list[dict]:
    try:
        return json.loads(fetch(f"{RAW.format(vol=vol)}/instruments/{category}.json"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    targets = TARGETS if which == "all" else {which: TARGETS[which]}

    for corpus, categories in targets.items():
        corpus_dir = ROOT / "data" / f"{corpus}s"
        seen = corpus_hashes(corpus_dir)
        out_dir = STAGING / f"{corpus}s" / "abuse"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in out_dir.glob("*.wav"):
            seen.add(hashlib.sha1(p.read_bytes()).hexdigest())

        jobs: list[tuple[str, str, str]] = []  # (url, basename, machine)
        for vol in VOLUMES:
            for cat in categories:
                for item in load_index(vol, cat):
                    url = f"{RAW.format(vol=vol)}/samples/{item['url_path']}/{item['file']}"
                    base = f"abuse{vol}_{slug(item['machine_id'])}_{slug(Path(item['file']).stem)}.wav"
                    jobs.append((url, base, item["machine_id"]))
        # de-dupe identical target names
        uniq: dict[str, str] = {}
        for url, base, _ in jobs:
            uniq.setdefault(base, url)
        print(f"{corpus}: {len(uniq)} candidate samples across vols {VOLUMES}")

        def grab(item: tuple[str, str]):
            base, url = item
            try:
                data = fetch(url, binary=True)
            except Exception as e:  # noqa: BLE001 - skip broken links, count them
                return base, None, str(e)
            return base, data, None

        written = dupes = failed = 0
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(grab, item) for item in uniq.items()]
            for i, fut in enumerate(as_completed(futures), 1):
                base, data, err = fut.result()
                if err or not data or not data[:4] == b"RIFF":
                    failed += 1
                    continue
                h = hashlib.sha1(data).hexdigest()
                if h in seen:
                    dupes += 1
                    continue
                seen.add(h)
                (out_dir / base).write_bytes(data)
                written += 1
                if i % 500 == 0:
                    print(f"  {corpus}: {i}/{len(uniq)} checked, {written} new")
        print(f"{corpus}: wrote {written} new files to {out_dir} "
              f"(dupes {dupes}, failed {failed})")


if __name__ == "__main__":
    main()
