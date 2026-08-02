#!/usr/bin/env python3
"""Enumerate and download the complete monthly fundingRate archive history.

The existing local mirror only holds 2026-01..2026-06. A trend strategy on
perpetuals holds directional exposure for weeks, so funding is a first-order
cost over the whole sample, not a 2026 footnote. This script lists every
monthly fundingRate zip that Binance has ever published for the USDT-margined
universe and downloads the missing ones into the shared mirror.

Idempotent: existing non-empty files are skipped.

Usage:
    python scripts/fetch_funding.py [--workers 32]
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from paths import MIRROR_DIR, ensure_directories  # noqa: E402

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BASE = "https://data.binance.vision/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
PREFIX = "data/futures/um/monthly/fundingRate/"


def list_keys(prefix: str, delimiter: str | None = None) -> list[str]:
    out: list[str] = []
    marker = None
    while True:
        query = {"prefix": prefix}
        if delimiter:
            query["delimiter"] = delimiter
        if marker:
            query["marker"] = marker
        base = urllib.parse.urlsplit(BUCKET)
        url = urllib.parse.urlunsplit(
            (base.scheme, base.netloc, base.path,
             urllib.parse.urlencode(query), ""))
        xml = None
        for attempt in range(5):
            try:
                xml = urllib.request.urlopen(url, timeout=30).read()
                break
            except Exception:
                if attempt == 4:
                    raise
        root = ET.fromstring(xml)
        if delimiter:
            page = [p.find(NS + "Prefix").text
                    for p in root.findall(NS + "CommonPrefixes")]
        else:
            page = [c.find(NS + "Key").text for c in root.findall(NS + "Contents")]
        out += page
        if root.find(NS + "IsTruncated").text == "true":
            nm = root.find(NS + "NextMarker")
            marker = nm.text if nm is not None else (page[-1] if page else None)
        else:
            return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    ensure_directories()

    symbols = sorted(p[len(PREFIX):].strip("/") for p in list_keys(PREFIX, delimiter="/"))
    symbols = [s for s in symbols if s.endswith("USDT")]
    print(f"{len(symbols)} symbols with published funding archives", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        per_symbol = list(ex.map(lambda s: list_keys(PREFIX + s + "/"), symbols))
    keys = [k for ks in per_symbol for k in ks if k.endswith(".zip")]
    print(f"{len(keys)} monthly archives listed", flush=True)

    manifest = MIRROR_DIR / "manifest_funding.txt"
    manifest.write_text("".join(k + "\n" for k in keys))

    failed: list[tuple[str, str]] = []

    def fetch(key: str) -> None:
        dest = MIRROR_DIR / key
        if dest.exists() and dest.stat().st_size > 0:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        err = None
        for _ in range(4):
            try:
                with urllib.request.urlopen(BASE + urllib.parse.quote(key), timeout=60) as r:
                    tmp.write_bytes(r.read())
                tmp.replace(dest)
                return
            except Exception as exc:  # noqa: BLE001 - logged, not raised
                err = exc
        failed.append((key, str(err)))

    with ThreadPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(fetch, keys), 1):
            if i % 2000 == 0:
                print(f"{i}/{len(keys)} processed, {len(failed)} failed", flush=True)

    print(f"finished: {len(keys)} archives, {len(failed)} failed", flush=True)
    (MIRROR_DIR / "failed_funding.txt").write_text(
        "".join(f"{k}\t{e}\n" for k, e in failed))


if __name__ == "__main__":
    main()
