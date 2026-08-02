"""Path configuration for the crypto trend-following project.

Raw archives live in a read-mostly mirror that can sit anywhere, since it is
several gigabytes and is usually shared between projects. Everything this
project produces stays inside the project directory.

    TREND_MIRROR_DIR    binance archive mirror   (default: ./data/mirror)
    TREND_DATA_DIR      derived panels           (default: ./data)
    TREND_RESULTS_DIR   tables, reports, figures (default: ./results)

All three are relative to the project root unless overridden, so a clone
runs without editing any path.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MIRROR_DIR = Path(os.environ.get("TREND_MIRROR_DIR", ROOT / "data" / "mirror"))
DATA_DIR = Path(os.environ.get("TREND_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("TREND_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"

KLINES_1H_DIR = MIRROR_DIR / "parquet_1h"
FUNDING_RAW_DIR = MIRROR_DIR / "data" / "futures" / "um" / "monthly" / "fundingRate"

PANEL_DIR = DATA_DIR / "panel"


def ensure_directories() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, PANEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
