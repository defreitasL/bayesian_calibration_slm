"""
Data preparation
================
Author: Lucas de Freitas Pereira

Builds the ``*_default.nc`` files that the calibration scripts read, from the
raw multi-transect IH-SET NetCDF files shipped in ``data/``.

This must be run **before** ``vitousek21_yates_Angourie.py`` and
``vitousek21_yates_LaJolla.py``.

Run
---
python examples/00_prepare_data.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Allow running this script without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slmcal.plotting.paper_figures import default_from_ihset

DATA_DIR = ROOT / "data"

SITE_FILES = {
    "Angourie Back Beach": DATA_DIR / "Angourie_CoastSat_NSWWaves.nc",
    "La Jolla Shores": DATA_DIR / "LaJolla_CoastSat_CMEMS.nc",
}


def main() -> None:
    missing = [str(p) for p in SITE_FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw input file(s):\n  " + "\n  ".join(missing)
        )

    written = default_from_ihset(SITE_FILES)

    print("[OK] Prepared datasets:")
    for label, path in written.items():
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  - {label}: {path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
