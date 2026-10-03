#!/usr/bin/env python3
"""Download and verify the public source data for this project.

The repository intentionally tracks only small, high-value source files.  The
large re-downloadable artifacts are fetched from their public sources and
verified against checksums recorded in ``docs/DATA.md``:

* GEO series GSE71620 (SOFT metadata incl. per-donor phenotype characteristics;
  optional supplementary CEL archive),
* the GPL11532 platform annotation (upstream of the tracked-out
  ``gene_names`` snapshot),
* local snapshot integrity checks for everything already in ``data/raw/``.

Everything is written to ``data/raw/upstream/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
UPSTREAM_DIR = RAW_DIR / "upstream"

GSE = "GSE71620"
PLATFORM = "GPL11532"
SERIES_SOFT_URL = (
    f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GSE}&targ=self&form=text&view=full"
)
PLATFORM_SOFT_URL = (
    f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={PLATFORM}&targ=self&form=text&view=full"
)
SUPPL_URL = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={GSE}"

# SHA-256 of the snapshots as last tracked in this repository (see docs/DATA.md).
LOCAL_SNAPSHOT_HASHES = {
    "GSE71620_Phenotype_GEO.csv": "a711c7428fcbd028230df3d16888c036ae22e04a1bfde20a6b2bb058fbe37e1e",
    "GSE71620_Phenotype_GEO.xlsx": "e89c02cf13780eaa31c228f242c2770bb1772d55f1946440c5f560ee3066ef74",
    "pnas.1508249112.sd01.csv": "def5b8571cab3f99a446098c2e11713cd227538db5381a86c9d437c4d6ef4211",
    "cause_of_death.csv": "ecfe7a6228435398d03440da2de0936fc2dfc79597713d373b7f660679309a07",
    # Large annotation snapshots no longer tracked (re-derivable upstream = GPL11532 table).
    "gene_names.csv": "40f63d7958c415f519266b3df368f11cccf689ba294d80767443b3096e3bc2dc",
    "gene_names.xlsx": "0df305426a67fea189ba5f34612bb3955cbb488f4841919547558a2e02199db8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}\n  -> {destination}")
    urllib.request.urlretrieve(url, destination)
    return destination


def verify_local_snapshots() -> int:
    print("Verifying checksums of local source snapshots in data/raw/ ...")
    failures = 0
    for name, expected in LOCAL_SNAPSHOT_HASHES.items():
        path = RAW_DIR / name
        if not path.exists():
            print(f"  [absent ] {name} (expected only if tracked; see docs/DATA.md)")
            continue
        actual = _sha256(path)
        status = "ok     " if actual == expected else "CHANGED"
        print(f"  [{status}] {name}")
        if actual != expected:
            failures += 1
    return failures


def fetch_gpl_annotation() -> Path:
    """Fetch the GPL11532 annotation table (upstream of the gene_names snapshot)."""
    target = UPSTREAM_DIR / f"{PLATFORM}_platform.soft"
    fetch(PLATFORM_SOFT_URL, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-expression", action="store_true",
        help=f"also download the {GSE} supplementary archive (several hundred MB)",
    )
    parser.add_argument(
        "--skip-annotation", action="store_true",
        help="skip the (large) platform annotation download",
    )
    args = parser.parse_args(argv)

    failures = verify_local_snapshots()

    print(f"\nFetching {GSE} series SOFT metadata (provenance for the phenotype table)...")
    fetch(SERIES_SOFT_URL, UPSTREAM_DIR / f"{GSE}_series.soft")

    if not args.skip_annotation:
        print(f"\nFetching {PLATFORM} platform annotation (upstream of gene_names snapshot)...")
        annotation = fetch_gpl_annotation()
        print(f"  Platform SOFT saved ({annotation.stat().st_size / 1e6:.1f} MB).")

    if args.with_expression:
        print(f"\nFetching {GSE} supplementary archive (this can take a while)...")
        fetch(SUPPL_URL, UPSTREAM_DIR / f"{GSE}_suppl.tar")

    print("\nDone. Downloads live in data/raw/upstream/ (git-ignored).")
    print("NOTE: data/raw/upstream/ holds public source data; do not commit it.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
