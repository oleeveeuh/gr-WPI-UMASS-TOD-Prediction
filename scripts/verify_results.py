#!/usr/bin/env python3
"""Verify and summarise the repository's authoritative result artifacts.

Two independent checks:

1. ``--check-paper`` — validate the transcribed paper constants (Table 2,
   hourly scale) for internal consistency (RMSE = sqrt(MSE), our method having
   the lowest hourly MAE per region).
2. Default — parse every workbook under ``results/sheets/`` into a tidy CSV
   (``results/derived/all_model_performance_tidy.csv``) and report the best
   configuration per option/window/region, exactly as the historical
   ``find_best_models.py`` did.  These metrics are on *transformed* targets
   and are documentation, not headline results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tod_pred.authoritative import (
    check_paper_internal_consistency,
    paper_table2_frame,
    parse_sheets_tree,
)

DEFAULT_SHEETS_DIR = REPO_ROOT / "results" / "sheets"
DEFAULT_TIDY_OUTPUT = REPO_ROOT / "results" / "derived" / "all_model_performance_tidy.csv"


def check_paper() -> int:
    problems = check_paper_internal_consistency()
    table = paper_table2_frame()
    columns = ["region", "model", "mae", "mse", "rmse", "stderr", "method"]
    print("Paper-reported hourly-scale results (Table 2, PDF p. 11 / proceedings p. 714):")
    print(table[columns].to_string(index=False))
    if problems:
        print("\nCONSISTENCY PROBLEMS FOUND:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nInternal consistency checks passed (RMSE=sqrt(MSE); ours lowest hourly MAE).")
    return 0


def parse_sheets(sheets_dir: Path, output_path: Path) -> int:
    tidy = parse_sheets_tree(sheets_dir)
    if tidy.empty:
        print(f"No performance workbooks found under {sheets_dir}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(output_path, index=False)
    best = tidy.sort_values("mae").groupby(["option", "window", "workbook"], dropna=False).head(1)
    print(f"Parsed {len(tidy)} configuration rows from results/sheets into {output_path}")
    print("\nBest row per option/window/region (metrics on TRANSFORMED targets — "
          "see docs/LIMITATIONS.md; not directly comparable across normalisation methods):")
    columns = ["option", "window", "workbook", "sheet", "model", "mae", "rmse"]
    print(best[columns].to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-paper", action="store_true",
                        help="only validate the transcribed paper constants")
    parser.add_argument("--sheets-dir", type=Path, default=DEFAULT_SHEETS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_TIDY_OUTPUT)
    args = parser.parse_args(argv)

    exit_code = 0
    if args.check_paper:
        exit_code = check_paper()
    else:
        exit_code = parse_sheets(args.sheets_dir, args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
