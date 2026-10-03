#!/usr/bin/env python3
"""Run the leakage-free nested-CV workflow for one brain region.

Example:
    python scripts/run_nested_cv.py --region BA11
    python scripts/run_nested_cv.py --region BA47 --n-iter 4 --verbose

Writes JSON results (clearly labelled as new, non-paper results) to
``results/leakage_free/`` and prints a summary table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tod_pred.data import (
    DEFAULT_PROCESSED_DIR,
    REGION_FILES,
    feature_matrix,
    load_region,
)
from tod_pred.nested_cv import run_nested_cv

DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "leakage_free"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(region: str, config_path: Path | None) -> dict:
    if config_path is None:
        config_path = DEFAULT_CONFIG_DIR / f"nested_cv_{region.lower()}.yaml"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("region") != region:
        raise ValueError(
            f"Config {config_path} targets region {config.get('region')!r}, expected {region!r}"
        )
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(REGION_FILES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-outer", type=int, default=None, help="override config")
    parser.add_argument("--n-inner", type=int, default=None, help="override config")
    parser.add_argument("--n-iter", type=int, default=None, help="override config")
    parser.add_argument("--estimators", nargs="*", default=None, help="override config")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.region, args.config)
    for key, override in (
        ("n_outer_folds", args.n_outer),
        ("n_inner_folds", args.n_inner),
        ("n_iter", args.n_iter),
        ("estimators", args.estimators),
    ):
        if override is not None:
            config[key] = override

    frame = load_region(args.region)
    features, target = feature_matrix(frame)

    results = run_nested_cv(
        features,
        target,
        estimator_names=config.get("estimators"),
        n_outer=config.get("n_outer_folds", 5),
        n_inner=config.get("n_inner_folds", 3),
        n_iter=config.get("n_iter", 12),
        stratification_bins=config.get("n_bins", 4),
        pca_variance=config.get("pca_variance", 0.90),
        random_state=config.get("random_state", 42),
        verbose=args.verbose,
    )
    results["region"] = args.region
    results["data_file"] = str(
        (DEFAULT_PROCESSED_DIR / REGION_FILES[args.region]).relative_to(REPO_ROOT)
    )
    results["data_sha256"] = _sha256(DEFAULT_PROCESSED_DIR / REGION_FILES[args.region])
    results["configuration"] = config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.region.lower()}_nested_cv_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved results to {output_path}")
    print(f"Region: {args.region} (n={len(target)} donors) — {results['warning']}")
    header = f"{'model':<24}{'MAE (h)':>10}{'+/-':>4}{'RMSE (h)':>10}{'StdDev (h)':>12}"
    print("\n" + header)
    print("-" * len(header))
    baseline = results["baseline_mean_predictor"]["aggregated"]
    print(
        f"{'mean-baseline':<24}{baseline['mae_mean']:>10.3f}{'+/-':>4}"
        f"{baseline['rmse_mean']:>10.3f}{baseline['stderr_mean']:>12.3f}"
    )
    for name, payload in results["models"].items():
        agg = payload["aggregated"]
        print(
            f"{name:<24}{agg['mae_mean']:>10.3f}{'+/-':>4}"
            f"{agg['rmse_mean']:>10.3f}{agg['stderr_mean']:>12.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
