"""Step 0/1: validate the inherited splits and build the modelled-space cache.

    python scripts/build_data.py
    python scripts/build_data.py --set data.n_hvg=5000
    python scripts/build_data.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import io, preprocess, splits


def _print_report(report: dict, config: dict) -> None:
    print(f"  reference folds are clean          : {report['reference_ok']}")
    print(f"  combinations derived from them     : {report['combinations_derived']}")
    print(f"  folds                              : {report['n_folds']}")
    print(f"  additive (train, test) per fold    : {report['additive_sizes']}")
    print(f"  combinations (doubles, held-out)   : {report['combinations_sizes']}")
    lo, hi = report["additive_test_pairwise_overlap"]
    print(f"  pairwise test overlap across folds : {lo}-{hi} conditions "
          f"(non-zero confirms independent draws, not a partition)")
    print(f"  doubles that are train in ALL folds: {report['doubles_train_in_every_fold']}")
    excluded = splits.held_out_conditions(config)
    print(f"  held out by ANY fold of ANY method : {len(excluded)} conditions "
          f"(exclusion set for leak-free work)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="config overrides in dot notation, e.g. data.n_hvg=5000")
    parser.add_argument("--validate-only", action="store_true",
                        help="check the splits and exit without touching the matrix")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the cache even if it already exists")
    parser.add_argument("--skip-validation", action="store_true",
                        help="the caller already validated; do not print it twice")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    raw_path = config["data"]["raw_h5ad"]
    cache_path = config["data"]["cache_h5ad"]

    # ---------------------------------------------------------------- splits
    if args.skip_validation and not args.validate_only:
        report = None
    else:
        print("=== 1. split validation ===")
        report = splits.validate(config)
    if report is not None:
        _print_report(report, config)

    if args.validate_only:
        return

    if os.path.exists(cache_path) and not args.force:
        print(f"\n{cache_path} already exists - pass --force to rebuild.")
        return

    # ---------------------------------------------------------------- genes
    print("\n=== 2. gene statistics (streaming pass over the raw matrix) ===")
    n_obs, n_var = io.shape(raw_path)
    print(f"  source: {raw_path}  ({n_obs:,} cells x {n_var:,} genes)")
    var_names = io.read_var_names(raw_path)
    conditions = io.read_obs_column(raw_path, "condition")

    started = time.time()
    mean, variance = preprocess.gene_statistics(raw_path, config["data"]["chunk_size"])
    print(f"  done in {time.time() - started:.1f}s")

    gene_indices, stats = preprocess.select_genes(config, var_names, conditions, mean, variance)
    print(f"\n=== 3. gene selection ({stats['criterion']}) ===")
    print(f"  top-n_hvg                    : {stats['n_hvg']:,}")
    print(f"  perturbation targets         : {stats['n_targets']}")
    print(f"  targets forced in beyond hvg : {stats['n_targets_forced_in']}")
    print(f"  targets missing from var     : {stats['n_targets_missing_from_var']} "
          f"{stats['missing_targets'] if stats['missing_targets'] else ''}")
    print(f"  modelled gene space          : {stats['n_selected']:,}")

    # ---------------------------------------------------------------- matrix
    print("\n=== 4. building the subset matrix ===")
    started = time.time()
    matrix = preprocess.build_matrix(raw_path, gene_indices, config["data"]["chunk_size"])
    print(f"  done in {time.time() - started:.1f}s  "
          f"shape={matrix.shape}  nnz={matrix.nnz:,}  density={matrix.nnz / np.prod(matrix.shape) * 100:.2f}%")

    # Carry over any obs column the split depends on. Without this the cache
    # loses obs['split'] / the group column, and every later load falls back to
    # the raw file or fails outright.
    obs_frame = {"condition": pd.Categorical(conditions)}
    for key in {config["split"].get("obs_key"), config["split"].get("group_key")}:
        if not key or key == "condition":
            continue
        try:
            obs_frame[key] = pd.Categorical(io.read_obs_column(raw_path, key))
            print(f"  carried over obs[{key!r}] for the split")
        except KeyError:
            pass

    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(obs_frame, index=io.read_obs_column(raw_path, "_index")),
        var=pd.DataFrame(index=pd.Index(var_names[gene_indices], name=None)),
    )
    adata.uns["build_config"] = json.dumps(config)
    adata.uns["gene_selection"] = json.dumps(
        {k: v for k, v in stats.items() if k != "missing_targets"})
    adata.uns["perturbation_targets"] = preprocess.perturbation_targets(
        conditions, config["data"]["control_label"])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    adata.write_h5ad(cache_path, compression="gzip")
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"\n-> wrote {cache_path}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
