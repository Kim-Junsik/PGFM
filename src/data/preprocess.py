"""Build the modelled-space cache from the raw Norman h5ad.

The only transformation is a gene subset. X in the source file is already log1p
normalised, and no raw-count layer survives, so re-running normalize_total/log1p
would double-normalise data that cannot be recovered.

Gene space = top `n_hvg` by the chosen criterion, unioned with every perturbation
target, kept in adata.var_names order. That order is the single alignment
convention for everything downstream (mask rows, decoder outputs, metrics); there
is no tokenizer and no alphabetical vocab.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from . import io


def perturbation_targets(conditions: np.ndarray, control_label: str = "ctrl") -> list[str]:
    """Genes named in any condition string, e.g. 'AHR+FEV' -> AHR, FEV."""
    targets = set()
    for condition in np.unique(conditions):
        if condition == control_label:
            continue
        for gene in condition.split("+"):
            if gene != control_label:
                targets.add(gene)
    return sorted(targets)


def gene_statistics(path: str, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-gene mean and variance over all cells, in one streaming pass."""
    n_obs, n_var = io.shape(path)
    total = np.zeros(n_var, dtype=np.float64)
    total_sq = np.zeros(n_var, dtype=np.float64)

    for start, stop, block in io.iter_row_chunks(path, chunk_size):
        total += np.asarray(block.sum(axis=0)).ravel()
        total_sq += np.asarray(block.multiply(block).sum(axis=0)).ravel()
        print(f"  variance pass: {stop:,} / {n_obs:,} cells", end="\r")
    print()

    mean = total / n_obs
    variance = total_sq / n_obs - mean ** 2
    return mean, np.maximum(variance, 0.0)


def select_genes(config: dict, var_names: np.ndarray, conditions: np.ndarray,
                 mean: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, dict]:
    """Choose the modelled gene set and return its indices in var_names order."""
    data_cfg = config["data"]
    criterion = data_cfg["hvg_criterion"]
    if criterion == "raw_variance":
        score = variance
    elif criterion == "dispersion":
        # variance-to-mean ratio; guard the genes that are all-zero
        score = np.where(mean > 0, variance / np.maximum(mean, 1e-12), 0.0)
    else:
        raise ValueError(f"unknown hvg_criterion {criterion!r}")

    n_hvg = data_cfg["n_hvg"]
    if n_hvg is None:
        selected = set(range(len(var_names)))
        hvg_indices: set[int] = set(selected)
    else:
        hvg_indices = set(np.argsort(-score)[:n_hvg].tolist())
        selected = set(hvg_indices)

    name_to_index = {name: i for i, name in enumerate(var_names)}
    targets = perturbation_targets(conditions, data_cfg["control_label"])
    missing = [t for t in targets if t not in name_to_index]
    forced: list[int] = []
    if data_cfg["force_include_targets"]:
        for target in targets:
            index = name_to_index.get(target)
            if index is not None and index not in selected:
                selected.add(index)
                forced.append(index)

    indices = np.array(sorted(selected), dtype=np.int64)
    stats = {
        "n_selected": len(indices),
        "n_hvg": len(hvg_indices),
        "n_targets": len(targets),
        "n_targets_forced_in": len(forced),
        "n_targets_missing_from_var": len(missing),
        "missing_targets": missing,
        "criterion": criterion,
    }
    return indices, stats


def build_matrix(path: str, gene_indices: np.ndarray, chunk_size: int) -> sparse.csr_matrix:
    """Second pass: stream the rows again, keeping only the selected columns."""
    n_obs, _ = io.shape(path)
    blocks = []
    for start, stop, block in io.iter_row_chunks(path, chunk_size):
        blocks.append(block[:, gene_indices])
        print(f"  subset pass:   {stop:,} / {n_obs:,} cells", end="\r")
    print()
    return sparse.vstack(blocks, format="csr")
