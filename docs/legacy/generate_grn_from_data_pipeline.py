# Verified working environment (Python 3.10.13):
#   pip install numpy==1.26.4 pandas==2.3.3 torch==2.2.0
#   pip install "dask==2024.2.1" "distributed==2024.2.1" arboreto==0.1.6
#   pip install scanpy==1.11.5 anndata==0.11.4
#   pip install jax==0.6.2 jaxlib==0.6.2          # imported transitively by src/data_process/data.py
#   pip install tqdm==4.67.3 POT==0.9.7 scikit-learn==1.7.2
#   pip install scipy==1.15.3 networkx            # used by src/utils/utils.py::process_vocab
#   pip install requests beautifulsoup4 rdkit==2025.9.5   # combosciplex only (compound annotation)
#
# jax/jaxlib MUST stay in sync with whatever this repo's flow_matching code needs
# (flax/optax/equinox/lineax) - do not downgrade jax to satisfy some other package
# (e.g. pertpy/scvi-tools want an old jax, which is why combosciplex's compound
# annotation is done pertpy-free in build_combosciplex_processed_cache.py instead).
#
# Shared script for norman / combosciplex:
#   1) Call Data.load_data / Data.process_data from src/data_process/data.py directly
#      ("as-is") to reproduce the exact same preprocessing (HVG selection + train/test
#      split) used by the real training run (run.py).
#   2) For the GRNBoost2 input, only use cells that stay leak-free no matter which fold
#      is used for training:
#      - norman: the double-perturbation combos assigned to "test" differ from fold to
#        fold (e.g. fold0's test set and fold1's test set overlap) because this is NOT
#        a non-overlapping partition k-fold, it's 5 independent random 70/30 splits with
#        different seeds. So we only keep double-perturbation combos that are "train" in
#        ALL 5 folds, plus every single-perturbation/control cell (measured on norman:
#        about 69.8% of cells kept - 100% of control/singles, only 20.8% of doubles).
#      - combosciplex: there is no fold concept at all, it's split by a single fixed
#        test_conditions list, so the adata_train produced by process_data() is already
#        free of test information.
#   3) The model does NOT actually read the mask via data.py's use_grn branch (that CSV
#      auto-realignment logic is unused). The real consumer is
#      src/models/origin/layers.py::GeneEncoder:
#          loaded_tensor = torch.load(mask_path)   # not a dict, the tensor itself, in __init__
#          mask_padded = softmax(mask_prior + alpha * mask_residual)  # recomputed every forward()
#          mask = mask_padded[x[0]]                 # plain integer indexing, no name matching
#      Since there is no name-based realignment, the row/column order of the saved
#      tensor must exactly match the vocab used at training time
#      (src/utils/utils.py::process_vocab, cached at
#      src/tokenizer/{data_name}_{n_top_genes}_highly_vocab.json). That vocab is NOT
#      ordered like adata.var_names - it's "4 special tokens (<pad>,<cls>,<mask>,control)
#      + genes sorted alphabetically" (see gene_tokenizer.py::_build_vocab_from_iterator).
#      So instead of reimplementing that ordering logic, this script calls
#      process_vocab() directly, builds an N x N dense tensor aligned to the resulting
#      token2idx mapping, and saves it with torch.save(tensor, path) - not wrapped in a
#      dict, since layers.py no longer does ['mask'].
#
#      A (source_gene, target_gene, weight) sparse CSV is also saved for reference, but
#      it's only for humans to inspect - the model never reads it directly.
#
#   4) GRNBoost2 alone is quite sparse (~1.5% density on norman), since it only assigns
#      edges from the ~450 canonical TFs it was given as candidate regulators, and only
#      where it found nonzero feature importance in THIS dataset. A DoRothEA-based
#      gap-filling step was tried and removed again: DoRothEA's regulons are not
#      cell-type-specific (this dataset is K562), the confidence-tier -> weight mapping
#      had no principled derivation, and the actual density gain was marginal
#      (+0.9% relative, from 1.44% to 1.45%) - not worth the added complexity/assumptions
#      until there's a validated need for it. The mask below is pure GRNBoost2.
#
# Note: combosciplex already has outputs from GRN.py (TF-priority approach) under
#       data/GRN/combosciplex/ (grn_edges.csv, grn_attention_mask.pt, etc). To avoid a
#       filename collision, the dense mask this script produces is saved as
#       weighted_grn_mask.pt instead of grn_attention_mask.pt.
#       (grn_edges.csv does share a name and will be overwritten - back it up first if
#       you need the old one.)

import argparse
import logging
import os
import sys
import time
import urllib.request
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import dask
dask.config.set({'dataframe.query-planning': False})

from arboreto.algo import grnboost2
from dask.distributed import Client, LocalCluster

# Quiets dask/distributed's verbose "Closing Nanny...", "Remove worker..." INFO logs
# that get printed when a LocalCluster shuts down. Purely cosmetic - doesn't affect
# whether anything actually succeeds - but this script starts/stops a fresh cluster
# once per dataset, so without this the terminal gets noisy (and can look alarming
# to someone unfamiliar with dask reading the output).
logging.getLogger('distributed').setLevel(logging.ERROR)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_process.data import Data
from src.utils.utils import process_vocab

SUPPORTED_DATASETS = ('norman', 'combosciplex')

# process_vocab() looks up the vocab cache using the relative path 'src/tokenizer/...',
# so this script must be run from the project root directory (same convention as run.py).
if os.path.abspath(os.getcwd()) != project_root:
    print(f"[WARN] Current working directory ({os.getcwd()}) is not the project root "
          f"({project_root}). process_vocab()'s relative-path cache lookup may break - "
          f"please run this script from the project root.")


def download_tf_list(save_path):
    """Downloads the Human TF list if it does not exist locally."""
    if not os.path.exists(save_path):
        print("Downloading Human TF list...")
        url = "https://raw.githubusercontent.com/aertslab/pySCENIC/master/resources/hs_hgnc_tfs.txt"
        urllib.request.urlretrieve(url, save_path)
    with open(save_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def run_process_data(data_name, data_base_path, dummy_mask_path, n_top_genes, infer_top_gene,
                      use_negative_edge, topk, split_method=None, test_conditions=None):
    """
    Calls Data.process_data() from data.py directly so we reproduce the exact same
    preprocessing as the real training pipeline (no copy/paste reimplementation).

    Passing use_grn=True makes process_data() try to read weighted_grn_mask_path, but we
    don't have a GRN result yet, so we pre-create a dummy .pt file holding an arbitrary
    tensor so it's recognized as "an existing mask file" - this prevents process_data()
    from instead running the expensive coexpression mask computation (pearson topk).
    This dummy mask is discarded and never actually used.

    data.py::process_data() now always does `self.mask = torch.load(mask_path)`
    regardless of the file extension (no CSV support, not a dict), so the dummy file
    must also be a tensor saved via torch.save() (a text CSV would raise
    UnpicklingError).

    The fold value doesn't matter here for norman (select_grn_input_adata() below
    re-selects cells by combining all 5 folds anyway). process_data() requires a fold
    kwarg, so we just pass 0 as a formality.
    """
    if data_name not in SUPPORTED_DATASETS:
        raise ValueError(f"{data_name} is not supported. Choose from {SUPPORTED_DATASETS}")

    os.makedirs(os.path.dirname(dummy_mask_path), exist_ok=True)
    if not os.path.exists(dummy_mask_path):
        torch.save(torch.zeros(1), dummy_mask_path)

    extra_kwargs = {}
    if data_name == 'norman':
        extra_kwargs['fold'] = 0
    elif data_name == 'combosciplex':
        if test_conditions is not None:
            extra_kwargs['test_conditions'] = test_conditions
        # If None, process_data() falls back to its own default test_conditions list.

    data_manager = Data(data_path=data_base_path)
    data_manager.load_data(data_name)
    data_manager.process_data(
        n_top_genes=n_top_genes,
        infer_top_gene=infer_top_gene,
        split_method=split_method or 'additive',
        use_negative_edge=use_negative_edge,
        k=topk,
        use_grn=True,
        weighted_grn_mask_path=dummy_mask_path,
        **extra_kwargs,
    )
    return data_manager


def select_grn_input_adata(data_name, data_manager):
    """
    Picks the set of cells to feed into GRNBoost2 such that no matter which fold is
    used for training, no test information leaks in.

    - norman: intersects the 'train' lists across all 5 folds stored in
      process_data()'s data_manager.split_results, keeping only double-perturbation
      combos that are 'train' in every single fold. Single-perturbation/control cells
      were never part of the fold split in the first place (their condition string
      contains 'control'), so they're always kept in full.
    - combosciplex: there's no fold concept, just one fixed test_conditions split, so
      the adata_train produced by process_data() is already free of test information.
    """
    if data_name == 'combosciplex':
        adata = data_manager.adata_train.copy()
        print(f"-> (combosciplex) using adata_train as-is: {adata.shape[0]} cells x {adata.shape[1]} genes")
        return adata

    # ---- norman: intersection across the 5 folds ----
    adata = data_manager.adata
    condition = adata.obs['condition'].astype(str)

    # process_data() replaces 'ctrl' -> 'control' internally. True double-perturbation
    # strings never contain 'ctrl'/'control' to begin with (e.g. 'AHR+FEV'), so this
    # substring check distinguishes them from single-perturbation cells
    # (e.g. 'AHR+control') and control cells ('control').
    is_double = ~condition.str.contains('control')
    all_doubles_in_data = set(condition[is_double].unique())

    always_train_doubles = set(all_doubles_in_data)
    for split in data_manager.split_results:
        always_train_doubles &= set(split['train'])

    n_total = adata.n_obs
    n_double_total = int(is_double.sum())
    safe_mask = (~is_double) | condition.isin(always_train_doubles)
    adata_safe = adata[safe_mask].copy()

    print(f"-> Applying norman 5-fold intersection:")
    print(f"   Out of {len(all_doubles_in_data)} total double-perturbation combos, "
          f"{len(always_train_doubles)} are 'train' in all 5 folds (leak-safe)")
    print(f"   Total cells {n_total} -> {adata_safe.n_obs} kept "
          f"({adata_safe.n_obs / n_total * 100:.2f}%), "
          f"out of {n_double_total} double-perturbation cells, "
          f"{int(condition.isin(always_train_doubles).sum())} are used")
    print(f"-> GRN input: {adata_safe.shape[0]} cells x {adata_safe.shape[1]} genes")
    return adata_safe


def run_grnboost2(adata, tf_names, n_workers=8):
    """Runs GRNBoost2 on the given adata to get a (TF, target, importance) network."""
    print("Converting expression matrix to dense format for GRNBoost2...")
    expr_matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    expr_df = pd.DataFrame(
        data=expr_matrix,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    valid_tfs = [tf for tf in tf_names if tf in expr_df.columns]
    print(f"Found {len(valid_tfs)} valid TFs among {expr_df.shape[1]} genes.")

    print(f"Starting GRN inference with {n_workers} workers...")
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)
    try:
        network_df = grnboost2(expression_data=expr_df, tf_names=valid_tfs, client_or_address=client)
    finally:
        client.close()
        cluster.close()
        # Dask's background comm threads sometimes fire a harmless CommClosedError
        # right at interpreter exit when a script creates/tears down more than one
        # LocalCluster in one process (e.g. looping over norman + combosciplex here).
        # This delay lets that teardown finish quietly instead of printing a scary
        # (but harmless) traceback after the real results have already been saved.
        time.sleep(1)

    return network_df


def normalize_and_save_sparse_prior(network_df, save_path):
    """
    Normalizes grn_edges.csv (TF, target, importance) into a
    (source_gene, target_gene, weight) sparse CSV. Same normalization as
    build_weighted_prior.py: log1p -> clip top 1% -> 0-1 max-scale.

    Note: this CSV is for humans to inspect/debug the results, not something the model
    reads directly (the model reads the .pt tensor built by build_and_save_dense_mask()).
    """
    log_importance = np.log1p(network_df['importance'])
    p99_val = np.percentile(log_importance, 99)
    log_importance_clipped = np.clip(log_importance, a_min=None, a_max=p99_val)
    max_val = log_importance_clipped.max()
    norm_importance = log_importance_clipped / max_val

    sparse_df = pd.DataFrame({
        'source_gene': network_df['TF'],
        'target_gene': network_df['target'],
        'weight': norm_importance,
    })
    sparse_df.to_csv(save_path, index=False)
    print(f"-> Saved {len(sparse_df):,} weighted edges to {save_path}")
    return sparse_df


def build_vocab_for_dataset(data_name, n_top_genes, data_manager):
    """
    Calls the same process_vocab() that run.py actually uses (not reimplemented).
    If the src/tokenizer/{data_name}_{n_top_genes}_highly_vocab.json cache already
    exists, it's loaded as-is; otherwise process_vocab() builds and saves a new one.
    """
    config_like = SimpleNamespace(data_name=data_name, n_top_genes=n_top_genes)
    vocab = process_vocab(data_manager, config_like)
    print(f"-> vocab size (including 4 special tokens): {len(vocab)}")
    return vocab


def build_and_save_dense_mask(vocab, network_df, save_path):
    """
    Builds an N x N dense tensor aligned to the vocab's token2idx order from the
    GRNBoost2 result (TF, target, importance) and saves it with
    torch.save(tensor, save_path) - not wrapped in a dict.

    models/origin/layers.py::GeneEncoder does:
        loaded_tensor = torch.load(mask_path)
        mask = self.mask_padded[x[0]]
    i.e. plain integer indexing with no name matching, so the row/column order of the
    tensor built here must exactly match the vocab's index order.
    """
    token2idx = vocab.get_stoi()
    n = len(token2idx)
    weight_matrix = np.zeros((n, n), dtype=np.float32)

    log_importance = np.log1p(network_df['importance'])
    p99_val = np.percentile(log_importance, 99)
    clipped = np.clip(log_importance, a_min=None, a_max=p99_val)
    norm_importance = clipped / clipped.max()

    matched = 0
    valid_weights = []
    for tf, target, w in zip(network_df['TF'], network_df['target'], norm_importance):
        tf_idx = token2idx.get(tf)
        target_idx = token2idx.get(target)
        if tf_idx is not None and target_idx is not None:
            weight_matrix[tf_idx, target_idx] = float(w)
            valid_weights.append(float(w))
            matched += 1

    # Fill the diagonal (self-loop) for every index, including the special tokens
    # <pad>/<cls>/<mask>/control (same convention as GRN.py's create_attention_mask()).
    mean_weight = float(np.mean(valid_weights)) if valid_weights else 0.5
    self_loop_weight = min(mean_weight * 1.5, 0.5)
    np.fill_diagonal(weight_matrix, self_loop_weight)

    total_off_diag_possible = n * n - n
    print(f"-> {matched:,} / {len(network_df):,} GRNBoost2 edges matched into the {n}x{n} vocab-aligned mask "
          f"({matched / total_off_diag_possible * 100:.2f}% density), self-loop weight={self_loop_weight:.4f}")

    tensor = torch.from_numpy(weight_matrix)
    torch.save(tensor, save_path)
    print(f"-> Saved dense vocab-aligned mask tensor to {save_path} (shape={tuple(tensor.shape)})")
    return tensor


def generate_grn_for_dataset(data_name, split_method='additive', n_top_genes=5000,
                              infer_top_gene=1000, n_workers=8, use_negative_edge=True, topk=30):
    data_base_path = os.path.join(project_root, "data", "scDFM", data_name)
    tf_list_path = os.path.join(project_root, "data", "GRN", "hs_hgnc_tfs.txt")
    out_dir = os.path.join(project_root, "data", "GRN", data_name, )
    dummy_mask_path = os.path.join(out_dir, "_dummy_mask.pt")
    processed_h5ad_path = os.path.join(out_dir, f"{data_name}_grn_input.h5ad")
    edge_save_path = os.path.join(out_dir, "grn_edges.csv")
    weighted_prior_save_path = os.path.join(out_dir, "weighted_grn_sparse_prior.csv")
    # grn_attention_mask.pt already holds a different (TF-priority) pipeline's output
    # for combosciplex, so we use a separate filename to avoid overwriting it. This
    # .pt file is the one to actually use for training.
    dense_mask_save_path = os.path.join(out_dir, "weighted_grn_mask.pt")

    os.makedirs(out_dir, exist_ok=True)

    print(f"=== [{data_name}] 1. Running the exact same preprocessing pipeline as data.py ===")
    data_manager = run_process_data(
        data_name=data_name,
        data_base_path=data_base_path,
        dummy_mask_path=dummy_mask_path,
        n_top_genes=n_top_genes,
        infer_top_gene=infer_top_gene,
        use_negative_edge=use_negative_edge,
        topk=topk,
        split_method=split_method,
    )

    print(f"\n=== [{data_name}] 2. Selecting leak-safe cells for GRN inference ===")
    grn_input_adata = select_grn_input_adata(data_name, data_manager)
    grn_input_adata.write_h5ad(processed_h5ad_path)
    print(f"-> Saved GRN input h5ad to {processed_h5ad_path}")

    print(f"\n=== [{data_name}] 3. Downloading TF list and running GRNBoost2 ===")
    os.makedirs(os.path.dirname(tf_list_path), exist_ok=True)
    tfs = download_tf_list(tf_list_path)
    network_df = run_grnboost2(grn_input_adata, tfs, n_workers=n_workers)
    network_df.to_csv(edge_save_path, index=False)
    print(f"-> Saved raw GRN edges to {edge_save_path}")

    print(f"\n=== [{data_name}] 4. Saving sparse edge CSV (for reference only) ===")
    normalize_and_save_sparse_prior(network_df, weighted_prior_save_path)

    print(f"\n=== [{data_name}] 5. Building vocab-aligned dense mask for the model ===")
    vocab = build_vocab_for_dataset(data_name, n_top_genes, data_manager)
    build_and_save_dense_mask(vocab, network_df, dense_mask_save_path)

    print(f"\n[{data_name}] Done. weighted_grn_mask_path -> {dense_mask_save_path}")
    return dense_mask_save_path


def main():
    parser = argparse.ArgumentParser(
        description="Build HVG data (same pipeline as data.py) + GRN prior for norman and/or combosciplex."
    )
    parser.add_argument('--datasets', nargs='+', choices=list(SUPPORTED_DATASETS),
                         default=list(SUPPORTED_DATASETS))
    parser.add_argument('--split_method', type=str, default='additive',
                         help='norman only (which cached split_results.pkl to load)')
    parser.add_argument('--n_top_genes', type=int, default=5000)
    parser.add_argument('--infer_top_gene', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()

    results = {}
    for data_name in args.datasets:
        results[data_name] = generate_grn_for_dataset(
            data_name=data_name,
            split_method=args.split_method,
            n_top_genes=args.n_top_genes,
            infer_top_gene=args.infer_top_gene,
            n_workers=args.n_workers,
        )

    print("\n=== Summary ===")
    for data_name, path in results.items():
        print(f"{data_name}: weighted_grn_mask_path -> {path}")


if __name__ == "__main__":
    main()
