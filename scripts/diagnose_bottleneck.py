"""What the representation path costs, and whether the task is sub-additive.

    python scripts/diagnose_bottleneck.py results/runs/pcab_lie_commutator
    python scripts/diagnose_bottleneck.py <run> --infer-top-gene 1000

Two measurements, neither of which uses the velocity field. Both run on a finished
checkpoint in minutes.

STEP 0 - the ceiling. Every prediction this model can make passes through
encode -> decode, so pushing the TRUE cells of each condition through that path
and re-deriving the non-additive residual gives an upper bound that no amount of
work on the dynamics can beat. Reported two ways:

  resid_R2 of the autoencoder   how much of the residual survives the whole path
  residual share, gene vs latent   ||r|| / ||delta|| on each side of the bottleneck

The second separates the two suspects. If the residual keeps its share of the
signal in the 64-d latent but loses it in gene space, the decoder is at fault; if
the share is already gone in the latent, the bottleneck is.

STEP 1 - is the task sub-additive. Measured on the true condition means alone, no
model involved: ||delta_AB|| / ||delta_A + delta_B||. This exists because the
transport diagnostic shows train singles reproduced at 0.588 of their true
displacement while test doubles come out at 0.973. A model that underfits the
conditions it is directly supervised on should not do BETTER on the combinations
built from them. If this ratio is near 0.6, the doubles are landing right because
two errors cancel, not because the singles are right.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import condition_groups, load_run, scdfm_eval_genes
from src.eval.predict import autoencode


def norm(v) -> float:
    return float(np.linalg.norm(v))


@torch.no_grad()
def measure(run_dir: str, device: str, n_cells: int, infer_top_gene: int | None):
    config, data, stats, fold, vae, field = load_run(run_dir, device)
    rng = np.random.default_rng(config["eval"]["seed"])
    genes = scdfm_eval_genes(data, fold, infer_top_gene) if infer_top_gene else None
    subset = (lambda v: v[genes]) if genes is not None else (lambda v: v)

    pairs = []
    for c in condition_groups(data, stats, fold, config["split"]["method"])["test doubles"]:
        a, b = data.naming.genes(c)
        sa, sb = stats.single_of(a), stats.single_of(b)
        if all(stats.has(x) for x in (c, sa, sb)):
            pairs.append((c, sa, sb))

    # Autoencoded and latent means, cached: the same singles recur across doubles.
    ae_cache: dict[str, np.ndarray] = {}
    z_cache: dict[str, np.ndarray] = {}

    def cells_of(condition):
        cells = data.cells(condition)
        if cells.shape[0] > n_cells:
            cells = cells[rng.choice(cells.shape[0], size=n_cells, replace=False)]
        return cells

    def ae_mean(condition):
        if condition not in ae_cache:
            cells = cells_of(condition)
            ae_cache[condition] = autoencode(vae, cells, device).mean(axis=0)
        return ae_cache[condition]

    def z_mean(condition):
        if condition not in z_cache:
            x = torch.as_tensor(cells_of(condition), device=device)
            z_cache[condition] = vae.encode_z(x)[0].mean(dim=0).cpu().numpy()
        return z_cache[condition]

    ctrl = stats.control_condition
    rows = []
    for c, sa, sb in pairs:
        # true residual and the autoencoder's reproduction of it
        r = subset(stats.mean[c] - stats.mean[sa] - stats.mean[sb] + stats.control)
        r_ae = subset(ae_mean(c) - ae_mean(sa) - ae_mean(sb) + ae_mean(ctrl))
        # the residual's share of the total signal, on each side of the bottleneck
        delta_gene = subset(stats.mean[c] - stats.control)
        r_z = z_mean(c) - z_mean(sa) - z_mean(sb) + z_mean(ctrl)
        delta_z = z_mean(c) - z_mean(ctrl)
        # step 1: is the double smaller than the sum of its singles
        sum_singles = subset((stats.mean[sa] - stats.control) +
                             (stats.mean[sb] - stats.control))
        rows.append({
            "num": norm(r_ae - r) ** 2,
            "den": norm(r) ** 2,
            "share_gene": norm(r) / norm(delta_gene) if norm(delta_gene) else np.nan,
            "share_latent": norm(r_z) / norm(delta_z) if norm(delta_z) else np.nan,
            "subadditive": norm(delta_gene) / norm(sum_singles) if norm(sum_singles) else np.nan,
        })
    return config, data, len(pairs), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=512,
                        help="cells per condition; every condition is encoded once")
    parser.add_argument("--infer-top-gene", type=int, default=None)
    args = parser.parse_args()

    config, data, n, rows = measure(args.run_dir, args.device, args.n_cells,
                                    args.infer_top_gene)
    scope = (f"top-{args.infer_top_gene} scanpy-HVG genes of the test subset"
             if args.infer_top_gene else f"all {data.n_genes:,} genes in the cache")
    print(f"{os.path.basename(args.run_dir.rstrip('/'))}   {n} test doubles   {scope}\n")

    ceiling = 1.0 - sum(r["num"] for r in rows) / sum(r["den"] for r in rows)
    print("STEP 0 - representation-path ceiling")
    print(f"  autoencoder resid_R2 (pooled)   {ceiling:8.4f}"
          "   <- no model through this path can beat this")
    print(f"  residual share, gene space      {np.nanmedian([r['share_gene'] for r in rows]):8.4f}"
          "   ||r|| / ||delta_AB||")
    print(f"  residual share, 64-d latent     {np.nanmedian([r['share_latent'] for r in rows]):8.4f}"
          "   same quantity after encoding")

    print("\nSTEP 1 - is the task sub-additive (true data only, no model)")
    sub = np.array([r["subadditive"] for r in rows])
    print(f"  ||delta_AB|| / ||delta_A + delta_B||   median {np.nanmedian(sub):.4f}"
          f"   mean {np.nanmean(sub):.4f}")
    print(f"  conditions below 0.8                  {int((sub < 0.8).sum())} / {len(sub)}")


if __name__ == "__main__":
    main()
