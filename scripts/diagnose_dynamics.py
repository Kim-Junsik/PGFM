"""Where the flow loses the 1.346 of resid_R2 that the autoencoder ceiling allows.

    python scripts/diagnose_dynamics.py results/runs/pcab_lie_commutator
    python scripts/diagnose_dynamics.py <run> --infer-top-gene 1000

diagnose_bottleneck.py established that encode-decode caps resid_R2 at +0.371 while
the model reaches -0.975, so the dynamics loses 2.1x more than the representation
path. This asks WHERE, with four measurements that need no retraining.

A. FLOW-MATCHING FLOOR. The stage-2 loss can only ever reach
   E[Var(z1 - z0 | z_t, t, S)]: many different pairs pass through the same z_t with
   different displacements, and the field can only learn their conditional mean. If
   the achieved fm is already near the variance of the target, the field is doing as
   well as this objective permits and the objective is the problem, not the fit.

B. INTEGRATION ERROR. Inference uses 20-step RK4. Re-integrating the SAME
   checkpoint at more steps costs nothing; if resid_R2 moves, the step count is a
   real term. If it does not, that hypothesis is closed.

C. CONSTITUENT SINGLES. The transport diagnostic samples 40 arbitrary train
   singles, which are not the singles that make up the test doubles. This measures
   the ones that actually do, since a double is built from exactly those two
   generators.

D. COMPOSITION NONLINEARITY. The composed velocity is u_a + u_b, but transport is
   not linear in the velocity: integrating a summed field differs from summing two
   integrations. This quantifies the gap, which is the model's own notion of
   "non-additivity" before any interaction term is added.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset import ConditionSampler, condition_genes
from src.eval import baselines
from src.eval.diagnostics import (_control_sample, condition_groups, load_run,
                                  scdfm_eval_genes)
from src.eval.predict import _head_aux
from src.models.flow import integrate
from src.train.coupling import sample_pairs


def norm(v) -> float:
    return float(np.linalg.norm(v))


def _pairs(data, stats, fold, method):
    out = []
    for c in condition_groups(data, stats, fold, method)["test doubles"]:
        a, b = data.naming.genes(c)
        sa, sb = stats.single_of(a), stats.single_of(b)
        if all(stats.has(x) for x in (c, sa, sb)):
            out.append((c, a, b, sa, sb))
    return out


@torch.no_grad()
def flow_matching_floor(vae, field, data, config, device, rng, n_batches):
    """A: what the stage-2 loss achieves, against what it could ever achieve.

    The floor is estimated as the variance of the coupled displacement itself. A
    predictor that ignores z_t entirely and returns the per-condition mean already
    reaches that variance, so fm at or above it means the field is explaining
    nothing beyond the mean displacement.
    """
    train_cfg = config["train"]
    conditions = [c for c in data.conditions if c != data.control_condition]
    sampler = ConditionSampler(data, conditions, train_cfg["batch_size"], rng)
    achieved, target_var, mean_only = [], [], []
    for condition in conditions[:n_batches]:
        source, target, _ = sampler.batch(condition)
        z0, _ = vae.encode_z(torch.as_tensor(source, device=device))
        z1, _ = vae.encode_z(torch.as_tensor(target, device=device))
        z0p, z1p = sample_pairs(z0, z1, train_cfg["coupling"], train_cfg["uot_reg"],
                                train_cfg["uot_reg_marginal"], rng)
        displacement = z1p - z0p
        t = torch.rand(z0p.shape[0], 1, device=device)
        z_t = (1.0 - t) * z0p + t * z1p
        perturbations = [data.pert_index[g] for g in condition_genes(condition)]
        predicted = field(z_t, t.reshape(-1), perturbations)
        achieved.append(float(torch.nn.functional.mse_loss(predicted, displacement)))
        target_var.append(float(displacement.var(dim=0).mean()))
        mean_only.append(float(torch.nn.functional.mse_loss(
            displacement.mean(dim=0, keepdim=True).expand_as(displacement), displacement)))
    return np.mean(achieved), np.mean(target_var), np.mean(mean_only)


@torch.no_grad()
def resid_r2(vae, field, data, stats, pairs, config, device, n_cells, n_steps, genes):
    rng = np.random.default_rng(config["eval"]["seed"])
    num = den = 0.0
    for c, a, b, sa, sb in pairs:
        x0 = torch.as_tensor(_control_sample(data, n_cells, rng), device=device)
        z1 = integrate(field, vae.encode_z(x0)[0],
                       [data.pert_index[a], data.pert_index[b]], n_steps)
        m_hat = vae.reconstruction(vae.decode_z(z1), **_head_aux(vae, x0)).mean(0).cpu().numpy()
        r = (stats.mean[c] - stats.mean[sa] - stats.mean[sb] + stats.control)[genes]
        r_hat = (m_hat - stats.mean[sa] - stats.mean[sb] + stats.control)[genes]
        num += float((r_hat - r) @ (r_hat - r))
        den += float(r @ r)
    return 1.0 - num / den


@torch.no_grad()
def singles_and_composition(vae, field, data, stats, pairs, config, device, n_cells, genes):
    """C and D, sharing one transport pass per condition."""
    rng = np.random.default_rng(config["eval"]["seed"])
    n_steps = config["train"]["n_integration_steps"]
    single_ratio, single_cos, compose_gap = [], [], []
    seen = set()
    for c, a, b, sa, sb in pairs:
        x0 = torch.as_tensor(_control_sample(data, n_cells, rng), device=device)
        z0 = vae.encode_z(x0)[0]
        origin = z0.mean(dim=0)
        ia, ib = data.pert_index[a], data.pert_index[b]

        za = integrate(field, z0, [ia], n_steps).mean(dim=0) - origin
        zb = integrate(field, z0, [ib], n_steps).mean(dim=0) - origin
        zab = integrate(field, z0, [ia, ib], n_steps).mean(dim=0) - origin

        # C: how well each constituent single transports, in gene space
        for single, index in ((sa, ia), (sb, ib)):
            if single in seen:
                continue
            seen.add(single)
            z1 = integrate(field, z0, [index], n_steps)
            m_hat = vae.reconstruction(vae.decode_z(z1), **_head_aux(vae, x0)).mean(0).cpu().numpy()
            hat = (m_hat - stats.control)[genes]
            true = (stats.mean[single] - stats.control)[genes]
            single_ratio.append(norm(hat) / norm(true) if norm(true) else np.nan)
            single_cos.append(float(hat @ true) / (norm(hat) * norm(true))
                              if norm(hat) * norm(true) else np.nan)

        # D: integrating the summed field vs summing the two integrations
        summed = (za + zb).cpu().numpy()
        composed = zab.cpu().numpy()
        compose_gap.append(norm(composed - summed) / norm(summed) if norm(summed) else np.nan)
    return single_ratio, single_cos, compose_gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=512)
    parser.add_argument("--infer-top-gene", type=int, default=None)
    parser.add_argument("--fm-batches", type=int, default=30)
    parser.add_argument("--steps", type=int, nargs="*", default=[20, 50, 100])
    args = parser.parse_args()

    config, data, stats, fold, vae, field = load_run(args.run_dir, args.device)
    genes = (scdfm_eval_genes(data, fold, args.infer_top_gene) if args.infer_top_gene
             else np.arange(data.n_genes))
    pairs = _pairs(data, stats, fold, config["split"]["method"])
    scope = (f"top-{args.infer_top_gene} scanpy-HVG genes" if args.infer_top_gene
             else f"all {data.n_genes:,} genes")
    print(f"{os.path.basename(args.run_dir.rstrip('/'))}   {len(pairs)} test doubles   {scope}\n")

    rng = np.random.default_rng(config["train"]["seed"])
    fm, var, mean_only = flow_matching_floor(vae, field, data, config, args.device,
                                             rng, args.fm_batches)
    print("A - flow-matching floor")
    print(f"  achieved fm (field)             {fm:8.4f}")
    print(f"  same loss for a MEAN predictor  {mean_only:8.4f}"
          "   <- ignores z_t, returns the per-condition mean")
    print(f"  variance of the target          {var:8.4f}")
    explained = 1.0 - fm / mean_only if mean_only else float("nan")
    print(f"  -> variance the field explains beyond the mean: {100*explained:5.1f}%")

    print("\nB - integration steps (same checkpoint, re-integrated)")
    for n_steps in args.steps:
        value = resid_r2(vae, field, data, stats, pairs, config, args.device,
                         args.n_cells, n_steps, genes)
        print(f"  RK4 {n_steps:4d} steps            resid_R2 {value:8.4f}")

    ratio, cos, gap = singles_and_composition(vae, field, data, stats, pairs, config,
                                              args.device, args.n_cells, genes)
    print(f"\nC - the singles that actually compose the test doubles ({len(ratio)} of them)")
    print(f"  gene-space displacement ratio   {np.nanmedian(ratio):8.4f}")
    print(f"  gene-space cosine               {np.nanmedian(cos):8.4f}")

    print("\nD - composition nonlinearity")
    print(f"  ||int(u_a+u_b) - (int u_a + int u_b)|| / ||sum||   {np.nanmedian(gap):.4f}")
    print("  the model's own non-additivity, before any interaction term")


if __name__ == "__main__":
    main()
