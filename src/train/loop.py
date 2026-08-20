"""Two-stage training: autoencode first, then learn the latent transport.

Stage 1 fits the VAE with the flow disabled, so the latent space exists before
anything tries to move through it.

Stage 2 fits the velocity field by flow matching along the straight interpolant
between OT-coupled latents. The encoder is fine-tuned here by default: freezing it
optimises the latent for reconstruction, whereas what we actually want is a space
in which the generators COMPOSE well - those are not the same objective.

Singles warm up first. Every u_a is identifiable from single perturbations alone,
so letting them settle stops the interaction term from absorbing error that really
belongs to the single-perturbation fields.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import ConditionSampler, PerturbationData, condition_genes
from ..models.flow import integrate
from ..data.counts import recover_counts
from .coupling import sample_pairs


def _to_device(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def _aux(x: torch.Tensor, config: dict) -> dict:
    """Extra inputs a head may need. Only zinb does; the rest ignore it."""
    if config["model"]["decoder_head"] != "zinb":
        return {}
    counts, library = recover_counts(x)
    return {"counts": counts, "library": library}


@torch.no_grad()
def latent_residual_targets(vae, data: PerturbationData, sampler: ConditionSampler,
                            device: str, n_cells: int = 512) -> dict:
    """z_AB - z_A - z_B + z_ctrl for every TRAINING double, computed once.

    This is the quantity resid_R2 scores, expressed in the latent space. Nothing
    in the stage-2 loss referred to it before, which is why the model was free to
    invent a composition non-additivity 8x larger than the data's.

    Computed once rather than per step: stage 2 freezes the VAE, so these targets
    cannot move between epochs, and re-encoding the same cells every step would
    cost the largest allocation in the loop for a constant.

    A double whose singles are not both present is skipped - its residual is not
    defined - so the returned dict may be smaller than sampler.doubles.
    """
    naming = data.naming
    have = set(data.rows)

    def single_of(gene: str):
        for form in naming.single_forms(gene):
            if form in have:
                return form
        return None

    def mean_latent(condition: str) -> torch.Tensor:
        cells = data.cells(condition)
        if cells.shape[0] > n_cells:
            cells = cells[:n_cells]
        return vae.encode_z(_to_device(cells, device))[0].mean(dim=0, keepdim=True)

    control = mean_latent(data.control_condition)
    cache: dict[str, torch.Tensor] = {}
    targets: dict[str, torch.Tensor] = {}
    for condition in sampler.doubles:
        a, b = naming.genes(condition)
        sa, sb = single_of(a), single_of(b)
        if sa is None or sb is None:
            continue
        for name in (condition, sa, sb):
            if name not in cache:
                cache[name] = mean_latent(name)
        targets[condition] = (cache[condition] - cache[sa] - cache[sb] + control)
    return {"targets": targets, "control": control}


def composition_residual(field, z0: torch.Tensor, perturbations: list[int],
                         n_steps: int) -> torch.Tensor:
    """Phi_ab(z0) - Phi_a(z0) - Phi_b(z0) + z0, on a single point.

    resid_R2 is a statement about population MEANS, so integrating the mean of z0
    supervises the same quantity as integrating every cell would, at 1/batch_size
    the cost. z0 is expected to be [1, latent_dim].
    """
    a, b = perturbations
    both = integrate(field, z0, [a, b], n_steps)
    only_a = integrate(field, z0, [a], n_steps)
    only_b = integrate(field, z0, [b], n_steps)
    return both - only_a - only_b + z0


def training_rows(data: PerturbationData, conditions) -> np.ndarray:
    """Row indices of the cells a model is allowed to see.

    Stage 1 used to draw from every cell in the cache, which meant the encoder and
    the latent standardisation both read the expression profiles of the held-out
    conditions - 10,643 cells, 12.5% of the dataset. That is transductive
    representation learning, and it invalidates a zero-shot combination claim even
    though no perturbation LABEL leaks.

    Control is added explicitly: it is never held out, every prediction starts from
    it, and it is not part of `training_conditions` for the additive split.
    """
    wanted = set(conditions) | {data.control_condition}
    rows = [data.rows[c] for c in sorted(wanted) if c in data.rows]
    return np.concatenate(rows) if rows else np.arange(data.x.shape[0])


def train_stage1(vae, data: PerturbationData, config: dict, device: str,
                 rng: np.random.Generator, log,
                 rows_allowed: np.ndarray | None = None) -> None:
    train_cfg = config["train"]
    optimiser = torch.optim.AdamW(vae.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    # None keeps the old behaviour (every cell). Callers that care about leakage
    # pass the training rows; scripts/train.py does.
    pool = (np.arange(data.x.shape[0]) if rows_allowed is None
            else np.asarray(rows_allowed))
    n_cells = len(pool)
    batch_size = train_cfg["batch_size"]
    steps = max(n_cells // batch_size, 1)
    if train_cfg["max_steps_per_epoch"]:
        steps = min(steps, train_cfg["max_steps_per_epoch"])

    vae.train()
    for epoch in range(train_cfg["stage1_epochs"]):
        totals: dict[str, float] = {}
        for _ in range(steps):
            rows = pool[rng.choice(n_cells, size=batch_size, replace=False)]
            x = _to_device(data.x[rows], device)
            params, mu, logvar = vae(x)
            recon, parts = vae.loss(params, x, **_aux(x, config))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + train_cfg["kl_weight"] * kl
            parts = {**parts, "kl": float(kl)}
            # Sparsity on the learned mask correction, so "how far from the
            # annotation" stays a budget rather than a free-for-all. Only pcab
            # has a mask; every other backbone skips this.
            if hasattr(vae, "mask_penalty"):
                penalty = vae.mask_penalty()
                loss = loss + config["model"]["mask_l1"] * penalty
                parts = {**parts, "mask_l1": float(penalty)}
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), train_cfg["grad_clip"])
            optimiser.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
        summary = "  ".join(f"{k} {v / steps:.5f}" for k, v in totals.items())
        log(f"  stage1 epoch {epoch + 1:3d}/{train_cfg['stage1_epochs']}  {summary}")


def train_stage2(vae, field, data: PerturbationData, sampler: ConditionSampler,
                 config: dict, device: str, rng: np.random.Generator, log,
                 rows_allowed: np.ndarray | None = None) -> None:
    train_cfg = config["train"]
    parameters = list(field.parameters())
    if train_cfg["finetune_vae_in_stage2"]:
        parameters += list(vae.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])

    # Same pool as stage 1: the standardisation statistics are part of the encoder,
    # so reading held-out cells here leaks exactly as much as training on them.
    pool = (np.arange(data.x.shape[0]) if rows_allowed is None
            else np.asarray(rows_allowed))
    scale_rows = pool[rng.choice(len(pool), size=min(8192, len(pool)), replace=False)]
    raw_norm, mean_std = vae.fit_latent_scale(_to_device(data.x[scale_rows], device))
    log(f"  latent standardised: raw ||std|| {raw_norm:.4f} -> unit "
        f"(mean per-dim std was {mean_std:.5f})")

    resid_weight = train_cfg.get("resid_weight", 0.0)
    resid = None
    if resid_weight > 0:
        resid = latent_residual_targets(vae, data, sampler, device)
        sizes = [float(v.norm()) for v in resid["targets"].values()]
        log(f"  composition residual on: {len(sizes)} training doubles, "
            f"target ||r|| median {float(np.median(sizes)):.4f}")

    for epoch in range(train_cfg["stage2_epochs"]):
        if train_cfg["latent_renorm_every"] and epoch and                 epoch % train_cfg["latent_renorm_every"] == 0:
            # Refreshing stale statistics, at the cost of moving the coordinate
            # system the field was trained in. Off by default - see config.
            vae.fit_latent_scale(_to_device(data.x[scale_rows], device))
            log(f"  [warn] latent renormalised at epoch {epoch + 1}; the field "
                f"was fitted in the previous coordinates")
        singles_only = epoch < train_cfg["single_warmup_epochs"]
        conditions = sampler.epoch(singles_only=singles_only)
        if train_cfg["max_steps_per_epoch"]:
            conditions = conditions[:train_cfg["max_steps_per_epoch"]]
        vae.train(train_cfg["finetune_vae_in_stage2"])
        field.train()

        total, total_match, count = 0.0, 0.0, 0
        totals_resid: list[float] = []
        for condition in conditions:
            source, target, _ = sampler.batch(condition)
            perturbations = [data.pert_index[g] for g in condition_genes(condition)]

            x0 = _to_device(source, device)
            x1 = _to_device(target, device)
            if train_cfg["finetune_vae_in_stage2"]:
                z0, _ = vae.encode_z(x0)
                z1, _ = vae.encode_z(x1)
            else:
                with torch.no_grad():
                    z0, _ = vae.encode_z(x0)
                    z1, _ = vae.encode_z(x1)

            z0p, z1p = sample_pairs(z0, z1, train_cfg["coupling"], train_cfg["uot_reg"],
                                    train_cfg["uot_reg_marginal"], rng)
            # One t PER SAMPLE. Drawing a single scalar for the whole batch gives
            # the time axis one sample per step instead of `batch_size`, so [0, 1]
            # is covered sparsely and the gradient is far noisier. Inference then
            # integrates through 20 RK4 times, passing through regions the field
            # barely saw, and the error accumulates as a displacement that is too
            # small - the measured prediction was 72 % of the true delta.
            t = torch.rand(z0p.shape[0], 1, device=device)
            z_t = (1.0 - t) * z0p + t * z1p
            predicted = field(z_t, t.reshape(-1), perturbations)
            matching = torch.nn.functional.mse_loss(predicted, z1p - z0p)
            loss = matching

            # The composition residual. Only doubles have one, and only the ones
            # whose singles both exist in the data - see latent_residual_targets.
            if resid is not None and condition in resid["targets"]:
                r_model = composition_residual(field, z0.mean(dim=0, keepdim=True),
                                               perturbations, train_cfg["resid_steps"])
                r_loss = torch.nn.functional.mse_loss(r_model, resid["targets"][condition])
                loss = loss + resid_weight * r_loss
                totals_resid.append(float(r_loss))

            # Flow matching ALONE has a degenerate optimum when the encoder is
            # trainable: collapse the latent, and z1 - z0 becomes 0 so any field
            # scores perfectly. Measured before this term was added: ||z1 - z0||
            # fell to 0.019 while ||z0|| stayed ~8, and no transport happened at
            # all. Keeping the reconstruction objective on removes that escape.
            if train_cfg["finetune_vae_in_stage2"]:
                params0 = vae.decode_z(z0)
                recon, _ = vae.loss(params0, x0, **_aux(x0, config))
                loss = loss + train_cfg["stage2_recon_weight"] * recon

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, train_cfg["grad_clip"])
            optimiser.step()
            total += float(loss)
            total_match += float(matching)
            count += 1

        phase = "singles" if singles_only else "all"
        # resid is logged separately from fm so the two can be read apart: the
        # whole point of the term is that it moves a quantity fm never saw.
        extra = (f"  resid {np.mean(totals_resid):.5f}" if totals_resid else "")
        log(f"  stage2 epoch {epoch + 1:3d}/{train_cfg['stage2_epochs']}  "
            f"[{phase:7s}] loss {total / max(count, 1):.5f}  "
            f"fm {total_match / max(count, 1):.5f}{extra}")
