"""Central configuration.

Every value an experiment might vary lives here, never as a literal inside the
code. Runs record the full resolved config next to their results, and any key can
be overridden from the command line with dot notation:

    python scripts/build_data.py --set data.n_hvg=5000 data.hvg_criterion=dispersion
"""

from __future__ import annotations

import copy
import json
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data": {
        "raw_h5ad": "data/norman/norman.h5ad",
        "kegg_dir": "assets/kegg",
        "pathway_min_genes": 10,
        "pathway_max_genes": 300,
        "drop_disease_pathways": True,
        "cache_h5ad": "assets/norman_modeled.h5ad",
        # X in the source file is ALREADY log1p-normalised (values 0.305-6.405,
        # non-integer, no raw-count layer). Never re-apply normalize_total/log1p.
        "assume_prenormalised": True,
        "n_hvg": 3000,  # null selects every gene
        "hvg_criterion": "raw_variance",  # raw_variance | dispersion
        "force_include_targets": True,
        "control_label": "ctrl",
        "chunk_size": 10000,
    },
    "split": {
        # The additive folds ship WITH the dataset, so there is nothing to
        # regenerate and nothing to copy: assets/splits_additive.pkl was
        # byte-identical to this file. The combinations folds are a deterministic
        # function of the additive ones and are derived at load time rather than
        # cached, so no split artifact can drift out of sync with its source.
        "reference_pkl": "data/norman/split_results.pkl",
        "method": "additive",  # additive | combinations
        "fold": 0,
    },
    "eval": {
        # power=1 is Szekely's energy distance. power=2 collapses to
        # 2*||mean_x - mean_y||^2 and would make the metric blind to everything
        # beyond first moments.
        "edist_power": 1,
        "n_gen_cells": 256,  # control cells the predicted shift is applied to
        # PROVISIONAL. Picking this by looking at test performance would not be
        # legitimate, so it is pinned rather than tuned; select it by inner CV on
        # the training doubles before quoting a final target line.
        "ridge_alpha": 1.0,
        "ridge_weight_by_cells": False,
        "device": "cuda",
        "seed": 0,
    },
    "model": {
        "latent_dim": 64,
        # --- representation backbone (ablation axis: the dynamics claim should
        #     hold on top of any of these, not just one) ---
        "backbone": "mlp",  # mlp | transformer | scvi | pcab (stage 3)
        "hidden": [1024, 512],
        "dropout": 0.1,
        # Decoder output head. 41.2 % of the data is exactly zero and an mse head
        # produces exact zeros 0.000 % of the time, so this matters for the
        # population-level metric.
        "decoder_head": "hurdle",  # mse | hurdle | zinb
        "hurdle_bce_weight": 1.0,
        # How the binary detection event is realised at inference.
        # sample is the right default for a distribution-level metric; soft is
        # optimal for mean-only metrics. See HurdleHead.point_estimate.
        "hurdle_gate": "sample",  # soft | hard | sample
        # point pins the magnitude to its conditional mean (what plain MSE does);
        # gaussian learns a dispersion so the magnitude can be drawn as well.
        "hurdle_magnitude": "gaussian",  # point | gaussian
        # --- transformer backbone ---
        "transformer_width": 128,
        "transformer_tokens": 32,
        "transformer_heads": 4,
        "transformer_layers": 2,
        # --- LieCFM latent dynamics ---
        "generator": "neural_field",  # affine | neural_field
        "generator_hidden": [256, 256],
        "interaction": "commutator",  # additive | commutator | free_mlp
        "gate_init": 0.0,  # antisymmetric gate Lambda, 0 = start exactly additive
        "time_embed_dim": 32,
        # --- P-CAB mask (stage 3) ---
        "n_pathway_tokens": None,  # None = however many KEGG pathways survive
        "n_free_tokens": 101,
        "d_key": 64,
        "d_value": 64,
        "mask_combine": "gate",  # gate (multiplicative) | logit_bias (control)
        "mask_mode": "hybrid",  # hybrid | prior_only | residual_only
        "mask_activation": "tanh",  # tanh (signed) | sigmoid (unsigned control)
        "mask_alpha": 1.0,
        "mask_share_enc_dec": True,
        "mask_l1": 1e-5,
        "mask_self_loop": True,
    },
    "train": {
        "stage1_epochs": 30,
        "stage2_epochs": 60,
        "batch_size": 256,
        # 0 = a full pass over the data. Set it low for smoke runs so a
        # configuration can be checked end-to-end in seconds.
        "max_steps_per_epoch": 0,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "kl_weight": 1e-3,
        "finetune_encoder_in_stage2": True,
        # Weight on the reconstruction term kept alive during stage 2. Without it
        # the encoder collapses the latent, which is the global optimum of flow
        # matching on its own.
        "stage2_recon_weight": 1.0,
        "single_warmup_epochs": 10,  # singles only before combinations join
        # --- minibatch OT coupling; never random pairing ---
        "coupling": "uot",  # uot | ot | random (random is a control only)
        "uot_reg": 0.05,
        "uot_reg_marginal": 1.0,
        # 0 = fit the latent standardisation once before stage 2 and keep it.
        "latent_renorm_every": 10,
        "n_integration_steps": 20,
        "grad_clip": 1.0,
        "device": "cuda",
        "seed": 0,
        "out_dir": "results/runs",
    },
}


def _coerce(text: str) -> Any:
    """Turn a command-line string into the value it obviously denotes."""
    lowered = text.lower()
    # Only "null" spells None. "none" stays a string, because it is a legitimate
    # value for at least one key (model.interaction) and silently turning it into
    # None made that config crash after the run had already started.
    if lowered == "null":
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply `a.b=value` strings onto a copy of `config`.

    Unknown keys raise instead of being silently created, so a typo in a sweep
    script fails loudly rather than running with the default.
    """
    resolved = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        path, raw = item.split("=", 1)
        node = resolved
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in node:
                raise KeyError(f"unknown config section {path!r}")
            node = node[key]
        if keys[-1] not in node:
            raise KeyError(f"unknown config key {path!r}")
        node[keys[-1]] = _coerce(raw)
    return resolved


def load(overrides: list[str] | None = None) -> dict:
    return apply_overrides(DEFAULTS, overrides)


def dumps(config: dict) -> str:
    return json.dumps(config, indent=2, sort_keys=True)
