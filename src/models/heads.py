"""Decoder output heads.

Measured on the modelled space: 41.2 % of entries are exactly zero, while an MSE
head produces exact zeros 0.000 % of the time and loses 24 % of the standard
deviation. Since the judging metric is an energy distance between POPULATIONS,
that lost spread is paid directly - which is why the head is a config axis rather
than a fixed choice.

    mse     Gaussian point estimate. Baseline and control arm.
    hurdle  gate * magnitude, the two-part (MAST-style) decomposition: a sigmoid
            decides zero vs non-zero, a softplus gives the magnitude. Distinct
            from zero-inflation - here zeros have exactly one source.
    zinb    scVI-style zero-inflated negative binomial on recovered counts.
            Available because raw counts turn out to be exactly recoverable
            (see src/data/counts.py), but it models a different space than the
            metric, so it carries a log1p round-trip.

All heads expose `point_estimate` in log1p space, because that is where every
metric and every baseline lives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEHead(nn.Module):
    def __init__(self, width: int, n_genes: int):
        super().__init__()
        self.out = nn.Linear(width, n_genes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mean": self.out(h)}

    def loss(self, params: dict, x: torch.Tensor, **_) -> tuple[torch.Tensor, dict]:
        mse = F.mse_loss(params["mean"], x)
        return mse, {"recon": float(mse)}

    def point_estimate(self, params: dict, **_) -> torch.Tensor:
        return params["mean"]


class HurdleHead(nn.Module):
    """x = B * M, with B ~ Bernoulli(sigma) and M drawn from p(M | z).

    A VAE decoder IS a distribution p(x|z) - that is what the ELBO's
    reconstruction term is a likelihood of. Training it with plain MSE silently
    fixes p(x|z) = N(f(z), sigma^2 I) with sigma^2 CONSTANT, so the network only
    ever learns the mean and inference returns E[x|z] rather than a sample. That
    is the same mechanism that makes VAE image reconstructions blurry; here it
    shows up as collapsed cell-to-cell variance (0.098 predicted vs 0.495 real).

    Both factors therefore have to be realised, not averaged:
      B  Bernoulli sampling  - restored edist_rel 6.64 -> 1.46
      M  a learned dispersion, so the magnitude is drawn rather than pinned to
         its conditional mean. Measured gap after fixing B alone: predicted std
         0.336 vs 0.495 real, i.e. std ~0.363 of magnitude spread still missing.
    """

    def __init__(self, width: int, n_genes: int, bce_weight: float = 1.0,
                 gate_mode: str = "sample", magnitude_mode: str = "gaussian"):
        super().__init__()
        self.gate = nn.Linear(width, n_genes)
        self.magnitude = nn.Linear(width, n_genes)
        self.bce_weight = bce_weight
        self.gate_mode = gate_mode
        self.magnitude_mode = magnitude_mode
        if magnitude_mode == "gaussian":
            self.log_scale = nn.Linear(width, n_genes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        params = {"gate_logit": self.gate(h), "magnitude": F.softplus(self.magnitude(h))}
        if self.magnitude_mode == "gaussian":
            params["log_scale"] = self.log_scale(h).clamp(-6.0, 2.0)
        return params

    def loss(self, params: dict, x: torch.Tensor, **_) -> tuple[torch.Tensor, dict]:
        observed = (x > 0).float()
        gate_loss = F.binary_cross_entropy_with_logits(params["gate_logit"], observed)
        # Magnitude is only supervised where something was actually detected;
        # training it on zeros would drag every prediction toward zero and undo
        # the point of separating the two parts.
        denominator = observed.sum().clamp(min=1.0)
        residual = params["magnitude"] - x
        if self.magnitude_mode == "gaussian":
            # Gaussian NLL, which is what MSE already was - except sigma is now
            # learned instead of pinned, so the decoder keeps the spread that a
            # point estimate throws away.
            log_scale = params["log_scale"]
            nll = 0.5 * (residual / log_scale.exp()) ** 2 + log_scale
            magnitude_loss = (nll * observed).sum() / denominator
        else:
            magnitude_loss = ((residual ** 2) * observed).sum() / denominator
        total = magnitude_loss + self.bce_weight * gate_loss
        return total, {"recon": float(magnitude_loss), "gate_bce": float(gate_loss)}

    def point_estimate(self, params: dict, **_) -> torch.Tensor:
        """How the binary event is realised at inference. Three different answers:

        soft    sigma * magnitude. The conditional expectation, so it is optimal
                for anything mean-based - but it never emits an exact zero, while
                41.2 % of the real data is exactly zero.
        hard    threshold at 0.5. Emits zeros, but deterministically per gene: a
                gene with sigma = 0.4 becomes zero in EVERY cell instead of 40 %
                of them, which destroys cell-to-cell variability.
        sample  Bernoulli(sigma). Reproduces the marginal zero rate AND the
                per-gene spread, and stays unbiased for the mean. This is the one
                a population-level metric wants.
        """
        probability = torch.sigmoid(params["gate_logit"])
        if self.gate_mode == "soft":
            gate = probability
        elif self.gate_mode == "hard":
            gate = (probability > 0.5).to(probability.dtype)
        elif self.gate_mode == "sample":
            gate = torch.bernoulli(probability)
        else:
            raise ValueError(f"unknown hurdle gate_mode {self.gate_mode!r}")

        magnitude = params["magnitude"]
        # Draw the magnitude only when the gate is also being drawn: mixing a
        # sampled factor with an averaged one would give neither the right mean
        # nor the right spread. Clamped at zero because log1p values cannot be
        # negative; the resulting bias is small next to the variance recovered.
        if self.magnitude_mode == "gaussian" and self.gate_mode == "sample":
            noise = torch.randn_like(magnitude) * params["log_scale"].exp()
            magnitude = (magnitude + noise).clamp(min=0.0)
        return gate * magnitude


class ZINBHead(nn.Module):
    """scVI-style: decoder emits (scale, dropout logit); dispersion is per-gene.

    NB mean is library_size * scale, and the reported point estimate converts the
    zero-inflated mean back to log1p so it is comparable with everything else.
    """

    def __init__(self, width: int, n_genes: int, target_sum: float = 1e4):
        super().__init__()
        self.scale = nn.Linear(width, n_genes)
        self.dropout = nn.Linear(width, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        self.target_sum = target_sum

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"scale": torch.softmax(self.scale(h), dim=-1),
                "dropout_logit": self.dropout(h)}

    def loss(self, params: dict, x: torch.Tensor, counts: torch.Tensor | None = None,
             library: torch.Tensor | None = None, **_) -> tuple[torch.Tensor, dict]:
        if counts is None or library is None:
            raise ValueError("the zinb head needs counts and library sizes")
        mu = params["scale"] * library.reshape(-1, 1)
        theta = torch.exp(self.log_theta).clamp(1e-4, 1e4)
        pi = params["dropout_logit"]
        eps = 1e-8

        log_theta_mu = torch.log(theta + mu + eps)
        nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
        nb_non_zero = (
            theta * (torch.log(theta + eps) - log_theta_mu)
            + counts * (torch.log(mu + eps) - log_theta_mu)
            + torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
        )
        softplus_pi = F.softplus(-pi)
        zero_case = F.softplus(nb_zero + pi) - softplus_pi
        non_zero_case = nb_non_zero - softplus_pi
        log_likelihood = torch.where(counts < eps, zero_case, non_zero_case)
        loss = -log_likelihood.mean()
        return loss, {"recon": float(loss)}

    def point_estimate(self, params: dict, library: torch.Tensor | None = None,
                       **_) -> torch.Tensor:
        if library is None:
            library = torch.full((params["scale"].shape[0], 1), self.target_sum,
                                 device=params["scale"].device)
        mu = params["scale"] * library.reshape(-1, 1)
        expected = (1.0 - torch.sigmoid(params["dropout_logit"])) * mu
        # Back to the space the metrics live in.
        return torch.log1p(expected * self.target_sum / library.reshape(-1, 1).clamp(min=1))


def build_head(config: dict, width: int, n_genes: int) -> nn.Module:
    model_cfg = config["model"]
    kind = model_cfg["decoder_head"]
    if kind == "mse":
        return MSEHead(width, n_genes)
    if kind == "hurdle":
        # .get with the old default so checkpoints written before the magnitude
        # dispersion existed still load and keep their original behaviour.
        return HurdleHead(width, n_genes, model_cfg["hurdle_bce_weight"],
                          model_cfg.get("hurdle_gate", "soft"),
                          model_cfg.get("hurdle_magnitude", "point"))
    if kind == "zinb":
        return ZINBHead(width, n_genes)
    raise ValueError(f"unknown decoder_head {kind!r}")
