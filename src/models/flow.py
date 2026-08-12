"""The composed LieCFM velocity field and its integrator.

    v(z, t, S) = sum_{a in S} u_a(z,t)  +  sum_{{a,b} in S} Lambda_ab [u_a, u_b](z,t)

Four properties hold by construction rather than by training, so each is a
numerical assertion (tests/test_structure.py) instead of a hope:

  1. control invariance      v(z, t, {}) = 0            - empty sums
  2. single supervision      v(z, t, {a}) = u_a(z,t)    - no pair term exists
  3. additive recovery       Lambda = 0 => exactly additive
  4. permutation invariance  Lambda and the bracket are both antisymmetric, so
                             their product does not depend on pair ordering

Nothing is indexed by a PAIR of perturbations: the interaction is built from
per-perturbation generators alone. That is why an unseen combination is
expressible at all - there are no parameters that only a seen pair could fit.
"""

from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn

from .commutator import AntisymmetricGate, FreeInteraction, lie_bracket
from .generators import build_generator


class LieCFMField(nn.Module):
    def __init__(self, config: dict, n_perturbations: int):
        super().__init__()
        model_cfg = config["model"]
        self.latent_dim = model_cfg["latent_dim"]
        self.interaction_kind = model_cfg["interaction"]
        self.generator = build_generator(config, n_perturbations)

        if self.interaction_kind in ("none", None):  # legacy spelling
            self.interaction_kind = "additive"
        if self.interaction_kind == "commutator":
            self.gate = AntisymmetricGate(n_perturbations, model_cfg["gate_init"])
        elif self.interaction_kind == "free_mlp":
            self.free = FreeInteraction(n_perturbations, self.latent_dim)
        elif self.interaction_kind != "additive":
            raise ValueError(f"unknown interaction {self.interaction_kind!r}")

    def forward(self, z: torch.Tensor, t: torch.Tensor,
                perturbations: list[int]) -> torch.Tensor:
        if not perturbations:  # control: structurally the zero field
            return torch.zeros_like(z)

        velocity = self.generator(z, t, perturbations[0])
        for pert in perturbations[1:]:
            velocity = velocity + self.generator(z, t, pert)

        if self.interaction_kind == "additive" or len(perturbations) < 2:
            return velocity

        for a, b in combinations(perturbations, 2):
            if self.interaction_kind == "commutator":
                field_a = lambda zz, p=a: self.generator(zz, t, p)  # noqa: E731
                field_b = lambda zz, p=b: self.generator(zz, t, p)  # noqa: E731
                velocity = velocity + self.gate(a, b) * lie_bracket(field_a, field_b, z)
            else:
                velocity = velocity + self.free(z, t, a, b)
        return velocity


def integrate(field: LieCFMField, z0: torch.Tensor, perturbations: list[int],
              n_steps: int) -> torch.Tensor:
    """RK4 from t=0 to t=1. Fixed step: the fields are smooth and an adaptive
    solver would make runs non-reproducible for no measurable accuracy gain."""
    z = z0
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t0 = torch.full((1,), step * dt, device=z.device, dtype=z.dtype)
        half = t0 + 0.5 * dt
        full = t0 + dt
        k1 = field(z, t0, perturbations)
        k2 = field(z + 0.5 * dt * k1, half, perturbations)
        k3 = field(z + 0.5 * dt * k2, half, perturbations)
        k4 = field(z + dt * k3, full, perturbations)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z
