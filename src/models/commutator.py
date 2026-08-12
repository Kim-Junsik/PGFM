"""The Lie bracket of two latent vector fields, without materialising a Jacobian.

For vector fields X, Y on R^d the bracket is

    [X, Y](z) = J_Y(z) X(z) - J_X(z) Y(z)

Each term is a Jacobian-vector product, so forward-mode AD gives both in two jvp
calls at O(d) memory. Materialising J would cost O(d^2) per sample and makes the
nonlinear generator unusable at any realistic latent size - that cost is the whole
reason the nonlinear case has not been done this way before.

Antisymmetry [X, Y] = -[Y, X] is exact by construction, not learned, which is what
makes permutation invariance of the composed field a structural guarantee.
"""

from __future__ import annotations

from typing import Callable

import torch


def lie_bracket(field_a: Callable[[torch.Tensor], torch.Tensor],
                field_b: Callable[[torch.Tensor], torch.Tensor],
                z: torch.Tensor) -> torch.Tensor:
    """[a, b](z), with both fields taken as functions of z alone (t held fixed).

    The networks act row-wise on the batch, so a batched jvp yields the per-sample
    products J(z_i) v_i rather than anything mixing samples.
    """
    value_a = field_a(z)
    value_b = field_b(z)
    _, jb_va = torch.func.jvp(field_b, (z,), (value_a,))
    _, ja_vb = torch.func.jvp(field_a, (z,), (value_b,))
    return jb_va - ja_vb


class AntisymmetricGate(torch.nn.Module):
    """Lambda_ab weighting the bracket, held antisymmetric by construction.

    Lambda_ab * [u_a, u_b] is invariant under swapping a and b because both
    factors flip sign, so the composed velocity does not depend on which member of
    a pair is written first - there is no ordering convention to get wrong.
    """

    def __init__(self, n_perturbations: int, init: float = 0.0):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.full((n_perturbations, n_perturbations), init))

    def forward(self, a: int, b: int) -> torch.Tensor:
        return 0.5 * (self.raw[a, b] - self.raw[b, a])

    def matrix(self) -> torch.Tensor:
        return 0.5 * (self.raw - self.raw.T)


class FreeInteraction(torch.nn.Module):
    """Control arm: an unconstrained MLP in place of the bracket.

    If this matches the commutator, the algebraic form contributes nothing and the
    paper's second claim does not stand. Reviewers ask for exactly this, so it is
    a first-class config value (`interaction=free_mlp`), not an afterthought.
    """

    def __init__(self, n_perturbations: int, latent_dim: int, hidden: int = 256,
                 embed_dim: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_perturbations, embed_dim)
        torch.nn.init.normal_(self.embedding.weight, std=0.02)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + 2 * embed_dim + 1, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, latent_dim))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, a: int, b: int) -> torch.Tensor:
        batch = z.shape[0]
        # Symmetrised over the pair so this control keeps the same permutation
        # invariance as the commutator; otherwise it would lose on a technicality
        # rather than on the substance of the comparison.
        ea = self.embedding.weight[a].reshape(1, -1).expand(batch, -1)
        eb = self.embedding.weight[b].reshape(1, -1).expand(batch, -1)
        time = t.reshape(-1, 1).expand(batch, 1)
        forward = self.net(torch.cat([z, ea, eb, time], dim=-1))
        backward = self.net(torch.cat([z, eb, ea, time], dim=-1))
        return 0.5 * (forward + backward)
