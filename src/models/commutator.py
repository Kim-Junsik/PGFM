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
    """Lambda_ab weighting the bracket, factorised so it is NOT indexed by a pair.

        Lambda_ab = scale * w_a^T J w_b,     J = -J^T

    The previous version stored a full [P, P] table. It trained fine and was
    exactly wrong for this problem: measured on a finished run, |Lambda| averaged
    0.076 over the 88 training doubles and was EXACTLY 0 over the 37 evaluation
    doubles, because a pair never seen in training keeps its initial value. The
    interaction term therefore vanished at evaluation and `commutator` was
    numerically identical to `additive` on precisely the conditions the paper is
    about - the measured relative size of the term was 0.0016 on training pairs
    and 0.0000 on test pairs.

    Factorising through per-perturbation vectors fixes that: an unseen pair (a, b)
    still has w_a and w_b, so Lambda_ab is defined. This is the same argument the
    generators already obey - nothing may be indexed by a pair, or unseen
    combinations are not expressible.

    Antisymmetry survives: w_a^T J w_b is a scalar, so it equals its own
    transpose w_b^T J^T w_a = -w_b^T J w_a. Hence Lambda_ab = -Lambda_ba and
    Lambda_aa = 0, and the composed field stays permutation invariant.

    `scale` starts at `init` (0 by default) so the model still begins exactly
    additive. It is the only parameter with a gradient at that point, so it opens
    the gate first and the factors learn afterwards.
    """

    def __init__(self, n_perturbations: int, init: float = 0.0, rank: int = 16):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(init)))
        self.factors = torch.nn.Parameter(torch.randn(n_perturbations, rank) * 0.3)
        self.coupling = torch.nn.Parameter(torch.randn(rank, rank) * 0.3)

    def _antisymmetric_coupling(self) -> torch.Tensor:
        return 0.5 * (self.coupling - self.coupling.T)

    def forward(self, a: int, b: int) -> torch.Tensor:
        j = self._antisymmetric_coupling()
        return self.scale * (self.factors[a] @ j @ self.factors[b])

    def matrix(self) -> torch.Tensor:
        j = self._antisymmetric_coupling()
        return self.scale * (self.factors @ j @ self.factors.T)


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
