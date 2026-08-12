"""The claims that must hold by construction, asserted numerically.

These are not accuracy tests. Each one checks a property the architecture is
supposed to guarantee without training, so a failure means the composition is
wrong rather than that the model needs more epochs.

    python -m pytest tests/test_structure.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.models.commutator import lie_bracket
from src.models.flow import LieCFMField, integrate

N_PERTURBATIONS = 6
LATENT = 8
BATCH = 16
TOLERANCE = 1e-5


def build(interaction="commutator", generator="neural_field", gate_init=0.7):
    config = config_module.load([
        f"model.interaction={interaction}",
        f"model.generator={generator}",
        f"model.gate_init={gate_init}",
        f"model.latent_dim={LATENT}",
    ])
    torch.manual_seed(0)
    field = LieCFMField(config, N_PERTURBATIONS).double()
    # The zero-initialised heads make every field identically zero, which would
    # satisfy the properties trivially. Randomise so the tests have real content.
    for parameter in field.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    return field


@pytest.fixture
def state():
    torch.manual_seed(1)
    return torch.randn(BATCH, LATENT, dtype=torch.float64), torch.rand(1, dtype=torch.float64)


# ---------------------------------------------------------------- bracket itself
def test_bracket_matches_explicit_jacobians(state):
    """jvp-based bracket equals J_b u_a - J_a u_b built from full Jacobians."""
    z, t = state
    field = build()
    z = z[:4]  # full Jacobians are only affordable on a few samples

    fa = lambda zz: field.generator(zz, t, 0)  # noqa: E731
    fb = lambda zz: field.generator(zz, t, 1)  # noqa: E731
    fast = lie_bracket(fa, fb, z)

    explicit = torch.empty_like(fast)
    for i in range(z.shape[0]):
        row = z[i:i + 1]
        ja = torch.autograd.functional.jacobian(lambda zz: fa(zz).squeeze(0), row)
        jb = torch.autograd.functional.jacobian(lambda zz: fb(zz).squeeze(0), row)
        ja, jb = ja.reshape(LATENT, LATENT), jb.reshape(LATENT, LATENT)
        explicit[i] = jb @ fa(row).squeeze(0) - ja @ fb(row).squeeze(0)

    assert torch.allclose(fast, explicit, atol=1e-8), (fast - explicit).abs().max()


def test_bracket_is_antisymmetric(state):
    z, t = state
    field = build()
    fa = lambda zz: field.generator(zz, t, 0)  # noqa: E731
    fb = lambda zz: field.generator(zz, t, 2)  # noqa: E731
    assert torch.allclose(lie_bracket(fa, fb, z), -lie_bracket(fb, fa, z), atol=1e-10)


def test_bracket_of_a_field_with_itself_vanishes(state):
    z, t = state
    field = build()
    fa = lambda zz: field.generator(zz, t, 3)  # noqa: E731
    assert lie_bracket(fa, fa, z).abs().max() < 1e-10


# ---------------------------------------------------------------- the four guarantees
@pytest.mark.parametrize("generator", ["affine", "neural_field"])
def test_control_invariance(state, generator):
    """v(z, t, {}) = 0 exactly - control is a structural fixed point."""
    z, t = state
    field = build(generator=generator)
    assert field(z, t, []).abs().max() == 0.0


@pytest.mark.parametrize("generator", ["affine", "neural_field"])
def test_single_perturbation_is_its_own_generator(state, generator):
    """v(z, t, {a}) = u_a(z, t): singles supervise the generators directly."""
    z, t = state
    field = build(generator=generator)
    assert torch.allclose(field(z, t, [4]), field.generator(z, t, 4), atol=1e-12)


@pytest.mark.parametrize("generator", ["affine", "neural_field"])
def test_zero_gate_recovers_exact_additivity(state, generator):
    """Lambda = 0 makes the composed field exactly the sum of its generators."""
    z, t = state
    field = build(generator=generator, gate_init=0.0)
    with torch.no_grad():
        field.gate.raw.zero_()
    additive = field.generator(z, t, 0) + field.generator(z, t, 1)
    assert torch.allclose(field(z, t, [0, 1]), additive, atol=1e-12)


@pytest.mark.parametrize("interaction", ["commutator", "free_mlp"])
def test_permutation_invariance(state, interaction):
    """v(z,t,{a,b}) = v(z,t,{b,a}): no ordering convention can leak in."""
    z, t = state
    field = build(interaction=interaction)
    assert torch.allclose(field(z, t, [1, 5]), field(z, t, [5, 1]), atol=TOLERANCE)


def test_gate_matrix_is_antisymmetric():
    field = build()
    matrix = field.gate.matrix()
    assert torch.allclose(matrix, -matrix.T, atol=1e-12)
    assert matrix.diagonal().abs().max() < 1e-12


# ---------------------------------------------------------------- integrator
def test_integration_of_the_zero_field_is_the_identity(state):
    """A control condition must transport nothing, at any number of steps."""
    z, t = state
    field = build()
    assert torch.allclose(integrate(field, z, [], n_steps=8), z, atol=1e-12)


def test_no_parameter_is_indexed_by_a_pair():
    """Unseen combinations are only expressible if nothing is fitted per pair.

    Every parameter must be indexed by at most ONE perturbation. The gate is the
    single exception and is checked separately: it is a scalar per ordered pair,
    so a pair never seen in training keeps its initial value and the model falls
    back to additivity rather than to an arbitrary fitted number.
    """
    field = build()
    for name, parameter in field.named_parameters():
        if name.startswith("gate."):
            continue
        assert N_PERTURBATIONS not in parameter.shape[1:], (
            f"{name} has shape {tuple(parameter.shape)}, which looks pair-indexed")
