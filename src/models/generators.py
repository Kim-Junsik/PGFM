"""Per-perturbation generators u_a(z, t) acting on the latent space.

Two families, kept as a config axis rather than a decision:

  affine        u_a(z,t) = A_a z + b_a. Its commutator is the bilinear Koopman
                normal form, which is already known - so if affine wins, the
                honest finding is that higher-order dynamics are unnecessary.
  neural_field  a shared trunk conditioned on a learned per-perturbation
                embedding. The novel part is taking the Lie bracket of NONLINEAR
                fields via forward-mode AD, which is what commutator.py does.

Both expose the same call signature so the interaction term never has to know
which one it is holding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal features so the field can vary along the transport path."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time_embed_dim must be even")
        self.dim = dim
        half = dim // 2
        self.register_buffer("frequencies", torch.exp(
            torch.linspace(0.0, 6.0, half)), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.reshape(-1, 1) * self.frequencies.reshape(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class AffineGenerator(nn.Module):
    """u_a(z,t) = A_a(t) z + b_a(t), with the time dependence factored out."""

    def __init__(self, n_perturbations: int, latent_dim: int, time_embed_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time = TimeEmbedding(time_embed_dim)
        # Start near zero: at initialisation every generator is the null field,
        # so the model begins as "predict no change" rather than as noise.
        self.a = nn.Parameter(torch.zeros(n_perturbations, latent_dim, latent_dim))
        self.b = nn.Parameter(torch.zeros(n_perturbations, latent_dim))
        self.time_scale = nn.Sequential(nn.Linear(time_embed_dim, 32), nn.GELU(),
                                        nn.Linear(32, 1))

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        scale = 1.0 + self.time_scale(self.time(t))
        return scale * (z @ self.a[pert].T + self.b[pert])


class NeuralFieldGenerator(nn.Module):
    """u_a(z,t) = f([z, phi(t), e_a]) with f shared and e_a learned per perturbation."""

    def __init__(self, n_perturbations: int, latent_dim: int, hidden: list[int],
                 time_embed_dim: int, embed_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.time = TimeEmbedding(time_embed_dim)
        self.embedding = nn.Embedding(n_perturbations, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        sizes = [latent_dim + time_embed_dim + embed_dim, *hidden]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(sizes[-1], latent_dim)
        # Zero-initialised head: the field starts as exactly zero, matching the
        # affine generator's "no change at init" behaviour.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor, pert: int) -> torch.Tensor:
        batch = z.shape[0]
        time_features = self.time(t).expand(batch, -1)
        embedding = self.embedding.weight[pert].reshape(1, -1).expand(batch, -1)
        return self.head(self.trunk(torch.cat([z, time_features, embedding], dim=-1)))


def build_generator(config: dict, n_perturbations: int) -> nn.Module:
    model_cfg = config["model"]
    kind = model_cfg["generator"]
    if kind == "affine":
        return AffineGenerator(n_perturbations, model_cfg["latent_dim"],
                               model_cfg["time_embed_dim"])
    if kind == "neural_field":
        return NeuralFieldGenerator(n_perturbations, model_cfg["latent_dim"],
                                    model_cfg["generator_hidden"],
                                    model_cfg["time_embed_dim"])
    raise ValueError(f"unknown generator {kind!r}")
