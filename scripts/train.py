"""Stage 1 + stage 2 training, then evaluation against the baseline target line.

    python scripts/train.py
    python scripts/train.py --set model.interaction=none          # stage-1 core
    python scripts/train.py --set model.interaction=free_mlp      # control arm
    python scripts/train.py --set model.generator=affine split.method=combinations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.data.dataset import ConditionSampler, PerturbationData
from src.eval import baselines
from src.eval.predict import evaluate_model
from src.models.flow import LieCFMField
from src.models.backbones import build_backbone
from src.train.loop import train_stage1, train_stage2


def build_logger(path: str):
    handle = open(path, "a", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--tag", default=None, help="run directory name")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    method = config["split"]["method"]
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu")
        device = config["train"]["device"] = config["eval"]["device"] = "cpu"

    tag = args.tag or (f"{config['model']['generator']}_{config['model']['interaction']}"
                       f"_{method}_{time.strftime('%m%d-%H%M')}")
    run_dir = os.path.join(config["train"]["out_dir"], tag)
    os.makedirs(run_dir, exist_ok=True)
    log, handle = build_logger(os.path.join(run_dir, "train.log"))

    torch.manual_seed(config["train"]["seed"])
    rng = np.random.default_rng(config["train"]["seed"])

    log(f"run {tag}")
    log(f"config {json.dumps(config['model'])}")

    data = PerturbationData(config["data"]["cache_h5ad"])
    fold = splits.folds(config, method)[config["split"]["fold"]]
    log(f"data: {data.x.shape[0]:,} cells x {data.n_genes:,} genes, "
        f"{data.n_perturbations} perturbations")

    stats = baselines.ConditionMeans(data.x, _condition_vector(data))
    train_conditions = baselines.training_conditions(stats, fold, method)
    sampler = ConditionSampler(data, train_conditions, config["train"]["batch_size"], rng)
    log(f"train conditions: {len(sampler.singles)} singles + {len(sampler.doubles)} doubles")

    vae = build_backbone(config, data.n_genes, data.gene_names).to(device)
    field = LieCFMField(config, data.n_perturbations).to(device)
    log(f"backbone={config['model']['backbone']} head={config['model']['decoder_head']}")
    log(f"params: vae {sum(p.numel() for p in vae.parameters()):,}  "
        f"field {sum(p.numel() for p in field.parameters()):,}")

    started = time.time()
    log("\n=== stage 1: autoencoding ===")
    train_stage1(vae, data, config, device, rng, log)

    log("\n=== stage 2: latent flow matching ===")
    train_stage2(vae, field, data, sampler, config, device, rng, log)
    log(f"\ntrained in {time.time() - started:.1f}s")

    log("\n=== evaluation (same protocol as the baselines) ===")
    results = evaluate_model(vae, field, data, stats, [fold], method, config, rng)
    for key, value in results.items():
        log(f"  {key:18s} {value}")

    torch.save({"vae": vae.state_dict(), "field": field.state_dict(), "config": config},
               os.path.join(run_dir, "checkpoint.pt"))
    with open(os.path.join(run_dir, "results.json"), "w") as out:
        json.dump({"config": config, "results": results}, out, indent=2)
    log(f"-> {run_dir}")
    handle.close()


def _condition_vector(data: PerturbationData) -> np.ndarray:
    """Rebuild the per-cell condition labels from the row index."""
    labels = np.empty(data.x.shape[0], dtype=object)
    for condition, rows in data.rows.items():
        labels[rows] = condition
    return labels


if __name__ == "__main__":
    main()
