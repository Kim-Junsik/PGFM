"""Score a trained run with cell-eval 0.5.42.

Runs in two steps because cell-eval needs its own interpreter:

  1. this environment  - load the checkpoint, transport control cells, write
                         pred.h5ad / real.h5ad
  2. .env-celleval     - run MetricsEvaluator on those two files

    python scripts/run_celleval.py results/runs/<tag>
    python scripts/run_celleval.py results/runs/<tag> --profile vcc
    python scripts/run_celleval.py results/runs/<tag> --export-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import splits
from src.data.conventions import ConditionNaming
from src.data.dataset import PerturbationData
from src.eval.celleval import CONTROL_LABEL, PERT_COL, export
from src.models.backbones import build_backbone
from src.models.flow import LieCFMField

# Absolute and OS-normalised: CreateProcess on Windows does not resolve a
# relative forward-slash path even when os.path.exists accepts it.
CELLEVAL_PYTHON = os.path.normpath(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env-celleval", "python.exe")))

# Kept as a string and run by the other interpreter; importing cell_eval here
# would fail, which is the whole reason for the split.
#
# The __main__ guard is REQUIRED, not stylistic. cell-eval parallelises with
# multiprocessing, and on Windows (spawn start method) each child re-imports the
# main module - without the guard every child re-runs the whole script and spawns
# more children, so the run never terminates and never errors either. That is what
# made the first attempts appear to hang.
SCORING_SCRIPT = '''
import sys, json
import multiprocessing


def main():
    from cell_eval import MetricsEvaluator
    pred, real, outdir, profile, control, pert_col = sys.argv[1:7]
    evaluator = MetricsEvaluator(
        adata_pred=pred, adata_real=real,
        control_pert=control, pert_col=pert_col,
        outdir=outdir, allow_discrete=False, num_threads=1)
    results, agg = evaluator.compute(profile=profile, break_on_error=False)
    print(json.dumps({"n_rows": int(results.height),
                      "columns": results.columns[:20]}))
    print(results.head(20))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="a directory holding checkpoint.pt")
    parser.add_argument("--profile", default="minimal",
                        choices=["full", "minimal", "vcc", "de", "anndata"])
    parser.add_argument("--export-only", action="store_true",
                        help="write the h5ad pair and stop")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="cap cells per condition; cell-eval runs a DE test per "
                             "condition, so the full export is slow to score")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.run_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"no checkpoint at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = config["train"]["device"] = config["eval"]["device"] = "cpu"

    print(f"loading data ({config['data']['cache_h5ad']}) ...")
    data = PerturbationData(config["data"]["cache_h5ad"],
                            naming=ConditionNaming.from_config(config))
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]

    vae = build_backbone(config, data.n_genes, data.gene_names).to(device)
    vae.load_state_dict(checkpoint["vae"])
    field = LieCFMField(config, data.n_perturbations).to(device)
    field.load_state_dict(checkpoint["field"])

    out_dir = os.path.join(args.run_dir, "celleval")
    rng = np.random.default_rng(config["eval"]["seed"])
    print("transporting control cells for every test condition ...")
    paths = export(vae, field, data, fold, config, out_dir, rng, args.max_cells)
    print(f"  wrote {paths['pred']} and {paths['real']}  "
          f"({paths['n_cells']} cells, {paths['n_conditions']} conditions)")

    if args.export_only:
        return

    if not os.path.exists(CELLEVAL_PYTHON):
        raise SystemExit(
            f"{CELLEVAL_PYTHON} not found - run with --export-only and score the "
            f"two h5ad files from wherever cell-eval is installed.")

    script_path = os.path.normpath(os.path.abspath(os.path.join(out_dir, "_score.py")))
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(SCORING_SCRIPT)

    print(f"\nscoring with cell-eval (profile={args.profile}) ...")
    completed = subprocess.run(
        [CELLEVAL_PYTHON, script_path,
         os.path.abspath(paths["pred"]), os.path.abspath(paths["real"]),
         os.path.abspath(out_dir), args.profile, CONTROL_LABEL, PERT_COL],
        capture_output=True, text=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"cell-eval failed (exit {completed.returncode})")

    with open(os.path.join(args.run_dir, "celleval_summary.json"), "w") as handle:
        json.dump({"profile": args.profile, "outdir": out_dir}, handle, indent=2)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
