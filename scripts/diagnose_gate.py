"""Did the interaction term ever open, and does it change the velocity?

    python scripts/diagnose_gate.py                     every s2_* run
    python scripts/diagnose_gate.py results/runs/foo    one run
    python scripts/diagnose_gate.py --device cuda       once the sweep is done

Without this, `commutator` scoring the same as `additive` is unreadable: it can
mean the Lie bracket contributes nothing, which is a result, or that the gate never
opened, which is a missing measurement. Only the first is a finding.

The number that decides it is `relative` - the size of the interaction term against
the additive part it corrects, along the trajectory RK4 actually walks. The failure
this exists to catch was measured at 0.0016 on training pairs and 0.0000 on test
pairs, i.e. the term was numerically absent exactly where the paper's claim lives.

Reads checkpoint.pt only, on cpu by default, so it does not disturb a running sweep.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import (condition_groups, load_run, measure_interaction,
                                  subsample)

HEADER = (f"{'run':32s} {'group':14s} {'n':>4s} {'mean|lambda|':>13s} "
          f"{'zero%':>7s} {'relative':>10s} {'add_norm':>10s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", default=None, help="run directories")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=128,
                        help="control cells per condition; the path is walked for each")
    parser.add_argument("--max-per-group", type=int, default=40,
                        help="cap per group; 0 for all")
    args = parser.parse_args()

    runs = args.runs or sorted(glob.glob("results/runs/s2_*"))
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        print("no finished run found (checkpoint.pt missing). still training?")
        return

    print(HEADER)
    print("-" * len(HEADER))
    for run_dir in runs:
        config, data, stats, fold, vae, field = load_run(run_dir, args.device)
        name = os.path.basename(run_dir.rstrip("/\\"))
        if field.interaction_kind == "additive":
            print(f"{name[:31]:32s} {'-':14s} {'-':>4s} "
                  f"{'additive arm: no interaction term by construction':>0s}")
            continue

        rng = np.random.default_rng(config["eval"]["seed"])
        groups = condition_groups(data, stats, fold, config["split"]["method"])
        scale = getattr(getattr(field, "gate", None), "scale", None)

        for group in ("train doubles", "test doubles"):
            conditions = subsample(groups[group], args.max_per_group, rng)
            if not conditions:
                continue
            rows = measure_interaction(vae, field, data, conditions, config, rng,
                                       args.device, args.n_cells)
            if not rows:
                continue
            lambdas = np.abs([r["lambda"] for r in rows])
            if np.all(np.isnan(lambdas)):
                # free_mlp has no scalar coefficient - only the term it produces
                # is defined - so these two columns are blank rather than nan.
                lambda_cell, zero_cell = f"{'-':>13s}", f"{'-':>7s}"
            else:
                # Exact zeros are the signature of the old pair-indexed table, where
                # an unseen pair kept its initial value. Counted separately from a
                # small mean: "tiny" and "structurally absent" are different bugs.
                lambda_cell = f"{float(np.nanmean(lambdas)):13.6f}"
                zero_cell = f"{float(np.mean(np.isclose(lambdas, 0.0))) * 100:6.1f}%"
            print(f"{name[:31]:32s} {group:14s} {len(rows):4d} "
                  f"{lambda_cell} {zero_cell} "
                  f"{float(np.nanmedian([r['relative'] for r in rows])):10.6f} "
                  f"{float(np.nanmedian([r['additive_norm'] for r in rows])):10.4f}")
            name = ""  # only label the first row of each run

        if scale is not None:
            print(f"{'':32s} {'gate.scale':14s} {float(scale):>10.6f}"
                  f"   (init {config['model']['gate_init']}; unchanged means "
                  f"the gate never opened)")

    print("\nrelative = ||interaction term|| / ||additive term||, median over"
          "\nconditions, measured at the points RK4 visits. Below ~1e-3 the term"
          "\nchanges no prediction, so `commutator` and `additive` are the same"
          "\nmodel and their scores cannot be compared as an ablation. Compare the"
          "\ntrain and test rows: a term that is present in training and absent in"
          "\ntest is the pair-indexing failure, not a null result.")


if __name__ == "__main__":
    main()
