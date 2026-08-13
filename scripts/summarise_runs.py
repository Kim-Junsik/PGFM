"""Read back what training already measured and put it next to the baselines.

Nothing is recomputed here - training writes results.json, data_prepare.py writes
results/baselines_*.json, and this only joins them so a run can be read against
the line it had to beat.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HEADER = (f"{'run':34s} {'backbone':11s} {'gen':13s} {'interaction':11s} "
          f"{'resid_R2':>9s} {'edist':>8s} {'floor':>8s}")


def load_baselines() -> dict:
    """(dataset, method) -> baseline results."""
    out = {}
    for path in glob.glob("results/baselines_*.json"):
        stem = os.path.basename(path)[len("baselines_"):-len(".json")]
        dataset, _, method = stem.rpartition("_")
        out[(dataset, method)] = json.load(open(path))["results"]
    return out


def row(name: str, payload: dict) -> str:
    model = payload["config"]["model"]
    r = payload["results"]
    return (f"{name[:33]:34s} {model.get('backbone', 'mlp'):11s} "
            f"{model.get('generator', '-'):13s} {model.get('interaction', '-'):11s} "
            f"{r['resid_R2_pooled']:9.4f} {r['edist_rel']:8.4f} "
            f"{r.get('edist_rel_autoencoder_floor', float('nan')):8.4f}")


def print_target_line(baselines: dict, dataset: str, method: str) -> None:
    ridge = baselines.get((dataset, method), {}).get("ridge_additive")
    if not ridge:
        print(f"\ntarget ({dataset}, {method}): none - run data_prepare.py first")
        return
    print(f"\ntarget ({dataset}, {method}, ridge_additive):  "
          f"resid_R2 > {ridge['resid_R2_pooled']:.4f}   "
          f"edist_rel < {ridge['edist_rel']:.4f}   "
          f"(n={ridge['n_evaluated']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="a single run directory")
    parser.add_argument("--filter", default=None,
                        help="only runs whose name contains this, e.g. s2_mlp")
    args = parser.parse_args()

    baselines = load_baselines()
    if not baselines:
        print("[warn] no baselines found - run: python data_prepare.py")

    paths = ([os.path.join(args.run, "results.json")] if args.run
             else sorted(glob.glob("results/runs/*/results.json")))
    paths = [p for p in paths if os.path.exists(p)]
    if args.filter:
        paths = [p for p in paths if args.filter in os.path.basename(os.path.dirname(p))]
    if not paths:
        print("no results.json found. train first:  sh train.sh")
        return

    print(HEADER)
    print("-" * len(HEADER))
    seen = set()
    for path in paths:
        payload = json.load(open(path))
        cache = payload["config"]["data"]["cache_h5ad"]
        seen.add((os.path.splitext(os.path.basename(cache))[0],
                  payload["config"]["split"]["method"]))
        print(row(os.path.basename(os.path.dirname(path)), payload))

    for dataset, method in sorted(seen):
        print_target_line(baselines, dataset, method)

    print("\nfloor is the autoencoder with transport switched off. edist equal to"
          "\nfloor means the flow did nothing, so comparing interactions in that"
          "\nstate says nothing.")


if __name__ == "__main__":
    main()
