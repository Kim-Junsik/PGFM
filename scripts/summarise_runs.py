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

HEADER = (f"{'run':30s} {'backbone':11s} {'gen':13s} {'interaction':11s} "
          f"{'resid_R2':>9s} {'edist':>8s} {'floor':>8s}")


def load_baselines() -> dict:
    out = {}
    for path in glob.glob("results/baselines_*.json"):
        method = os.path.basename(path)[len("baselines_"):-len(".json")]
        out[method] = json.load(open(path))["results"]
    return out


def row(name: str, payload: dict) -> str:
    model = payload["config"]["model"]
    r = payload["results"]
    return (f"{name[:29]:30s} {model.get('backbone', 'mlp'):11s} "
            f"{model.get('generator', '-'):13s} {model.get('interaction', '-'):11s} "
            f"{r['resid_R2_pooled']:9.4f} {r['edist_rel']:8.4f} "
            f"{r.get('edist_rel_autoencoder_floor', float('nan')):8.4f}")


def print_target_line(baselines: dict, method: str) -> None:
    if method not in baselines:
        return
    ridge = baselines[method].get("ridge_additive", {})
    if not ridge:
        return
    print(f"\n목표선 ({method}, ridge_additive):  "
          f"resid_R2 > {ridge['resid_R2_pooled']:.4f}   "
          f"edist_rel < {ridge['edist_rel']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="a single run directory")
    args = parser.parse_args()

    baselines = load_baselines()
    if not baselines:
        print("[warn] no baselines found - run: python data_prepare.py")

    paths = ([os.path.join(args.run, "results.json")] if args.run
             else sorted(glob.glob("results/runs/*/results.json")))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("no results.json found. train first:  sh train.sh")
        return

    print(HEADER)
    print("-" * len(HEADER))
    methods = set()
    for path in paths:
        payload = json.load(open(path))
        methods.add(payload["config"]["split"]["method"])
        print(row(os.path.basename(os.path.dirname(path)), payload))

    for method in sorted(methods):
        print_target_line(baselines, method)

    print("\nfloor 는 수송을 끈 오토인코더 성능이다. edist 가 floor 와 같으면 flow 가"
          "\n아무 일도 하지 않은 것이므로, 그 상태의 interaction 비교는 의미가 없다.")


if __name__ == "__main__":
    main()
