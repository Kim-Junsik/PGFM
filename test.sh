#!/bin/sh
# Evaluate a trained run against the target line from data_prepare.py.
#
#   sh test.sh                        the most recent run
#   sh test.sh results/runs/<tag>     a specific run
#   sh test.sh --summary              every run in one table
#   sh test.sh <run> --celleval       also score with cell-eval 0.5.42
#
# The internal metrics are already written by training; this reads them back and
# puts them next to the baselines. --celleval additionally transports control
# cells and hands the result to the separate Python 3.11 environment.
set -e

RUN=""
SUMMARY=0
CELLEVAL=0
FILTER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --summary)  SUMMARY=1; shift ;;
    --celleval) CELLEVAL=1; shift ;;
    --filter)   FILTER="$2"; shift 2 ;;
    *)          RUN="$1"; shift ;;
  esac
done

if [ "$SUMMARY" -eq 1 ]; then
  python scripts/summarise_runs.py ${FILTER:+--filter "$FILTER"}
  exit 0
fi

if [ -z "$RUN" ]; then
  RUN=$(ls -dt results/runs/*/ 2>/dev/null | head -1)
  if [ -z "$RUN" ]; then
    echo "no run found under results/runs - train one first:  sh train.sh"
    exit 1
  fi
  echo "using the most recent run: $RUN"
fi

python scripts/summarise_runs.py --run "$RUN"

if [ "$CELLEVAL" -eq 1 ]; then
  echo ""
  python scripts/run_celleval.py "$RUN" --profile minimal
fi
