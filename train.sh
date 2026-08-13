#!/bin/sh
# Train one model. Run data_prepare.py first.
#
#   sh train.sh                                     defaults
#   sh train.sh --tag run1 model.backbone=pcab      named run, config overrides
#   sh train.sh model.interaction=additive split.method=combinations
#   sh train.sh --ablation                          the stage-2 verdict sweep
#
# Any config key can be overridden by writing key.path=value; an unknown key
# raises instead of silently running with the default.
set -e

TAG=""
ABLATION=0
OVERRIDES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)       TAG="$2"; shift 2 ;;
    --ablation)  ABLATION=1; shift ;;
    *)           OVERRIDES="$OVERRIDES $1"; shift ;;
  esac
done

if [ "$ABLATION" -eq 1 ]; then
  # Does the Lie bracket contribute at all? Judged on resid_R2, which is
  # mean-based and so unaffected by the decoder's variance behaviour.
  # The backbone goes into the run name. Without it a second sweep on another
  # backbone writes into the same directories and silently replaces the first,
  # and `sh test.sh --summary` would then show only the survivor.
  BACKBONE=mlp
  case "$OVERRIDES" in
    *model.backbone=*)
      BACKBONE=$(echo "$OVERRIDES" | sed -n 's/.*model\.backbone=\([^ ]*\).*/\1/p') ;;
  esac

  echo "stage-2 ablation on backbone=$BACKBONE: generator x interaction (6 runs)"
  for GEN in neural_field affine; do
    for INT in additive commutator free_mlp; do
      echo "--- $BACKBONE / $GEN / $INT ---"
      python scripts/train.py --tag "s2_${BACKBONE}_${GEN}_${INT}" --set \
        model.generator=$GEN model.interaction=$INT $OVERRIDES
    done
  done
  echo ""
  echo "compare with:  sh test.sh --summary"
  exit 0
fi

if [ -n "$TAG" ]; then
  python scripts/train.py --tag "$TAG" ${OVERRIDES:+--set $OVERRIDES}
else
  python scripts/train.py ${OVERRIDES:+--set $OVERRIDES}
fi

echo ""
echo "next:  sh test.sh"
