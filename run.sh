#!/bin/sh
# One experiment end to end, with every knob in one place.
#
#   sh run.sh                 build cache if missing -> train both arms -> evaluate
#   sh run.sh --eval-only     skip training, re-read what is already on disk
#   sh run.sh --no-control    target arm only (you then cannot attribute anything
#                             to the interaction term - see below)
#   sh run.sh --celleval      also score with cell-eval, ~11 min per run
#
# Edit the EDIT block. Do not edit the commands underneath it: the two arms have
# to differ in `interaction` and NOTHING ELSE, or the comparison measures the
# settings rather than the Lie bracket.
set -e

# --------------------------------------------------------------------- EDIT
TAG=pcab_lie             # run names become ${TAG}_${INTERACTION} and ${TAG}_additive

BACKBONE=pcab            # mlp | pcab | transformer | scvi
                         # pcab is P-CAB encoder + E-RCA decoder - one flag, both.
GENERATOR=neural_field   # affine | neural_field
                         # affine's commutator is the bilinear Koopman normal
                         # form, which is already published; neural_field is the
                         # one the novelty claim rests on.
INTERACTION=commutator   # additive | commutator | free_mlp

# --- gene space: these are what make our numbers comparable to scDFM ---
N_HVG=5000               # scDFM's run.sh uses --n_top_genes=5000
HVG_CRITERION=dispersion # raw_variance | dispersion
                         # raw_variance picks the largest ABSOLUTE effects, a
                         # harder gene set, and is most of why our L2 read so much
                         # worse than the published table. Measured on the same
                         # cells and conditions, 1,000 genes give Control L2 5.69
                         # under raw_variance and 3.51 under dispersion, against
                         # their 3.99. Match the criterion, not the count: raising
                         # the count alone RAISES L2, which is a sum over genes.
INFER_TOP_GENE=1000      # scDFM runs scanpy's HVG a SECOND time, over the test
                         # subset alone (--infer_top_gene=1000), and reports on
                         # those genes. 0 disables and scores the whole cache.
CACHE=assets/norman_disp5000.h5ad
                         # MUST be a new path whenever N_HVG or HVG_CRITERION
                         # changes. Overwriting assets/norman_modeled.h5ad makes
                         # every existing checkpoint unloadable - their decoders
                         # emit 3,074 genes and would meet a different count.

FOLD=1                   # scDFM's run.sh uses --fold=1; we had been on fold 0.
                         # NEITHER side runs k-fold: the shipped split file holds
                         # five independent 70/30 draws and each side picks one.
                         # Folds 0 and 1 share only 12 of their 37 test doubles,
                         # so this has to match before any number is comparable.

BATCH=256               # Applies to BOTH stages; there is only one key.
                         # Stage 2 draws one batch per condition and conditions
                         # hold 48-1,005 cells (median 262), so a batch above
                         # ~512 resamples WITH REPLACEMENT and feeds duplicate
                         # cells to the OT coupling. 256 is near the median on
                         # purpose. Raise it for stage-1 throughput knowing that.
LR=1e-3                  # constant, no scheduler. Scale it with BATCH.
STAGE1=500               # stage 1 was flat by epoch 28 at the old setting
STAGE2=1000              # stage 2 was still falling at epoch 60 - not converged
WARMUP=10                # singles-only epochs before combinations join stage 2

N_GEN=1024               # control cells transported per condition AT EVAL TIME.
                         # The old default of 256 left sampling noise in every
                         # mean-based metric while the additive baselines it is
                         # compared against pay none - they use full populations.
                         # Measured on one run: L2 4.11 at 256 vs 3.89 at 1024.

DEVICE=cuda
# ----------------------------------------------------------------- END EDIT

EVAL_ONLY=0
NO_CONTROL=0
CELLEVAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --eval-only)  EVAL_ONLY=1; shift ;;
    --no-control) NO_CONTROL=1; shift ;;
    --celleval)   CELLEVAL=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

TARGET_TAG="${TAG}_${INTERACTION}"
CONTROL_TAG="${TAG}_additive"
# When the target IS the additive arm there is nothing to control against.
if [ "$INTERACTION" = "additive" ]; then NO_CONTROL=1; fi

INFER_FLAG=""
if [ "$INFER_TOP_GENE" -gt 0 ]; then
  INFER_FLAG="--infer-top-gene $INFER_TOP_GENE"
fi

if [ "$BATCH" -gt 512 ]; then
  echo "[warn] batch=$BATCH exceeds most conditions' cell count; stage 2 will"
  echo "       resample with replacement and the OT coupling sees duplicates."
fi

# Every override both arms share. Written once so they cannot drift apart.
COMMON="data.cache_h5ad=$CACHE data.n_hvg=$N_HVG  data.hvg_criterion=$HVG_CRITERION split.fold=$FOLD \
model.backbone=$BACKBONE model.generator=$GENERATOR \
train.batch_size=$BATCH train.lr=$LR \
train.stage1_epochs=$STAGE1 train.stage2_epochs=$STAGE2 \
train.single_warmup_epochs=$WARMUP train.device=$DEVICE \
eval.n_gen_cells=$N_GEN"

echo "=== configuration ==="
echo "  backbone=$BACKBONE generator=$GENERATOR interaction=$INTERACTION"
echo "  n_hvg=$N_HVG criterion=$HVG_CRITERION fold=$FOLD cache=$CACHE"
echo "  batch=$BATCH lr=$LR stage1=$STAGE1 stage2=$STAGE2 warmup=$WARMUP"
echo "  eval n_gen_cells=$N_GEN infer_top_gene=$INFER_TOP_GENE device=$DEVICE"
echo "  runs: $TARGET_TAG$([ "$NO_CONTROL" -eq 0 ] && echo ", $CONTROL_TAG")"
echo ""

if [ "$EVAL_ONLY" -eq 0 ]; then
  # --- data ---------------------------------------------------------------
  # Rebuilt only when the cache is absent. data_prepare.py also recomputes the
  # baselines, which are a property of the gene space and so change with N_HVG;
  # they land in results/baselines_<cache stem>_<method>.json, keyed by the cache
  # name, so the 3,074-gene target line is not overwritten.
  if [ ! -f "$CACHE" ]; then
    echo "=== building $CACHE (n_hvg=$N_HVG) ==="
    python data_prepare.py --set data.n_hvg=$N_HVG data.hvg_criterion=$HVG_CRITERION data.cache_h5ad=$CACHE
    echo ""
  else
    echo "using existing cache $CACHE"
    echo ""
  fi

  # --- train --------------------------------------------------------------
  echo "=== training $TARGET_TAG ==="
  python scripts/train.py --tag "$TARGET_TAG" --set $COMMON model.interaction=$INTERACTION

  if [ "$NO_CONTROL" -eq 0 ]; then
    echo ""
    echo "=== training $CONTROL_TAG (same schedule, interaction=additive) ==="
    python scripts/train.py --tag "$CONTROL_TAG" --set $COMMON model.interaction=additive
  fi
  echo ""
fi

# --- evaluate -------------------------------------------------------------
echo "=== internal metrics + diagnostics ==="
# diagnose_gate first: if `relative` is under ~1e-3 the interaction term changes
# no prediction, and the table above it is comparing a model against itself.
sh test.sh --summary --filter "$TAG" --diagnose

if [ "$CELLEVAL" -eq 1 ]; then
  echo ""
  echo "=== cell-eval (full profile) ==="
  for d in results/runs/${TAG}_*/; do
    python scripts/run_celleval.py "$d" --profile full $INFER_FLAG
  done
fi

echo ""
echo "=== reported-table metrics ==="
python scripts/paper_table.py --filter "$TAG" --n-cells $N_GEN $INFER_FLAG \
  --csv "results/${TAG}_table.csv"

echo ""
echo "convergence check - fm must be flat over the last ~30 epochs:"
echo "  grep 'stage2 epoch' results/runs/$TARGET_TAG/train.log | tail -30"
