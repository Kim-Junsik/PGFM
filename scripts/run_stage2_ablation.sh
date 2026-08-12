#!/bin/sh
# Stage 2 verdict: does the Lie bracket contribute at all?
# Judged on resid_R2, which is mean-based and therefore unaffected by the
# decoder's variance problem. edist_rel is recorded but treated as provisional.
for GEN in neural_field affine; do
  for INT in additive commutator free_mlp; do
    python scripts/train.py --tag "s2_${GEN}_${INT}" --set \
      model.generator=$GEN model.interaction=$INT \
      model.decoder_head=hurdle model.hurdle_gate=sample \
      train.stage1_epochs=20 train.stage2_epochs=30 train.batch_size=128 \
      eval.n_gen_cells=128 > "results/runs/s2_${GEN}_${INT}.out" 2>&1 || echo "FAILED $GEN $INT"
    echo "done $GEN $INT"
  done
done
