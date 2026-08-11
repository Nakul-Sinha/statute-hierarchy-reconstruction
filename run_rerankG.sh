#!/bin/bash
# Wave G: trained K-best reranker.
#   phase 1  24 inner-CV OOF jobs (4 folds x 6 NN seeds; seed 42 also does the
#            fold LGBM), 8 concurrent x 4 threads = 32 cores.
#   phase 2  candidate features + lambdarank fit + outer-holdout adoption gate.
# Idempotent: a (fold,seed) whose .npy already exists is skipped, so the script
# can be rerun after an interruption.
set -e
cd /home/ec2-user/ch3
export CH3_DATA=/home/ec2-user/ch3
export CH3_SCRATCH=/home/ec2-user/ch3/scratch
PY=/home/ec2-user/venv/bin/python
ARCH="NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_CNG=64"
SEEDS=(42 1 2 3 7 11)
SLOTS=8
mkdir -p "$CH3_SCRATCH"

echo "=== wave G phase 1: inner-CV OOF generation start $(date) ==="
running=0
for fold in 0 1 2 3; do
  for seed in "${SEEDS[@]}"; do
    if [ -f "$CH3_SCRATCH/oof_nn_f${fold}_s${seed}.npy" ] &&
       { [ "$seed" != "42" ] || [ -f "$CH3_SCRATCH/oof_lgbm_f${fold}.npy" ]; }; then
      echo "skip f${fold} s${seed} (outputs present)"
      continue
    fi
    echo "launch f${fold} s${seed} $(date)"
    env $ARCH NN_THREADS=4 OOF_FOLD=$fold OOF_SEED=$seed \
      $PY exp_oofgen.py > oof_f${fold}_s${seed}.log 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$SLOTS" ]; then
      wait
      running=0
      echo "--- batch drained $(date) ---"
    fi
  done
done
wait
echo "=== wave G phase 1 done $(date) ==="

missing=0
for fold in 0 1 2 3; do
  for seed in "${SEEDS[@]}"; do
    f="$CH3_SCRATCH/oof_nn_f${fold}_s${seed}.npy"
    [ -f "$f" ] || { echo "MISSING $f (see oof_f${fold}_s${seed}.log)"; missing=1; }
  done
  f="$CH3_SCRATCH/oof_lgbm_f${fold}.npy"
  [ -f "$f" ] || { echo "MISSING $f (see oof_f${fold}_s42.log)"; missing=1; }
done
grep -h "best outer-holdout depth acc" oof_f*_s*.log || true
if [ "$missing" -ne 0 ]; then
  echo "=== wave G aborting: OOF outputs incomplete $(date) ==="
  exit 1
fi

for f in va_emis_nn_c2.npy va_emis_nn_s1.npy va_emis_nn_s2.npy \
         va_emis_nn_s3.npy va_emis_nn_s7.npy va_emis_nn_s11.npy \
         va_emis_lgbm_hand.npy; do
  [ -f "$CH3_SCRATCH/$f" ] || { echo "MISSING production emission $CH3_SCRATCH/$f"; exit 1; }
done

echo "=== wave G phase 2: rerank fit + holdout gate $(date) ==="
env RR_THREADS=32 $PY exp_rerank.py 2>&1 | tee exp_rerank.log
echo "=== wave G done $(date) ==="
