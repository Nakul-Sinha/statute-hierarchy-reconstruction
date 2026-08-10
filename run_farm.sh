#!/bin/bash
# Seed farm for the chosen NN config (default: b2 arch = SVD-input emb160/hid256).
# Runs pairs in parallel at 4 threads each; emissions land in scratch as
# va_emis_nn_s<seed>.npy. b2 (seed 42) and b2s1 (seed 1) already exist.
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python
ARCH="NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=2"
SEEDS=(2 3 7 11 13 17)

for ((i = 0; i < ${#SEEDS[@]}; i += 2)); do
  s1=${SEEDS[i]}; s2=${SEEDS[i+1]:-}
  echo "=== farm pair $s1 $s2 $(date) ==="
  env $ARCH NN_THREADS=4 NN_SEED=$s1 CH3_EMIS_SUFFIX=_s$s1 \
    $PY exp_nn.py > exp_s$s1.log 2>&1 &
  if [ -n "$s2" ]; then
    env $ARCH NN_THREADS=4 NN_SEED=$s2 CH3_EMIS_SUFFIX=_s$s2 \
      $PY exp_nn.py > exp_s$s2.log 2>&1 &
  fi
  wait
done
echo "=== farm done $(date) ==="
grep -H "best val depth acc" exp_s*.log
