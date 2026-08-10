#!/bin/bash
# Wave D (TF-IDF-free refinement): CNG channel + cosine schedule variants.
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python
BASE="NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_CNG=64 NN_COS=1 NN_EPOCHS=40 NN_PATIENCE=40"

while pgrep -f "[e]xp_nn.py" > /dev/null; do sleep 30; done
echo "=== waveD start $(date) ==="

env $BASE NN_THREADS=4 CH3_EMIS_SUFFIX=_d1 $PY exp_nn.py > exp_d1.log 2>&1 &
env $BASE NN_THREADS=4 NN_CNG=128 CH3_EMIS_SUFFIX=_d2 $PY exp_nn.py > exp_d2.log 2>&1 &
wait
echo "=== waveD wave2 $(date) ==="
env $BASE NN_THREADS=4 NN_FL=16 CH3_EMIS_SUFFIX=_d3 $PY exp_nn.py > exp_d3.log 2>&1 &
env $BASE NN_THREADS=4 NN_EMB=192 NN_HID=320 CH3_EMIS_SUFFIX=_d4 $PY exp_nn.py > exp_d4.log 2>&1 &
wait
echo "=== waveD done $(date) ==="
grep -H "BEST lam" exp_d1.log exp_d2.log exp_d3.log exp_d4.log
grep -H "best val depth acc" exp_d1.log exp_d2.log exp_d3.log exp_d4.log
