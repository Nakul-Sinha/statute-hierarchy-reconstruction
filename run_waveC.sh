#!/bin/bash
# Wave C (TF-IDF-free): hand-only LGBM emissions + scaled arch without SVD
# input (c1, the scaling-vs-SVD decomposition) + c1 with hashed char-ngram
# EmbeddingBag channel (c2). Waits for any running exp_nn jobs to finish.
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python

while pgrep -f "[e]xp_nn.py" > /dev/null; do sleep 30; done
echo "=== waveC start $(date) ==="

LGBM_TFIDF=0 CH3_LGB_OUT=va_emis_lgbm_hand.npy \
  $PY exp_lgbm.py > exp_lgbm_hand.log 2>&1
echo "=== lgbm_hand done $(date) ==="

NN_THREADS=4 NN_EMB=160 NN_HID=256 NN_LAYERS=2 \
  CH3_EMIS_SUFFIX=_c1 $PY exp_nn.py > exp_c1.log 2>&1 &
NN_THREADS=4 NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_CNG=64 \
  CH3_EMIS_SUFFIX=_c2 $PY exp_nn.py > exp_c2.log 2>&1 &
wait
echo "=== waveC done $(date) ==="
grep -H "BEST lam" exp_c1.log exp_c2.log
