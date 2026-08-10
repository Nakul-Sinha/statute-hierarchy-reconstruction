#!/bin/bash
# Screening battery: LGBM emissions regen + NN arch variants around b2
# (b2 = NN_SVD=1 emb=160 hid=256 layers=2, single-seed 0.5942 NN-only holdout).
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python

echo "=== stage0 lgbm $(date) ==="
$PY exp_lgbm.py > exp_lgbm_regen.log 2>&1

echo "=== wave1 b3(attn) b5(fl16) $(date) ==="
NN_THREADS=4 NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_ATTN=1 \
  CH3_EMIS_SUFFIX=_b3 $PY exp_nn.py > exp_b3.log 2>&1 &
NN_THREADS=4 NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_FL=16 \
  CH3_EMIS_SUFFIX=_b5 $PY exp_nn.py > exp_b5.log 2>&1 &
wait

echo "=== wave2 b4(cosine40) b6(3layer) $(date) ==="
NN_THREADS=4 NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_COS=1 \
  NN_EPOCHS=40 NN_PATIENCE=40 CH3_EMIS_SUFFIX=_b4 $PY exp_nn.py > exp_b4.log 2>&1 &
NN_THREADS=4 NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=3 \
  CH3_EMIS_SUFFIX=_b6 $PY exp_nn.py > exp_b6.log 2>&1 &
wait

echo "=== wave3 b7(bigger) b2s1(seed-noise read) $(date) ==="
NN_THREADS=4 NN_SVD=1 NN_EMB=192 NN_HID=320 NN_LAYERS=2 \
  CH3_EMIS_SUFFIX=_b7 $PY exp_nn.py > exp_b7.log 2>&1 &
NN_THREADS=4 NN_SVD=1 NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_SEED=1 \
  CH3_EMIS_SUFFIX=_b2s1 $PY exp_nn.py > exp_b2s1.log 2>&1 &
wait

echo "=== screen done $(date) ==="
grep -H "BEST lam" exp_b3.log exp_b5.log exp_b4.log exp_b6.log exp_b7.log exp_b2s1.log
