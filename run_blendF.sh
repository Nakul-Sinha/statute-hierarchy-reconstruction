#!/bin/bash
# Blend F: extended tau/alpha/lambda grids over seed subsets.
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
export CH3_LGB_FILE=va_emis_lgbm_hand.npy
PY=/home/nakul/venv/bin/python

blend () {
  echo "=== blendF $1 ==="
  CH3_NN_FILES="$2" $PY exp_blend2.py 2>&1 | grep -E "TOP|BEST|blend of"
}

C2=va_emis_nn_c2.npy
E6="$C2,va_emis_nn_s1.npy,va_emis_nn_s2.npy,va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy"
E7="$C2,va_emis_nn_s2.npy,va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy,va_emis_nn_s13.npy,va_emis_nn_s17.npy"
E8="$C2,va_emis_nn_s1.npy,va_emis_nn_s2.npy,va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy,va_emis_nn_s13.npy,va_emis_nn_s17.npy"
T4="va_emis_nn_s11.npy,va_emis_nn_s13.npy,va_emis_nn_s17.npy,va_emis_nn_s7.npy"
T5="$T4,$C2"
T6="$T5,va_emis_nn_s3.npy"

echo "=== blendF start $(date) ==="
blend e6ext "$E6"
blend e8ext "$E8"
blend e7-no-s1 "$E7"
blend top4 "$T4"
blend top5 "$T5"
blend top6 "$T6"
echo "=== blendF done $(date) ==="
