#!/bin/bash
# Blend-combo study over saved holdout emission files (no retraining).
# Each line: label | comma-separated NN emission files in /home/nakul/scratch
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python

run () {
  local label=$1 files=$2
  echo "=== combo $label: $files ==="
  CH3_NN_FILES="$files" $PY exp_blend.py 2>&1 | tail -n 8
}

run single_b2        "va_emis_nn_b2.npy"
run same2_b2_s1      "va_emis_nn_b2.npy,va_emis_nn_b2s1.npy"
run mix_b2_b3        "va_emis_nn_b2.npy,va_emis_nn_b3.npy"
run mix_b2_b4        "va_emis_nn_b2.npy,va_emis_nn_b4.npy"
run mix_b2_b5        "va_emis_nn_b2.npy,va_emis_nn_b5.npy"
run mix_b2_b6        "va_emis_nn_b2.npy,va_emis_nn_b6.npy"
run mix_b2_b7        "va_emis_nn_b2.npy,va_emis_nn_b7.npy"
run big7             "va_emis_nn_b2.npy,va_emis_nn_b2s1.npy,va_emis_nn_b3.npy,va_emis_nn_b4.npy,va_emis_nn_b5.npy,va_emis_nn_b6.npy,va_emis_nn_b7.npy"
run all8             "va_emis_nn_b1.npy,va_emis_nn_b2.npy,va_emis_nn_b2s1.npy,va_emis_nn_b3.npy,va_emis_nn_b4.npy,va_emis_nn_b5.npy,va_emis_nn_b6.npy,va_emis_nn_b7.npy"
echo "=== blend combos done ==="
