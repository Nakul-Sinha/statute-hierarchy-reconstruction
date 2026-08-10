#!/bin/bash
# Farm E: 7 more seeds of the c2 config (TF-IDF-free: CNG64, emb160/hid256,
# plateau early-stop) + ensemble-size blend study vs hand-only LGBM.
cd /home/nakul
export CH3_DATA=/home/nakul
export CH3_SCRATCH=/home/nakul/scratch
PY=/home/nakul/venv/bin/python
ARCH="NN_EMB=160 NN_HID=256 NN_LAYERS=2 NN_CNG=64"
SEEDS=(1 2 3 7 11 13 17)

while pgrep -f "[e]xp_nn.py" > /dev/null; do sleep 30; done
echo "=== farmE start $(date) ==="
for ((i = 0; i < ${#SEEDS[@]}; i += 2)); do
  s1=${SEEDS[i]}; s2=${SEEDS[i+1]:-}
  echo "=== farm pair $s1 ${s2:-none} $(date) ==="
  env $ARCH NN_THREADS=4 NN_SEED=$s1 CH3_EMIS_SUFFIX=_s$s1 \
    $PY exp_nn.py > exp_s$s1.log 2>&1 &
  if [ -n "$s2" ]; then
    env $ARCH NN_THREADS=4 NN_SEED=$s2 CH3_EMIS_SUFFIX=_s$s2 \
      $PY exp_nn.py > exp_s$s2.log 2>&1 &
  fi
  wait
done
echo "=== farmE seeds done $(date) ==="
grep -H "best val depth acc" exp_s1.log exp_s2.log exp_s3.log exp_s7.log \
  exp_s11.log exp_s13.log exp_s17.log

export CH3_LGB_FILE=va_emis_lgbm_hand.npy
blend () {
  echo "=== blend $1 ==="
  CH3_NN_FILES="$2" $PY exp_blend.py 2>&1 | tail -n 3
}
E8="va_emis_nn_c2.npy,va_emis_nn_s1.npy,va_emis_nn_s2.npy,va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy,va_emis_nn_s13.npy,va_emis_nn_s17.npy"
E6="va_emis_nn_c2.npy,va_emis_nn_s1.npy,va_emis_nn_s2.npy,va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy"
E4="va_emis_nn_c2.npy,va_emis_nn_s1.npy,va_emis_nn_s2.npy,va_emis_nn_s3.npy"
blend e1 "va_emis_nn_c2.npy"
blend e4 "$E4"
blend e6 "$E6"
blend e8 "$E8"
echo "=== farmE done $(date) ==="
