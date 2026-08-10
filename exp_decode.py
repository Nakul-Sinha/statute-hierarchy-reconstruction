"""Decode comparison: MAP (sum log p) vs expected-accuracy (sum p) Viterbi,
on saved val emissions; includes ensemble grid over alpha for both."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components
from pipeline import attach_from_depths, viterbi

DATA = r"G:\Datacurve\Latest_Chals\Challenge 3\dataset"
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"C:\Users\nakul\AppData\Local\Temp\claude\G--Datacurve-Latest-Chals\c252d314-4a06-4b8d-a7b1-0935a59ec986\scratchpad")
SEED = 42
t0 = time.time()
N_STATES = 7


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


MASK0 = np.full((N_STATES, N_STATES), -np.inf)
for a in range(N_STATES):
    for b in range(min(a + 1, N_STATES - 1) + 1):
        MASK0[a, b] = 0.0

train = pd.read_csv(rf"{DATA}\train.csv")
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH, "va_emis_lgbm.npy"))
p_nn = np.load(os.path.join(SCRATCH, "va_emis_nn.npy"))


def decode_all(emis_flat):
    preds = []
    for k in range(len(va_provs)):
        seq = viterbi(emis_flat[offsets[k]:offsets[k + 1]], MASK0, 1.0)
        preds.append(attach_from_depths(seq))
    return preds


for alpha in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]:
    p_mix_log = np.exp(alpha * np.log(np.clip(p_nn, 1e-12, None))
                       + (1 - alpha) * np.log(np.clip(p_lgb, 1e-12, None)))
    p_mix_lin = alpha * p_nn + (1 - alpha) * p_lgb
    for name, emis in [
        ("MAP-logmix ", np.log(np.clip(p_mix_log, 1e-12, None))),
        ("EXP-logmix ", p_mix_log),
        ("EXP-linmix ", p_mix_lin),
    ]:
        comp = components(decode_all(emis), va_pars)
        log(f"alpha={alpha:.1f} {name} norm={comp['normalized']:.4f} "
            f"pacc={comp['parent_acc']:.4f} depth={comp['depth']:.4f} "
            f"sibF1={comp['sib_f1']:.4f}")
