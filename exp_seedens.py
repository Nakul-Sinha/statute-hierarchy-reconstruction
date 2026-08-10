"""Seed-averaged NN ensemble + LGBM: final holdout evaluation."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components
from pipeline import attach_from_depths, transition_matrix, viterbi

DATA = r"G:\Datacurve\Latest_Chals\Challenge 3\dataset"
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"C:\Users\nakul\AppData\Local\Temp\claude\G--Datacurve-Latest-Chals\c252d314-4a06-4b8d-a7b1-0935a59ec986\scratchpad")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


train = pd.read_csv(rf"{DATA}\train.csv")
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH, "va_emis_lgbm.npy"))
nns = []
for name in ["va_emis_nn_big.npy", "va_emis_nn_s1.npy", "va_emis_nn_s2.npy"]:
    path = os.path.join(SCRATCH, name)
    if os.path.exists(path):
        nns.append(np.load(path))
        log(f"loaded {name}")
p_nn = np.mean(nns, axis=0)
log(f"seed-averaged {len(nns)} NNs")

lp_lgb = np.log(np.clip(p_lgb, 1e-12, None))
lp_nn = np.log(np.clip(p_nn, 1e-12, None))
MASK0 = np.full((7, 7), -np.inf)
for a in range(7):
    for b in range(min(a + 1, 6) + 1):
        MASK0[a, b] = 0.0


def decode_all(le):
    preds = []
    for k in range(len(va_provs)):
        seq = viterbi(le[offsets[k]:offsets[k + 1]], MASK0, 1.0)
        preds.append(attach_from_depths(seq))
    return preds


best = None
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    le = alpha * lp_nn + (1 - alpha) * lp_lgb
    comp = components(decode_all(le), va_pars)
    log(f"alpha={alpha:.1f} norm={comp['normalized']:.4f} pacc={comp['parent_acc']:.4f} "
        f"depth={comp['depth']:.4f} sibF1={comp['sib_f1']:.4f}")
    if best is None or comp["normalized"] > best[1]["normalized"]:
        best = (alpha, comp)
log(f"BEST alpha={best[0]} -> {best[1]}")
