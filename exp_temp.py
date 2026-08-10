"""Fine alpha grid + per-model temperature for the NN-avg + LGBM blend."""
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


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


train = pd.read_csv(rf"{DATA}\train.csv")
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH, "va_emis_lgbm.npy"))
nns = [np.load(os.path.join(SCRATCH, f)) for f in
       ["va_emis_nn_big.npy", "va_emis_nn_s1.npy", "va_emis_nn_s2.npy"]]
p_nn = np.mean(nns, axis=0)
lp_lgb = np.log(np.clip(p_lgb, 1e-12, None))
lp_nn = np.log(np.clip(p_nn, 1e-12, None))

MASK = np.full((7, 7), -np.inf)
for a in range(7):
    for b in range(min(a + 1, 6) + 1):
        MASK[a, b] = 0.0


def evaluate(le):
    preds = []
    for k in range(len(va_provs)):
        preds.append(attach_from_depths(viterbi(le[offsets[k]:offsets[k + 1]], MASK, 1.0)))
    return components(preds, va_pars)


best = None
for tau_l in [0.5, 0.75, 1.0, 1.5]:
    for alpha in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        le = alpha * lp_nn + (1 - alpha) * tau_l * lp_lgb
        comp = evaluate(le)
        if best is None or comp["normalized"] > best[2]["normalized"]:
            best = (tau_l, alpha, comp)
        log(f"tau_lgb={tau_l:.2f} alpha={alpha:.2f} norm={comp['normalized']:.4f}")
log(f"BEST tau={best[0]} alpha={best[1]} -> {best[2]}")
