"""Audit item 5: per-class prior adjustment (le - tau*log prior) on holdout."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
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
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

prior = np.bincount(np.concatenate([np.asarray(d) for d in tr_depths]), minlength=7)
prior = (prior + 0.5) / (prior.sum() + 3.5)
log(f"train-split depth prior: {np.round(prior, 4)}")

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

for alpha in [0.7, 0.8]:
    le0 = alpha * lp_nn + (1 - alpha) * lp_lgb
    for tau in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75]:
        le = le0 - tau * np.log(prior)[None, :]
        preds = []
        deep = 0
        for k in range(len(va_provs)):
            seq = viterbi(le[offsets[k]:offsets[k + 1]], MASK, 1.0)
            deep += sum(1 for d in seq if d >= 3)
            preds.append(attach_from_depths(seq))
        comp = components(preds, va_pars)
        log(f"alpha={alpha:.1f} tau={tau:.2f} norm={comp['normalized']:.4f} "
            f"pacc={comp['parent_acc']:.4f} depth={comp['depth']:.4f} "
            f"sibF1={comp['sib_f1']:.4f} d3plus={deep/offsets[-1]:.4f}")
