"""Blend evaluator: LGBM emissions + N NN emission files, extended grids.

Env: CH3_DATA, CH3_SCRATCH, CH3_NN_FILES (comma-separated .npy names in scratch).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
from pipeline import attach_from_depths, transition_matrix, viterbi

DATA = os.environ.get("CH3_DATA", r"G:\ml\Latest_Chals\Challenge 3\dataset")
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"D:\scratch\ch3")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH,
                             os.environ.get("CH3_LGB_FILE", "va_emis_lgbm.npy")))
names = os.environ.get("CH3_NN_FILES", "").split(",")
nns = [np.load(os.path.join(SCRATCH, n.strip())) for n in names if n.strip()]
p_nn = np.mean(nns, axis=0)
log(f"blend of {len(nns)} NN files: {names}")

y_tr = np.concatenate([np.asarray(d) for d in tr_depths])
prior = np.bincount(y_tr, minlength=7).astype(np.float64)
prior = (prior + 0.5) / (prior.sum() + 3.5)
log_prior = np.log(prior)
logT = transition_matrix(tr_depths)
lp_lgb = np.log(np.clip(p_lgb, 1e-12, None))
lp_nn = np.log(np.clip(p_nn, 1e-12, None))

best = None
for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    le_base = alpha * lp_nn + (1 - alpha) * lp_lgb
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4]:
        le = le_base - tau * log_prior[None, :]
        for lam in [0.0, 0.1, 0.2]:
            preds = []
            for k in range(len(va_provs)):
                preds.append(attach_from_depths(
                    viterbi(le[offsets[k]:offsets[k + 1]], logT, lam)))
            comp = components(preds, va_pars)
            if best is None or comp["normalized"] > best[3]["normalized"]:
                best = (alpha, tau, lam, comp)
    log(f"alpha={alpha:.1f} scanned; running best: a={best[0]} tau={best[1]} "
        f"lam={best[2]} norm={best[3]['normalized']:.4f}")
log(f"BEST alpha={best[0]} tau={best[1]} lam={best[2]} -> {best[3]}")
