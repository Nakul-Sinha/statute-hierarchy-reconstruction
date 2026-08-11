"""Milestone 5: ensemble alpha/lambda joint grid on saved val emissions."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
from pipeline import attach_from_depths, transition_matrix, viterbi

DATA = r"G:\ml\Latest_Chals\Challenge 3\dataset"
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"D:\scratch\ch3")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


train = pd.read_csv(rf"{DATA}\train.csv")
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]

p_lgb = np.load(os.path.join(SCRATCH, "va_emis_lgbm.npy"))
p_nn = np.load(os.path.join(SCRATCH, "va_emis_nn.npy"))
assert p_lgb.shape == p_nn.shape, (p_lgb.shape, p_nn.shape)
log(f"emissions {p_lgb.shape}")

logT = transition_matrix(tr_depths)
offsets = np.cumsum([0] + [len(a) for a in va_provs])
lp_lgb = np.log(np.clip(p_lgb, 1e-12, None))
lp_nn = np.log(np.clip(p_nn, 1e-12, None))

rows = []
for alpha in np.arange(0.0, 1.01, 0.1):
    le = alpha * lp_nn + (1 - alpha) * lp_lgb
    for lam in [0.0, 0.1, 0.2, 0.4, 0.6]:
        preds = []
        for k in range(len(va_provs)):
            seq = viterbi(le[offsets[k]:offsets[k + 1]], logT, lam)
            preds.append(attach_from_depths(seq))
        comp = components(preds, va_pars)
        rows.append((alpha, lam, comp))
        log(f"alpha={alpha:.1f} lam={lam:.1f} norm={comp['normalized']:.4f} "
            f"pacc={comp['parent_acc']:.4f} depth={comp['depth']:.4f} sibF1={comp['sib_f1']:.4f}")

best = max(rows, key=lambda r: r[2]["normalized"])
log(f"BEST alpha={best[0]:.1f} lam={best[1]:.1f} -> {best[2]}")
