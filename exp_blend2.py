"""Blend evaluator v2: env-configurable grids, reports top-5 combos.

Env: CH3_DATA, CH3_SCRATCH, CH3_NN_FILES, CH3_LGB_FILE,
     CH3_ALPHAS, CH3_TAUS, CH3_LAMS (comma-separated floats).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
from pipeline import attach_from_depths, transition_matrix, viterbi

DATA = os.environ.get("CH3_DATA", "/home/nakul")
SCRATCH = os.environ.get("CH3_SCRATCH", "/home/nakul/scratch")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def envgrid(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",")]


ALPHAS = envgrid("CH3_ALPHAS", "0.6,0.7,0.75,0.8,0.85,0.9,0.95,1.0")
TAUS = envgrid("CH3_TAUS", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
LAMS = envgrid("CH3_LAMS", "0.0,0.05,0.1,0.15,0.2")

train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH,
                             os.environ.get("CH3_LGB_FILE", "va_emis_lgbm_hand.npy")))
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

results = []
for alpha in ALPHAS:
    le_base = alpha * lp_nn + (1 - alpha) * lp_lgb
    for tau in TAUS:
        le = le_base - tau * log_prior[None, :]
        for lam in LAMS:
            preds = []
            for k in range(len(va_provs)):
                preds.append(attach_from_depths(
                    viterbi(le[offsets[k]:offsets[k + 1]], logT, lam)))
            comp = components(preds, va_pars)
            results.append((comp["normalized"], alpha, tau, lam, comp))
    rb = max(results)
    log(f"alpha={alpha:.2f} scanned; running best: a={rb[1]} tau={rb[2]} "
        f"lam={rb[3]} norm={rb[0]:.4f}")
results.sort(key=lambda r: -r[0])
for norm, alpha, tau, lam, comp in results[:5]:
    log(f"TOP a={alpha:.2f} tau={tau:.2f} lam={lam:.2f} norm={norm:.4f} "
        f"par={comp['parent_acc']:.4f} dep={comp['depth']:.4f} "
        f"sib={comp['sib_f1']:.4f}")
n, a, t, l, c = results[0]
log(f"BEST alpha={a} tau={t} lam={l} -> {c}")
