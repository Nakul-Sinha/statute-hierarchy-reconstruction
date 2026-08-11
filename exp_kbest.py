"""K-best Viterbi study: oracle ceiling + lightweight tree-level reranking.

Env: CH3_DATA, CH3_SCRATCH, CH3_NN_FILES (comma-separated NN emission .npy in
scratch), KB_ALPHA, KB_TAU, KB_LAM, KB_K.

Outputs: top-1 baseline, oracle-over-K ceiling (both on the 15% act holdout),
then a small fixed-grid rerank study over tree-level penalty features.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import act_score, components, depths_from_parents
from pipeline import attach_from_depths, transition_matrix

DATA = os.environ.get("CH3_DATA", r"G:\ml\Latest_Chals\Challenge 3\dataset")
SCRATCH = os.environ.get("CH3_SCRATCH", ".")
SEED = 42
K = int(os.environ.get("KB_K", "16"))
ALPHA = float(os.environ.get("KB_ALPHA", "0.8"))
TAU = float(os.environ.get("KB_TAU", "0.3"))
LAM = float(os.environ.get("KB_LAM", "0.0"))
N_STATES = 7
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def kbest_viterbi(log_emis, logT, lam, k):
    """Returns list of (score, depth_seq) sorted desc; constrained d0=0,
    d[i] <= d[i-1]+1. Beam is exact k-best (per-state k lists)."""
    n = log_emis.shape[0]
    finite = np.isfinite(logT)
    trans = np.where(finite, lam * np.where(finite, logT, 0.0), -np.inf)
    # dp[s] = list of (score, path_tuple)
    dp = [[] for _ in range(N_STATES)]
    dp[0] = [(log_emis[0, 0], (0,))]
    for i in range(1, n):
        ndp = [[] for _ in range(N_STATES)]
        for b in range(N_STATES):
            cands = []
            for a in range(N_STATES):
                if not np.isfinite(trans[a, b]) or not dp[a]:
                    continue
                for (sc, path) in dp[a]:
                    cands.append((sc + trans[a, b] + log_emis[i, b], path + (b,)))
            cands.sort(key=lambda x: -x[0])
            ndp[b] = cands[:k]
        dp = ndp
    allc = [c for b in range(N_STATES) for c in dp[b]]
    allc.sort(key=lambda x: -x[0])
    return allc[:k]


def tree_feats(seq, prior):
    """Tree-level features of a candidate depth sequence."""
    seq = np.asarray(seq)
    n = len(seq)
    hist = np.bincount(seq, minlength=N_STATES) / n
    # KL(candidate hist || train prior)
    kl = float(np.sum(np.where(hist > 0, hist * (np.log(hist + 1e-12) -
                                                 np.log(prior)), 0.0)))
    deep = float((seq >= 3).mean())
    # sibling run stats: mean run length of constant depth
    runs, cur = [], 1
    for i in range(1, n):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return kl, deep, float(np.mean(runs))


train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH,
                             os.environ.get("CH3_LGB_FILE", "va_emis_lgbm.npy")))
names = [n.strip() for n in os.environ.get("CH3_NN_FILES", "").split(",") if n.strip()]
p_nn = np.mean([np.load(os.path.join(SCRATCH, n)) for n in names], axis=0)
log(f"NN files: {names}")

y_tr = np.concatenate([np.asarray(d) for d in tr_depths])
prior = np.bincount(y_tr, minlength=N_STATES).astype(np.float64)
prior = (prior + 0.5) / (prior.sum() + 0.5 * N_STATES)
logT = transition_matrix(tr_depths)
le = (ALPHA * np.log(np.clip(p_nn, 1e-12, None))
      + (1 - ALPHA) * np.log(np.clip(p_lgb, 1e-12, None))
      - TAU * np.log(prior)[None, :])
log(f"blend alpha={ALPHA} tau={TAU} lam={LAM} K={K}")

# ---- K-best per act; measure top-1 vs oracle ----
cand_all = []
for k_act in range(len(va_provs)):
    cand_all.append(kbest_viterbi(le[offsets[k_act]:offsets[k_act + 1]],
                                  logT, LAM, K))
log("k-best decoded")

top1 = [attach_from_depths(list(c[0][1])) for c in cand_all]
comp = components(top1, va_pars)
log(f"top1   norm={comp['normalized']:.4f} pacc={comp['parent_acc']:.4f} "
    f"sibF1={comp['sib_f1']:.4f}")

oracle_scores = []
pick_rank = []
for cands, true in zip(cand_all, va_pars):
    best_s, best_r = -1.0, 0
    for r, (sc, seq) in enumerate(cands):
        s = act_score(attach_from_depths(list(seq)), true)
        if s > best_s:
            best_s, best_r = s, r
    oracle_scores.append(best_s)
    pick_rank.append(best_r)
log(f"ORACLE@{K} norm={np.mean(oracle_scores):.4f} "
    f"(top1 in {np.mean([r == 0 for r in pick_rank]):.2%} of acts; "
    f"mean best rank={np.mean(pick_rank):.2f})")

# ---- lightweight rerank: score' = logprob/n + w1*kl + w2*|deep-prior_deep| ----
prior_deep = float((y_tr >= 3).mean())
feat_cache = []
for cands in cand_all:
    feat_cache.append([(sc / len(seq),) + tree_feats(seq, prior)
                       for (sc, seq) in cands])

best = None
for w1 in [0.0, -0.02, -0.05, -0.1, -0.2]:
    for w2 in [0.0, -0.2, -0.5, -1.0]:
        preds = []
        for cands, feats in zip(cand_all, feat_cache):
            vals = [f[0] + w1 * f[1] + w2 * abs(f[2] - prior_deep)
                    for f in feats]
            preds.append(attach_from_depths(list(cands[int(np.argmax(vals))][1])))
        comp = components(preds, va_pars)
        if best is None or comp["normalized"] > best[2]["normalized"]:
            best = (w1, w2, comp)
        log(f"w1={w1:+.2f} w2={w2:+.2f} norm={comp['normalized']:.4f}")
log(f"BEST rerank w1={best[0]} w2={best[1]} -> {best[2]}")
