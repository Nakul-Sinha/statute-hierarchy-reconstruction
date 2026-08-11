"""MBR decoding: K-best beam candidates rescored by expected challenge metric
against posterior samples (FFBS with structural mask), vs MAP baseline."""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metric import act_score, components
from pipeline import attach_from_depths, viterbi

DATA = r"G:\ml\Latest_Chals\Challenge 3\dataset"
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"D:\scratch\ch3")
SEED = 42
N = 7
t0 = time.time()
rng = np.random.RandomState(SEED)


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


MASK = np.full((N, N), -np.inf)
ALLOWED = {}
for a in range(N):
    allowed = list(range(min(a + 1, N - 1) + 1))
    ALLOWED[a] = allowed
    for b in allowed:
        MASK[a, b] = 0.0


def beam_kbest(le, K=8):
    """K best depth sequences by sum log p under mask; d0=0."""
    n = le.shape[0]
    beams = [(le[0, 0], [0])]
    for i in range(1, n):
        cand = []
        for sc, seq in beams:
            for b in ALLOWED[seq[-1]]:
                cand.append((sc + le[i, b], seq + [b]))
        cand.sort(key=lambda x: -x[0])
        beams = cand[:K]
    return [seq for _, seq in beams]


def ffbs_samples(le, M=48):
    """M posterior samples under emissions + uniform-allowed transitions."""
    n = le.shape[0]
    p = np.exp(le - le.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    # forward messages with mask (uniform over allowed transitions)
    A = np.exp(MASK)  # 1 for allowed, 0 else
    fwd = np.zeros((n, N))
    fwd[0, 0] = 1.0
    for i in range(1, n):
        fwd[i] = (fwd[i - 1] @ A) * p[i]
        s = fwd[i].sum()
        if s <= 0:
            fwd[i] = p[i] * (fwd[i - 1] @ A > 0)
            s = fwd[i].sum()
        fwd[i] /= s
    samples = np.zeros((M, n), dtype=np.int8)
    # backward sampling
    last = rng.choice(N, size=M, p=fwd[-1] / fwd[-1].sum())
    samples[:, -1] = last
    for i in range(n - 2, -1, -1):
        # P(d_i | d_{i+1}) prop fwd[i, a] * A[a, b]
        for m in range(M):
            b = samples[m, i + 1]
            w = fwd[i] * A[:, b]
            s = w.sum()
            if s <= 0:
                w = fwd[i]
                s = w.sum()
            samples[m, i] = rng.choice(N, p=w / s)
    return samples


train = pd.read_csv(rf"{DATA}\train.csv")
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
offsets = np.cumsum([0] + [len(a) for a in va_provs])

p_lgb = np.load(os.path.join(SCRATCH, "va_emis_lgbm.npy"))
nns = [np.load(os.path.join(SCRATCH, f)) for f in
       ["va_emis_nn_big.npy", "va_emis_nn_s1.npy", "va_emis_nn_s2.npy"]
       if os.path.exists(os.path.join(SCRATCH, f))]
p_nn = np.mean(nns, axis=0)
ALPHA = 0.6
le_all = ALPHA * np.log(np.clip(p_nn, 1e-12, None)) + \
    (1 - ALPHA) * np.log(np.clip(p_lgb, 1e-12, None))
log(f"emissions ready ({len(nns)} NNs, alpha={ALPHA})")

preds_map, preds_mbr = [], []
n_changed = 0
for k in range(len(va_provs)):
    le = le_all[offsets[k]:offsets[k + 1]]
    mapseq = viterbi(le, MASK, 1.0)
    preds_map.append(attach_from_depths(mapseq))
    cands = beam_kbest(le, K=8)
    if mapseq not in cands:
        cands.append(mapseq)
    samples = ffbs_samples(le, M=48)
    sample_parents = [attach_from_depths(list(s)) for s in samples]
    best_sc, best_par = -1, None
    for c in cands:
        cp = attach_from_depths(c)
        sc = np.mean([act_score(cp, sp) for sp in sample_parents])
        if sc > best_sc:
            best_sc, best_par = sc, cp
    preds_mbr.append(best_par)
    if best_par != preds_map[-1]:
        n_changed += 1
    if (k + 1) % 100 == 0:
        log(f"  {k+1}/{len(va_provs)} acts (changed={n_changed})")

comp_map = components(preds_map, va_pars)
comp_mbr = components(preds_mbr, va_pars)
log(f"MAP norm={comp_map['normalized']:.4f} pacc={comp_map['parent_acc']:.4f} "
    f"depth={comp_map['depth']:.4f} sibF1={comp_map['sib_f1']:.4f}")
log(f"MBR norm={comp_mbr['normalized']:.4f} pacc={comp_mbr['parent_acc']:.4f} "
    f"depth={comp_mbr['depth']:.4f} sibF1={comp_mbr['sib_f1']:.4f} "
    f"(changed {n_changed} acts)")
