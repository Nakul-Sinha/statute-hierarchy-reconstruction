"""Trained K-best reranker: OOF candidates -> LambdaRank -> holdout gate.

  A. For every inner-CV fold (exp_oofgen.py outputs) blend the 6 NN seeds with
     the fold LGBM at the frozen e6 operating point, K-best decode each
     inner-val act and label every candidate with its TRUE act score.
  B. Per-candidate features: path-likelihood shape, NN-vs-LGBM disagreement,
     depth histogram / bigram-transition / sibling-run profile, act length.
  C. LightGBM lambdarank with acts as groups: fit on folds 0-2, early stop on
     fold 3 NDCG@1, refit on all four folds at the best iteration.
  D. Apply to the 562-act outer holdout decoded from the production emission
     files; report top-1 vs reranked and the +0.008 adoption gate.

Env: CH3_DATA, CH3_SCRATCH, RR_K (16), RR_ALPHA (0.8), RR_TAU (0.4),
     RR_LAM (0.1), RR_THREADS, CH3_NN_FILES / CH3_LGB_FILE (the production
     holdout emissions, same names as the blend drivers).

Statistics choice: the transition matrix, class prior and empirical bigram
distribution are fit ONCE on the full 85% train split and reused for every
inner fold as well as for the holdout. They are 7- and 49-cell aggregates over
~69k provisions, so the leak into an inner fold is negligible, and using one
set of statistics keeps the fold decode numerically identical in scale to the
holdout decode - which is what the reranker's features have to transfer across.
No TF-IDF or any other fitted text statistic appears anywhere.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from metric import act_score, components, depths_from_parents
from pipeline import attach_from_depths, transition_matrix

DATA = os.environ.get("CH3_DATA", "/home/ec2-user/ch3")
SCRATCH = os.environ.get("CH3_SCRATCH", "/home/ec2-user/ch3/scratch")
SEED = 42
N_STATES = 7
N_FOLDS = 4
NN_SEEDS = (42, 1, 2, 3, 7, 11)
K = int(os.environ.get("RR_K", "16"))
ALPHA = float(os.environ.get("RR_ALPHA", "0.8"))
TAU = float(os.environ.get("RR_TAU", "0.4"))
LAM = float(os.environ.get("RR_LAM", "0.1"))
THREADS = int(os.environ.get("RR_THREADS", str(os.cpu_count() or 8)))
GATE = 0.008
MAX_GRADE = 7
PROD_NN = [n.strip() for n in os.environ.get(
    "CH3_NN_FILES",
    "va_emis_nn_c2.npy,va_emis_nn_s1.npy,va_emis_nn_s2.npy,"
    "va_emis_nn_s3.npy,va_emis_nn_s7.npy,va_emis_nn_s11.npy").split(",")
    if n.strip()]
PROD_LGB = os.environ.get("CH3_LGB_FILE", "va_emis_lgbm_hand.npy")

FEAT_NAMES = [
    "rank", "score_norm", "gap_norm", "mean_lp", "min_lp",
    "mean_margin_nn_lgb", "nn_disagree", "lgb_disagree",
    "kl_depth_hist", "deep_frac", "max_depth", "mean_depth",
    "kl_trans", "run_mean", "run_max", "run_count_norm", "log_len",
]
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def load_npy(name):
    path = os.path.join(SCRATCH, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing emission file: {path}")
    return np.load(path)


# ----------------------------------------------------------------------------
# K-best decode
# ----------------------------------------------------------------------------

def kbest_viterbi(log_emis, logT, lam, k):
    """K-best constrained Viterbi (d0=0, d[i] <= d[i-1]+1), best first.

    Backpointer variant of exp_kbest.kbest_viterbi: identical scores and
    identical tie ordering (stable sort over ascending prev-state then
    ascending prev-rank), but it stores (prev_state, prev_rank) instead of
    growing path tuples, so cost is linear rather than quadratic in act length.
    """
    n = log_emis.shape[0]
    finite = np.isfinite(logT)
    trans = np.where(finite, lam * np.where(finite, logT, 0.0), -np.inf)
    scores = [[[] for _ in range(N_STATES)] for _ in range(n)]
    back = [[[] for _ in range(N_STATES)] for _ in range(n)]
    scores[0][0] = [float(log_emis[0, 0])]
    back[0][0] = [(-1, -1)]
    for i in range(1, n):
        em = log_emis[i]
        prev = scores[i - 1]
        for b in range(N_STATES):
            cands = []
            for a in range(N_STATES):
                if not np.isfinite(trans[a, b]) or not prev[a]:
                    continue
                step = float(trans[a, b]) + float(em[b])
                for r, sc in enumerate(prev[a]):
                    cands.append((sc + step, a, r))
            cands.sort(key=lambda x: -x[0])
            del cands[k:]
            scores[i][b] = [c[0] for c in cands]
            back[i][b] = [(c[1], c[2]) for c in cands]
    last = [(scores[n - 1][b][r], b, r)
            for b in range(N_STATES) for r in range(len(scores[n - 1][b]))]
    last.sort(key=lambda x: -x[0])
    out = []
    for sc, b, r in last[:k]:
        seq = [0] * n
        cb, cr = b, r
        for i in range(n - 1, -1, -1):
            seq[i] = cb
            cb, cr = back[i][cb][cr]
        out.append((sc, tuple(seq)))
    return out


# ----------------------------------------------------------------------------
# Candidate features
# ----------------------------------------------------------------------------

def cand_features(rank, sc, top_sc, seq, le_a, lp_nn_a, lp_lgb_a, nn_arg,
                  lgb_arg, log_prior, log_trans_flat):
    """Feature row for one candidate depth sequence of one act."""
    arr = np.asarray(seq, dtype=np.int64)
    n = arr.shape[0]
    rows = np.arange(n)
    path_lp = le_a[rows, arr]
    margin = lp_nn_a[rows, arr] - lp_lgb_a[rows, arr]
    hist = np.bincount(arr, minlength=N_STATES) / n
    nz = hist > 0
    kl_hist = float(np.sum(hist[nz] * (np.log(hist[nz]) - log_prior[nz])))
    if n > 1:
        big = np.bincount(arr[:-1] * N_STATES + arr[1:],
                          minlength=N_STATES * N_STATES).astype(np.float64)
        big /= big.sum()
        m = big > 0
        kl_trans = float(np.sum(big[m] * (np.log(big[m]) - log_trans_flat[m])))
        brk = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        runs = np.diff(np.concatenate(([0], brk, [n])))
    else:
        kl_trans = 0.0
        runs = np.ones(1, dtype=np.int64)
    return [
        float(rank),
        float(sc) / n,
        float(top_sc - sc) / n,
        float(path_lp.mean()),
        float(path_lp.min()),
        float(margin.mean()),
        float((nn_arg != arr).mean()),
        float((lgb_arg != arr).mean()),
        kl_hist,
        float((arr >= 3).mean()),
        float(arr.max()),
        float(arr.mean()),
        kl_trans,
        float(runs.mean()),
        float(runs.max()),
        float(len(runs)) / n,
        float(np.log(n)),
    ]


def decode_block(act_lens, p_nn, p_lgb, logT, log_prior, log_trans_flat):
    """K-best decode + featurize a block of acts at the frozen blend point.

    Returns (candidates per act, feature matrix stacked in act order,
    group sizes)."""
    lp_nn = np.log(np.clip(p_nn, 1e-12, None))
    lp_lgb = np.log(np.clip(p_lgb, 1e-12, None))
    le = ALPHA * lp_nn + (1 - ALPHA) * lp_lgb - TAU * log_prior[None, :]
    offs = np.cumsum([0] + list(act_lens))
    cand_all, feats, groups = [], [], []
    for i in range(len(act_lens)):
        a, b = int(offs[i]), int(offs[i + 1])
        le_a, lp_nn_a, lp_lgb_a = le[a:b], lp_nn[a:b], lp_lgb[a:b]
        cands = kbest_viterbi(le_a, logT, LAM, K)
        nn_arg, lgb_arg = lp_nn_a.argmax(1), lp_lgb_a.argmax(1)
        top_sc = cands[0][0]
        feats.append(np.asarray(
            [cand_features(r, s, top_sc, seq, le_a, lp_nn_a, lp_lgb_a, nn_arg,
                           lgb_arg, log_prior, log_trans_flat)
             for r, (s, seq) in enumerate(cands)], dtype=np.float32))
        cand_all.append(cands)
        groups.append(len(cands))
    return cand_all, np.vstack(feats), np.asarray(groups, dtype=np.int64)


def grades_from_scores(scores):
    """Dense-rank true act scores (desc) into integer relevance grades.

    Ties share a grade, which matters here because near-duplicate candidate
    depth sequences very often score identically; the best distinct score gets
    MAX_GRADE and each next distinct score drops one, floored at 0. Scores are
    rounded to 1e-12 first so float noise cannot split a genuine tie."""
    vals = [round(float(s), 12) for s in scores]
    order = sorted(set(vals), reverse=True)
    rank_of = {v: i for i, v in enumerate(order)}
    return [max(0, MAX_GRADE - rank_of[v]) for v in vals]


# ----------------------------------------------------------------------------
# Split + split-fit statistics (fit once on the full 85% train split)
# ----------------------------------------------------------------------------

train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_depths = [depths_from_parents(json.loads(s)) for s in tr_rows["parents_json"]]
y_tr = np.concatenate([np.asarray(d) for d in tr_depths])
prior = np.bincount(y_tr, minlength=N_STATES).astype(np.float64)
prior = (prior + 0.5) / (prior.sum() + 3.5)
log_prior = np.log(prior)
logT = transition_matrix(tr_depths)
tcnt = np.zeros((N_STATES, N_STATES), dtype=np.float64)
for ds in tr_depths:
    for a, b in zip(ds[:-1], ds[1:]):
        tcnt[a, b] += 1
trans_p = (tcnt + 0.5) / (tcnt.sum() + 0.5 * N_STATES * N_STATES)
log_trans_flat = np.log(trans_p).ravel()
log(f"split: train={len(tr_rows)} holdout={len(va_rows)} acts; "
    f"alpha={ALPHA} tau={TAU} lam={LAM} K={K} threads={THREADS}")

# ----------------------------------------------------------------------------
# Step A/B: fold candidates + labels
# ----------------------------------------------------------------------------

folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
             .split(np.arange(len(tr_rows))))
acts_X, acts_y, acts_fold = [], [], []
for f, (_, iva_idx) in enumerate(folds):
    rows = tr_rows.iloc[iva_idx]
    provs = [json.loads(s) for s in rows["provisions_json"]]
    pars = [json.loads(s) for s in rows["parents_json"]]
    lens = [len(p) for p in provs]
    p_nn = np.mean([load_npy(f"oof_nn_f{f}_s{s}.npy") for s in NN_SEEDS], axis=0)
    p_lgb = load_npy(f"oof_lgbm_f{f}.npy")
    if p_nn.shape[0] != sum(lens) or p_lgb.shape != p_nn.shape:
        raise SystemExit(f"fold {f} emission rows {p_nn.shape}/{p_lgb.shape} "
                         f"!= {sum(lens)} provisions")
    cand_all, F, G = decode_block(lens, p_nn, p_lgb, logT, log_prior,
                                  log_trans_flat)
    pos = 0
    top1, best_s = [], []
    for cands, true in zip(cand_all, pars):
        sc = [act_score(attach_from_depths(list(seq)), true) for (_, seq) in cands]
        acts_X.append(F[pos:pos + len(cands)])
        acts_y.append(np.asarray(grades_from_scores(sc), dtype=np.int32))
        acts_fold.append(f)
        pos += len(cands)
        top1.append(sc[0])
        best_s.append(max(sc))
    log(f"fold {f}: acts={len(cand_all)} cands={len(F)} "
        f"top1={np.mean(top1):.4f} oracle@{K}={np.mean(best_s):.4f}")

useful = [i for i in range(len(acts_X)) if len(set(acts_y[i].tolist())) > 1]
log(f"training groups: {len(useful)}/{len(acts_X)} informative "
    f"(dropped {len(acts_X)-len(useful)} acts whose candidates all tie)")


def assemble(idxs):
    X = np.vstack([acts_X[i] for i in idxs])
    y = np.concatenate([acts_y[i] for i in idxs])
    g = np.asarray([len(acts_y[i]) for i in idxs], dtype=np.int64)
    return X, y, g


tr_idx = [i for i in useful if acts_fold[i] < N_FOLDS - 1]
va_idx = [i for i in useful if acts_fold[i] == N_FOLDS - 1]
Xtr, ytr, gtr = assemble(tr_idx)
Xva, yva, gva = assemble(va_idx)
Xall, yall, gall = assemble(useful)
log(f"ltr matrices: fit {Xtr.shape} ({len(gtr)} groups) / "
    f"es {Xva.shape} ({len(gva)} groups) / all {Xall.shape}")

# ----------------------------------------------------------------------------
# Step C: LambdaRank
# ----------------------------------------------------------------------------

params = dict(objective="lambdarank", metric="ndcg", eval_at=[1],
              lambdarank_truncation_level=K, num_leaves=31,
              learning_rate=0.05, min_child_samples=20, num_threads=THREADS,
              seed=SEED, verbose=-1, deterministic=True, force_row_wise=True)
dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=FEAT_NAMES)
dva = lgb.Dataset(Xva, label=yva, group=gva, reference=dtr,
                  feature_name=FEAT_NAMES)
probe = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(50)])
best_iter = probe.best_iteration or 400
bs = dict(probe.best_score.get("valid_0", {}))
log(f"lambdarank best_iter={best_iter} valid_best={bs}")
dall = lgb.Dataset(Xall, label=yall, group=gall, feature_name=FEAT_NAMES)
ranker = lgb.train(params, dall, num_boost_round=best_iter)
ranker.save_model(os.path.join(SCRATCH, "reranker.txt"))
log(f"ranker refit on all folds and saved -> {SCRATCH}/reranker.txt")

gains = ranker.feature_importance(importance_type="gain")
tot = gains.sum() or 1.0
for name, g in sorted(zip(FEAT_NAMES, gains), key=lambda x: -x[1]):
    log(f"  imp {name:<20s} gain={g:12.1f} ({100*g/tot:5.1f}%)")

# ----------------------------------------------------------------------------
# Step D: outer-holdout evaluation + adoption gate
# ----------------------------------------------------------------------------

va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
va_lens = [len(p) for p in va_provs]
p_nn = np.mean([load_npy(n) for n in PROD_NN], axis=0)
p_lgb = load_npy(PROD_LGB)
if p_nn.shape[0] != sum(va_lens) or p_lgb.shape != p_nn.shape:
    raise SystemExit(f"holdout emission rows {p_nn.shape}/{p_lgb.shape} "
                     f"!= {sum(va_lens)} provisions")
log(f"holdout emissions: {len(PROD_NN)} NN files + {PROD_LGB}")

cand_all, Xho, Gho = decode_block(va_lens, p_nn, p_lgb, logT, log_prior,
                                  log_trans_flat)
log(f"holdout k-best decoded: {len(cand_all)} acts, {len(Xho)} candidates")

top1 = [attach_from_depths(list(c[0][1])) for c in cand_all]
comp1 = components(top1, va_pars)
log(f"top1     norm={comp1['normalized']:.4f} pacc={comp1['parent_acc']:.4f} "
    f"depth={comp1['depth']:.4f} sibF1={comp1['sib_f1']:.4f}")

pred = ranker.predict(Xho)
picks, ranks, oracle = [], [], []
pos = 0
for cands, true in zip(cand_all, va_pars):
    m = len(cands)
    r = int(np.argmax(pred[pos:pos + m]))
    pos += m
    ranks.append(r)
    picks.append(attach_from_depths(list(cands[r][1])))
    oracle.append(max(act_score(attach_from_depths(list(seq)), true)
                      for (_, seq) in cands))
comp2 = components(picks, va_pars)
log(f"rerank   norm={comp2['normalized']:.4f} pacc={comp2['parent_acc']:.4f} "
    f"depth={comp2['depth']:.4f} sibF1={comp2['sib_f1']:.4f}")
log(f"oracle@{K} norm={np.mean(oracle):.4f}")

hist = np.bincount(np.asarray(ranks), minlength=K)
log(f"picked-rank histogram: {list(map(int, hist))} "
    f"(kept top-1 in {hist[0]/len(ranks):.1%} of acts)")

delta = comp2["normalized"] - comp1["normalized"]
verdict = "ADOPT" if delta >= GATE else "REJECT"
log(f"delta={delta:+.4f} vs gate {GATE:+.4f} "
    f"(top1={comp1['normalized']:.4f} -> gate {comp1['normalized']+GATE:.4f}, "
    f"reranked={comp2['normalized']:.4f})")
log(f"VERDICT: {verdict}")
