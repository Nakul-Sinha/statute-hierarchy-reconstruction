"""Milestone 3: LightGBM + constrained Viterbi backup path, holdout-scored."""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
from pipeline import attach_from_depths, build_features, transition_matrix, viterbi

import os
DATA = os.environ.get("CH3_DATA", r"G:\ml\Latest_Chals\Challenge 3\dataset")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
log(f"acts: train={len(tr_rows)} val={len(va_rows)}")

tr_provs = [json.loads(s) for s in tr_rows["provisions_json"]]
tr_pars = [json.loads(s) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
tr_depths = [depths_from_parents(p) for p in tr_pars]
va_depths = [depths_from_parents(p) for p in va_pars]

# hand features
Xh_tr = build_features(tr_provs)
Xh_va = build_features(va_provs)
y_tr = np.concatenate([np.asarray(d) for d in tr_depths])
y_va = np.concatenate([np.asarray(d) for d in va_depths])
log(f"hand features: {Xh_tr.shape} / {Xh_va.shape}")

# TF-IDF -> SVD (fit on training acts only). LGBM_TFIDF=0 disables the block
# entirely (TF-IDF-free mode: hand+neighbor features only).
USE_TFIDF = int(os.environ.get("LGBM_TFIDF", "1"))
if USE_TFIDF:
    flat_tr = [t for act in tr_provs for t in act]
    flat_va = [t for act in va_provs for t in act]

    tw = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=150_000,
                         lowercase=True, sublinear_tf=True)
    Tw_tr = tw.fit_transform(flat_tr)
    Tw_va = tw.transform(flat_va)
    svd_w = TruncatedSVD(n_components=60, random_state=SEED)
    Sw_tr = svd_w.fit_transform(Tw_tr)
    Sw_va = svd_w.transform(Tw_va)
    log(f"word tfidf {Tw_tr.shape} -> svd 60 (evr={svd_w.explained_variance_ratio_.sum():.3f})")

    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=200_000, sublinear_tf=True)
    Tc_tr = tc.fit_transform(flat_tr)
    Tc_va = tc.transform(flat_va)
    svd_c = TruncatedSVD(n_components=40, random_state=SEED)
    Sc_tr = svd_c.fit_transform(Tc_tr)
    Sc_va = svd_c.transform(Tc_va)
    log(f"char tfidf {Tc_tr.shape} -> svd 40 (evr={svd_c.explained_variance_ratio_.sum():.3f})")

    X_tr = np.hstack([Xh_tr, Sw_tr, Sc_tr]).astype(np.float32)
    X_va = np.hstack([Xh_va, Sw_va, Sc_va]).astype(np.float32)
else:
    X_tr = Xh_tr.astype(np.float32)
    X_va = Xh_va.astype(np.float32)
    log("TF-IDF disabled (LGBM_TFIDF=0): hand features only")
log(f"full X: {X_tr.shape}")

params = dict(
    objective="multiclass", num_class=7, num_leaves=127, learning_rate=0.06,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=30, num_threads=-1, seed=SEED, verbose=-1,
    metric="multi_logloss")
dtr = lgb.Dataset(X_tr, label=y_tr)
dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
log(f"lgbm best_iter={booster.best_iteration}")

proba_va = booster.predict(X_va, num_iteration=booster.best_iteration)
log(f"raw argmax depth acc (val): {(proba_va.argmax(1) == y_va).mean():.4f}")
import os
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"D:\scratch\ch3")
np.save(os.path.join(SCRATCH, os.environ.get("CH3_LGB_OUT", "va_emis_lgbm.npy")),
        proba_va)

logT = transition_matrix(tr_depths)
log_emis_all = np.log(np.clip(proba_va, 1e-12, None))

# split emissions back into acts
offsets = np.cumsum([0] + [len(a) for a in va_provs])
best = None
for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
    preds = []
    for k in range(len(va_provs)):
        emis = log_emis_all[offsets[k]:offsets[k + 1]]
        seq = viterbi(emis, logT, lam)
        preds.append(attach_from_depths(seq))
    comp = components(preds, va_pars)
    dep_acc = np.mean([
        np.mean(np.asarray(depths_from_parents(p)) == np.asarray(d))
        for p, d in zip(preds, va_depths)])
    log(f"lam={lam:.1f} norm={comp['normalized']:.4f} pacc={comp['parent_acc']:.4f} "
        f"depth={comp['depth']:.4f} sibF1={comp['sib_f1']:.4f} depth_acc={dep_acc:.4f}")
    if best is None or comp["normalized"] > best[1]["normalized"]:
        best = (lam, comp)

log(f"BEST lam={best[0]} -> {best[1]}")
