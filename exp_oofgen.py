"""Inner-CV OOF depth emissions on the 85% train split (one fold+seed/process).

Produces the training material for the K-best reranker (exp_rerank.py): for
fold f a model is trained on the other three quarters of the train split and
scores the held-out quarter, so every candidate the reranker is fit on comes
from a model that never saw that act.

Split reproduction (identical to every other driver in this repo):
  train.csv -> train_test_split(test_size=0.15, random_state=42)
            -> tr_rows (85%, the "train split") + va_rows (15% outer holdout)
  KFold(n_splits=4, shuffle=True, random_state=42) over tr_rows positions.

The NN is the frozen production c2 config (emb 160 / hid 256 / 2 BiLSTM layers /
hashed char-ngram channel 64, plateau early-stop) and is early-stopped on the
OUTER holdout depth accuracy, matching solution.py's convention exactly.
No TF-IDF or any other fitted text statistic is used anywhere.

Env knobs: CH3_DATA, CH3_SCRATCH, OOF_FOLD (0..3), OOF_SEED (NN seed; seed 42
additionally trains the fold's hand-only LGBM), NN_THREADS (torch + LightGBM
thread cap), plus the usual NN_EMB / NN_HID / NN_LAYERS / NN_EPOCHS /
NN_PATIENCE overrides.

Outputs in CH3_SCRATCH, row-aligned with the concatenated provisions of the
fold's inner-val acts in inner-val dataframe order:
  oof_nn_f{fold}_s{seed}.npy   (n_prov, 7) probabilities
  oof_lgbm_f{fold}.npy         (n_prov, 7) probabilities  [seed 42 only]
Existing outputs are never rewritten, so the driver can resume idempotently.
"""
import json
import os
import time

# nn_model reads NN_CNG / NN_MAXTOK / NN_FL at import time; pin the production
# c2 char-ngram width unless the caller already set it.
os.environ.setdefault("NN_CNG", "64")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402

from metric import depths_from_parents  # noqa: E402
from nn_model import (ActDataset, DepthTagger, build_vocab,  # noqa: E402
                      predict_proba, train_tagger)
from pipeline import act_features  # noqa: E402

DATA = os.environ.get("CH3_DATA", "/home/ec2-user/ch3")
SCRATCH = os.environ.get("CH3_SCRATCH", "/home/ec2-user/ch3/scratch")
SEED = 42
N_STATES = 7
N_FOLDS = 4
FOLD = int(os.environ.get("OOF_FOLD", "0"))
NN_SEED = int(os.environ.get("OOF_SEED", "42"))
THREADS = int(os.environ.get("NN_THREADS", "4"))
EMB = int(os.environ.get("NN_EMB", "160"))
HID = int(os.environ.get("NN_HID", "256"))
LAYERS = int(os.environ.get("NN_LAYERS", "2"))
EPOCHS = int(os.environ.get("NN_EPOCHS", "40"))
PATIENCE = int(os.environ.get("NN_PATIENCE", "6"))
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


np.random.seed(SEED)
torch.set_num_threads(THREADS)
device = "cpu"
os.makedirs(SCRATCH, exist_ok=True)

nn_out = os.path.join(SCRATCH, f"oof_nn_f{FOLD}_s{NN_SEED}.npy")
lgb_out = os.path.join(SCRATCH, f"oof_lgbm_f{FOLD}.npy")
want_nn = not os.path.exists(nn_out)
want_lgb = (NN_SEED == SEED) and not os.path.exists(lgb_out)
log(f"fold={FOLD} seed={NN_SEED} threads={THREADS} "
    f"want_nn={want_nn} want_lgb={want_lgb}")
if not want_nn and not want_lgb:
    log("nothing to do; outputs already present")
    raise SystemExit(0)

# ---- splits (sacred: identical in every driver) ----
train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
             .split(np.arange(len(tr_rows))))
itr_idx, iva_idx = folds[FOLD]
itr_rows, iva_rows = tr_rows.iloc[itr_idx], tr_rows.iloc[iva_idx]


def load_rows(rows):
    provs = [json.loads(s) for s in rows["provisions_json"]]
    pars = [json.loads(s) for s in rows["parents_json"]]
    return provs, [depths_from_parents(p) for p in pars]


itr_provs, itr_depths = load_rows(itr_rows)
iva_provs, iva_depths = load_rows(iva_rows)
ova_provs, ova_depths = load_rows(va_rows)
log(f"acts: inner_tr={len(itr_provs)} inner_va={len(iva_provs)} "
    f"outer_va={len(ova_provs)}")

feats_itr = [act_features(p) for p in itr_provs]
feats_iva = [act_features(p) for p in iva_provs]
feats_ova = [act_features(p) for p in ova_provs]
allf = np.vstack(feats_itr)
mu, sd = allf.mean(0), allf.std(0) + 1e-6
y_itr = np.concatenate([np.asarray(d) for d in itr_depths])
y_iva = np.concatenate([np.asarray(d) for d in iva_depths])
y_ova = np.concatenate([np.asarray(d) for d in ova_depths])
log(f"hand features dim={allf.shape[1]} inner_tr_rows={allf.shape[0]} "
    f"inner_va_rows={len(y_iva)}")

# ---- Model B: hand-only LightGBM (solution.py params verbatim) ----
if want_lgb:
    import lightgbm as lgb

    params = dict(objective="multiclass", num_class=N_STATES, num_leaves=127,
                  learning_rate=0.06, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=30, num_threads=THREADS,
                  seed=SEED, verbose=-1, metric="multi_logloss",
                  deterministic=True, force_row_wise=True)
    X_iva = np.vstack(feats_iva)
    dtr = lgb.Dataset(allf, label=y_itr)
    dva = lgb.Dataset(np.vstack(feats_ova), label=y_ova, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    proba_lgb = booster.predict(X_iva, num_iteration=booster.best_iteration)
    np.save(lgb_out, proba_lgb)
    log(f"lgbm best_iter={booster.best_iteration} "
        f"oof_depth_acc={(proba_lgb.argmax(1) == y_iva).mean():.4f} "
        f"-> {os.path.basename(lgb_out)} {proba_lgb.shape}")

# ---- Model A: one neural tagger seed ----
if want_nn:
    vocab = build_vocab([t for a in itr_provs for t in a], min_count=2)
    ds_itr = ActDataset(itr_provs, vocab, feats_itr, mu, sd)
    ds_ova = ActDataset(ova_provs, vocab, feats_ova, mu, sd)
    ds_iva = ActDataset(iva_provs, vocab, feats_iva, mu, sd)
    log(f"vocab size={len(vocab)}; datasets encoded")

    torch.manual_seed(NN_SEED)
    model = DepthTagger(len(vocab), allf.shape[1], emb_dim=EMB,
                        lstm_hidden=HID, num_layers=LAYERS)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params={n_params/1e6:.2f}M (emb={EMB} hid={HID} layers={LAYERS} "
        f"cng={os.environ['NN_CNG']} seed={NN_SEED} epochs={EPOCHS} "
        f"patience={PATIENCE})")

    best_state, best_acc = train_tagger(
        model, ds_itr, itr_depths, ds_ova, y_ova, device,
        max_epochs=EPOCHS, patience=PATIENCE, batch_acts=32, lr=1e-3,
        seed=NN_SEED, log=log, cosine=False)
    model.load_state_dict(best_state)
    log(f"best outer-holdout depth acc={best_acc:.4f}")

    proba_nn = predict_proba(model, ds_iva, device)
    np.save(nn_out, proba_nn)
    log(f"oof_depth_acc={(proba_nn.argmax(1) == y_iva).mean():.4f} "
        f"-> {os.path.basename(nn_out)} {proba_nn.shape}")

log("done")
