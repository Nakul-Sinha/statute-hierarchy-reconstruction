"""NN tagger experiment driver (portable; knobs via env vars).

Env knobs: CH3_DATA, CH3_SCRATCH, CH3_EMIS_SUFFIX, NN_SEED, NN_EMB, NN_HID,
NN_LAYERS, NN_ATTN, NN_COS, NN_EPOCHS, NN_PATIENCE, NN_SVD (append TF-IDF/SVD
features to the NN input), NN_MAXTOK, NN_FL (read inside nn_model).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from metric import components, depths_from_parents
from nn_model import ActDataset, DepthTagger, build_vocab, predict_proba, train_tagger
from pipeline import act_features, attach_from_depths, transition_matrix, viterbi

DATA = os.environ.get("CH3_DATA", r"G:\Datacurve\Latest_Chals\Challenge 3\dataset")
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"C:\Users\nakul\AppData\Local\Temp\claude\G--Datacurve-Latest-Chals\c252d314-4a06-4b8d-a7b1-0935a59ec986\scratchpad")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("NN_THREADS", str(os.cpu_count() or 8))))
device = "cpu"

train = pd.read_csv(os.path.join(DATA, "train.csv"))
tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
tr_provs = [json.loads(s) for s in tr_rows["provisions_json"]]
tr_pars = [json.loads(s) for s in tr_rows["parents_json"]]
va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
va_pars = [json.loads(s) for s in va_rows["parents_json"]]
tr_depths = [depths_from_parents(p) for p in tr_pars]
va_depths = [depths_from_parents(p) for p in va_pars]
log(f"acts: train={len(tr_provs)} val={len(va_provs)}")

feats_tr = [act_features(p) for p in tr_provs]
feats_va = [act_features(p) for p in va_provs]

if int(os.environ.get("NN_SVD", "0")):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    flat_tr = [t for a in tr_provs for t in a]
    flat_va = [t for a in va_provs for t in a]
    tw = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=150_000,
                         sublinear_tf=True)
    Sw = TruncatedSVD(n_components=60, random_state=SEED)
    sw_tr = Sw.fit_transform(tw.fit_transform(flat_tr))
    sw_va = Sw.transform(tw.transform(flat_va))
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=200_000, sublinear_tf=True)
    Sc = TruncatedSVD(n_components=40, random_state=SEED)
    sc_tr = Sc.fit_transform(tc.fit_transform(flat_tr))
    sc_va = Sc.transform(tc.transform(flat_va))
    log("tfidf/svd features for NN input ready")

    def _split(mat, provs):
        out, pos = [], 0
        for p in provs:
            out.append(mat[pos:pos + len(p)].astype(np.float32))
            pos += len(p)
        return out
    sw_tr_a, sc_tr_a = _split(sw_tr, tr_provs), _split(sc_tr, tr_provs)
    sw_va_a, sc_va_a = _split(sw_va, va_provs), _split(sc_va, va_provs)
    feats_tr = [np.hstack([f, a, b]) for f, a, b in zip(feats_tr, sw_tr_a, sc_tr_a)]
    feats_va = [np.hstack([f, a, b]) for f, a, b in zip(feats_va, sw_va_a, sc_va_a)]

allf = np.vstack(feats_tr)
mu, sd = allf.mean(0), allf.std(0) + 1e-6
log(f"hand features dim={allf.shape[1]}")

vocab = build_vocab([t for act in tr_provs for t in act], min_count=2)
log(f"vocab size={len(vocab)}")

ds_tr = ActDataset(tr_provs, vocab, feats_tr, mu, sd)
ds_va = ActDataset(va_provs, vocab, feats_va, mu, sd)
y_va_flat = np.concatenate([np.asarray(d) for d in va_depths])
log("datasets encoded")

EMB = int(os.environ.get("NN_EMB", "100"))
HID = int(os.environ.get("NN_HID", "128"))
LAYERS = int(os.environ.get("NN_LAYERS", "1"))
ATTN = bool(int(os.environ.get("NN_ATTN", "0")))
COS = bool(int(os.environ.get("NN_COS", "0")))
EPOCHS = int(os.environ.get("NN_EPOCHS", "40"))
PATIENCE = int(os.environ.get("NN_PATIENCE", "6"))
NN_SEED = int(os.environ.get("NN_SEED", str(SEED)))
SUFFIX = os.environ.get("CH3_EMIS_SUFFIX", "")
torch.manual_seed(NN_SEED)

model = DepthTagger(len(vocab), allf.shape[1], emb_dim=EMB,
                    lstm_hidden=HID, num_layers=LAYERS, use_attn=ATTN)
n_params = sum(p.numel() for p in model.parameters())
log(f"model params={n_params/1e6:.2f}M (emb={EMB} hid={HID} layers={LAYERS} "
    f"attn={ATTN} cos={COS} seed={NN_SEED} epochs={EPOCHS})")

best_state, best_acc = train_tagger(
    model, ds_tr, tr_depths, ds_va, y_va_flat, device,
    max_epochs=EPOCHS, patience=PATIENCE, batch_acts=32, lr=1e-3,
    seed=NN_SEED, log=log, cosine=COS)
model.load_state_dict(best_state)
log(f"best val depth acc={best_acc:.4f}")

proba_va = predict_proba(model, ds_va, device)
np.save(os.path.join(SCRATCH, f"va_emis_nn{SUFFIX}.npy"), proba_va)
log("val emissions saved")

logT = transition_matrix(tr_depths)
log_emis_all = np.log(np.clip(proba_va, 1e-12, None))
offsets = np.cumsum([0] + [len(a) for a in va_provs])
best = None
for lam in [0.0, 0.1, 0.2]:
    preds = []
    for k in range(len(va_provs)):
        emis = log_emis_all[offsets[k]:offsets[k + 1]]
        seq = viterbi(emis, logT, lam)
        preds.append(attach_from_depths(seq))
    comp = components(preds, va_pars)
    log(f"lam={lam:.1f} norm={comp['normalized']:.4f} pacc={comp['parent_acc']:.4f} "
        f"depth={comp['depth']:.4f} sibF1={comp['sib_f1']:.4f}")
    if best is None or comp["normalized"] > best[1]["normalized"]:
        best = (lam, comp)
log(f"BEST lam={best[0]} -> {best[1]}")
