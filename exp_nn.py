"""Milestone 4: neural depth tagger (BiLSTM over provision encoders)."""
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

DATA = r"G:\Datacurve\Latest_Chals\Challenge 3\dataset"
SCRATCH = os.environ.get(
    "CH3_SCRATCH",
    r"C:\Users\nakul\AppData\Local\Temp\claude\G--Datacurve-Latest-Chals\c252d314-4a06-4b8d-a7b1-0935a59ec986\scratchpad")
SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 8)
device = "cpu"

train = pd.read_csv(rf"{DATA}\train.csv")
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
allf = np.vstack(feats_tr)
mu, sd = allf.mean(0), allf.std(0) + 1e-6
log(f"hand features dim={allf.shape[1]}")

vocab = build_vocab([t for act in tr_provs for t in act], min_count=2)
log(f"vocab size={len(vocab)}")

ds_tr = ActDataset(tr_provs, vocab, feats_tr, mu, sd)
ds_va = ActDataset(va_provs, vocab, feats_va, mu, sd)
y_va_flat = np.concatenate([np.asarray(d) for d in va_depths])
log("datasets encoded")

model = DepthTagger(len(vocab), allf.shape[1]).to(device)
n_params = sum(p.numel() for p in model.parameters())
log(f"model params={n_params/1e6:.2f}M")

best_state, best_acc = train_tagger(
    model, ds_tr, tr_depths, ds_va, y_va_flat, device,
    max_epochs=30, patience=5, batch_acts=32, lr=1e-3, seed=SEED, log=log)
model.load_state_dict(best_state)
log(f"best val depth acc={best_acc:.4f}")

proba_va = predict_proba(model, ds_va, device)
np.save(os.path.join(SCRATCH, "va_emis_nn.npy"), proba_va)
log("val emissions saved")

logT = transition_matrix(tr_depths)
log_emis_all = np.log(np.clip(proba_va, 1e-12, None))
offsets = np.cumsum([0] + [len(a) for a in va_provs])
best = None
for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
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
