"""Statutory Outline Reconstruction — end-to-end solution.

Usage: python3 solution.py <public_dir> <submission_out>

Approach: document order is a pre-order traversal of the statute tree, so the
tree is fully determined by the per-provision depth sequence (verified lossless
on train). We train two depth-emission models from scratch in-script —
(A) a neural act-level BiLSTM tagger over learned provision encoders and
(B) a LightGBM multiclass model on hand + TF-IDF/SVD features — ensemble their
log-probabilities, decode each act with a constrained Viterbi (d0=0,
d[i] <= d[i-1]+1, empirical smoothed transition prior), and attach
parent[i] = nearest earlier provision at depth-1.

All training, feature fitting, and hyper-selection (ensemble weight alpha,
transition weight lambda) happen inside this run on an internal 15% act-level
holdout scored with a local replication of the challenge metric.
"""
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

T0 = time.time()
SEED = 42
BUDGET_S = 78 * 60          # aim to be fully done well before the 90-min cap
NN_START_CUTOFF_S = 45 * 60  # don't even start NN if this much time has passed
NN_STOP_ELAPSED_S = 58 * 60  # stop NN training when elapsed passes this


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def elapsed():
    return time.time() - T0


# ----------------------------------------------------------------------------
# Metric replication (calibrated: chain scores 0.0185 on train vs official
# ~0.019 anchor; all-(-1) scores exactly 0; oracle scores exactly 1).
# Convention: top-level provisions (parent -1) DO form a sibling group.
# ----------------------------------------------------------------------------

def depths_from_parents(parents):
    d = [0] * len(parents)
    for i, p in enumerate(parents):
        d[i] = 0 if p == -1 else d[p] + 1
    return d


def _sibling_pairs(parents):
    groups = {}
    for i, p in enumerate(parents):
        groups.setdefault(p, []).append(i)
    pairs = set()
    for members in groups.values():
        m = len(members)
        for a in range(m):
            for b in range(a + 1, m):
                pairs.add((members[a], members[b]))
    return pairs


def _sibling_f1(pred, true):
    pp, tp = _sibling_pairs(pred), _sibling_pairs(true)
    if not pp and not tp:
        return 1.0
    if not pp or not tp:
        return 0.0
    inter = len(pp & tp)
    prec, rec = inter / len(pp), inter / len(tp)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def _raw(pred, true, td):
    n = len(true)
    pacc = sum(1 for a, b in zip(pred, true) if a == b) / n
    pd_ = depths_from_parents(pred)
    span = max(max(td), 1)
    depth = sum(max(0.0, 1.0 - abs(a - b) / span) for a, b in zip(pd_, td)) / n
    return 0.55 * pacc + 0.25 * depth + 0.20 * _sibling_f1(pred, true)


def act_score(pred, true):
    td = depths_from_parents(true)
    raw = _raw(pred, true, td)
    trivial = _raw([-1] * len(true), true, td)
    if trivial >= 1.0:
        return 1.0 if raw >= 1.0 else 0.0
    return max(0.0, (raw - trivial) / (1.0 - trivial))


def score_sets(preds, trues):
    return float(np.mean([act_score(p, t) for p, t in zip(preds, trues)]))


# ----------------------------------------------------------------------------
# Hand features
# ----------------------------------------------------------------------------

_BY_GER = re.compile(r"^by\s+\w+ing\b")
_SEC_REF = re.compile(r"^Section\s+\d")
_PARA = re.compile(r"(?<!sub)paragraph \(")
_SUBPARA = re.compile(r"subparagraph \(")
_CLAUSE = re.compile(r"(?<!sub)clause \(")
_SUBSEC = re.compile(r"subsection \(")

CORE_DIM = 27
PREV_IDX = [0, 2, 26, 9, 7, 6, 21, 11, 8]
NEXT_IDX = [0, 11, 26, 2, 21, 1]
PREV2_IDX = [2, 26, 0]


def core_feats(t):
    s = t.rstrip()
    low = t.lower()
    first = s[0] if s else " "
    wc = len(t.split())
    return [
        first.islower(), first.isdigit(),
        s.endswith((":", "—", "--", "–")),
        s.endswith(";"), s.endswith(("; and", "; or")),
        s.endswith((", and", ", or")), s.endswith("."),
        ("is amended" in t) or ("are amended" in t),
        "may be cited as" in t, "the following" in low[-60:], "$" in t,
        bool(_BY_GER.match(low)), "notwithstanding" in low,
        len(_SUBSEC.findall(low)), len(_PARA.findall(low)),
        len(_SUBPARA.findall(low)), len(_CLAUSE.findall(low)),
        ("the term" in low) and ("means" in low),
        wc, np.log1p(wc), min(len(t), 1200) / 1200.0, wc < 15,
        low.endswith("the following:"), s.startswith("That "),
        bool(_SEC_REF.match(s)), "this act" in low,
        s.endswith((";", "; and", "; or", ", and", ", or")),
    ]


def act_features(provs):
    n = len(provs)
    core = np.asarray([core_feats(t) for t in provs], dtype=np.float32)
    zeros = np.zeros(CORE_DIM, dtype=np.float32)
    prev1 = np.vstack([zeros, core[:-1]]) if n > 1 else zeros[None, :]
    next1 = np.vstack([core[1:], zeros]) if n > 1 else zeros[None, :]
    prev2 = (np.vstack([zeros, zeros, core[:-2]]) if n > 2
             else np.zeros((n, CORE_DIM), dtype=np.float32))
    idx = np.arange(n, dtype=np.float32)
    pos = np.stack([idx, idx / max(n - 1, 1), np.full(n, n, np.float32),
                    (idx == 0).astype(np.float32),
                    (idx == n - 1).astype(np.float32)], axis=1)
    return np.hstack([core, prev1[:, PREV_IDX], next1[:, NEXT_IDX],
                      prev2[:, PREV2_IDX], pos]).astype(np.float32)


# ----------------------------------------------------------------------------
# Structured decode
# ----------------------------------------------------------------------------

N_STATES = 7


def transition_matrix(depth_seqs, laplace=0.5):
    counts = np.zeros((N_STATES, N_STATES))
    for ds in depth_seqs:
        for a, b in zip(ds[:-1], ds[1:]):
            counts[a][b] += 1
    logT = np.full((N_STATES, N_STATES), -np.inf)
    for a in range(N_STATES):
        allowed = list(range(min(a + 1, N_STATES - 1) + 1))
        tot = counts[a, allowed].sum() + laplace * len(allowed)
        for b in allowed:
            logT[a, b] = np.log((counts[a, b] + laplace) / tot)
    return logT


def viterbi(log_emis, logT, lam):
    n = log_emis.shape[0]
    finite = np.isfinite(logT)
    trans = np.where(finite, lam * np.where(finite, logT, 0.0), -np.inf)
    dp = np.full(N_STATES, -np.inf)
    dp[0] = log_emis[0, 0]
    back = np.zeros((n, N_STATES), dtype=np.int8)
    for i in range(1, n):
        cand = dp[:, None] + trans
        best_prev = np.argmax(cand, axis=0)
        dp = cand[best_prev, np.arange(N_STATES)] + log_emis[i]
        back[i] = best_prev
    seq = [int(np.argmax(dp))]
    for i in range(n - 1, 0, -1):
        seq.append(int(back[i, seq[-1]]))
    return seq[::-1]


def attach_from_depths(depths):
    parents, last_at = [], {}
    for i, d in enumerate(depths):
        # defensive clamp: repair any (impossible-by-construction) gap
        while d > 0 and (d - 1) not in last_at:
            d -= 1
        parents.append(-1 if d == 0 else last_at[d - 1])
        last_at[d] = i
    return parents


def decode_acts(log_emis_flat, act_lens, logT, lam):
    """Per-act Viterbi + attach. Returns list of parent lists."""
    preds, pos = [], 0
    for n in act_lens:
        seq = viterbi(log_emis_flat[pos:pos + n], logT, lam)
        preds.append(attach_from_depths(seq))
        pos += n
    return preds


# ----------------------------------------------------------------------------
# Neural tagger
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn

_TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
PAD, UNK = 0, 1
MAX_TOK = 140


def tokenize(text):
    return _TOK.findall(text.lower())


def build_vocab(texts, min_count=2):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, c in cnt.most_common():
        if c >= min_count:
            vocab[w] = len(vocab)
    return vocab


def encode_provision(text, vocab):
    ids = [vocab.get(w, UNK) for w in tokenize(text)]
    ids_t = ids[:120] + ids[-20:] if len(ids) > MAX_TOK else ids
    if not ids_t:
        ids_t = [UNK]
    return ids_t, (ids[:8] or [UNK]), (ids[-8:] or [UNK])


class DepthTagger(nn.Module):
    def __init__(self, vocab_size, feat_dim, emb_dim=100, enc_dim=256, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.enc = nn.Sequential(
            nn.Linear(emb_dim * 3 + feat_dim, enc_dim), nn.ReLU(), nn.Dropout(0.3))
        self.lstm = nn.LSTM(enc_dim, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hid, N_STATES)

    def _mmean(self, ids):
        mask = (ids != PAD).float().unsqueeze(-1)
        return (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, ids, f8, l8, feats, act_lens):
        v = torch.cat([self._mmean(ids), self._mmean(f8), self._mmean(l8), feats], 1)
        h = self.enc(v)
        B, L = len(act_lens), max(act_lens)
        padded = torch.zeros(B, L, h.shape[1])
        pos = 0
        for b, n in enumerate(act_lens):
            padded[b, :n] = h[pos:pos + n]
            pos += n
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, torch.as_tensor(act_lens), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.head(out)
        return torch.cat([logits[b, :n] for b, n in enumerate(act_lens)], 0)


def pad_batch(seqs):
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out


class ActDataset:
    def __init__(self, provs_per_act, vocab, feats_per_act, mu, sd):
        self.acts = [
            ([encode_provision(t, vocab) for t in provs],
             ((feats - mu) / sd).astype(np.float32))
            for provs, feats in zip(provs_per_act, feats_per_act)]

    def batch(self, act_indices):
        ids, f8, l8, feats, act_lens = [], [], [], [], []
        for ai in act_indices:
            enc, z = self.acts[ai]
            act_lens.append(len(enc))
            for (a, b, c) in enc:
                ids.append(a); f8.append(b); l8.append(c)
            feats.append(z)
        return (pad_batch(ids), pad_batch(f8), pad_batch(l8),
                torch.as_tensor(np.vstack(feats)), act_lens)


@torch.no_grad()
def nn_proba(model, ds, batch_acts=64):
    model.eval()
    out = []
    for s in range(0, len(ds.acts), batch_acts):
        idxs = list(range(s, min(s + batch_acts, len(ds.acts))))
        ids, f8, l8, feats, act_lens = ds.batch(idxs)
        out.append(torch.softmax(model(ids, f8, l8, feats, act_lens), 1).numpy())
    return np.vstack(out)


def train_tagger(model, ds_tr, y_tr_per_act, ds_va, y_va_flat,
                 max_epochs=30, patience=5, batch_acts=32, lr=1e-3):
    rng = np.random.RandomState(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    ys = [torch.as_tensor(y, dtype=torch.long) for y in y_tr_per_act]
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        order = rng.permutation(len(ds_tr.acts))
        tot = cnt = 0.0
        for s in range(0, len(order), batch_acts):
            idxs = order[s:s + batch_acts]
            ids, f8, l8, feats, act_lens = ds_tr.batch(idxs)
            y = torch.cat([ys[i] for i in idxs])
            opt.zero_grad()
            loss = lossf(model(ids, f8, l8, feats, act_lens), y)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y); cnt += len(y)
        acc = float((nn_proba(model, ds_va).argmax(1) == y_va_flat).mean())
        log(f"  nn epoch {ep}: loss={tot/cnt:.4f} val_depth_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("  nn early stop")
                break
        if elapsed() > NN_STOP_ELAPSED_S:
            log("  nn time budget reached")
            break
    return best_state, best_acc


# ----------------------------------------------------------------------------
# Submission IO
# ----------------------------------------------------------------------------

def write_submission(sub_path, act_ids, preds, test_df):
    sub = pd.DataFrame({
        "act_id": act_ids,
        "parents_json": [json.dumps(p, separators=(",", ":")) for p in preds]})
    # validate: exact id set, lengths, well-foundedness
    lens = {a: len(json.loads(s)) for a, s in
            zip(test_df["act_id"], test_df["provisions_json"])}
    assert list(sub.columns) == ["act_id", "parents_json"]
    assert sub["act_id"].is_unique and set(sub["act_id"]) == set(lens)
    for a, pj in zip(sub["act_id"], sub["parents_json"]):
        arr = json.loads(pj)
        assert len(arr) == lens[a], f"len mismatch {a}"
        for i, v in enumerate(arr):
            assert isinstance(v, int) and (v == -1 or 0 <= v < i), f"bad {v}@{i} {a}"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(sub_path, index=False)
    log(f"submission written: {sub_path} ({len(sub)} acts)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    from pathlib import Path
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb

    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 8)

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    log(f"loaded train={len(train)} acts, test={len(test)} acts")

    tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
    tr_provs = [json.loads(s) for s in tr_rows["provisions_json"]]
    tr_pars = [json.loads(s) for s in tr_rows["parents_json"]]
    va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
    va_pars = [json.loads(s) for s in va_rows["parents_json"]]
    te_provs = [json.loads(s) for s in test["provisions_json"]]
    tr_depths = [depths_from_parents(p) for p in tr_pars]
    va_depths = [depths_from_parents(p) for p in va_pars]
    all_depths = [depths_from_parents(json.loads(s)) for s in train["parents_json"]]

    # ---- features ----
    feats_tr = [act_features(p) for p in tr_provs]
    feats_va = [act_features(p) for p in va_provs]
    feats_te = [act_features(p) for p in te_provs]
    Xh_tr, Xh_va, Xh_te = map(np.vstack, (feats_tr, feats_va, feats_te))
    y_tr = np.concatenate([np.asarray(d) for d in tr_depths])
    y_va = np.concatenate([np.asarray(d) for d in va_depths])
    va_lens = [len(p) for p in va_provs]
    te_lens = [len(p) for p in te_provs]
    log(f"hand features {Xh_tr.shape} (+val {Xh_va.shape}, +test {Xh_te.shape})")

    # ---- TF-IDF -> SVD (fit on train-split text only) ----
    flat_tr = [t for a in tr_provs for t in a]
    flat_va = [t for a in va_provs for t in a]
    flat_te = [t for a in te_provs for t in a]
    tw = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=150_000,
                         sublinear_tf=True)
    Tw_tr = tw.fit_transform(flat_tr)
    svd_w = TruncatedSVD(n_components=60, random_state=SEED)
    Sw_tr = svd_w.fit_transform(Tw_tr)
    Sw_va = svd_w.transform(tw.transform(flat_va))
    Sw_te = svd_w.transform(tw.transform(flat_te))
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                         max_features=200_000, sublinear_tf=True)
    Tc_tr = tc.fit_transform(flat_tr)
    svd_c = TruncatedSVD(n_components=40, random_state=SEED)
    Sc_tr = svd_c.fit_transform(Tc_tr)
    Sc_va = svd_c.transform(tc.transform(flat_va))
    Sc_te = svd_c.transform(tc.transform(flat_te))
    X_tr = np.hstack([Xh_tr, Sw_tr, Sc_tr]).astype(np.float32)
    X_va = np.hstack([Xh_va, Sw_va, Sc_va]).astype(np.float32)
    X_te = np.hstack([Xh_te, Sw_te, Sc_te]).astype(np.float32)
    log(f"full X {X_tr.shape}; tfidf+svd done")

    # ---- Model B: LightGBM ----
    params = dict(objective="multiclass", num_class=N_STATES, num_leaves=127,
                  learning_rate=0.06, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=30, num_threads=-1,
                  seed=SEED, verbose=-1, metric="multi_logloss")
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = booster.best_iteration
    p_lgb_va = booster.predict(X_va, num_iteration=best_iter)
    log(f"lgbm trained best_iter={best_iter} "
        f"val_depth_acc={(p_lgb_va.argmax(1) == y_va).mean():.4f}")

    # ---- Model A: neural tagger (guarded) ----
    p_nn_va = None
    nn_pack = None
    if elapsed() < NN_START_CUTOFF_S:
        try:
            allf = np.vstack(feats_tr)
            mu, sd = allf.mean(0), allf.std(0) + 1e-6
            vocab = build_vocab(flat_tr, min_count=2)
            ds_tr = ActDataset(tr_provs, vocab, feats_tr, mu, sd)
            ds_va = ActDataset(va_provs, vocab, feats_va, mu, sd)
            log(f"nn vocab={len(vocab)}")
            model = DepthTagger(len(vocab), allf.shape[1])
            best_state, best_acc = train_tagger(
                model, ds_tr, tr_depths, ds_va, y_va)
            if best_state is not None:
                model.load_state_dict(best_state)
                p_nn_va = nn_proba(model, ds_va)
                nn_pack = (model, vocab, mu, sd)
                log(f"nn trained val_depth_acc={best_acc:.4f}")
        except Exception as e:  # noqa: BLE001
            log(f"NN branch failed ({type(e).__name__}: {e}); LGBM-only fallback")
            p_nn_va = None
            nn_pack = None
    else:
        log("skipping NN (time guard)")

    # ---- ensemble + decode tuning on holdout ----
    logT = transition_matrix(tr_depths)
    lp_lgb = np.log(np.clip(p_lgb_va, 1e-12, None))
    alphas = [0.0] if p_nn_va is None else [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
    lp_nn = None if p_nn_va is None else np.log(np.clip(p_nn_va, 1e-12, None))
    best_cfg, best_sc = (0.0, 0.6), -1.0
    for alpha in alphas:
        le = lp_lgb if lp_nn is None else alpha * lp_nn + (1 - alpha) * lp_lgb
        for lam in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
            preds = decode_acts(le, va_lens, logT, lam)
            sc = score_sets(preds, va_pars)
            if sc > best_sc:
                best_cfg, best_sc = (alpha, lam), sc
    alpha, lam = best_cfg
    log(f"tuned alpha={alpha} lam={lam} holdout_norm_score={best_sc:.4f}")

    # ---- final LGBM refit on ALL train (cheap; NN stays split-trained) ----
    final_booster = booster
    if elapsed() < BUDGET_S - 20 * 60:
        try:
            X_all = np.vstack([X_tr, X_va])
            y_all = np.concatenate([y_tr, y_va])
            dall = lgb.Dataset(X_all, label=y_all)
            final_booster = lgb.train(params, dall, num_boost_round=best_iter)
            log("lgbm refit on all train done")
        except Exception as e:  # noqa: BLE001
            log(f"lgbm refit failed ({e}); using split-trained booster")
            final_booster = booster
    logT_full = transition_matrix(all_depths)

    # ---- test inference ----
    p_lgb_te = final_booster.predict(
        X_te, num_iteration=getattr(final_booster, "best_iteration", None) or best_iter)
    lp_te = np.log(np.clip(p_lgb_te, 1e-12, None))
    if nn_pack is not None and alpha > 0:
        try:
            model, vocab, mu, sd = nn_pack
            ds_te = ActDataset(te_provs, vocab, feats_te, mu, sd)
            p_nn_te = nn_proba(model, ds_te)
            lp_te = alpha * np.log(np.clip(p_nn_te, 1e-12, None)) + (1 - alpha) * lp_te
        except Exception as e:  # noqa: BLE001
            log(f"NN test inference failed ({e}); LGBM-only emissions")
    preds_te = decode_acts(lp_te, te_lens, logT_full, lam)
    write_submission(submission_out, list(test["act_id"]), preds_te, test)
    log(f"done. estimated_score(15% act holdout)={best_sc:.4f} "
        f"elapsed={elapsed()/60:.1f}min")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # last-resort safety net: never leave the grader without a valid file
        log(f"FATAL in main ({type(exc).__name__}: {exc}); writing chain fallback")
        from pathlib import Path
        public_dir = Path(sys.argv[1])
        submission_out = Path(sys.argv[2])
        test = pd.read_csv(public_dir / "test.csv")
        preds = [[-1] + list(range(len(json.loads(s)) - 1))
                 for s in test["provisions_json"]]
        write_submission(submission_out, list(test["act_id"]), preds, test)
        raise
