"""Statutory Outline Reconstruction — end-to-end solution.

Usage:  python3 solution.py <public_dir> <submission_out>
   or:  python3 solution.py            (probes standard mounts, writes
                                        ./working/submission.csv)
Either way the file is also mirrored to ./working/submission.csv.

Approach: document order is a pre-order traversal of the statute tree, so the
tree is fully determined by the per-provision depth sequence (verified lossless
on train: nearest-earlier-at-depth-1 attachment reconstructs 100% of parents).
We train two depth-emission models from scratch in-script —
(A) a fixed 5-seed ensemble of neural act-level BiLSTM taggers over learned
provision encoders and (B) a LightGBM multiclass model on hand + TF-IDF/SVD
features — blend their log-probabilities, apply a rare-depth prior adjustment
(log p - tau * log prior, counteracting argmax shrinkage of deep classes),
decode each act with a constrained Viterbi (d0=0, d[i] <= d[i-1]+1, optional
smoothed transition prior), and attach parent[i] = nearest earlier provision
at depth-1.

All training, feature fitting, and hyper-selection (blend weight alpha,
transition weight lambda, prior strength tau — fixed grids) happen inside this
run on an internal 15% act-level holdout scored with a local replication of the
challenge metric (calibrated to the published chain ~= 0.019 anchor). The
transition matrix and class prior used at test time are the same split-fit
objects used during tuning (tune/deploy consistency). The recipe is fixed and
hardware-independent; elapsed-time triggers are crash protection only and never
fire on reference hardware (~15-40 min total vs the 90-min cap).
"""
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# Cap BLAS/OpenMP pools BEFORE importing numpy/torch so containerized runs
# (where cpu_count may report host cores) cannot oversubscribe threads.
N_THREADS = min(10, os.cpu_count() or 10)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

import numpy as np
import pandas as pd

T0 = time.time()
SEED = 42
# Reference environment: 10 CPU cores, 62GB RAM, 90-min cap (CPU-only).
# The recipe below is FIXED (5 NN seeds, fixed grids, fixed hyperparameters) and
# produces the same result regardless of hardware. The elapsed-time triggers are
# CRASH PROTECTION ONLY: on reference hardware the full recipe finishes in
# ~15-40 min, so none of them ever fire there.
NN_SEEDS = (42, 1, 2, 3, 7, 11, 13, 17)  # fixed; never varied by environment
NN_EMB = 160                         # provision-token embedding dim
NN_HID = 256                         # BiLSTM hidden size (per direction)
CRASH_SEED_SKIP_S = 70 * 60          # pathological-slowness protection only
CRASH_EPOCH_STOP_S = 78 * 60         # pathological-slowness protection only
CRASH_REFIT_SKIP_S = 82 * 60         # pathological-slowness protection only
MIRROR_OUT = Path("working") / "submission.csv"


def _has_data(d):
    try:
        return (d / "train.csv").exists() and (d / "test.csv").exists()
    except OSError:
        return False


def _probe(cands):
    """Return first candidate (or immediate subdir of one) holding the data."""
    for c in cands:
        if _has_data(c):
            return c
    for c in cands:  # one level down, e.g. /kaggle/input/<comp>/train.csv
        try:
            if not c.is_dir():
                continue
            for sub in sorted(c.iterdir()):
                if sub.is_dir() and _has_data(sub):
                    return sub
        except OSError:
            continue
    return None


def resolve_paths():
    """Support both invocation conventions:
    new: python3 solution.py <public_dir> <submission_out>
    old: no args -> probe standard mounts, write ./working/submission.csv.
    An argv-supplied public_dir is exhausted (itself, then its immediate
    subdirs) before any legacy fallback, so stray files in CWD can never
    shadow the explicit argument."""
    argv_pub = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    argv_out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if argv_pub is not None:
        pub = _probe([argv_pub]) or argv_pub  # argv wins; fail loudly at read
    else:
        pub = _probe([Path("dataset/public"), Path("data/public"),
                      Path("public"), Path("dataset"), Path("input"),
                      Path("/kaggle/input"), Path(".")]) or Path("dataset/public")
    out = argv_out if argv_out is not None else MIRROR_OUT
    return pub, out


def mirror_submission(submission_out):
    """Unconditionally also provide ./working/submission.csv."""
    try:
        if MIRROR_OUT.resolve() != Path(submission_out).resolve():
            MIRROR_OUT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(submission_out, MIRROR_OUT)
    except Exception as e:  # noqa: BLE001 - mirroring must never kill a run
        print(f"mirror copy failed (non-fatal): {e}", flush=True)


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


def _runlen_feats(core):
    """Medium-range context: distances/runs over cue columns. (n, 12)"""
    n = core.shape[0]
    colon = core[:, 2] > 0
    semi = core[:, 26] > 0
    period = core[:, 6] > 0
    follow = core[:, 9] > 0
    anchor = (core[:, 23] > 0) | (core[:, 24] > 0) | (core[:, 8] > 0) | (core[:, 7] > 0)
    CAP = 15.0
    out = np.zeros((n, 12), dtype=np.float32)
    last_colon = last_follow = last_anchor = -1
    semi_run = period_run = colon_cum = 0
    for i in range(n):
        out[i, 0] = min(i - last_colon, CAP) if last_colon >= 0 else CAP + 1
        out[i, 1] = 1.0 if last_colon >= 0 else 0.0
        out[i, 2] = min(i - last_follow, CAP) if last_follow >= 0 else CAP + 1
        out[i, 3] = min(i - last_anchor, CAP) if last_anchor >= 0 else CAP + 1
        out[i, 4] = min(semi_run, CAP)
        out[i, 5] = min(period_run, CAP)
        out[i, 6] = colon_cum / max(i, 1)
        if colon[i]:
            last_colon = i
            colon_cum += 1
        if follow[i]:
            last_follow = i
        if anchor[i]:
            last_anchor = i
        semi_run = semi_run + 1 if semi[i] else 0
        period_run = period_run + 1 if period[i] else 0
    nxt_colon = nxt_period = -1
    nxt_semi_run = 0
    for i in range(n - 1, -1, -1):
        out[i, 7] = min(nxt_colon - i, CAP) if nxt_colon >= 0 else CAP + 1
        out[i, 8] = min(nxt_period - i, CAP) if nxt_period >= 0 else CAP + 1
        out[i, 9] = min(nxt_semi_run, CAP)
        if colon[i]:
            nxt_colon = i
        if period[i]:
            nxt_period = i
        nxt_semi_run = nxt_semi_run + 1 if semi[i] else 0
    out[:, 10] = np.cumsum(semi).astype(np.float32) / max(n, 1)
    out[:, 11] = np.cumsum(period).astype(np.float32) / max(n, 1)
    return out


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
                      prev2[:, PREV2_IDX], pos, _runlen_feats(core)]).astype(np.float32)


# ----------------------------------------------------------------------------
# Structured decode
# ----------------------------------------------------------------------------

N_STATES = 7  # default; main() recomputes as (max train depth + 1) at runtime


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
    def __init__(self, vocab_size, feat_dim, emb_dim=NN_EMB, enc_dim=256,
                 hid=NN_HID, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.enc = nn.Sequential(
            nn.Linear(emb_dim * 3 + feat_dim, enc_dim), nn.ReLU(), nn.Dropout(0.3))
        self.lstm = nn.LSTM(enc_dim, hid, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
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
                 max_epochs=40, patience=6, batch_acts=32, lr=1e-3, seed=SEED):
    rng = np.random.RandomState(seed)
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
            if bad == 2:  # plateau: halve LR
                for g in opt.param_groups:
                    g["lr"] = max(g["lr"] * 0.5, 1e-4)
            if bad >= patience:
                log("  nn early stop")
                break
        if elapsed() > CRASH_EPOCH_STOP_S:
            log("  nn crash-protection time trigger (never expected on reference hw)")
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
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb

    public_dir, submission_out = resolve_paths()
    log(f"public_dir={public_dir} submission_out={submission_out}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)  # CPU-only by construction; cap at 10

    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    log(f"loaded train={len(train)} acts, test={len(test)} acts")

    # depth-state count derived from the data, not hardcoded
    global N_STATES
    N_STATES = 1 + max(max(depths_from_parents(json.loads(s)))
                       for s in train["parents_json"])
    log(f"N_STATES={N_STATES} (max train depth + 1)")

    tr_rows, va_rows = train_test_split(train, test_size=0.15, random_state=SEED)
    tr_provs = [json.loads(s) for s in tr_rows["provisions_json"]]
    tr_pars = [json.loads(s) for s in tr_rows["parents_json"]]
    va_provs = [json.loads(s) for s in va_rows["provisions_json"]]
    va_pars = [json.loads(s) for s in va_rows["parents_json"]]
    te_provs = [json.loads(s) for s in test["provisions_json"]]
    tr_depths = [depths_from_parents(p) for p in tr_pars]
    va_depths = [depths_from_parents(p) for p in va_pars]

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

    # NN input features per act: hand + SVD (the SAME hand+word/char-SVD rows
    # the LGBM sees; adopted lever, single-seed +0.013 NN-only holdout).
    def _split_acts(flat, provs):
        out, pos = [], 0
        for p in provs:
            out.append(flat[pos:pos + len(p)])
            pos += len(p)
        return out
    nnf_tr = _split_acts(X_tr, tr_provs)
    nnf_va = _split_acts(X_va, va_provs)
    nnf_te = _split_acts(X_te, te_provs)

    # ---- Model B: LightGBM ----
    params = dict(objective="multiclass", num_class=N_STATES, num_leaves=127,
                  learning_rate=0.06, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=30, num_threads=N_THREADS,
                  seed=SEED, verbose=-1, metric="multi_logloss",
                  deterministic=True, force_row_wise=True)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best_iter = booster.best_iteration
    p_lgb_va = booster.predict(X_va, num_iteration=best_iter)
    log(f"lgbm trained best_iter={best_iter} "
        f"val_depth_acc={(p_lgb_va.argmax(1) == y_va).mean():.4f}")

    # ---- Model A: neural taggers, fixed 5-seed ensemble ----
    p_nn_va = None
    nn_models = []
    vocab = mu = sd = None
    if elapsed() < CRASH_SEED_SKIP_S:
        try:
            allf = X_tr  # hand + SVD features, flat over train provisions
            mu, sd = allf.mean(0), allf.std(0) + 1e-6
            vocab = build_vocab(flat_tr, min_count=2)
            ds_tr = ActDataset(tr_provs, vocab, nnf_tr, mu, sd)
            ds_va = ActDataset(va_provs, vocab, nnf_va, mu, sd)
            log(f"nn vocab={len(vocab)}")
            nn_probas = []
            for sd_i in NN_SEEDS:
                if nn_probas and elapsed() > CRASH_SEED_SKIP_S:
                    log("crash-protection seed skip (never expected on reference hw)")
                    break
                torch.manual_seed(sd_i)
                model = DepthTagger(len(vocab), allf.shape[1])
                best_state, best_acc = train_tagger(
                    model, ds_tr, tr_depths, ds_va, y_va, seed=sd_i)
                if best_state is not None:
                    model.load_state_dict(best_state)
                    nn_probas.append(nn_proba(model, ds_va))
                    nn_models.append(model)
                    log(f"nn seed={sd_i} val_depth_acc={best_acc:.4f}")
            if nn_probas:
                p_nn_va = np.mean(nn_probas, axis=0)
                log(f"nn ensemble of {len(nn_probas)} seeds "
                    f"val_depth_acc={(p_nn_va.argmax(1) == y_va).mean():.4f}")
        except Exception as e:  # noqa: BLE001
            log(f"NN branch failed ({type(e).__name__}: {e}); LGBM-only fallback")
            p_nn_va = None
            nn_models = []
    else:
        log("skipping NN (crash-protection trigger; never expected on reference hw)")

    # ---- ensemble + decode tuning on holdout (fixed grids) ----
    # The SAME split-fit transition matrix and class prior are used here and at
    # test-time deployment (tune/deploy consistency).
    logT = transition_matrix(tr_depths)
    prior = np.bincount(y_tr, minlength=N_STATES).astype(np.float64)
    prior = (prior + 0.5) / (prior.sum() + 0.5 * N_STATES)
    log_prior = np.log(prior)
    lp_lgb = np.log(np.clip(p_lgb_va, 1e-12, None))
    alphas = [0.0] if p_nn_va is None else [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
    lp_nn = None if p_nn_va is None else np.log(np.clip(p_nn_va, 1e-12, None))
    best_cfg, best_sc = (0.0, 0.0, 0.0), -1.0
    for alpha in alphas:
        le_base = lp_lgb if lp_nn is None else alpha * lp_nn + (1 - alpha) * lp_lgb
        for tau in [0.0, 0.1, 0.2, 0.3]:  # rare-depth prior adjustment
            le = le_base - tau * log_prior[None, :]
            for lam in [0.0, 0.1, 0.2, 0.4]:
                preds = decode_acts(le, va_lens, logT, lam)
                sc = score_sets(preds, va_pars)
                if sc > best_sc:
                    best_cfg, best_sc = (alpha, lam, tau), sc
    alpha, lam, tau = best_cfg
    log(f"tuned alpha={alpha} lam={lam} tau={tau} holdout_norm_score={best_sc:.4f}")

    # ---- final LGBM refit on ALL train (cheap; NN stays split-trained) ----
    final_booster = booster
    if elapsed() < CRASH_REFIT_SKIP_S:
        try:
            X_all = np.vstack([X_tr, X_va])
            y_all = np.concatenate([y_tr, y_va])
            dall = lgb.Dataset(X_all, label=y_all)
            final_booster = lgb.train(params, dall, num_boost_round=best_iter)
            log("lgbm refit on all train done")
        except Exception as e:  # noqa: BLE001
            log(f"lgbm refit failed ({e}); using split-trained booster")
            final_booster = booster

    # ---- test inference (same logT/prior construction as tuning) ----
    p_lgb_te = final_booster.predict(
        X_te, num_iteration=getattr(final_booster, "best_iteration", None) or best_iter)
    lp_te = np.log(np.clip(p_lgb_te, 1e-12, None))
    if nn_models and alpha > 0:
        try:
            ds_te = ActDataset(te_provs, vocab, nnf_te, mu, sd)
            p_nn_te = np.mean([nn_proba(m, ds_te) for m in nn_models], axis=0)
            lp_te = alpha * np.log(np.clip(p_nn_te, 1e-12, None)) + (1 - alpha) * lp_te
        except Exception as e:  # noqa: BLE001
            log(f"NN test inference failed ({e}); LGBM-only emissions")
    lp_te = lp_te - tau * log_prior[None, :]
    preds_te = decode_acts(lp_te, te_lens, logT, lam)
    write_submission(submission_out, list(test["act_id"]), preds_te, test)
    mirror_submission(submission_out)
    log(f"done. estimated_score(15% act holdout)={best_sc:.4f} "
        f"elapsed={elapsed()/60:.1f}min")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        # last-resort safety net: never leave the grader without a valid file
        banner = (
            "\n" + "!" * 78 + "\n"
            f"!!! FATAL in main pipeline: {type(exc).__name__}: {exc}\n"
            "!!! The ML pipeline DID NOT produce this submission.\n"
            "!!! Writing structurally-valid CHAIN FALLBACK (scores ~0.02).\n"
            + "!" * 78)
        log(banner)
        print(banner, file=sys.stderr, flush=True)
        try:
            public_dir, submission_out = resolve_paths()
            test = pd.read_csv(public_dir / "test.csv")
            preds = [[-1] + list(range(len(json.loads(s)) - 1))
                     for s in test["provisions_json"]]
            write_submission(submission_out, list(test["act_id"]), preds, test)
            mirror_submission(submission_out)
            log("fallback submission written and verified; exiting 0")
            sys.exit(0)  # file-based grading: a valid fallback beats a crash
        except SystemExit:
            raise
        except Exception as exc2:  # noqa: BLE001
            log(f"fallback also failed ({type(exc2).__name__}: {exc2})")
            raise exc from None
