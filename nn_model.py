"""Neural depth tagger: provision encoder + act BiLSTM, trained from scratch."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

_TOK = re.compile(r"[a-z]+|\d+|[^\sa-z\d]")
PAD, UNK = 0, 1
MAX_TOK = 140  # first 120 + last 20


def tokenize(text: str) -> list:
    return _TOK.findall(text.lower())


def build_vocab(texts, min_count=2) -> dict:
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, c in cnt.most_common():
        if c >= min_count:
            vocab[w] = len(vocab)
    return vocab


def encode_provision(text: str, vocab: dict):
    """Returns (ids_truncated, first8, last8) as int lists."""
    toks = tokenize(text)
    ids = [vocab.get(w, UNK) for w in toks]
    if len(ids) > MAX_TOK:
        ids_t = ids[:120] + ids[-20:]
    else:
        ids_t = ids
    if not ids_t:
        ids_t = [UNK]
    f8 = ids[:8] if ids else [UNK]
    l8 = ids[-8:] if ids else [UNK]
    return ids_t, f8, l8


class DepthTagger(nn.Module):
    def __init__(self, vocab_size, feat_dim, emb_dim=100, enc_dim=256, lstm_hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        in_dim = emb_dim * 3 + feat_dim
        self.enc = nn.Sequential(nn.Linear(in_dim, enc_dim), nn.ReLU(), nn.Dropout(0.3))
        self.lstm = nn.LSTM(enc_dim, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * lstm_hidden, 7)

    def _masked_mean(self, ids):
        # ids: (P, T) padded with PAD
        mask = (ids != PAD).float().unsqueeze(-1)  # (P, T, 1)
        emb = self.emb(ids)  # (P, T, E)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, ids, f8, l8, feats, act_lens):
        """ids/f8/l8: padded (P, *) token ids; feats: (P, F);
        act_lens: list of provisions per act (sum = P). Returns (P, 7) logits."""
        v = torch.cat(
            [self._masked_mean(ids), self._masked_mean(f8), self._masked_mean(l8), feats],
            dim=1)
        h = self.enc(v)  # (P, enc)
        # scatter into (B, L, enc)
        B = len(act_lens)
        L = max(act_lens)
        dev = h.device
        padded = torch.zeros(B, L, h.shape[1], device=dev)
        pos = 0
        for b, n in enumerate(act_lens):
            padded[b, :n] = h[pos:pos + n]
            pos += n
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, torch.as_tensor(act_lens), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.head(out)  # (B, L, 7)
        flat = []
        for b, n in enumerate(act_lens):
            flat.append(logits[b, :n])
        return torch.cat(flat, dim=0)


def pad_batch(seqs, device):
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out.to(device)


class ActDataset:
    """Precomputed per-act encodings + standardized hand features."""

    def __init__(self, provs_per_act, vocab, feats_per_act, mu, sd):
        self.acts = []
        for provs, feats in zip(provs_per_act, feats_per_act):
            enc = [encode_provision(t, vocab) for t in provs]
            z = ((feats - mu) / sd).astype(np.float32)
            self.acts.append((enc, z))

    def batch(self, act_indices, device):
        ids, f8, l8, feats, act_lens = [], [], [], [], []
        for ai in act_indices:
            enc, z = self.acts[ai]
            act_lens.append(len(enc))
            for (i_, a_, b_) in enc:
                ids.append(i_); f8.append(a_); l8.append(b_)
            feats.append(z)
        return (pad_batch(ids, device), pad_batch(f8, device), pad_batch(l8, device),
                torch.as_tensor(np.vstack(feats), device=device), act_lens)


def train_tagger(model, ds_tr, y_tr_per_act, ds_va, y_va_flat, device,
                 max_epochs=30, patience=5, batch_acts=32, lr=1e-3,
                 seed=42, time_left_fn=None, log=print):
    """Early stopping on val depth accuracy. Returns best state dict."""
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    n_acts = len(ds_tr.acts)
    y_flat_per_act = [torch.as_tensor(y, dtype=torch.long) for y in y_tr_per_act]
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        order = rng.permutation(n_acts)
        tot_loss = tot_n = 0
        for s in range(0, n_acts, batch_acts):
            idxs = order[s:s + batch_acts]
            ids, f8, l8, feats, act_lens = ds_tr.batch(idxs, device)
            y = torch.cat([y_flat_per_act[i] for i in idxs]).to(device)
            opt.zero_grad()
            logits = model(ids, f8, l8, feats, act_lens)
            loss = lossf(logits, y)
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(y)
            tot_n += len(y)
        acc = eval_depth_acc(model, ds_va, y_va_flat, device)
        log(f"epoch {ep}: loss={tot_loss/tot_n:.4f} val_depth_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {ep}")
                break
        if time_left_fn is not None and not time_left_fn():
            log("time budget reached; stopping training")
            break
    return best_state, best_acc


@torch.no_grad()
def predict_proba(model, ds, device, batch_acts=64):
    model.eval()
    out = []
    n = len(ds.acts)
    for s in range(0, n, batch_acts):
        idxs = list(range(s, min(s + batch_acts, n)))
        ids, f8, l8, feats, act_lens = ds.batch(idxs, device)
        logits = model(ids, f8, l8, feats, act_lens)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def eval_depth_acc(model, ds, y_flat, device):
    proba = predict_proba(model, ds, device)
    return float((proba.argmax(1) == y_flat).mean())
