"""Shared pipeline components: features, transitions, Viterbi, attach, writer."""
from __future__ import annotations

import json
import re

import numpy as np

MAX_DEPTH = 6
N_STATES = MAX_DEPTH + 1

_BY_GER = re.compile(r"^by\s+\w+ing\b")
_SEC_REF = re.compile(r"^Section\s+\d")
_PARA = re.compile(r"(?<!sub)paragraph \(")
_SUBPARA = re.compile(r"subparagraph \(")
_CLAUSE = re.compile(r"(?<!sub)clause \(")
_SUBSEC = re.compile(r"subsection \(")

CORE_DIM = 27


def core_feats(t: str) -> list:
    """Per-provision cue vector (order fixed; neighbors reuse indices)."""
    s = t.rstrip()
    low = t.lower()
    first = s[0] if s else " "
    words = t.split()
    wc = len(words)
    return [
        first.islower(),                                   # 0 starts_lower
        first.isdigit(),                                   # 1 starts_digit
        s.endswith((":", "—", "--", "–")),       # 2 ends_colon/dash
        s.endswith(";"),                                   # 3 ends_semi
        s.endswith(("; and", "; or")),                     # 4 ends_semi_andor
        s.endswith((", and", ", or")),                     # 5 ends_comma_andor
        s.endswith("."),                                   # 6 ends_period
        ("is amended" in t) or ("are amended" in t),       # 7 has_amended
        "may be cited as" in t,                            # 8 has_cited
        "the following" in low[-60:],                      # 9 following_tail
        "$" in t,                                          # 10 has_dollar
        bool(_BY_GER.match(low)),                          # 11 starts_by_ger
        "notwithstanding" in low,                          # 12 notwithstanding
        len(_SUBSEC.findall(low)),                         # 13 xref_subsection
        len(_PARA.findall(low)),                           # 14 xref_paragraph
        len(_SUBPARA.findall(low)),                        # 15 xref_subparagraph
        len(_CLAUSE.findall(low)),                         # 16 xref_clause
        ("the term" in low) and ("means" in low),          # 17 term_means
        wc,                                                # 18 word_count
        np.log1p(wc),                                      # 19 log_wc
        min(len(t), 1200) / 1200.0,                        # 20 char_count_norm
        wc < 15,                                           # 21 short15
        low.endswith("the following:"),                    # 22 ends_following_colon
        s.startswith("That "),                             # 23 starts_that
        bool(_SEC_REF.match(s)),                           # 24 starts_section_ref
        "this act" in low,                                 # 25 has_this_act
        s.endswith((";", "; and", "; or", ", and", ", or")),  # 26 semi_family
    ]


# neighbor feature index lists (into core vector)
PREV_IDX = [0, 2, 26, 9, 7, 6, 21, 11, 8]
NEXT_IDX = [0, 11, 26, 2, 21, 1]
PREV2_IDX = [2, 26, 0]


def _runlen_feats(core: np.ndarray) -> np.ndarray:
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
    semi_run = 0
    period_run = 0
    colon_cum = 0
    for i in range(n):
        out[i, 0] = min(i - last_colon, CAP) if last_colon >= 0 else CAP + 1
        out[i, 1] = 1.0 if last_colon >= 0 else 0.0
        out[i, 2] = min(i - last_follow, CAP) if last_follow >= 0 else CAP + 1
        out[i, 3] = min(i - last_anchor, CAP) if last_anchor >= 0 else CAP + 1
        out[i, 4] = min(semi_run, CAP)      # consecutive semi-enders just before i
        out[i, 5] = min(period_run, CAP)
        out[i, 6] = colon_cum / max(i, 1)   # fraction of colon-enders before i
        if colon[i]:
            last_colon = i
            colon_cum += 1
        if follow[i]:
            last_follow = i
        if anchor[i]:
            last_anchor = i
        semi_run = semi_run + 1 if semi[i] else 0
        period_run = period_run + 1 if period[i] else 0
    # lookahead distances
    nxt_colon = nxt_period = -1
    nxt_semi_run = 0
    for i in range(n - 1, -1, -1):
        out[i, 7] = min(nxt_colon - i, CAP) if nxt_colon >= 0 else CAP + 1
        out[i, 8] = min(nxt_period - i, CAP) if nxt_period >= 0 else CAP + 1
        out[i, 9] = min(nxt_semi_run, CAP)  # consecutive semi-enders just after i
        if colon[i]:
            nxt_colon = i
        if period[i]:
            nxt_period = i
        nxt_semi_run = nxt_semi_run + 1 if semi[i] else 0
    out[:, 10] = np.cumsum(semi).astype(np.float32) / max(n, 1)
    out[:, 11] = np.cumsum(period).astype(np.float32) / max(n, 1)
    return out


def act_features(provs: list) -> np.ndarray:
    """Feature matrix (n, F) for one act's provisions."""
    n = len(provs)
    core = np.asarray([core_feats(t) for t in provs], dtype=np.float32)
    zeros = np.zeros(CORE_DIM, dtype=np.float32)
    prev1 = np.vstack([zeros, core[:-1]]) if n > 1 else zeros[None, :]
    next1 = np.vstack([core[1:], zeros]) if n > 1 else zeros[None, :]
    prev2 = np.vstack([zeros, zeros, core[:-2]]) if n > 2 else np.zeros((n, CORE_DIM), dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    pos = np.stack(
        [
            idx,
            idx / max(n - 1, 1),
            np.full(n, n, dtype=np.float32),
            (idx == 0).astype(np.float32),
            (idx == n - 1).astype(np.float32),
        ],
        axis=1,
    )
    return np.hstack(
        [core, prev1[:, PREV_IDX], next1[:, NEXT_IDX], prev2[:, PREV2_IDX], pos,
         _runlen_feats(core)]
    ).astype(np.float32)


def build_features(provs_per_act: list) -> np.ndarray:
    """Stack act_features over acts -> (total_provisions, F)."""
    return np.vstack([act_features(p) for p in provs_per_act])


def transition_matrix(depth_seqs, laplace=0.5) -> np.ndarray:
    """Smoothed P(d_i | d_{i-1}); hard mask b > a+1. Returns log matrix."""
    counts = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for ds in depth_seqs:
        for a, b in zip(ds[:-1], ds[1:]):
            counts[a][b] += 1
    logT = np.full((N_STATES, N_STATES), -np.inf)
    for a in range(N_STATES):
        allowed = [b for b in range(N_STATES) if b <= a + 1]
        tot = counts[a, allowed].sum() + laplace * len(allowed)
        for b in allowed:
            logT[a, b] = np.log((counts[a, b] + laplace) / tot)
    return logT


def viterbi(log_emis: np.ndarray, logT: np.ndarray, lam: float) -> list:
    """Best depth seq: d0=0, d_i <= d_{i-1}+1. log_emis: (n, 7)."""
    n = log_emis.shape[0]
    finite = np.isfinite(logT)
    trans = np.where(finite, lam * np.where(finite, logT, 0.0), -np.inf)
    dp = np.full(N_STATES, -np.inf)
    dp[0] = log_emis[0, 0]
    back = np.zeros((n, N_STATES), dtype=np.int8)
    for i in range(1, n):
        cand = dp[:, None] + trans  # (prev, cur)
        best_prev = np.argmax(cand, axis=0)
        dp = cand[best_prev, np.arange(N_STATES)] + log_emis[i]
        back[i] = best_prev
    seq = [int(np.argmax(dp))]
    for i in range(n - 1, 0, -1):
        seq.append(int(back[i, seq[-1]]))
    return seq[::-1]


def attach_from_depths(depths: list) -> list:
    parents = []
    last_at = {}
    for i, d in enumerate(depths):
        parents.append(-1 if d == 0 else last_at[d - 1])
        last_at[d] = i
    return parents


def validate_submission(sub_df, test_df) -> None:
    """Raise AssertionError on any grader-fatal format problem."""
    assert list(sub_df.columns) == ["act_id", "parents_json"], "bad columns"
    assert len(sub_df) == len(test_df), "row count mismatch"
    assert sub_df["act_id"].is_unique, "duplicate act ids"
    assert set(sub_df["act_id"]) == set(test_df["act_id"]), "act id set mismatch"
    lens = {a: len(json.loads(p)) for a, p in zip(test_df["act_id"], test_df["provisions_json"])}
    for a, pj in zip(sub_df["act_id"], sub_df["parents_json"]):
        arr = json.loads(pj)
        assert len(arr) == lens[a], f"length mismatch for {a}"
        for i, v in enumerate(arr):
            assert isinstance(v, int) and (v == -1 or 0 <= v < i), f"bad parent {v}@{i} in {a}"
