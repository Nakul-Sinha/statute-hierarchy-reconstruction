"""Replicated metric for Statutory Outline Reconstruction.

Per act:
  parent_acc (0.55) + depth_agreement (0.25) + sibling_pair_F1 (0.20) = raw
  score_act = max(0, (raw - trivial) / (1 - trivial))   [trivial = all -1 on same act]
Final = mean over acts.

Ambiguities (resolved empirically against chain ~= 0.019 anchor):
  root_group: whether provisions with parent -1 form a sibling group for F1
  empty_conv: F1 value when BOTH pred and true pair sets are empty (1.0 or 0.0)
"""
from __future__ import annotations


def depths_from_parents(parents):
    d = [0] * len(parents)
    for i, p in enumerate(parents):
        d[i] = 0 if p == -1 else d[p] + 1
    return d


def sibling_pairs(parents, root_group=True):
    """Set of unordered index pairs sharing a parent."""
    groups = {}
    for i, p in enumerate(parents):
        groups.setdefault(p, []).append(i)
    pairs = set()
    for p, members in groups.items():
        if p == -1 and not root_group:
            continue
        m = len(members)
        for a in range(m):
            for b in range(a + 1, m):
                pairs.add((members[a], members[b]))
    return pairs


def sibling_f1(pred, true, root_group=True, empty_conv=1.0):
    pp = sibling_pairs(pred, root_group)
    tp = sibling_pairs(true, root_group)
    if not pp and not tp:
        return empty_conv
    if not pp or not tp:
        return 0.0
    inter = len(pp & tp)
    prec = inter / len(pp)
    rec = inter / len(tp)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def raw_score(pred, true, root_group=True, empty_conv=1.0, true_depths=None):
    n = len(true)
    pacc = sum(1 for a, b in zip(pred, true) if a == b) / n
    td = true_depths if true_depths is not None else depths_from_parents(true)
    pd_ = depths_from_parents(pred)
    span = max(max(td), 1)
    depth_comp = sum(max(0.0, 1.0 - abs(a - b) / span) for a, b in zip(pd_, td)) / n
    f1 = sibling_f1(pred, true, root_group, empty_conv)
    return 0.55 * pacc + 0.25 * depth_comp + 0.20 * f1


def act_score(pred, true, root_group=True, empty_conv=1.0):
    td = depths_from_parents(true)
    raw = raw_score(pred, true, root_group, empty_conv, true_depths=td)
    trivial_pred = [-1] * len(true)
    trivial = raw_score(trivial_pred, true, root_group, empty_conv, true_depths=td)
    if trivial >= 1.0:
        return 1.0 if raw >= 1.0 else 0.0
    return max(0.0, (raw - trivial) / (1.0 - trivial))


def mean_score(preds, trues, root_group=True, empty_conv=1.0):
    """preds/trues: lists of parent lists, aligned by act."""
    tot = 0.0
    for p, t in zip(preds, trues):
        tot += act_score(p, t, root_group, empty_conv)
    return tot / len(preds)


def components(preds, trues, root_group=True, empty_conv=1.0):
    """Diagnostic per-component means (unnormalized) + normalized score."""
    n_acts = len(preds)
    pacc_s = depth_s = f1_s = norm_s = 0.0
    for p, t in zip(preds, trues):
        n = len(t)
        td = depths_from_parents(t)
        pd_ = depths_from_parents(p)
        span = max(max(td), 1)
        pacc_s += sum(1 for a, b in zip(p, t) if a == b) / n
        depth_s += sum(max(0.0, 1.0 - abs(a - b) / span) for a, b in zip(pd_, td)) / n
        f1_s += sibling_f1(p, t, root_group, empty_conv)
        norm_s += act_score(p, t, root_group, empty_conv)
    return {
        "parent_acc": pacc_s / n_acts,
        "depth": depth_s / n_acts,
        "sib_f1": f1_s / n_acts,
        "normalized": norm_s / n_acts,
    }
