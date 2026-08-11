"""Milestones 1+2: calibrate metric variants against chain~=0.019 anchor,
verify all-(-1)=0, oracle=1, and lossless nearest-earlier attach on train."""
import json
import sys
import time

import pandas as pd

from metric import act_score, depths_from_parents

DATA = r"G:\ml\Latest_Chals\Challenge 3\dataset"


def attach_from_depths(depths):
    parents = []
    last_at_depth = {}
    for i, d in enumerate(depths):
        if d == 0:
            parents.append(-1)
        else:
            parents.append(last_at_depth[d - 1])
        last_at_depth[d] = i
        # clear deeper levels not needed: nearest-earlier lookup only ever
        # sees the most recent index at each depth, which we overwrite.
    return parents


def main():
    t0 = time.time()
    train = pd.read_csv(rf"{DATA}\train.csv")
    trues = [json.loads(s) for s in train["parents_json"]]
    n_acts = len(trues)
    print(f"acts={n_acts} provisions={sum(len(t) for t in trues)}")

    # Milestone 2: lossless attach check
    bad = 0
    for t in trues:
        d = depths_from_parents(t)
        if attach_from_depths(d) != t:
            bad += 1
    print(f"[M2] lossless attach mismatches: {bad}/{n_acts}")

    variants = [
        ("rootgrp=T empty=1", True, 1.0),
        ("rootgrp=T empty=0", True, 0.0),
        ("rootgrp=F empty=1", False, 1.0),
        ("rootgrp=F empty=0", False, 0.0),
    ]
    for name, rg, ec in variants:
        chain_s = flat_s = oracle_s = 0.0
        for t in trues:
            n = len(t)
            chain = [-1] + list(range(n - 1))
            flat = [-1] * n
            chain_s += act_score(chain, t, rg, ec)
            flat_s += act_score(flat, t, rg, ec)
            oracle_s += act_score(list(t), t, rg, ec)
        print(
            f"[M1] {name}: chain={chain_s/n_acts:.4f} "
            f"flat={flat_s/n_acts:.4f} oracle={oracle_s/n_acts:.4f}"
        )
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
