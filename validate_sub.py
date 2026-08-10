"""Independent grader-style validation of a submission.csv against test.csv."""
import json
import sys

import pandas as pd

test_path, sub_path = sys.argv[1], sys.argv[2]
test = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)

errors = []
if set(sub.columns) != {"act_id", "parents_json"}:
    errors.append(f"bad columns: {list(sub.columns)}")
if len(sub) != len(test):
    errors.append(f"row count {len(sub)} != {len(test)}")
if sub["act_id"].duplicated().any():
    errors.append("duplicate act ids")
if set(sub["act_id"]) != set(test["act_id"]):
    errors.append("act id set mismatch")

lens = {a: len(json.loads(s)) for a, s in zip(test["act_id"], test["provisions_json"])}
n_bad_rows = 0
depth_hist = {}
for a, pj in zip(sub["act_id"], sub["parents_json"]):
    try:
        arr = json.loads(pj)
    except Exception:
        n_bad_rows += 1
        continue
    ok = (isinstance(arr, list) and len(arr) == lens.get(a, -1)
          and all(isinstance(v, int) and (v == -1 or 0 <= v < i)
                  for i, v in enumerate(arr)))
    if not ok:
        n_bad_rows += 1
        continue
    d = [0] * len(arr)
    for i, p in enumerate(arr):
        d[i] = 0 if p == -1 else d[p] + 1
    for x in d:
        depth_hist[x] = depth_hist.get(x, 0) + 1

if n_bad_rows:
    errors.append(f"{n_bad_rows} structurally invalid rows")

if errors:
    print("FAIL:", "; ".join(errors))
    sys.exit(1)
tot = sum(depth_hist.values())
print(f"PASS: {len(sub)} acts, {tot} provisions")
print("pred depth distribution:",
      {k: round(v / tot, 4) for k, v in sorted(depth_hist.items())})
