# Eris ch3 — Statutory Outline Reconstruction

Rebuild the nesting tree of US public-law provisions (parent index per provision).
Metric: 0.55 parent-acc + 0.25 depth agreement + 0.20 sibling pair-F1, normalized
per act against the all-top-level trivial answer (chain baseline ~0.019).

## Core insight

Document order is an exact pre-order traversal: given true depths,
`parent(i) = nearest earlier provision at depth-1` reconstructs 100% of train
parents. So the task reduces losslessly to tagging the per-provision depth
sequence under `d[0]=0`, `d[i] <= d[i-1]+1`, `d <= 6`.

## Solution (`solution.py`, single end-to-end script)

1. 15% act-level internal holdout (seed 42) for early stopping + tuning.
2. Hand features (62 dims): drafting cues (case, terminal punctuation,
   amendment/citation phrases, xref granularity, lengths) + neighbor flags +
   run-length/distance context (distance since last colon-ender, semicolon runs,
   block position).
3. Model B: LightGBM 7-class depth on hand features + word/char TF-IDF->SVD
   (fit on train split only), early-stopped, then refit on all train.
4. Model A: 3-seed ensemble of act-level BiLSTM taggers (from-scratch embeddings,
   provision encoder = mean/first8/last8 embedding pools + hand features -> MLP;
   2-layer BiLSTM hid=160; 7-way head), early-stopped on holdout depth acc.
5. Ensemble log-probs (alpha tuned in-run, typically 0.8 NN-heavy), decode each
   act with constrained Viterbi (hard pre-order mask; empirical transition prior
   weight lambda tuned in-run, typically 0), attach parents losslessly, validate,
   write.
6. Wall-clock guards: seed loop stops if slow; NN failure falls back to
   LGBM-only; fatal failure still writes a structurally valid submission.

Holdout (562 unseen acts): normalized 0.587, parent-acc 0.715, depth 0.856,
sibling-F1 0.708. Runtime ~35-50 min on 10-12 CPU threads.

## Files

- `solution.py` — the submission script (`python3 solution.py <public_dir> <out_csv>`)
- `metric.py` / `calibrate.py` — replicated metric, calibrated to the chain=0.019 anchor
- `pipeline.py`, `nn_model.py` — shared components used by experiments
- `exp_*.py` — experiment drivers (LGBM, NN, ensemble, decode/MBR/temperature studies)
- `validate_sub.py` — independent grader-style submission validation

Negative results kept for the record: transition-prior weighting hurts
(lambda=0 optimal), MBR decoding over FFBS samples hurts, per-model temperature
in the blend is a no-op. Data lives outside the repo.
