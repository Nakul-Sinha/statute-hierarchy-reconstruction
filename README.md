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
4. Model A: FIXED 5-seed ensemble (seeds 42/1/2/3/7) of act-level BiLSTM taggers
   (from-scratch embeddings, provision encoder = mean/first8/last8 embedding
   pools + hand features -> MLP; 2-layer BiLSTM hid=160; 7-way head),
   early-stopped on holdout depth acc.
5. Ensemble log-probs with a rare-depth prior adjustment
   (log p - tau*log prior, counteracting argmax shrinkage of deep classes),
   decode each act with constrained Viterbi (hard pre-order mask + optional
   smoothed transition prior). alpha/lambda/tau tuned in-run on the internal
   holdout with fixed grids; the SAME split-fit transition matrix and class
   prior are used for tuning and test decode. Attach parents losslessly,
   validate, write.
6. The recipe is hardware-independent: CPU-only, threads capped at
   min(10, cores) with BLAS pools pinned before numpy import, fixed seed count
   and grids, N_STATES derived from train at runtime. Both invocation
   conventions supported (argv paths, or no-args probing with output mirrored
   to ./working/submission.csv). Elapsed-time triggers exist only as crash
   protection (65-75 min) and never fire on reference hardware; fatal failure
   still writes a structurally valid chain submission and exits 0 with a loud
   FATAL banner.

Final confirmation run of the frozen revision (8-thread Zen5, pandas 3.0.5 /
torch 2.13-cpu / lgbm 4.7): holdout (562 unseen acts) normalized 0.5816
(tuned alpha=0.8, lambda=0.1, tau=0.3; 5-seed NN val depth acc 0.6733);
23.3 min wall-clock including heavy external CPU contention during the first
20 min (~17 min clean). The tau depth-prior adjustment was adopted on evidence
(+0.008 holdout in the isolated study, 0.5768 -> 0.5816 in the final run);
same-machine reruns are bit-identical; cross-machine holdout varies
~+/-0.01-0.02 from CPU float nondeterminism in torch training.

## Files

- `solution.py` — the submission script (`python3 solution.py <public_dir> <out_csv>`)
- `metric.py` / `calibrate.py` — replicated metric, calibrated to the chain=0.019 anchor
- `pipeline.py`, `nn_model.py` — shared components used by experiments
- `exp_*.py` — experiment drivers (LGBM, NN, ensemble, decode/MBR/temperature studies)
- `validate_sub.py` — independent grader-style submission validation

Negative results kept for the record: transition-prior weighting hurts
(lambda=0 optimal), MBR decoding over FFBS samples hurts, per-model temperature
in the blend is a no-op. Data lives outside the repo.
