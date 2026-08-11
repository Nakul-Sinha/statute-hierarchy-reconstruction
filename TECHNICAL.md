# the platform ch3 — Statutory Outline Reconstruction

Rebuild the nesting tree of US public-law provisions (parent index per provision).
Metric: 0.55 parent-acc + 0.25 depth agreement + 0.20 sibling pair-F1, normalized
per act against the all-top-level trivial answer (chain baseline ~0.019).

## Core insight

Document order is an exact pre-order traversal: given true depths,
`parent(i) = nearest earlier provision at depth-1` reconstructs 100% of train
parents. So the task reduces losslessly to tagging the per-provision depth
sequence under `d[0]=0`, `d[i] <= d[i-1]+1`, `d <= 6`.

## Solution (`solution.py`, single end-to-end script, fully TF-IDF-free)

No fitted text statistics anywhere (TF-IDF/SVD banned by challenge guidance);
all text signal is learned inside the NN.

1. 15% act-level internal holdout (seed 42) for early stopping + tuning.
2. Hand features (62 dims): drafting cues (case, terminal punctuation,
   amendment/citation phrases, xref granularity, lengths) + neighbor flags +
   run-length/distance context (distance since last colon-ender, semicolon runs,
   block position).
3. Model B: LightGBM 7-class depth on hand/neighbor features ONLY,
   early-stopped on the holdout, then refit on all train.
4. Model A: FIXED 6-seed ensemble (seeds 42/1/2/3/7/11) of act-level BiLSTM
   taggers: provision encoder = trained word embeddings (mean/first8/last8
   pools) + hashed char-3/4gram EmbeddingBag channel (crc32, 64 dims) + hand
   features -> MLP; 2-layer BiLSTM emb160/hid256; 7-way head; plateau-decay
   early stop on holdout depth acc.
5. Ensemble log-probs (alpha grid), rare-depth prior adjustment
   (log p - tau*log prior, tau grid 0-0.4), constrained Viterbi per act
   (hard pre-order mask + lambda-smoothed transition prior). alpha/tau/lambda
   tuned in-run on the internal holdout with fixed grids; the SAME split-fit
   transition matrix and class prior are used for tuning and test decode.
   Attach parents losslessly, validate, write.
6. Hardware-independent recipe: CPU-only, threads capped at min(10, cores)
   with BLAS pools pinned before numpy import, fixed seed count and grids,
   N_STATES derived from train at runtime. Both invocation conventions
   supported (argv paths, or no-args probing with output mirrored to
   ./working/submission.csv). Elapsed-time triggers exist only as crash
   protection (70-82 min) and never fire on reference hardware; fatal failure
   still writes a structurally valid chain submission and exits 0 with a loud
   FATAL banner.

## Campaign numbers (all on the same 562-act 15% holdout)

- LGBM hand-only alone: 0.4379. Single-seed NN+LGBM blend: 0.5846.
- Seed farm (8 c2-config seeds, val depth acc 0.6477-0.6577); ensemble-size
  blend study: 1/4/6/8 seeds -> 0.5846 / 0.5971 / 0.5986 / 0.5942.
  6-seed blend adopted (alpha=0.8, tau=0.4, lambda=0.1) = 0.5986.
- Extended alpha/tau/lambda grids and seed-subset selection (drop-worst,
  top-k-by-val): no gain over 0.5986 (tau optimum is a plateau, not an edge).
- K-best study at the adopted config: oracle@16 = 0.7645, but a lightweight
  tree-feature rerank recovers none of it (best rerank = top-1). A trained
  learning-to-rank reranker over OOF inner-CV candidates is an open stretch
  lever (adoption gate >= +0.008 holdout).
- Previous frozen SVD-era pipeline: holdout 0.5816, public 0.583
  (near-1:1 transfer). The TF-IDF-free freeze improves on it by +0.017.

Final confirmation run of the frozen revision (322a626) on the box
(4-core Zen5, 8 threads): 32.0 min wall-clock, in-run tuning chose the SAME
operating point as the study (alpha=0.8, lambda=0.1, tau=0.4), 6-seed NN
ensemble val depth acc 0.6665, holdout normalized 0.5944 (vs 0.5986 at the
emission level in the blend study — within the documented +/-0.01-0.02
same-recipe CPU float nondeterminism across retrains). Submission validated:
1,588 acts, 33,392 provisions, well-founded. The reference grader box has
10 cores, so the 90-min cap holds with ample margin.

Honesty note: the outer 15% holdout is used for NN early stopping AND
alpha/tau/lambda selection (production convention all campaign), so in-run
"estimated_score" figures are model-selected on that split; the near-1:1
holdout-to-public transfer observed for the previous freeze (0.5816 -> 0.583)
is the empirical check that this selection is not overfit.

## Files

- `solution.py` — the submission script (`python3 solution.py <public_dir> <out_csv>`)
- `metric.py` / `calibrate.py` — replicated metric, calibrated to the chain=0.019 anchor
- `pipeline.py`, `nn_model.py` — shared components used by experiments
- `exp_*.py` — experiment drivers (LGBM, NN, blend/subset studies, K-best/rerank)
- `run_*.sh` — box drivers for the experiment waves
- `validate_sub.py` — independent grader-style submission validation

Negative results kept for the record: transition-prior weighting beyond
lambda~0.1 hurts, MBR decoding over FFBS samples hurts, per-model temperature
in the blend is a no-op, extended tau grid (0.5-1.0) flat, seed-subset
selection no better than the first six seeds. Data lives outside the repo.
