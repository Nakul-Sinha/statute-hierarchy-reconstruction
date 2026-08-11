# Statute Hierarchy Reconstruction

## The problem

I get the provisions of US public laws as a flat list in document order, and I
have to rebuild the nesting tree by naming the parent of every provision. Scoring
is 0.55 parent accuracy, plus 0.25 depth agreement, plus 0.20 sibling pair F1,
normalized per act against the trivial answer that calls everything top level.

## What I did

The observation that made this tractable is that document order is an exact
pre-order traversal. If I know the true depth of every provision, then the parent
of provision i is simply the nearest earlier provision at depth minus one, and
that reconstructs 100 percent of the training parents. So the task collapses,
without loss, into tagging a depth sequence.

From there I predict depths two ways and combine them. A LightGBM classifier runs
on 62 hand built drafting features: casing, terminal punctuation, amendment and
citation phrases, cross reference granularity, and run length context. A six seed
ensemble of act level BiLSTM taggers runs on learned word embeddings, hashed
character n-grams, and those same hand features. I blend the log probabilities,
adjust for rare depth priors, then decode each act with a constrained Viterbi
pass that enforces the pre-order rules, and finally attach parents from the
depths.

No TF-IDF or fitted text statistics anywhere, since those were off limits, so all
the text signal has to be learned inside the network. CPU only.

## Layout

`solution.py` is the single end to end script. The `exp_*.py` files are the
experiments that chose the recipe. `TECHNICAL.md` has the details.
