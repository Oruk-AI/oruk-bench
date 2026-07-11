# Statistics & Reporting Methodology

This document describes the statistical machinery behind every number on the
Oruk speech-emotion leaderboard, implemented in `src/oruk_bench/stats.py`.
The benchmark scores speech-emotion models on a 7-class task (anger,
happiness, sadness, fear, disgust, surprise, neutral) over 64,384 clips
drawn from ~50 source corpora, plus a 5,000-clip stratified subsample used
for closed / API-priced models.

## Why clustering matters

Clips are not independent samples. They arrive in clusters: the same source
corpus (recording conditions, label taxonomy, acting style) and often the
same speaker contribute many clips. Errors within a cluster are correlated —
a model that misreads one CREMA-D actor's "happy" tends to misread that
actor's other clips too, and corpus-level effects are even stronger (compare
per-source macro-F1 in any of our result JSONs: the same model can score
0.97 on TESS and 0.24 on MELD).

Treating the 64,384 clips as iid would divide variance by roughly the number
of clips, when the *effective* sample size is closer to the number of
clusters. The result is confidence intervals that are far too narrow and
leaderboard gaps that look "significant" but do not replicate when the
corpus mix changes.

All interval estimates therefore use a **cluster bootstrap**: we resample
whole clusters (source corpus by default; speaker where speaker ids exist)
with replacement, keeping every clip of a sampled cluster together. This
preserves within-cluster correlation in each replicate. The unit test
`test_cluster_ci_wider_than_iid_on_clustered_data` demonstrates the effect:
on synthetic data with cluster-level accuracy shifts, the cluster-aware CI
is strictly wider than the naive item-level CI.

## Methods

### `bootstrap_ci(y_true, y_pred, clusters, metric_fn, n_boot=2000, alpha=0.05)`

Percentile bootstrap CI for a metric ("accuracy", "macro_f1", or any
callable `f(y_true, y_pred) -> float`). Each of the `n_boot` replicates
draws `n_clusters` clusters with replacement and recomputes the metric on
the concatenated items. Returns the point estimate (computed once on the
full data), the `alpha/2` and `1 - alpha/2` percentiles of the replicate
distribution, and the replicates themselves.

Macro-F1 uses a frozen label set (the classes present in `y_true`): a
bootstrap replicate can drop a rare class, and letting sklearn re-infer the
label set would silently change the macro average's denominator between
replicates.

With `clusters=None` the function degrades to an item-level (iid) bootstrap;
the CLI warns loudly when this happens.

### `paired_bootstrap_test(y_true, pred_a, pred_b, clusters, metric_fn, n_boot=2000)`

Paired comparison of two models scored on **identical items**. Each
replicate resamples clusters once and evaluates both models on the same
resample, so shared item difficulty cancels out of the difference — this is
much more powerful than comparing two independent CIs (which is a common
leaderboard mistake: overlapping CIs do *not* imply "no significant
difference").

Returns:

- `delta` = metric(A) − metric(B) on the full data,
- a percentile CI for the delta,
- a two-sided sign-flip p-value: twice the (add-one smoothed) fraction of
  replicates whose delta sign contradicts the observed sign. Smoothing
  means the smallest reportable p is `2/(n_boot+1)`, never an artifactual 0.

This test powers "is rank *i* really above rank *i+1*" on the leaderboard.

### `soft_label_metrics(pred_probs_or_onehot, soft_targets)`

Emotion perception is genuinely ambiguous; for the human-relabeled subset we
collect multiple ratings per clip and keep the full rater distribution
rather than a majority vote. Against these soft targets we report:

- **cross-entropy**: mean of −Σ target·log(pred). Predictions are clipped
  at 1e-12 so hard (one-hot) predictions stay finite, at the cost of a
  large-but-bounded penalty for confidently contradicting the raters.
- **Jensen–Shannon divergence** (base 2, so bounded in [0, 1]): a symmetric
  distributional distance that, unlike cross-entropy, does not explode for
  hard predictions and equals 0 iff the model reproduces the rater
  distribution exactly.
- **soft accuracy**: the mean probability mass raters assigned to the
  model's argmax class. A model that answers "happiness" on a clip 60% of
  raters called happiness earns 0.6, not 1.0 or 0.0. This is the headline
  soft metric because it is a strict generalization of plain accuracy
  (it reduces to accuracy when raters are unanimous).

Hard predictions are accepted as integer column indices or as class-name
strings (with `labels=` naming the soft-target columns) and are converted
to one-hot rows internally.

### `prompt_ensemble_summary(list_of_pred_arrays, y_true)`

Audio LLMs are sensitive to prompt wording in ways classifier heads are
not. Every audio-LLM entry is run with *k* prompt variants; this function
reports the per-prompt metric plus mean, sample standard deviation, min,
max, and **spread** (max − min). The spread is published alongside the
mean so readers can tell a robust model from one that got lucky with our
default prompt.

### `min_detectable_delta(n_items, base_acc, ...)`

Power simulation answering: *on an eval of `n_items` clips, how large must
a true accuracy difference be before the paired test reliably detects it?*
We simulate two models with independent per-item correctness at `base_acc`
and `base_acc + delta`, bootstrap the paired difference (the per-item
difference takes values in {−1, 0, +1}, so the bootstrap reduces to a fast
multinomial draw), and binary-search for the smallest delta rejected at
95% confidence in ≥ 80% of simulated evals.

Two deliberate conservatisms/idealizations:

- **Independent correctness is conservative** — real models' errors
  correlate positively, which shrinks the paired variance, so real
  detectable deltas are at or below the simulated threshold.
- **Items are iid in the simulation** — clustering reduces effective n and
  pushes the real threshold up. This is exactly why leaderboard CIs come
  from the cluster bootstrap rather than from this idealized bound; the
  table below is a *floor* on distinguishable differences.

Results at 80% power, alpha = 0.05 (reproduce with
`python -m oruk_bench.stats mdd --n 1000 5000 17823 64384 --base-acc 0.45 0.77`
after `pip install -e .`, or with `PYTHONPATH=src`; also stored in
`stats/min_detectable_delta.json`):

| n items | min detectable Δacc @ base 0.45 | min detectable Δacc @ base 0.77 |
| ------: | ------------------------------: | ------------------------------: |
|   1,000 |                          0.0625 |                          0.0488 |
|   5,000 |                          0.0285 |                          0.0246 |
|  17,823 |                          0.0148 |                          0.0121 |
|  64,384 |                          0.0082 |                          0.0070 |

Reading the table: at the full 64,384-clip set, sub-1-point accuracy gaps
are resolvable (before clustering penalties); on the 5,000-clip closed-model
subsample, only gaps of roughly 2.5–3 accuracy points are reliably
detectable. This drives two policy decisions:

1. **The 5k subsample is adequate for its purpose** — separating closed
   models whose scores differ by several points — while keeping API cost
   bounded; near-ties among closed models are reported as ties rather than
   resolved by rank order.
2. **Per-cell reporting rules**: per-language / per-source cells inherit the
   same math at much smaller n. A 1,000-clip cell cannot support claims
   about differences under ~5–6 points, so cells below a minimum n are
   flagged, and we do not publish per-cell rankings — only per-cell scores
   with CIs.

## Leaderboard CLI

```
python -m oruk_bench.stats leaderboard \
    --results-dir benchmarks/seb_results \
    --preds-dir benchmarks/seb_preds \
    --labels benchmarks/seb_labels.npz \
    --out leaderboard_ci.json
```

Inputs:

- `--results-dir` — the existing per-model summary JSONs (`*.json`,
  `*.errors.json` sidecars are ignored). Each `<stem>.json` must have a
  matching `<stem>.npz` in `--preds-dir`.
- `--preds-dir` — per-clip predictions, one `.npz` per model with key
  `y_pred` (aliases `pred`/`predictions` accepted) and optional `clip_ids`
  for explicit alignment.
- `--labels` — ground-truth `.npz` with `y_true`, `clusters` (cluster id
  per clip: source corpus or speaker), and optional `clip_ids`.

Output JSON contains, per model, accuracy and macro-F1 each with point
estimate and cluster-bootstrap CI, sorted by the chosen `--metric`
(default macro-F1), plus `adjacent_tests`: the paired bootstrap test
between every pair of neighboring ranks, and `display_rank` where
non-separable neighbors share a rank.

**Real-data wiring (one command, once artifacts land):** per-clip
prediction `.npz` files currently live in GCS and the ground-truth label
array has not been exported yet. Until both are present the CLI refuses to
run and prints the exact list of missing files. Once they are downloaded
to the paths above, the same command produces the final
`leaderboard_ci.json` with no code changes.

## Reporting policy

1. **Every public leaderboard number ships with a 95% CI** from the
   cluster bootstrap (cluster = source corpus, or speaker where available).
   Numbers without intervals are not published.
2. **Adjacent ranks that the paired test cannot separate at 95% confidence
   are displayed as ties** (shared `display_rank`). We do not let point-
   estimate ordering imply distinctions the data cannot support.
3. **Audio-LLM scores report prompt-ensemble spread**: mean over k prompt
   variants ± std, with min/max spread shown. Single-prompt audio-LLM runs
   are marked as such and excluded from headline rankings.
4. **Per-language and per-source cells** are published with CIs and a
   minimum-n flag; no rankings are derived from cells whose
   min-detectable-delta exceeds the observed differences.
5. **Soft-label metrics** (JSD, soft accuracy) are reported on the
   multi-rater subset alongside hard accuracy, acknowledging that the
   ceiling on this task is set by human disagreement, not 100%.
