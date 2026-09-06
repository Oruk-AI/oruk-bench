# Benchmark Card — oruk-bench

## Snapshot and model identity — September 6, 2026

The `oruk-spectra` result in this repository (**77.6% accuracy, 0.810 macro-F1**)
is the earlier published checkpoint snapshot. The [website benchmark](https://oruk.ai/benchmarks)
reports a later Spectra 1 checkpoint at **77.8%, 0.816**, with its
[methodology and limitations](https://oruk.ai/benchmarks/methodology).
These are different checkpoint snapshots; the historical result files here
have not been replaced or relabeled.

Both results describe historical Spectra models, retired from the hosted API
on September 4, 2026. **Neither is a Resonance evaluation.**
[Resonance is Oruk's current flagship speech recognition model](https://oruk.ai/models#resonance).
Use the [current API documentation](https://oruk.ai/docs) for model IDs,
supported outputs, and access; do not infer current API support from a benchmark adapter.

## Task definition

Single-label speech-emotion classification. Given an audio clip of speech, the
system predicts exactly one of seven emotion categories:

`anger, happiness, sadness, fear, disgust, surprise, neutral`

- **Input:** 16 kHz mono audio, truncated to the first 16 seconds.
- **Output:** one label from the 7-class space. Models with different native
  taxonomies are mapped through a published alias table
  (`oruk_bench.core.LABEL_ALIASES`); label mass that cannot be mapped is
  ignored at argmax time.
- **Population:** 64,384 held-out clips spanning ~20 languages and a mix of
  acted, elicited, and spontaneous speech.

## Data provenance

**Public eval shards.** The evaluation set is assembled from multiple
multilingual emotional-speech corpora. Clips were held out from any Oruk
training runs *except* for oruk models explicitly marked in-distribution (see
GOVERNANCE.md). The shards are **not redistributed in this repository** because
the underlying corpora carry heterogeneous licenses; researchers can request
access or reconstruct equivalents from the source corpora. Shards are parquet
files with `audio_flac` / `label` / `language` / `source_id` columns, read in
sorted-filename order.

**Private label layer (private-v1, upcoming).** A proprietary multi-rater
human annotation layer collected via Prolific is being added as an escrowed
private split. Per-clip labels are aggregated from multiple independent raters
with quality controls (attention checks, gold-standard calibration items,
rater agreement filters). The private split will ship as:

- a published SHA-256 hash of the frozen private set at freeze time,
- public/private score-gap monitoring for every leaderboard entrant, so
  overfitting to the public shards is detectable and disclosed.

No participant-identifying data, rater data, or raw annotation records are
published in this repository — only aggregate benchmark statistics.

## Scoring protocol

One scoring implementation (`oruk_bench.core.score`) is used for every arm of
the benchmark:

- **Metrics:** accuracy, macro-F1 (primary), weighted-F1, per-class
  precision/recall/F1, per-language macro-F1 (languages with ≥100 clips),
  per-source macro-F1.
- **Fair-subset scores:** models that structurally cannot emit some classes
  (e.g. 4-class IEMOCAP models) receive their penalized full-set score *and* a
  fair-subset score restricted to clips whose gold label is in their supported
  set. Both are published.
- **Deterministic order:** shards load in sorted-filename order; full-set runs
  see every clip exactly once.
- **API subsample:** closed API models and local audio-LLMs are scored on a
  fixed label-stratified 5,000-clip subsample (`stratified_subsample`, seed 0).
  The subsample was validated by rescoring all open models on it: scores shift
  by less than 2 points versus the full set. Subsampled scores are flagged
  (`"subsample": true`) everywhere they are reported.
- **Best-chance prompting:** prompted models (Gemini, OpenAI audio,
  audio-LLMs) all receive the same published expert-annotator prompt
  (`oruk_bench.core.PROMPT`) with per-class acoustic definitions, temperature 0,
  and constrained output where the API supports it.

## Refusal / error accounting

API errors, refusals, and unparseable outputs are never dropped:

- After bounded retries (exponential backoff; refusal-prone audio models get
  extra re-asks), a failed clip is scored as a **neutral** prediction.
- The count is published per run as `api_errors_defaulted_to_neutral` and
  carried into `leaderboard/leaderboard.json`.
- A run whose error count equals the full subsample (100% failure) is invalid,
  excluded from the leaderboard, and rerun.

Anthropic models have no audio input; they classify a cached
`gpt-4o-mini-transcribe` transcript instead, and are flagged
`transcript-only` — not directly comparable to audio-modality entries.

## Versioning & refresh policy

- Benchmark versions are tagged (`v0.1.0` = current). Any change to the label
  space, alias table, clipping rule, subsample, or scoring code bumps the
  version, and leaderboard entries are only comparable within a version.
- The eval set is frozen per version. New corpora or the private-v1 label
  layer arrive as a new version with re-runs of all entrants, not as silent
  edits.
- Result JSONs record the harness name, model id, adapter, and error counts so
  any published number can be traced to a run artifact.
- Known-broken runs are marked and redone; they never appear on the
  leaderboard (see the validity rule in `scripts/build_leaderboard.py`).
