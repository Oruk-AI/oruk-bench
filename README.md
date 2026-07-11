# oruk-bench

**A speech-emotion recognition benchmark: 64,384 held-out clips, 7 emotion
classes, ~20 languages, one scoring protocol for every entrant.**

oruk-bench evaluates any speech-emotion model — open-source checkpoints,
open-weight audio-LLMs, and closed API models — on a single held-out evaluation
set with identical audio preprocessing (16 kHz mono, 16 s truncation), an
identical 7-class label space (anger, happiness, sadness, fear, disgust,
surprise, neutral), and a single scoring implementation.

Maintained by [Oruk AI](https://github.com/Oruk-AI). **Disclosure: Oruk trains
models that appear on this leaderboard.** They are marked **OURS** and
*in-distribution* everywhere they appear. See [GOVERNANCE.md](GOVERNANCE.md).

## Leaderboard highlights

| Model | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---|
| **oruk-spectra** (Whisper-L-v3 FT, 640M) | **77.6%** | **0.810** | **OURS — trained in-distribution** |
| emotion2vec-plus-seed | 68.7% | 0.680 | |
| emotion2vec-plus-large | 68.6% | 0.677 | |
| emotion2vec-plus-base | 68.5% | 0.683 | |
| emotion2vec-base-finetuned | 63.6% | 0.616 | |
| EmotionThinker (Qwen2.5-Omni-7B FT) | 60.5% | 0.504 | 5k subsample* |
| SenseVoice-small | 55.7% | 0.469 | |
| Gemini 3 Flash Preview (best closed) | 46.0% | 0.373 | 5k subsample* |
| Gemini 2.5 Flash | 45.4% | 0.370 | 5k subsample* |
| OpenAI gpt-audio-1.5 | 43.3% | 0.347 | 5k subsample* |

\* Closed/API models and audio-LLMs are scored on a fixed 5,000-clip stratified
subsample (seed 0), validated by rescoring open models on the same subsample
(scores shift <2 points). Full results, including per-class F1 and
refusal/error counts, are in
[`leaderboard/leaderboard.json`](leaderboard/leaderboard.json).

**Fairness note:** oruk-spectra is trained on this benchmark's training split
(in-distribution); every other entrant is zero-shot cross-corpus. This is
disclosed on every surface where the number appears. Any lab can request a free
evaluation of their model — see [GOVERNANCE.md](GOVERNANCE.md).

## Install

```bash
pip install oruk-bench                # core: data loading + scoring only
pip install "oruk-bench[open]"        # + open-source SER adapters (torch, funasr, ...)
pip install "oruk-bench[audiollm]"    # + open-weight audio-LLM adapters
pip install "oruk-bench[api]"         # + OpenAI/Anthropic API arms
```

Or from source:

```bash
git clone https://github.com/Oruk-AI/oruk-bench
cd oruk-bench
pip install -e ".[dev]"
```

## Quickstart

```bash
# what can I run?
oruk-bench list-models

# evaluate a registered model on your copy of the eval shards
oruk-bench eval --model emotion2vec-plus-large --data-dir path/to/shards --device cuda:0

# closed models run on the fixed 5k stratified subsample automatically
oruk-bench eval --model gemini-2.5-flash --data-dir path/to/shards --project my-gcp-project

# score your own predictions offline (JSON list, npz, or one label per line)
oruk-bench score --preds preds.json --labels labels.json
```

### Data format

The eval shards are **not distributed with this repository** (see
[BENCHMARK_CARD.md](BENCHMARK_CARD.md) for why and how to request access).
`--data-dir` must point at a directory of parquet shards with columns:

| column | type | meaning |
|---|---|---|
| `audio_flac` | binary | FLAC-encoded audio clip |
| `label` | int64 | index into `["anger", "happiness", "sadness", "fear", "disgust", "surprise", "neutral"]` |
| `language` | string | language tag (null → "unknown") |
| `source_id` | string | source corpus identifier |

Shards are read in sorted-filename order, which fixes the deterministic example
order used by every run. A synthetic fixture generator
([`tests/make_fixture.py`](tests/make_fixture.py)) produces shards in this exact
schema so you can test the pipeline end to end without the real data.

## Repository layout

```
src/oruk_bench/
  core.py               # label space, aliases, data loading, clipping, scoring
  cli.py                # oruk-bench eval / score / list-models
  adapters/
    open_models.py      # HF / FunASR / SpeechBrain adapters + model registry
    gemini.py           # Gemini via Vertex AI (JSON-schema constrained output)
    openai_api.py       # OpenAI audio models + Anthropic (transcript-only)
    audio_llm.py        # Qwen-Omni, Voxtral, Audio Flamingo, MERaLiON
leaderboard/
  leaderboard.json      # machine-readable current results
scripts/
  build_leaderboard.py  # regenerates leaderboard.json from result files
tests/
  make_fixture.py       # synthetic parquet fixture (sine waves, ~24 clips)
```

## Documentation

- [BENCHMARK_CARD.md](BENCHMARK_CARD.md) — task definition, data provenance, scoring protocol, versioning
- [GOVERNANCE.md](GOVERNANCE.md) — conflict-of-interest disclosure, free-evaluation policy, private split escrow
- [INTENDED_USE.md](INTENDED_USE.md) — what this benchmark is (and explicitly is not) for
- [CONTRIBUTING.md](CONTRIBUTING.md) — adding a model adapter, submitting results

## License

[Apache-2.0](LICENSE). The benchmark *code* is Apache-2.0; the evaluation *data*
is governed separately (see BENCHMARK_CARD.md).
