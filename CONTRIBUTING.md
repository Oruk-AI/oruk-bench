# Contributing

## Adding a model adapter

An adapter is a small class that turns a batch (or single clip) of 16 kHz
float32 waveforms into predictions in the shared 7-label space. Pick the arm
that matches your model:

1. **Open-source SER checkpoint** (`src/oruk_bench/adapters/open_models.py`):
   implement a class with `__init__(self, cfg, device)` and
   `predict(self, audios) -> list[int]` (indices into `core.LABELS`). Import
   heavy dependencies *inside* `__init__`, not at module top — the base install
   must stay light. Register it in `ADAPTERS` and add an entry to `MODELS`
   with `name`, `adapter`, `model_id` (public checkpoint), `params_m`, and
   `batch_size`.

2. **Open-weight audio-LLM** (`src/oruk_bench/adapters/audio_llm.py`):
   implement `__init__(self, model_id, device)` and
   `classify(self, audio) -> str` (raw text; the shared parser extracts the
   label and unparseable outputs are counted and defaulted to neutral). Add it
   to `REGISTRY`.

3. **Closed API model** (`gemini.py` / `openai_api.py`): open an issue first —
   API arms need credentials and cost budgeting on our side.

Ground rules for all adapters:

- No prompt tuning beyond the shared published prompt (`core.PROMPT`). The
  benchmark measures models, not prompt engineering.
- Label mapping goes through `core.norm_label` / `LABEL_ALIASES` only. If your
  model's taxonomy needs a new alias, add it to the table in a PR so it applies
  to everyone.
- If your model structurally lacks some classes, declare `"supported"` in its
  registry entry; you get full-set and fair-subset scores.
- `pip install -e ".[dev]" && ruff check src tests && pytest` must pass. CI
  runs the score path on a synthetic fixture; it must not download models.

## Submitting results

- **Preferred: request a run.** Open an issue titled
  `[eval request] <model name>` (see GOVERNANCE.md — evaluation is free and
  results are published unedited). We run all official numbers ourselves so
  the leaderboard stays one-protocol.
- **Self-reported numbers** are accepted only as *unverified* pending
  replication: include your adapter PR, the exact command, the produced result
  JSON, and the `.npz` predictions file so we can rescore with
  `oruk-bench score`.
- Leaderboard updates regenerate `leaderboard/leaderboard.json` via
  `scripts/build_leaderboard.py`; don't hand-edit that file.

## Development setup

```bash
git clone https://github.com/Oruk-AI/oruk-bench
cd oruk-bench
python -m venv .venv && . .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pytest
ruff check src tests
```

The test suite runs entirely on a synthetic fixture
(`tests/make_fixture.py`) — no eval data or GPU required.

## Reporting problems

- Scoring/protocol bugs: open an issue with a minimal repro against the
  synthetic fixture if possible.
- Label disputes or data-quality reports for specific clips: open an issue
  with the `source_id` and shard row; do not post audio content in the issue.
- Governance concerns: see GOVERNANCE.md.
