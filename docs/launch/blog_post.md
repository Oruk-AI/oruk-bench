# Frontier AI can transcribe you perfectly — but can't hear how you feel

The best score any frontier multimodal model achieves on our speech-emotion benchmark is 46.0%.

That number belongs to Gemini 3 Flash Preview, asked to do something a human listener does without thinking: hear a short audio clip and say whether the speaker sounds angry, happy, sad, afraid, disgusted, surprised, or neutral. Random guessing on seven classes scores 14.3%. The entire frontier cohort — Gemini 3 Flash Preview, Gemini 2.5 Pro and Flash, OpenAI's gpt-audio models — clusters between 40% and 46%. Barely three times random, on a task people perform implicitly in every conversation.

These same models transcribe speech at near-human accuracy. They can tell you *what* was said in dozens of languages. They cannot reliably tell you *how* it was said.

## The benchmark

**speech-emotion bench** is 64,384 held-out audio clips spanning roughly 20 languages, each labeled with one of seven emotion classes: anger, happiness, sadness, fear, disgust, surprise, neutral. Every model — open-weight, closed API, or ours — is scored with the identical harness and identical scoring rules.

Closed and API-priced models are scored on a validated 5,000-clip stratified subsample. We checked the subsample against the full set by rescoring open models on both: scores shift by less than 2 points. The subsample composition is published with the harness.

## The leaderboard

*Chart: horizontal bar chart of accuracy by model, sorted descending. A dashed vertical line at 14.3% marks the random baseline; a second dashed line at 77.6% marks oruk-spectra. The frontier-API cluster sits in a shaded band between 40% and 46%.*

| Model | Type | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|
| oruk-spectra (ours) | Specialized, trained in-distribution | **77.6%** | **0.810** | See disclosure below |
| emotion2vec+ family | Open zero-shot | 68.5–68.7% | — | Best open zero-shot |
| EmotionThinker | Open audio-LLM (emotion-specialized Qwen-Omni FT, ICLR 2026) | 60.5% | 0.504 | |
| Gemini 3 Flash Preview | Frontier API | 46.0% | — | |
| Gemini 2.5 Flash | Frontier API | 45.4% | — | |
| Gemini 2.5 Pro | Frontier API | 44.0% | — | |
| gpt-audio-1.5 | Frontier API | 43.3% | — | 161/5,000 refusals |
| Gemini 2.5 Flash-Lite | Frontier API | 40.0% | — | |
| gpt-audio-mini | Frontier API | 39.7% | — | 473/5,000 refusals |
| Claude (transcript-only) | Text-only control | 39–40% | — | No audio input available |
| Voxtral-Mini-3B | Open audio-LLM | 36.6% | 0.204 | Mistral lists emotion as a future feature |
| Random baseline | — | 14.3% | — | |

Two details worth pausing on.

First, Claude. Anthropic's models accept no audio input — we verified this against the live API, including claude-sonnet-5 and claude-fable-5. So we ran Claude on transcripts alone, and it scores 39–40%. That is the most useful control in the table: **words alone get you about 40%.** Several audio-capable frontier models score within a point or two of a model that never heard the audio.

Second, refusals. gpt-audio-1.5 declined to answer 161 of 5,000 clips; gpt-audio-mini declined 473. Refusals are scored as errors, because a production system that won't answer hasn't answered. Refusal counts are reported alongside every score.

We gave frontier models their best chance: all seven options listed in the prompt with acoustic definitions of each, temperature 0, constrained output, thinking enabled. The 40–46% cluster is what best-chance prompting produces.

## Why frontier models fail

**Prosody versus content.** Emotion in speech lives in pitch, energy, rhythm, and voice quality — not primarily in words. The Claude transcript-only control at 39–40% shows how much of the frontier cohort's performance is explained by lexical content. Models with full access to the waveform barely beat a model reading a transcript.

**Neutral bias.** Frontier models lean hard on the safest answer. Their best per-class F1 is neutral, at roughly 0.53. When unsure, they default to "neutral" — which is exactly the failure mode you don't want in any application that exists to detect when something is *not* neutral.

**Disgust blindness.** Frontier per-class F1 on disgust ranges from 0.13 to 0.19. They nearly cannot hear it. For comparison, oruk-spectra scores 0.905 on disgust and 0.864 on fear — evidence that the acoustic signal is present in the data and learnable; the frontier models simply have not learned it.

## Our model, and our conflict of interest

We publish this benchmark, and we also publish a model that leads it. State that plainly and handle it with structure, not asterisks.

**oruk-spectra** is our specialized speech-emotion model, trained in-distribution for this task. It scores 77.6% accuracy and 0.810 macro-F1 on the held-out set. It is not a zero-shot result and it is never presented as one: the "trained in-distribution" disclosure appears on every leaderboard row, every table, and every figure where the number appears. The right zero-shot comparison for frontier labs is the emotion2vec+ family at 68.5–68.7% — still more than 20 points above the best frontier API model.

The honest claim is narrower than "our model beats Gemini." It is: *a small specialized model trained on task data solves most of this problem, and trillion-parameter frontier models with best-chance prompting do not.* The gap is a supervision gap, not a scale gap.

## What's next: a split no one can train on

Static benchmarks decay. Test sets leak into training corpora, and scores inflate without capability improving. Our roadmap addresses this directly.

We are building a proprietary human label layer: multi-annotator distributions from a Prolific study with 3-rater overlap, roughly 18,000 annotated clips across 6 studio corpora. It will ship as an escrowed private split — held off the public internet entirely, with a published SHA-256 hash committing us to its contents before any model is scored on it. When private-v1 scores land, no one, including us, will have been able to train on them.

## Submit your model

The harness is open and the barrier is one command:

```
pip install oruk-bench
```

One command per model produces a full scored run. Evaluation is free for any lab — including on the private set once it lands — and results are published unedited, refusals and all. If your model beats ours, the leaderboard says so.

The repository is at [github.com/Oruk-AI/oruk-bench](https://github.com/Oruk-AI/oruk-bench). The full technical report, with per-class breakdowns, prompting protocol, and subsample validation, ships alongside it.

Frontier models will get better at this. When they do, this leaderboard is where it will show up first.
