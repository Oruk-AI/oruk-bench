# Launch thread + Hacker News post

## X/Twitter thread (8 tweets)

**1/**
We tested frontier AI models on a task every human does implicitly: listen to a voice and name the emotion.

Best frontier score: 46.0%.
Random guessing: 14.3%.
7 classes, 64,384 held-out clips, ~20 languages, identical scoring for every model.

github.com/Oruk-AI/oruk-bench

**2/**
The board:

oruk-spectra (ours, fine-tuned*) — 77.6%
emotion2vec+ (best open zero-shot) — 68.5–68.7%
EmotionThinker — 60.5%
Gemini 3 Flash Preview — 46.0%
Gemini 2.5 Flash — 45.4%
Gemini 2.5 Pro — 44.0%
gpt-audio-1.5 — 43.3%
gpt-audio-mini — 39.7%
Voxtral-Mini-3B — 36.6%

*trained on the benchmark's training split. Disclosed on every row.

**3/**
The most telling number isn't a frontier model's.

Claude has no audio input (we verified against the live API). So we gave it transcripts only: 39–40%.

Words alone get you ~40%. Audio-capable frontier models score 40–46%. They're mostly reading, not listening.

**4/**
These weren't adversarial prompts. Best-chance configuration: all 7 options listed with acoustic definitions, temperature 0, constrained output, thinking enabled.

40–46% is the ceiling we found, not the floor.

**5/**
Where they fail, per class:

Frontier models' best class is neutral (~0.53 F1) — the default answer when unsure.
Disgust: 0.13–0.19 F1. They nearly can't hear it.

Our 640M model: disgust 0.905, fear 0.864. The signal is in the audio. It's learnable.

**6/**
Refusals count as errors. gpt-audio-1.5 declined 161 of 5,000 clips; gpt-audio-mini declined 473. A judge that won't answer hasn't answered. All refusal counts published.

**7/**
Contamination roadmap: an escrowed private split — ~18k clips, 3 raters each via a Prolific study, 6 studio corpora, multi-annotator distributions — never on the public internet, SHA-256 published before any model is scored on it.

**8/**
pip install oruk-bench

Open harness. One command per model. Free evaluation for any lab, results published unedited. If your model beats ours, the leaderboard says so.

github.com/Oruk-AI/oruk-bench

---

## Hacker News

**Title:**
Show HN: Speech-emotion bench – frontier models score 40–46% on vocal emotion (random is 14.3%)

**First comment (from submitter):**
Author here. We built a 7-class vocal emotion benchmark: 64,384 held-out clips, ~20 languages, one open harness for everything from frontier APIs to open-weight encoders (pip install oruk-bench). Closed models run on a validated 5,000-clip stratified subsample — open models rescored on both full set and subsample shift <2 points.

Three results I think this crowd will find interesting:

1. Frontier multimodal models cluster at 40.0–46.0% (Gemini 3 Flash Preview is top at 46.0%), with best-chance prompting: acoustic definitions for all 7 classes, temperature 0, constrained output, thinking enabled. Random is 14.3%.

2. Claude has no audio input at all (verified against the live API), so we ran it on transcripts only: 39–40%. That's the control that matters — audio-capable models barely beat a model that never hears the clip. Per-class, frontier models lean on "neutral" (~0.53 F1, their best class) and nearly miss disgust (0.13–0.19 F1).

3. Conflict of interest, stated plainly: our own model (oruk-spectra, 640M, fine-tuned Whisper-large-v3 encoder) tops the board at 77.6% / 0.810 macro-F1, and it's trained on the benchmark's training split. That's disclosed on every row — the fair zero-shot comparison is emotion2vec+ at ~68.5%. The point isn't "we beat Gemini"; it's that a 640M model with task supervision solves most of a problem trillion-parameter models don't.

Refusals count as errors (gpt-audio-1.5: 161/5,000; gpt-audio-mini: 473/5,000). Next up is an escrowed private split (~18k clips, 3-rater Prolific annotations, published SHA-256) so nobody — including us — can train on the test set. Evaluation is free for any lab and results go up unedited. Happy to answer questions about the protocol.
