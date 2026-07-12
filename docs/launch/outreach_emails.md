# Outreach email templates — speech-emotion bench launch

---

## (a) Frontier-lab audio teams

**Subject:** Your audio model scored [X]% on 7-class vocal emotion — free private-set eval, 48h turnaround

Hi [name],

We just published speech-emotion bench: 64,384 held-out clips, 7 emotion classes, ~20 languages, one open harness for every model ([github.com/Oruk-AI/oruk-bench](https://github.com/Oruk-AI/oruk-bench)).

Under best-chance prompting (all 7 options with acoustic definitions, temperature 0, constrained output, thinking enabled), [model] scored [X]% on the validated 5,000-clip subsample. The frontier cohort clusters at 40–46% against a 14.3% random baseline — and a transcript-only control scores 39–40%, so words alone explain most of it. Full protocol and per-class breakdowns are in the technical report.

Two offers:

1. **Free evaluation of any unreleased or updated model on our private set** — escrowed, never on the public internet, SHA-256 published in advance. 48-hour turnaround from checkpoint or API access to full scored report.
2. If you believe our prompting undersells your model, send us your preferred prompt and we'll run it and publish both configurations, unedited.

Disclosure up front: our own model (oruk-spectra, trained in-distribution — labeled as such everywhere) scores 77.6%. The gap we're measuring is supervision, not scale, and we'd genuinely like your models to close it.

Nathan Roll, Oruk AI

---

## (b) Agent-QA companies (Coval, Hamming, Roark, Bluejay)

**Subject:** The judge models scoring your voice agents can't hear emotion — data inside

Hi [name],

If you use frontier multimodal models to judge voice-agent calls for frustration, escalation, or sentiment, we have measurements you'll want to see.

We run speech-emotion bench (64,384 held-out clips, 7 emotion classes, identical scoring for every model). The models commonly used as judges score 40–46% — Gemini 3 Flash Preview 46.0%, Gemini 2.5 Pro 44.0%, gpt-audio-1.5 43.3% with 161/5,000 refusals. Random is 14.3%. A transcript-only Claude control scores 39–40%: the audio judges are barely beating a model that never hears the call. Their strongest class is "neutral" (~0.53 F1) and they nearly miss disgust entirely (0.13–0.19 F1) — the exact profile that under-reports unhappy customers.

We're exploring judge-calibration work with agent-QA platforms: benchmark your current judge pipeline, quantify the miss rate on emotionally loaded calls, and compare against specialized encoders. Evaluation is free and results go to you first.

Worth 20 minutes?

Nathan Roll, Oruk AI

---

## (c) Speech academics

**Subject:** Advisor invitation — multilingual speech-emotion benchmark with escrowed multi-rater private split

Dear Professor [name],

We've released speech-emotion bench, an open benchmark for vocal emotion recognition: 64,384 held-out clips, 7 classes, ~20 languages, with frontier multimodal APIs, open audio-LLMs, and fine-tuned encoders scored under one harness ([github.com/Oruk-AI/oruk-bench](https://github.com/Oruk-AI/oruk-bench), `pip install oruk-bench`). Headline finding: frontier models cluster at 40–46% accuracy versus a 39–40% transcript-only control — they are largely reading the words, not the voice.

We're forming a small advisory group for the next phase: a multi-rater human label layer (Prolific study, 3-rater overlap, ~18k clips over 6 studio corpora, released as multi-annotator distributions on an escrowed private split with published SHA-256). Open design questions where your work on [specific area] is directly relevant: annotator-distribution metrics vs. majority-vote labels, acted/spontaneous composition, and per-language reporting standards.

The ask is light — a few design reviews per year, acknowledged in the report (co-authorship on the benchmark paper where contribution warrants). Evaluation compute for your group's models is free.

May I send the technical report?

Nathan Roll, Oruk AI
