# speech-emotion bench: A Multilingual Benchmark for Vocal Emotion Recognition in Foundation Models

*Technical report skeleton — arXiv preprint draft. TODO markers indicate sections pending private-v1 results.*

---

## Abstract

We introduce **speech-emotion bench**, a benchmark of 64,384 held-out audio clips across approximately 20 languages, labeled with seven emotion classes (anger, happiness, sadness, fear, disgust, surprise, neutral), scored identically for all models. We evaluate frontier multimodal APIs, open-weight audio-LLMs, open zero-shot speech-emotion models, and our own specialized model under a single open harness. Frontier multimodal models cluster at 40.0–46.0% accuracy under best-chance prompting — barely 3× the 14.3% random baseline — while a transcript-only text-model control scores 39–40%, indicating that much of frontier audio-model performance is attributable to lexical content rather than prosody. The best open zero-shot models (emotion2vec+ family) reach 68.5–68.7%. Our specialized model, oruk-spectra (trained in-distribution — disclosed in every result), reaches 77.6% accuracy / 0.810 macro-F1, demonstrating that the task is largely solvable from the acoustic signal with task supervision. Per-class analysis shows frontier models depend on the neutral class (best per-class F1 ≈ 0.53) and nearly fail on disgust (F1 0.13–0.19). We release an open harness (`pip install oruk-bench`), free evaluation for any lab with unedited result publication, and announce an escrowed private split with published SHA-256 to address contamination.

---

## 1. Introduction

- Speech understanding in foundation models has been benchmarked overwhelmingly on *content* (ASR, translation, spoken QA); paralinguistic understanding remains under-measured.
- Gap statement: models that transcribe at near-human accuracy score barely 3× random on 7-class vocal emotion.
- Contributions: (i) 64,384-clip multilingual held-out benchmark with identical scoring for all model families; (ii) validated 5,000-clip stratified subsample protocol for API models; (iii) best-chance prompting protocol with refusal accounting; (iv) per-class failure analysis (neutral bias, disgust blindness); (v) open harness and free-evaluation policy; (vi) escrowed private-split roadmap.

## 2. Benchmark Construction

### 2.1 Corpus and labels
- 64,384 held-out clips; 7 emotion classes: anger, happiness, sadness, fear, disgust, surprise, neutral.
- ~20 languages represented. **TODO:** per-language clip counts table.
- **TODO:** source corpora enumeration, licensing summary, acted vs. spontaneous composition per corpus.
- Train/held-out split discipline: the held-out set is disjoint from any data used to train oruk-spectra (§4.1).

### 2.2 Class balance
- **TODO:** class distribution table for full set and subsample.
- Random baseline: 14.3% (uniform over 7 classes).

### 2.3 Held-out subsample for closed/API models
- Closed and API models are scored on a validated 5,000-clip stratified subsample (stratified over class, language, and source corpus).
- **Validation:** open models rescored on both the full 64,384-clip set and the 5,000-clip subsample shift by **<2 accuracy points**. Subsample composition is published with the harness.

## 3. Evaluation Protocol

### 3.1 Scoring
- Identical scoring for all models: single-label classification accuracy and macro-F1 over 7 classes.
- Per-class F1 reported for all models where output permits.

### 3.2 Prompting protocol for instruction-following models (best-chance configuration)
- Prompt lists all 7 options with acoustic definitions of each class.
- Temperature 0; constrained output format; thinking/reasoning enabled where the API supports it.
- **TODO:** verbatim prompt text appendix; per-provider constrained-decoding notes.

### 3.3 Refusal accounting
- Refusals (declined answers, safety deflections, malformed constrained output after retry policy) are scored as errors and reported as separate counts.
- Observed: gpt-audio-1.5 refused 161/5,000; gpt-audio-mini refused 473/5,000.
- Rationale: deployed systems that decline have not classified; excluding refusals would inflate scores non-comparably.

### 3.4 Text-only control
- Anthropic Claude models accept no audio input (verified against the live API, including claude-sonnet-5 and claude-fable-5).
- Claude evaluated on transcripts only as a lexical-content control: 39–40% accuracy. This bounds the contribution of words alone at ≈40%.
- **TODO:** transcript source (reference vs. ASR) and Claude model/version table.

## 4. Models Evaluated

### 4.1 oruk-spectra (ours)
- Specialized speech-emotion model developed by Oruk AI; architecture and training details are proprietary.
- **Disclosure (repeated wherever the number appears):** oruk-spectra is trained in-distribution. It is not a zero-shot result and is not comparable to zero-shot rows without this caveat.

### 4.2 Open zero-shot speech-emotion models
- emotion2vec+ family (best open zero-shot): 68.5–68.7% accuracy.
- **TODO:** full open-model zoo table with checkpoints and versions.

### 4.3 Open-weight audio-LLMs (zero-shot prompted, same 5,000-clip subsample)
- EmotionThinker (ICLR 2026; emotion-specialized Qwen-Omni fine-tune): 60.5% / 0.504 macro-F1.
- Voxtral-Mini-3B: 36.6% / 0.204 macro-F1 (Mistral documents emotion as a future feature).
- In flight: Qwen2.5-Omni-7B, Audio Flamingo 3. **TODO:** results.

### 4.4 Frontier multimodal APIs
- Gemini 3 Flash Preview, Gemini 2.5 Pro / Flash / Flash-Lite, OpenAI gpt-audio-1.5, gpt-audio-mini. Versions and access dates: **TODO** appendix.

## 5. Results

### 5.1 Main leaderboard

| Model | Type | Accuracy | Macro-F1 | Refusals |
|---|---|---|---|---|
| oruk-spectra† | Specialized (in-distribution) | 77.6% | 0.810 | 0 |
| emotion2vec+ family | Open zero-shot | 68.5–68.7% | — | 0 |
| EmotionThinker | Open audio-LLM | 60.5% | 0.504 | — |
| Gemini 3 Flash Preview | Frontier API | 46.0% | — | — |
| Gemini 2.5 Flash | Frontier API | 45.4% | — | — |
| Gemini 2.5 Pro | Frontier API | 44.0% | — | — |
| gpt-audio-1.5 | Frontier API | 43.3% | — | 161/5,000 |
| Gemini 2.5 Flash-Lite | Frontier API | 40.0% | — | — |
| gpt-audio-mini | Frontier API | 39.7% | — | 473/5,000 |
| Claude (transcript-only control) | Text-only | 39–40% | — | — |
| Voxtral-Mini-3B | Open audio-LLM | 36.6% | 0.204 | — |
| Random | Baseline | 14.3% | — | — |

† Trained in-distribution (see §4.1 disclosure).

### 5.2 Per-class analysis
- Frontier models: best per-class F1 is neutral (≈0.53); disgust F1 ranges 0.13–0.19 across the frontier cohort.
- oruk-spectra: disgust F1 0.905, fear F1 0.864 — the acoustic signal for these classes is present and learnable.
- **TODO:** full 7-class × model F1 matrix; confusion matrices per model family.

### 5.3 Content vs. prosody decomposition
- Audio-capable frontier models score within 0–6 points of the transcript-only control (39–40%), consistent with predominantly lexical strategies.
- **TODO:** paired analysis on clips where transcript is emotion-neutral but prosody is not.

## 6. Limitations

1. **In-distribution advantage of our model.** oruk-spectra is trained in-distribution; its 77.6% is an in-distribution supervised result, not evidence of general zero-shot superiority. The fair zero-shot reference is the emotion2vec+ family.
2. **English-heavy evaluation.** Although ~20 languages are present, the distribution skews English. Per-language results (**TODO**) should be consulted before multilingual claims.
3. **Acted vs. spontaneous mix.** Source corpora include acted studio speech; acted emotion is more prototypical than spontaneous emotion, and absolute scores likely overstate real-world performance for all models.
4. **Single-label taxonomy.** Forced single-label choice under-represents mixed and ambiguous affect; the multi-annotator distribution layer (§7) is intended to address this.
5. **Intended use (EU AI Act statement).** This benchmark measures model capability on emotion *recognition from voice* for research comparison. It is not an endorsement of emotion-recognition deployment. Under the EU AI Act, emotion recognition systems are prohibited in workplace and educational contexts (Art. 5) and otherwise regulated as high-risk (Annex III). Benchmark scores must not be construed as fitness claims for such deployments.

## 7. Private Split and Contamination Roadmap

- Proprietary multi-rater human label layer: multi-annotator distributions from a Prolific study, 3-rater overlap, ~18k annotated clips across 6 studio corpora.
- Ships as an escrowed private split, never published; SHA-256 of the split published in advance, committing contents before any model is scored.
- **TODO (private-v1):** results tables on the private split; public-vs-private score deltas as a contamination estimate; annotator-distribution-aware metrics (soft-label cross-entropy, distribution calibration).

## 8. Reproducibility and Submission

- `pip install oruk-bench`; open harness; one command per model.
- Free evaluation for any lab; results published unedited, including refusal counts.
- **TODO:** exact harness version pin, per-model command lines, API access dates, and cost accounting appendix.

## References

- **TODO:** emotion2vec+, EmotionThinker (ICLR 2026), Voxtral, source corpora citations, EU AI Act.

## Appendix A: Prompt text (verbatim) — **TODO**
## Appendix B: Subsample validation details (full-vs-subsample deltas per model) — **TODO**
## Appendix C: Per-language results — **TODO**
## Appendix D: Refusal taxonomy and examples — **TODO**
