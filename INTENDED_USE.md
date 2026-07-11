# Intended Use

## What this benchmark is for

oruk-bench measures **model capability**: how well a speech-emotion recognition
system reproduces aggregated human annotations of *how a speaker sounds* on a
fixed, held-out, multilingual corpus. Appropriate uses:

- Comparing SER architectures, training recipes, and scale on a common
  protocol.
- Tracking research progress on multilingual, cross-corpus emotion
  recognition.
- Auditing closed audio models' capability claims with published prompts and
  error accounting.
- Studying failure modes (per-class, per-language, per-source breakdowns are
  published for every run).

The labels are perception-level annotations of vocal expression. A high score
means a model matches how human raters *perceive* a clip — not that it can
determine what anyone actually feels.

## Explicitly out of scope

**Prohibited-context uses (EU AI Act Article 5(1)(f)).** This benchmark must
not be used to develop, market, or validate systems that infer the emotions of
natural persons in **workplace or educational contexts**, which the EU AI Act
prohibits (with narrow medical/safety exceptions). A leaderboard score here is
not, and must not be presented as, evidence of fitness for any such system.

**Individual-level inference claims.** Benchmark scores are corpus-level
aggregates. They do not support claims about any individual person's emotional
state, truthfulness, intent, mental health, or personality — in hiring,
lending, insurance, law enforcement, border control, or anywhere else. Emotion
expression varies across cultures, languages, contexts, and individuals;
perceived affect is not internal state.

**High-stakes decision-making.** No score here qualifies a system to make or
inform consequential decisions about people (employment, education, credit,
medical triage, legal outcomes) without domain-specific validation, which this
benchmark does not provide.

**Surveillance.** The benchmark is not a validation instrument for emotion
surveillance of non-consenting people, at scale or otherwise.

## Interpretation cautions

- Scores are specific to this label taxonomy (7 classes), this clip
  distribution, and this protocol version. They do not transfer to other
  taxonomies (e.g. dimensional affect) or acoustic conditions.
- Closed-model scores come from a 5,000-clip subsample and include
  refusal-to-neutral defaults; read `api_errors_defaulted_to_neutral` before
  citing them.
- In-distribution entries (marked OURS) are not comparable to zero-shot
  entries; see GOVERNANCE.md.

If you are unsure whether a use is in scope, open an issue before building on
the benchmark.
