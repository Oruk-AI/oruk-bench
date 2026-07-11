# Governance

## Conflict-of-interest disclosure

**Oruk AI both maintains this benchmark and trains models that appear on its
leaderboard.** We do not pretend otherwise, and we mitigate it structurally:

- Every oruk model is marked **OURS** and **in-distribution** in the README,
  in `leaderboard/leaderboard.json` (`"ours": true, "in_distribution": true`),
  and in any derived reporting. oruk models are trained on this benchmark's
  training split; every other entrant is evaluated zero-shot cross-corpus.
  These numbers answer "how well can a model do on this distribution when
  trained for it" — they are not a like-for-like comparison with zero-shot
  entrants, and we say so wherever they appear.
- All entrants — including ours — run through the same public harness in this
  repository: same audio preprocessing, same label mapping, same scoring
  function, same error accounting.
- The scoring code, prompts, and subsampling procedure are public and
  versioned, so any reported number can be reproduced by a third party with
  data access.

## Free-evaluation policy

Any lab may request an evaluation run of their model at no cost:

- Open an issue titled `[eval request] <model name>` with a link to public
  weights (or an API endpoint plus temporary credentials), a suggested adapter
  configuration, and any inference caveats.
- We run the standard protocol and publish the results **unedited** — including
  scores that are unflattering to the submitter or that beat our own models.
- Submitters may ask us to annotate results with caveats (e.g. "checkpoint was
  not tuned for long audio"), which we publish alongside — never instead of —
  the numbers.
- We do not accept resubmission churn: one listed result per released
  checkpoint; a new checkpoint is a new entry.

## No payment for placement

- Leaderboard placement cannot be bought. We accept no payment, sponsorship, or
  in-kind consideration in exchange for inclusion, exclusion, ranking, or
  framing of any result.
- Compute or API credits donated to run third-party evaluations are disclosed
  in the result metadata if accepted.

## Escrowed private split

To make overfitting to the public benchmark detectable:

- A private evaluation split (private-v1, multi-rater human labels; see
  BENCHMARK_CARD.md) is held in escrow and never distributed.
- At freeze time we publish the **SHA-256 hash** of the frozen private split
  manifest, so its contents are cryptographically committed before any model is
  scored against it.
- Every leaderboard entrant is scored on both the public shards and the private
  split, and we publish **public/private gap monitoring**: a persistent,
  unexplained positive gap on the public side is flagged on the entry and
  investigated as possible benchmark contamination.
- The private split refreshes on a published cadence; hashes for retired splits
  are kept so historical results remain auditable.

## Changes to this document

Material governance changes are made by pull request to this file, are
versioned with the benchmark, and never apply retroactively to already
published results.
