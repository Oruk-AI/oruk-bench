# Governance

## Conflict-of-interest disclosure

**Oruk AI both maintains this benchmark and trains models that appear on its
leaderboard.** We do not pretend otherwise, and we mitigate it structurally:

- Every oruk model is marked **OURS** and **in-distribution** in the README,
  in `leaderboard/leaderboard.json` (`"ours": true, "in_distribution": true`),
  and in any derived reporting. oruk models are trained in-distribution;
  every other entrant is evaluated zero-shot cross-corpus.
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

## Private labels: current evidence and planned evaluation

The [published annotation QC report](prolific_qc/QC_SUMMARY.md) describes a
frozen private-v1 label file with **13,716 clips**. Of these, 13,704 have one
included rater, 12 have two, and none have three. The planned three-rater
coverage was not reached. Most selection fractions therefore describe one
listener's choices, not an estimated distribution across listeners.

The underlying audio comes from public studio corpora. Private storage of fresh
labels does not establish that the recordings, speakers or sources were excluded
from every evaluated model's development. The QC report marks 1,889 clips from
CREMA-D and RAVDESS as in-distribution audio; the other clips' fresh labels alone
do not establish model independence.

The QC report records a private Hugging Face upload and a hash retained with the
private artifacts. This repository does not publish that frozen-file hash or
completed private-v1 results for every leaderboard entrant. It does not establish
custody by an independent external evaluator. **The historical leaderboard is
not a completed public/private comparison.**

Before publishing such a comparison, we must document the permitted use and
development-exclusion scope, complete the planned annotation coverage, freeze
and publish a permission-safe manifest commitment before inference, and register
the scorer and evaluated model revisions. An external evaluator's role must be
stated separately from Oruk's role. Results must retain failures and the
limitations of each population; a public/private score gap alone does not
establish contamination.

Private-split comparisons and a refresh cadence remain planned work. Historical
results and their original scoring rules are unchanged by this clarification.

## Changes to this document

Material governance changes are made by pull request to this file, are
versioned with the benchmark, and never apply retroactively to already
published results.
