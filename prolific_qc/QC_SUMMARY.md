# Prolific annotation QC summary (public-safe)

Data-QC track for the Oruk speech-emotion benchmark. This document contains
methodology and aggregate statistics only: no raw labels, no participant
identifiers, no clip-level data. All audio in the underlying pool comes from
public studio corpora (EARS, Expresso, LibriTTS-R, VCTK, CREMA-D, RAVDESS);
what is proprietary is the **label distributions**, not the audio. The honest
claim is: *proprietary label distributions on public audio*.

## Source data

- Prolific annotation export of 2026-07-06 (production app v48, schema
  3.7.0): 461 studio-metacorpus-pool sessions from 460 unique participants,
  each session = 39 random pool clips + 1 repeated consistency clip
  + 2 expert-labeled practice clips + 1 attention check.
- Task: unordered multi-select over a flat 31-label vocabulary (15 emotions
  + 16 speaking styles), at least 2 labels per trial, submit locked for the
  first 3 s of every clip.
- The clip pool is a 20,162-clip studio meta-corpus (2–5 s English mono).
- A small pre-cleanup pilot batch (2026-07-03, 40 submissions on earlier
  pool versions) is analyzed separately and **never merged** into v1.

## Pipeline (scripts in this folder)

| script | purpose |
|---|---|
| `qc_common.py` | shared loaders, label constants, 7-class mapping |
| `run_rater_qc.py` | per-rater metrics + include/exclude decision |
| `build_labels.py` | per-clip soft label aggregation + frozen parquet |
| `run_agreement.py` | Krippendorff alpha (MASI/Jaccard/per-label/7-class) |
| `pilot_summary.py` | separate descriptive stats for the pilot export |
| `freeze_and_upload.py` | SHA-256 freeze + escrow upload (private repo) |

Run order: `run_rater_qc.py` → `build_labels.py` → `run_agreement.py` →
`freeze_and_upload.py`. Outputs go to a private directory / private HF
dataset and are not part of this repository.

## Rater QC

Starting point is the server payability signal (`bonus_tier == "standard"`,
which alone would keep 457 of 460 raters), tightened with recomputed checks.
A rater is excluded if ANY check fails:

1. server not payable on any session
2. attention check failed (named emotion not among selections)
3. fewer than 1 of 2 expert-labeled practice clips correct
4. zero label overlap (Jaccard = 0) between the two annotations of the
   repeated consistency clip
5. median per-trial response time < 4 s
6. more than 10% of real trials submitted under 4 s
7. fewer than 90% of real trials fully played (ended event or
   played fraction ≥ 0.9)
8. straight-lining: one identical label set on > 50% of trials
9. label overuse: mean labels per trial > 6

**Result: 351 of 460 raters included, 109 excluded.** Failure counts (a
rater can fail several): probe no-overlap 89, practice incorrect 16, label
overuse 8, attention failed 5, server not payable 3. No rater failed the
timing, playback, or straight-lining checks — the 3 s submit lock and the
server-side playback verification appear to have prevented rushing entirely
(median per-rater response time was ~22 s per trial; playback completion was
100% across the board).

## Label aggregation and coverage

For included raters only, each clip gets a soft distribution: fraction of its
raters selecting each of the 31 canonical labels. Only the first presentation
of the repeated consistency clip enters aggregation; audio-failure rows are
dropped; the retired label "explaining" (valid pre-3.7.0) is excluded from
aggregated columns.

Coverage of the 20,162-clip pool by included annotations:

| annotations per clip | clips |
|---|---|
| 0 | 6,446 |
| 1 | 13,704 |
| 2 | 12 |
| 3+ | 0 |

13,716 clips (13,728 annotations) enter the frozen v1 file. **The
3-annotators-per-clip design target was not reached** — collection ended
after roughly one coverage pass, so v1 soft labels are predominantly
single-rater and should be treated as such (see Limitations).

## 7-class mapping

The benchmark's 7 classes and the Prolific emotion labels mapped to them
(mapping shipped in `qc_common.py`; direct rows follow the schema's own
canonical source mapping, family rows follow common SER merges):

| 7-class | mapped labels | kind |
|---|---|---|
| anger | angry; frustrated | direct; family |
| happiness | happy; excited | direct; family |
| sadness | sad; disappointed | direct; family |
| fear | scared; worried | direct; family |
| disgust | disgusted | direct |
| surprise | surprised | direct |
| neutral | neutral | direct |

Unmapped emotions (kept only in the 31-label soft columns): hopeful,
embarrassed, proud, relieved. All 16 style tags stay separate.

Per clip, each rater votes for every mapped class among their selections; the
hard 7-class label is the unique modal class (ties or zero emotion votes →
no hard label). **7,720 of 13,716 clips receive a hard 7-class label**
(neutral 3,867; happiness 1,538; anger 759; fear 553; sadness 498; surprise
398; disgust 107).

## Agreement (Krippendorff's alpha)

Because pool overlap collapsed (see above), only 193 pool clips have 2
annotations from distinct raters — and ~90% of the first-side annotations in
those pairs come from 5 low-quality pre-cleanup sessions whose clips were
reassigned, so the all-rater pool alpha is a junk-vs-good comparison and
lands at chance by construction. The 2 practice clips (annotated by all
raters) give the most stable estimates; the acted-emotion check below is the
strongest external validity signal.

| measure | pool (all raters) | pool (included) | practice (included) |
|---|---|---|---|
| pairable units | 193 | 12 | 2 |
| alpha, MASI set distance | −0.00 | 0.03 | 0.06 |
| alpha, Jaccard set distance | −0.01 | 0.09 | 0.12 |
| alpha, mapped 7-class hard (nominal) | 0.01 | 0.40 | 0.44 |

- The 7-class hard-label alpha (~0.4 on both usable views) matches the
  expected ballpark for perceived-emotion tasks.
- Full multi-select alphas are much lower (~0.05–0.12): raters agree on the
  headline emotion far more than on the exact 31-label set, and per-label
  binary alphas are < 0.2 for most labels (confident/neutral/formal/deadpan
  among the more reliable; style tags generally noisier). This is flagged:
  the 31-label soft distributions are usable as distributional targets, not
  as reliable per-label ground truth.
- External validity: on CREMA-D/RAVDESS clips, the acted emotion from the
  source corpus is among an included rater's mapped selections 37.6% of the
  time vs a 14.1% shuffled-annotation chance baseline (~2.7× chance).

## Contamination / provenance flags

Each clip in the frozen file carries `in_distribution_audio`:

- **True** (1,889 clips): CREMA-D, RAVDESS — emotion-labeled in our model's
  training distribution.
- **False** (11,827 clips): EARS, Expresso, LibriTTS-R, VCTK — audio public
  but not emotion-labeled in our training; labels are fresh.

## Freeze

v1 labels are frozen as a parquet (13,716 rows, 48 columns), hashed with
SHA-256, escrowed with timestamp and row count, and uploaded together with
the rater-quality and agreement reports to a **private** HF dataset
(`oruk/speech-emotion-bench-private-v1`, test-only, never-train). The hash
and row-level data live only in the private artifacts.

## Limitations (read before citing numbers)

1. Coverage is ~1 annotation/clip, not the 3 targeted; soft distributions
   for single-rater clips are point masses. Hard 7-class labels on 1-rater
   clips reflect one listener. A second collection pass is the obvious fix.
2. Pool-level inter-rater alpha is effectively unmeasurable from this batch
   (12 clean pairs); the reported ~0.4 7-class agreement comes from the
   practice clips and the small clean-pair sample.
3. Perceived-emotion labels on non-acted corpora have no external ground
   truth; the CREMA-D/RAVDESS acted-emotion check is a proxy.
