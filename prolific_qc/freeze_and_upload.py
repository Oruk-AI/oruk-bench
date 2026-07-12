"""Freeze the private v1 label file and escrow it to a private HF dataset.

Steps:
  1. SHA-256 the frozen private_bench/private_v1_labels.parquet
  2. Write private_bench/ESCROW.md (hash + UTC timestamp + row count)
  3. Create the PRIVATE dataset repo oruk/speech-emotion-bench-private-v1
     (private=True, exist_ok) and upload: the parquet, the rater quality
     report (.json/.md), the agreement report (.json/.md), ESCROW.md, and a
     README stating test-only / never-train / private license.

Usage:  python freeze_and_upload.py [--no-upload]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os

import pandas as pd

from qc_common import utf8_stdout, write_text

PRIVATE_DIR = os.environ.get("ORUK_PRIVATE_DIR", "private_bench")
ESCROW_REPO = "oruk/speech-emotion-bench-private-v1"

README = """---
license: other
license_name: private-research-data
viewer: false
tags:
- speech-emotion-recognition
- benchmark
- test-only
- oruk
pretty_name: Oruk Speech Emotion Bench (private v1 labels)
---

# speech-emotion-bench-private-v1

**TEST-ONLY. NEVER TRAIN ON THIS DATA.** License: private research data —
do not redistribute.

Frozen v1 of the Oruk private speech-emotion benchmark labels:
per-clip soft label distributions over a 31-label vocabulary (15 emotions +
16 speaking styles) plus a mapped 7-class view, aggregated from
QC-filtered Prolific annotations
(`oruk/oruk-prolific-annotation-results`, export 2026-07-06).

All audio comes from public studio corpora (EARS, Expresso, LibriTTS-R,
VCTK, CREMA-D, RAVDESS); what is proprietary here is the **label
distributions**, not the audio. The honest claim for papers/marketing is
"proprietary label distributions on public audio". Clips from CREMA-D and
RAVDESS are marked `in_distribution_audio=True` because those corpora are
emotion-labeled in our model training distribution; the other four corpora
carry fresh labels.

## Files

- `private_v1_labels.parquet` — frozen label table (integrity hash in
  `ESCROW.md`). One row per clip with >= 1 included-rater annotation:
  clip/corpus/speaker provenance, `n_raters`, `label_*` selection fractions
  (31 columns), `cls7_*` mapped 7-class fractions, `hard_label_7`,
  `in_distribution_audio`.
- `ESCROW.md` — SHA-256, freeze timestamp, row count.
- `rater_quality_report.json` / `.md` — per-rater QC and include/exclude
  decisions (contains hashed participant ids — keep private).
- `agreement_report.json` / `.md` — Krippendorff alpha (MASI / Jaccard /
  per-label / 7-class) with sample-size caveats.
"""


def main() -> None:
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true", help="freeze locally only")
    args = ap.parse_args()

    parquet = os.path.join(PRIVATE_DIR, "private_v1_labels.parquet")
    digest = hashlib.sha256(open(parquet, "rb").read()).hexdigest()
    n_rows = len(pd.read_parquet(parquet, columns=["clip_id"]))
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    escrow = "\n".join([
        "# ESCROW — speech-emotion-bench private v1",
        "",
        "- file: `private_v1_labels.parquet`",
        f"- sha256: `{digest}`",
        f"- frozen_at_utc: {stamp}",
        f"- rows: {n_rows}",
        "",
        "This hash freezes v1 of the private benchmark labels. Any file that",
        "does not match this digest is not v1. TEST-ONLY: never train on it.",
        "",
    ])
    write_text(os.path.join(PRIVATE_DIR, "ESCROW.md"), escrow)
    print(f"sha256={digest}")
    print(f"rows={n_rows} frozen_at={stamp}")

    if args.no_upload:
        print("skipping upload (--no-upload)")
        return

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(ESCROW_REPO, repo_type="dataset", private=True, exist_ok=True)
    readme_path = os.path.join(PRIVATE_DIR, "_escrow_README.md")
    write_text(readme_path, README)
    uploads = [
        (readme_path, "README.md"),
        (parquet, "private_v1_labels.parquet"),
        (os.path.join(PRIVATE_DIR, "ESCROW.md"), "ESCROW.md"),
        (os.path.join(PRIVATE_DIR, "rater_quality_report.json"), "rater_quality_report.json"),
        (os.path.join(PRIVATE_DIR, "rater_quality_report.md"), "rater_quality_report.md"),
        (os.path.join(PRIVATE_DIR, "agreement_report.json"), "agreement_report.json"),
        (os.path.join(PRIVATE_DIR, "agreement_report.md"), "agreement_report.md"),
    ]
    for local, remote in uploads:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=ESCROW_REPO, repo_type="dataset")
        print(f"uploaded {remote}")
    info = api.repo_info(ESCROW_REPO, repo_type="dataset")
    print(f"repo private={info.private} sha={info.sha}")


if __name__ == "__main__":
    main()
