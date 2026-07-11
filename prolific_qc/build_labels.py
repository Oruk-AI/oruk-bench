"""Aggregate included-rater annotations into per-clip soft label distributions.

Covers tasks: label aggregation, 7-class mapping, contamination flags.

Design decisions (documented here and in reports):
  * Included raters only (from private_bench/rater_quality_report.json).
  * One annotation per (rater, clip): the repeated consistency clip is
    annotated twice by the same rater; only the FIRST presentation (trial_role
    == "pool") enters aggregation. probe_repeat rows are used solely for QC.
  * audio_failed rows are dropped (labels are empty/unreliable by design).
  * The retired label "explaining" (valid in pre-3.7.0 rows) is excluded from
    the 31 aggregated label columns; rows containing it keep their other labels.
  * Soft label = fraction of that clip's included raters who selected the label.
  * 7-class view: each rater votes for every 7-class that any of their selected
    emotion labels maps to (EMOTION_TO_7CLASS). cls7_* columns are the fraction
    of the clip's raters voting each class. The hard label is the unique modal
    class by vote count; ties or zero emotion votes leave it null.
  * in_distribution_audio: True for CREMA-D and RAVDESS (emotion-labeled in our
    model's training distribution), False for EARS/Expresso/LibriTTS-R/VCTK
    (audio public, but our labels are fresh). All audio is public; the honest
    claim is "proprietary label distributions on public audio".

Outputs (PRIVATE):
  private_bench/private_v1_labels.parquet
  private_bench/label_aggregation_stats.json  (coverage histogram etc.)

Usage:  python build_labels.py
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import pandas as pd

from qc_common import (
    ALL_LABELS,
    EMOTION_TO_7CLASS,
    IN_DISTRIBUTION_CORPORA,
    SEVEN_CLASSES,
    download_inputs,
    load_manifest,
    load_studio_sessions,
    utf8_stdout,
    write_json,
)

PRIVATE_DIR = os.environ.get("ORUK_PRIVATE_DIR", "private_bench")


def load_included_raters() -> set[str]:
    path = os.path.join(PRIVATE_DIR, "rater_quality_report.json")
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    return {pid for pid, m in rep["participants"].items() if m["include"]}


def collect_annotations(sessions, included: set[str]) -> dict[str, list[set]]:
    """{source_pool_id -> [selected-label sets, one per included rater]}"""
    per_clip: dict[str, list[set]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for s in sessions:
        if s.participant_id not in included:
            continue
        for r in s.pool_rows:  # probe_repeat intentionally excluded
            if r["audio_failed"] or not r["source_pool_id"]:
                continue
            pair = (s.participant_id, r["source_pool_id"])
            if pair in seen_pairs:  # participant with 2 sessions could repeat a clip
                continue
            seen_pairs.add(pair)
            per_clip[r["source_pool_id"]].append(set(r["selected"]))
    return per_clip


def main() -> None:
    utf8_stdout()
    paths = download_inputs()
    manifest = load_manifest(paths["manifest"])
    sessions = load_studio_sessions(paths["submissions"])
    included = load_included_raters()
    print(f"included raters: {len(included)}")

    per_clip = collect_annotations(sessions, included)
    print(f"clips with >=1 included annotation: {len(per_clip)}")

    # coverage over the FULL 20,162-clip pool
    coverage = Counter()
    for pool_id in manifest:
        n = len(per_clip.get(pool_id, []))
        coverage["3+" if n >= 3 else str(n)] += 1
    unknown_ids = [pid for pid in per_clip if pid not in manifest]
    if unknown_ids:
        raise RuntimeError(f"{len(unknown_ids)} pool ids missing from manifest")

    records = []
    for pool_id, annots in per_clip.items():
        m = manifest[pool_id]
        n = len(annots)
        rec = {
            "clip_id": m["clipId"],
            "source_pool_id": pool_id,
            "corpus": m["corpus"],
            "source_file": m["path"],
            "speaker_id": m["speakerKey"],
            "duration_sec": m["durationSec"],
            "n_raters": n,
            "in_distribution_audio": m["corpus"] in IN_DISTRIBUTION_CORPORA,
        }
        for lab in ALL_LABELS:
            rec[f"label_{lab}"] = sum(1 for a in annots if lab in a) / n

        # 7-class votes: one rater may vote several classes
        votes = Counter()
        for a in annots:
            classes = {EMOTION_TO_7CLASS[e] for e in a if e in EMOTION_TO_7CLASS}
            for c in classes:
                votes[c] += 1
        for c in SEVEN_CLASSES:
            rec[f"cls7_{c}"] = votes.get(c, 0) / n
        hard, hard_votes = None, 0
        if votes:
            best = votes.most_common()
            if len(best) == 1 or best[0][1] > best[1][1]:
                hard, hard_votes = best[0]
        rec["hard_label_7"] = hard
        rec["hard_label_votes"] = hard_votes
        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("clip_id").reset_index(drop=True)
    out_parquet = os.path.join(PRIVATE_DIR, "private_v1_labels.parquet")
    df.to_parquet(out_parquet, index=False)

    n_hard = int(df["hard_label_7"].notna().sum())
    hard_dist = df["hard_label_7"].value_counts(dropna=True).to_dict()
    hard_2plus = int(df.loc[df["n_raters"] >= 2, "hard_label_7"].notna().sum())
    stats = {
        "n_included_raters": len(included),
        "n_annotations_used": int(sum(len(v) for v in per_clip.values())),
        "n_clips": int(len(df)),
        "pool_size": len(manifest),
        "coverage_histogram": {k: coverage[k] for k in ["0", "1", "2", "3+"]},
        "n_hard_label_7": n_hard,
        "n_hard_label_7_among_2plus_raters": hard_2plus,
        "hard_label_distribution": {k: int(v) for k, v in hard_dist.items()},
        "corpus_counts": df["corpus"].value_counts().to_dict(),
        "in_distribution_clip_count": int(df["in_distribution_audio"].sum()),
        "mapping": EMOTION_TO_7CLASS,
    }
    write_json(os.path.join(PRIVATE_DIR, "label_aggregation_stats.json"), stats)
    print(json.dumps(stats, indent=1))
    print(f"wrote {out_parquet} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
