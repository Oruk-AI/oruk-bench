"""Shared utilities for the Oruk Prolific annotation QC pipeline.

Public-safe module: contains no participant data, no clip-level labels.
All data access goes through the (private) HuggingFace dataset
`oruk/oruk-prolific-annotation-results`; outputs are written to a private
directory that never ships with public code.

Schema reference: schema/schema.json (v3.7.0) in the dataset repo. That file
is authoritative; notes below document where the actual export deviates from
or refines the schema doc:

  * There is no `payable` field in `metadata.server_quality`. Payability is
    encoded as `server_quality.bonus_tier` == "standard" (vs "none").
  * `source_pool_id` is the first 20 hex chars of SHA-1(manifest clipId).
    Verified exact for 100% of studio-pool rows. `keys/source_key.json` only
    covers the 41 fixed participant-facing items (practice etc.) of the
    legacy design, so the manifest join goes through the SHA-1 key instead.
  * The 2026-07-06 export contains legacy packages from earlier pool versions
    (stage42-english-train-10000-v1, nv-mixed-english-pool-20000-v1, none).
    The main analysis is restricted to poolVersion == studio-metacorpus-20162-v1.
  * Rows from schema <3.7.0 may contain the retired label "explaining"
    (320 selections). It is not part of the canonical 31-label vocabulary and
    is kept out of the aggregated label columns (documented in reports).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from collections import Counter
from dataclasses import dataclass, field


def utf8_stdout() -> None:
    """Windows cp1252 console crashes on non-ASCII; force UTF-8 stdout."""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


HF_DATA_REPO = "oruk/oruk-prolific-annotation-results"
MAIN_EXPORT = "exports/2026-07-06T22-14-26-submissions.ndjson"
PILOT_EXPORT = "exports/2026-07-03T19-49-01-submissions.ndjson"
MANIFEST_FILE = "keys/studio_metacorpus_manifest_2026-07-04.json"
STUDIO_POOL_VERSION = "studio-metacorpus-20162-v1"

# Canonical 31-label vocabulary (schema 3.7.0).
EMOTION_LABELS = [
    "happy", "excited", "hopeful", "sad", "worried", "angry", "frustrated",
    "disappointed", "scared", "disgusted", "surprised", "embarrassed",
    "proud", "relieved", "neutral",
]
STYLE_LABELS = [
    "energetic", "passionate", "irritated", "warm", "playful", "sarcastic",
    "deadpan", "hesitant", "confident", "sincere", "skeptical", "tired",
    "formal", "casual", "impatient", "distracted",
]
ALL_LABELS = EMOTION_LABELS + STYLE_LABELS
RETIRED_LABELS = {"explaining"}  # valid in pre-3.7.0 rows, excluded from aggregation

# 7-class benchmark space and the emotion-label mapping onto it.
SEVEN_CLASSES = ["anger", "happiness", "sadness", "fear", "disgust", "surprise", "neutral"]

# Direct mappings come from the schema's own canonicalSourceMapping.
# Family mappings follow common SER practice (IEMOCAP merges excited into
# happiness; frustration is anger-family; worry is fear/anxiety-family;
# disappointment is sadness-family in Plutchik-style taxonomies).
EMOTION_TO_7CLASS = {
    # direct (schema canonicalSourceMapping)
    "happy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "scared": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "neutral": "neutral",
    # family-level (documented as approximate in reports)
    "excited": "happiness",
    "frustrated": "anger",
    "worried": "fear",
    "disappointed": "sadness",
}
UNMAPPED_EMOTIONS = ["hopeful", "embarrassed", "proud", "relieved"]

# Corpora that appear in our model's training distribution with emotion labels.
IN_DISTRIBUTION_CORPORA = {"cremad", "ravdess"}
FRESH_LABEL_CORPORA = {"ears", "expresso", "libritts", "vctk"}


def download_inputs() -> dict:
    """Fetch (cached) required files from the HF dataset; return local paths."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    files = {
        "submissions": MAIN_EXPORT,
        "pilot_submissions": PILOT_EXPORT,
        "manifest": MANIFEST_FILE,
        "schema": "schema/schema.json",
    }
    return {
        key: hf_hub_download(HF_DATA_REPO, name, repo_type="dataset")
        for key, name in files.items()
    }


def pool_hash(clip_id: str) -> str:
    return hashlib.sha1(clip_id.encode("utf-8")).hexdigest()[:20]


def load_manifest(path: str) -> dict[str, dict]:
    """Return {source_pool_id -> manifest entry} for the 20,162-clip pool."""
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    return {pool_hash(e["clipId"]): e for e in entries}


@dataclass
class SessionRecord:
    """One submission package restricted to the fields the QC pipeline needs."""

    session_id: str
    participant_id: str
    schema_version: str
    completed_at: str | None
    bonus_tier: str | None
    server_flags: list[str]
    server_quality: dict
    demographics_flags: list[str]
    practice_rows: list[dict] = field(default_factory=list)
    pool_rows: list[dict] = field(default_factory=list)
    probe_rows: list[dict] = field(default_factory=list)
    attention_rows: list[dict] = field(default_factory=list)


def _slim_row(row: dict) -> dict:
    labels = row.get("labels") or {}
    return {
        "trial_id": row.get("trial_id"),
        "trial_role": row.get("trial_role"),
        "source_pool_id": row.get("source_pool_id"),
        "selected": list(labels.get("selected") or []),
        "response_ms": row.get("response_ms"),
        "played_fraction": row.get("played_fraction"),
        "ended_once": bool(row.get("ended_once")),
        "audio_failed": bool(row.get("audio_failed")),
        "attention_passed": row.get("attention_passed"),
        "gold_primary_correct": row.get("gold_primary_correct"),
        "submitted_at": row.get("submitted_at"),
        "schema_version": row.get("schema_version"),
    }


def load_studio_sessions(submissions_path: str) -> list[SessionRecord]:
    """Parse the main export, keeping only studio-metacorpus-pool sessions."""
    sessions: list[SessionRecord] = []
    with open(submissions_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pkg = json.loads(line)
            md = pkg.get("metadata") or {}
            pool = (md.get("trial_pool") or {}).get("poolVersion")
            if pool != STUDIO_POOL_VERSION:
                continue
            sq = md.get("server_quality") or {}
            demo = md.get("demographics") or {}
            rec = SessionRecord(
                session_id=md.get("session_id"),
                participant_id=md.get("participant_id"),
                schema_version=md.get("schema_version"),
                completed_at=md.get("completed_at"),
                bonus_tier=sq.get("bonus_tier"),
                server_flags=list(sq.get("flags") or []),
                server_quality=sq,
                demographics_flags=list(demo.get("flags") or []),
            )
            for row in pkg.get("rows") or []:
                role = row.get("trial_role")
                slim = _slim_row(row)
                if role == "practice":
                    rec.practice_rows.append(slim)
                elif role == "pool":
                    slim["source"] = row.get("source") or {}
                    rec.pool_rows.append(slim)
                elif role == "probe_repeat":
                    rec.probe_rows.append(slim)
                elif role == "attention":
                    rec.attention_rows.append(slim)
            sessions.append(rec)
    return sessions


def jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    union = a | b
    return len(a & b) / len(union)


def label_set_counter(rows: list[dict]) -> Counter:
    return Counter(frozenset(r["selected"]) for r in rows)


def write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
