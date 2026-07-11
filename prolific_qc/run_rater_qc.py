"""Rater QC for the Oruk Prolific speech-understanding annotation batch.

Computes per-participant quality metrics from the main 2026-07-06 export
(studio-metacorpus pool sessions only) and combines them into an
include/exclude decision per rater.

Metrics per participant (pooled over their sessions; 1 participant has 2):
  * practice correctness (server gold_primary_correct on the 2 practice rows)
  * attention-check pass/fail
  * consistency probe agreement: Jaccard between the two annotations of the
    repeated clip (pool row vs probe_repeat row with the same source_pool_id)
  * median per-trial response time; fraction of real trials under 4 s
    (suspicious: submit is locked for the first 3 s of every clip)
  * mean labels selected per trial
  * straight-lining: share of trials carrying the participant's most frequent
    identical label set (max_set_share) and overall repetition rate
  * playback completion: share of non-audio-failed real trials with an ended
    event or played_fraction >= 0.9

Decision rule (all documented in the report): start from the server payable
signal (bonus_tier == "standard") and tighten with local recomputation.

Outputs (PRIVATE):
  private_bench/rater_quality_report.json
  private_bench/rater_quality_report.md

Usage:  python run_rater_qc.py
"""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict

from qc_common import (
    SessionRecord,
    download_inputs,
    jaccard,
    load_studio_sessions,
    utf8_stdout,
    write_json,
    write_text,
)

PRIVATE_DIR = os.environ.get("ORUK_PRIVATE_DIR", "private_bench")

# ---------------------------------------------------------------------------
# Documented thresholds. A rater is EXCLUDED when any check fails.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # baseline: server scoring says the submission is payable
    "server_payable": "bonus_tier == 'standard' on every session",
    # server verifies the named emotion was among selected labels
    "attention_passed": "attention check passed on every session",
    # 2 expert-labeled practice clips; require at least 1 of 2 correct
    "min_practice_correct": 1,
    # the two annotations of the repeated clip must share at least one label
    "min_probe_jaccard": 1e-9,  # i.e. strictly > 0
    # timing: median >= 4 s (server hard gate) and few near-lock submits
    "min_median_response_ms": 4000,
    "max_frac_under_4s": 0.10,
    # playback: at least 90% of real trials fully listened
    "min_playback_complete_frac": 0.90,
    # straight-lining: no identical label set on more than 50% of trials
    "max_set_share": 0.50,
    # label overuse (server review flag threshold): mean labels/trial <= 6
    "max_mean_labels": 6.0,
}


def participant_metrics(sessions: list[SessionRecord]) -> dict:
    """Pool one participant's sessions into a single metric dict."""
    real_rows = []  # pool + probe rows, the trials that matter for behavior
    practice_correct = 0
    practice_total = 0
    attention_pass = []
    probe_jaccards = []
    payable = []
    server_flags = set()
    session_ids = []

    for s in sessions:
        session_ids.append(s.session_id)
        payable.append(s.bonus_tier == "standard")
        server_flags.update(s.server_flags)
        for r in s.practice_rows:
            practice_total += 1
            if r.get("gold_primary_correct"):
                practice_correct += 1
        for r in s.attention_rows:
            attention_pass.append(bool(r.get("attention_passed")))
        real_rows.extend(s.pool_rows + s.probe_rows)

        # consistency probe: pool row and probe_repeat row share source_pool_id
        pool_by_id = {r["source_pool_id"]: r for r in s.pool_rows}
        for probe in s.probe_rows:
            first = pool_by_id.get(probe["source_pool_id"])
            if first is None:
                continue
            j = jaccard(set(first["selected"]), set(probe["selected"]))
            if j is not None:
                probe_jaccards.append(j)

    ok_rows = [r for r in real_rows if not r["audio_failed"]]
    resp = [r["response_ms"] for r in ok_rows if r["response_ms"] is not None]
    sets = [frozenset(r["selected"]) for r in ok_rows]
    set_counts = Counter(sets)
    n_labels = [len(r["selected"]) for r in ok_rows]
    playback_ok = [
        r["ended_once"] or (r["played_fraction"] is not None and r["played_fraction"] >= 0.9)
        for r in ok_rows
    ]

    return {
        "sessions": session_ids,
        "n_sessions": len(sessions),
        "n_real_trials": len(real_rows),
        "n_audio_failed": sum(1 for r in real_rows if r["audio_failed"]),
        "server_payable": all(payable),
        "server_flags": sorted(server_flags),
        "practice_correct": practice_correct,
        "practice_total": practice_total,
        "attention_passed": all(attention_pass) if attention_pass else False,
        "probe_jaccard": (sum(probe_jaccards) / len(probe_jaccards)) if probe_jaccards else None,
        "median_response_ms": statistics.median(resp) if resp else None,
        "frac_under_4s": (sum(1 for x in resp if x < 4000) / len(resp)) if resp else None,
        "mean_labels_per_trial": (sum(n_labels) / len(n_labels)) if n_labels else None,
        "max_set_share": (set_counts.most_common(1)[0][1] / len(sets)) if sets else None,
        "repetition_rate": (1 - len(set_counts) / len(sets)) if sets else None,
        "playback_complete_frac": (sum(playback_ok) / len(playback_ok)) if playback_ok else None,
    }


def decide(m: dict) -> list[str]:
    """Return the list of failed checks (empty list = include)."""
    t = THRESHOLDS
    fails = []
    if not m["server_payable"]:
        fails.append("server_not_payable")
    if not m["attention_passed"]:
        fails.append("attention_failed")
    if m["practice_correct"] < t["min_practice_correct"]:
        fails.append("practice_incorrect")
    if m["probe_jaccard"] is None or m["probe_jaccard"] <= 0:
        fails.append("probe_no_overlap")
    if m["median_response_ms"] is None or m["median_response_ms"] < t["min_median_response_ms"]:
        fails.append("median_response_fast")
    if m["frac_under_4s"] is None or m["frac_under_4s"] > t["max_frac_under_4s"]:
        fails.append("too_many_fast_trials")
    if m["playback_complete_frac"] is None or m["playback_complete_frac"] < t["min_playback_complete_frac"]:
        fails.append("playback_incomplete")
    if m["max_set_share"] is None or m["max_set_share"] > t["max_set_share"]:
        fails.append("straight_lining")
    if m["mean_labels_per_trial"] is None or m["mean_labels_per_trial"] > t["max_mean_labels"]:
        fails.append("label_overuse")
    return fails


def pctiles(vals: list[float], name: str) -> str:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return f"{name}: no data"
    import math

    def q(p: float) -> float:
        return vals[min(len(vals) - 1, math.floor(p * len(vals)))]

    return (
        f"{name}: min={vals[0]:.3f} p10={q(0.10):.3f} p25={q(0.25):.3f} "
        f"p50={q(0.50):.3f} p75={q(0.75):.3f} p90={q(0.90):.3f} max={vals[-1]:.3f}"
    )


def main() -> None:
    utf8_stdout()
    os.makedirs(PRIVATE_DIR, exist_ok=True)
    paths = download_inputs()
    sessions = load_studio_sessions(paths["submissions"])
    by_part: dict[str, list[SessionRecord]] = defaultdict(list)
    for s in sessions:
        by_part[s.participant_id].append(s)
    print(f"studio sessions: {len(sessions)}, participants: {len(by_part)}")

    report = {}
    for pid, sess in sorted(by_part.items()):
        m = participant_metrics(sess)
        m["failed_checks"] = decide(m)
        m["include"] = not m["failed_checks"]
        report[pid] = m

    included = [p for p, m in report.items() if m["include"]]
    excluded = [p for p, m in report.items() if not m["include"]]
    fail_counts = Counter(f for m in report.values() for f in m["failed_checks"])

    # distribution printout for threshold sanity-checking
    for key in [
        "probe_jaccard", "median_response_ms", "frac_under_4s",
        "mean_labels_per_trial", "max_set_share", "repetition_rate",
        "playback_complete_frac",
    ]:
        print(pctiles([m[key] for m in report.values()], key))

    summary = {
        "n_participants": len(report),
        "n_included": len(included),
        "n_excluded": len(excluded),
        "server_payable_only_would_include": sum(1 for m in report.values() if m["server_payable"]),
        "failed_check_counts": dict(fail_counts.most_common()),
        "thresholds": {k: str(v) for k, v in THRESHOLDS.items()},
    }
    out = {"summary": summary, "participants": report}
    write_json(os.path.join(PRIVATE_DIR, "rater_quality_report.json"), out)

    lines = [
        "# Rater quality report (PRIVATE)",
        "",
        "Source: `oruk/oruk-prolific-annotation-results`, export "
        "`2026-07-06T22-14-26-submissions.ndjson`, studio-metacorpus pool "
        "sessions only (`poolVersion == studio-metacorpus-20162-v1`).",
        "",
        f"- Studio sessions: **{len(sessions)}**",
        f"- Unique participants: **{len(report)}**",
        f"- Included: **{len(included)}**",
        f"- Excluded: **{len(excluded)}**",
        f"- (Server payable flag alone would have included "
        f"{summary['server_payable_only_would_include']})",
        "",
        "## Exclusion reasons (a rater can fail several)",
        "",
        "| check | raters failing |",
        "|---|---|",
    ]
    for name, cnt in fail_counts.most_common():
        lines.append(f"| {name} | {cnt} |")
    lines += [
        "",
        "## Thresholds",
        "",
        "A rater is excluded when ANY check fails:",
        "",
        "1. `server_not_payable` — server `bonus_tier != 'standard'` on any session",
        "2. `attention_failed` — attention check failed (named emotion not among selections)",
        f"3. `practice_incorrect` — fewer than {THRESHOLDS['min_practice_correct']} of 2 "
        "expert-labeled practice clips correct",
        "4. `probe_no_overlap` — Jaccard = 0 between the two annotations of the repeated clip",
        f"5. `median_response_fast` — median per-trial response < "
        f"{THRESHOLDS['min_median_response_ms']} ms",
        f"6. `too_many_fast_trials` — more than {THRESHOLDS['max_frac_under_4s']:.0%} of real "
        "trials submitted under 4 s (submit locked for first 3 s)",
        f"7. `playback_incomplete` — fewer than {THRESHOLDS['min_playback_complete_frac']:.0%} "
        "of real trials fully played (ended event or played_fraction >= 0.9)",
        f"8. `straight_lining` — one identical label set on more than "
        f"{THRESHOLDS['max_set_share']:.0%} of trials",
        f"9. `label_overuse` — mean labels per trial > {THRESHOLDS['max_mean_labels']}",
        "",
        "## Metric distributions (all raters)",
        "",
        "```",
    ]
    for key in [
        "probe_jaccard", "median_response_ms", "frac_under_4s",
        "mean_labels_per_trial", "max_set_share", "repetition_rate",
        "playback_complete_frac",
    ]:
        lines.append(pctiles([m[key] for m in report.values()], key))
    lines += ["```", ""]
    write_text(os.path.join(PRIVATE_DIR, "rater_quality_report.md"), "\n".join(lines))
    print(f"included {len(included)} / excluded {len(excluded)} of {len(report)}")
    print("wrote rater_quality_report.json/.md")


if __name__ == "__main__":
    main()
