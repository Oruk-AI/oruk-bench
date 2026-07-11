"""Inter-rater agreement for the Oruk Prolific annotation batch.

Krippendorff's alpha with set-valued annotations:
  * MASI distance:    d = 1 - Jaccard * monotonicity  (Passonneau 2006)
  * Jaccard distance: d = 1 - |A&B| / |A|B|
  * per-label binary alpha (nominal) for each of the 31 canonical labels
  * nominal alpha over mapped 7-class hard labels (per-annotation hard label =
    the single 7-class its mapped emotion selections point to; annotations
    mapping to zero or 2+ classes carry no hard label)

IMPORTANT CAVEATS (also written into the report):
  * Collection stopped far short of the 3-annotators-per-clip target: only
    193 pool clips have 2 annotations from distinct raters, none have 3+.
  * Those pairs are NOT a random sample: ~90% of first-side annotations come
    from 5 low-quality pre-cleanup sessions (attention failures / fast
    sessions / probe mismatches) whose clips the server reassigned. Pair
    agreement over all raters is therefore junk-vs-good and lands at chance.
    After rater exclusions only 12 pairable clips remain — too few for a
    stable alpha. Both views are reported with these caveats.
  * The two practice clips (annotated by every rater) are reported
    separately — 2 units only, but hundreds of raters each; they give the
    most stable available agreement estimate, on those specific clips.
  * As an external validity check we score annotations against the acted
    emotion encoded in CREMA-D/RAVDESS filenames (hit = acted emotion among
    the rater's mapped 7-class selections), compared to a shuffled-annotation
    chance baseline.

Repeat-probe rows are intra-rater duplicates and are excluded from
inter-rater agreement (they feed the rater QC consistency metric instead).

Outputs (PRIVATE):
  private_bench/agreement_report.json
  private_bench/agreement_report.md

Usage:  python run_agreement.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from qc_common import (
    ALL_LABELS,
    EMOTION_TO_7CLASS,
    download_inputs,
    load_studio_sessions,
    utf8_stdout,
    write_json,
    write_text,
)

PRIVATE_DIR = os.environ.get("ORUK_PRIVATE_DIR", "private_bench")


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def jaccard_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


def masi_distance(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    inter = len(a & b)
    if inter == 0:
        return 1.0
    j = inter / len(a | b)
    if a == b:
        m = 1.0
    elif a <= b or b <= a:
        m = 2 / 3
    else:
        m = 1 / 3
    return 1.0 - j * m


def nominal_distance(a, b) -> float:
    return 0.0 if a == b else 1.0


def krippendorff_alpha(units: list[list], distance) -> dict:
    """Generic distance-based alpha. `units` = list of value-lists (len >= 1);
    units with fewer than 2 values are ignored (standard missing-data rule)."""
    pairable = [u for u in units if len(u) >= 2]
    values = [v for u in pairable for v in u]
    n = len(values)
    if n < 2 or not pairable:
        return {"alpha": None, "n_units": len(pairable), "n_values": n}
    d_o = 0.0
    for u in pairable:
        m = len(u)
        s = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                s += distance(u[i], u[j])
        d_o += 2.0 * s / (m - 1)
    d_o /= n
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += distance(values[i], values[j])
    d_e = 2.0 * s / (n * (n - 1))
    alpha = None if d_e == 0 else 1.0 - d_o / d_e
    return {"alpha": alpha, "n_units": len(pairable), "n_values": n,
            "observed_disagreement": d_o, "expected_disagreement": d_e}


def annotation_hard7(selected: set) -> str | None:
    classes = {EMOTION_TO_7CLASS[e] for e in selected if e in EMOTION_TO_7CLASS}
    return next(iter(classes)) if len(classes) == 1 else None


# ---------------------------------------------------------------------------
# external validity: acted emotion encoded in CREMA-D / RAVDESS file names
# ---------------------------------------------------------------------------
CREMAD_CODES = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
                "HAP": "happiness", "NEU": "neutral", "SAD": "sadness"}
RAVDESS_CODES = {"01": "neutral", "02": "neutral", "03": "happiness",
                 "04": "sadness", "05": "anger", "06": "fear",
                 "07": "disgust", "08": "surprise"}


def acted_emotion(source: dict) -> str | None:
    corpus = source.get("corpus")
    usid = source.get("upstreamSourceId") or ""
    if corpus == "cremad":
        parts = usid.split("_")
        return CREMAD_CODES.get(parts[2]) if len(parts) >= 3 else None
    if corpus == "ravdess":
        parts = usid.split("-")
        return RAVDESS_CODES.get(parts[2]) if len(parts) >= 3 else None
    return None


def acted_emotion_check(sessions, raters: set[str] | None) -> dict:
    """Hit rate of acted emotion among each rater's mapped 7-class selections,
    vs a label-shuffle chance baseline."""
    import random

    mapped_sets, golds = [], []
    for s in sessions:
        if raters is not None and s.participant_id not in raters:
            continue
        for r in s.pool_rows:
            if r["audio_failed"]:
                continue
            mset = {EMOTION_TO_7CLASS[e] for e in r["selected"] if e in EMOTION_TO_7CLASS}
            gold = acted_emotion(r.get("source") or {})
            mapped_sets.append(mset)
            if gold is not None:
                golds.append((gold, mset))
    hits = sum(1 for g, mset in golds if g in mset)
    rng = random.Random(0)
    chance = sum(1 for g, _ in golds if g in rng.choice(mapped_sets)) / len(golds)
    return {"n_acted_trials": len(golds), "hit_rate": hits / len(golds),
            "chance_baseline": round(chance, 4)}


def collect(sessions, raters: set[str] | None):
    """Return ({pool_id: [(rater, frozenset)]}, {practice_id: [(rater, frozenset)]})."""
    pool = defaultdict(list)
    practice = defaultdict(list)
    seen = set()
    for s in sessions:
        if raters is not None and s.participant_id not in raters:
            continue
        for r in s.pool_rows:
            if r["audio_failed"] or not r["source_pool_id"]:
                continue
            key = (s.participant_id, r["source_pool_id"])
            if key in seen:
                continue
            seen.add(key)
            pool[r["source_pool_id"]].append(frozenset(r["selected"]))
        for r in s.practice_rows:
            practice[r["trial_id"]].append(frozenset(r["selected"]))
    return pool, practice


def agreement_suite(units: list[list[frozenset]]) -> dict:
    out = {
        "masi": krippendorff_alpha(units, masi_distance),
        "jaccard": krippendorff_alpha(units, jaccard_distance),
        "per_label_binary": {},
    }
    for lab in ALL_LABELS:
        bin_units = [[lab in v for v in u] for u in units]
        out["per_label_binary"][lab] = krippendorff_alpha(bin_units, nominal_distance)
    hard_units = []
    for u in units:
        hu = [h for h in (annotation_hard7(set(v)) for v in u) if h is not None]
        if hu:
            hard_units.append(hu)
    out["hard7_nominal"] = krippendorff_alpha(hard_units, nominal_distance)
    return out


def fmt(a) -> str:
    return "n/a" if a is None else f"{a:.3f}"


def main() -> None:
    utf8_stdout()
    paths = download_inputs()
    sessions = load_studio_sessions(paths["submissions"])
    with open(os.path.join(PRIVATE_DIR, "rater_quality_report.json"), encoding="utf-8") as f:
        included = {p for p, m in json.load(f)["participants"].items() if m["include"]}

    report = {}
    for name, raters in [("all_raters", None), ("included_raters", included)]:
        pool, practice = collect(sessions, raters)
        pool_units = [v for v in pool.values() if len(v) >= 2]
        report[name] = {
            "pool": agreement_suite(list(pool.values())),
            "practice": agreement_suite(list(practice.values())),
            "n_pool_clips_pairable": len(pool_units),
            "acted_emotion_check": acted_emotion_check(sessions, raters),
        }

    # low-agreement flags on the practice-clip per-label alphas (most stable
    # inter-rater sample: hundreds of raters, but only 2 clips)
    primary = report["included_raters"]["practice"]["per_label_binary"]
    low = {lab: r["alpha"] for lab, r in primary.items()
           if r["alpha"] is not None and r["alpha"] < 0.2}
    report["flags"] = {
        "low_per_label_alpha_below_0.2_practice": low,
        "caveats": [
            "Pool pairable sample is small (193 clips all-raters, 12 "
            "included-only); collection ended before the 3x overlap target.",
            "~90% of pool pair first-sides come from 5 low-quality "
            "pre-cleanup sessions whose clips were reassigned, so all-rater "
            "pool alpha is a junk-vs-good comparison and sits at chance; it "
            "does NOT measure included-rater reliability.",
            "Practice-clip alphas rest on 2 clips only and characterize "
            "those clips, not the pool.",
        ],
    }
    write_json(os.path.join(PRIVATE_DIR, "agreement_report.json"), report)

    ar, ir = report["all_raters"], report["included_raters"]
    lines = [
        "# Agreement report (PRIVATE)",
        "",
        "Krippendorff's alpha over multi-select annotations. Repeat-probe rows",
        "(intra-rater) are excluded. Pool clips have at most 2 distinct raters",
        "because collection stopped before the 3x overlap target; ~90% of the",
        "pair first-sides come from 5 low-quality pre-cleanup sessions whose",
        "clips were reassigned, so all-rater pool alpha compares junk vs good",
        "annotations and sits at chance. Practice-clip columns (2 clips, all",
        "460 raters) give the most stable estimates available.",
        "",
        "| measure | pool (all) | pool (included) | practice (all) | practice (incl) |",
        "|---|---|---|---|---|",
        f"| pairable units | {ar['n_pool_clips_pairable']} | "
        f"{ir['n_pool_clips_pairable']} | 2 | 2 |",
        f"| alpha MASI | {fmt(ar['pool']['masi']['alpha'])} | "
        f"{fmt(ir['pool']['masi']['alpha'])} | {fmt(ar['practice']['masi']['alpha'])} | "
        f"{fmt(ir['practice']['masi']['alpha'])} |",
        f"| alpha Jaccard | {fmt(ar['pool']['jaccard']['alpha'])} | "
        f"{fmt(ir['pool']['jaccard']['alpha'])} | {fmt(ar['practice']['jaccard']['alpha'])} | "
        f"{fmt(ir['practice']['jaccard']['alpha'])} |",
        f"| alpha 7-class hard (nominal) | {fmt(ar['pool']['hard7_nominal']['alpha'])} | "
        f"{fmt(ir['pool']['hard7_nominal']['alpha'])} | "
        f"{fmt(ar['practice']['hard7_nominal']['alpha'])} | "
        f"{fmt(ir['practice']['hard7_nominal']['alpha'])} |",
        "",
        "## External validity: acted-emotion hit rate (CREMA-D + RAVDESS)",
        "",
        "Hit = acted emotion from the source filename is among the rater's",
        "mapped 7-class selections. Chance = shuffled-annotation baseline.",
        "",
        "| raters | acted trials | hit rate | chance |",
        "|---|---|---|---|",
        f"| all | {ar['acted_emotion_check']['n_acted_trials']} | "
        f"{ar['acted_emotion_check']['hit_rate']:.3f} | "
        f"{ar['acted_emotion_check']['chance_baseline']:.3f} |",
        f"| included | {ir['acted_emotion_check']['n_acted_trials']} | "
        f"{ir['acted_emotion_check']['hit_rate']:.3f} | "
        f"{ir['acted_emotion_check']['chance_baseline']:.3f} |",
        "",
        "## Per-label binary alpha (included raters, practice clips)",
        "",
        "| label | alpha | | label | alpha |",
        "|---|---|---|---|---|",
    ]
    labs = [(lab, primary[lab]["alpha"]) for lab in ALL_LABELS]
    half = (len(labs) + 1) // 2
    for i in range(half):
        left = labs[i]
        right = labs[i + half] if i + half < len(labs) else ("", None)
        lines.append(f"| {left[0]} | {fmt(left[1])} | | {right[0]} | {fmt(right[1])} |")
    lines += [
        "",
        "## Flags and caveats",
        "",
        f"- Practice labels with alpha < 0.2: "
        f"{', '.join(sorted(low)) if low else 'none'}",
    ]
    lines += [f"- {c}" for c in report["flags"]["caveats"]]
    lines.append("")
    write_text(os.path.join(PRIVATE_DIR, "agreement_report.md"), "\n".join(lines))

    print("pool pairable clips: all =", ar["n_pool_clips_pairable"],
          " included =", ir["n_pool_clips_pairable"])
    for name in ("all_raters", "included_raters"):
        p = report[name]["pool"]
        pr = report[name]["practice"]
        ac = report[name]["acted_emotion_check"]
        print(f"{name}: pool MASI={fmt(p['masi']['alpha'])} "
              f"Jaccard={fmt(p['jaccard']['alpha'])} hard7={fmt(p['hard7_nominal']['alpha'])} "
              f"| practice MASI={fmt(pr['masi']['alpha'])} "
              f"Jaccard={fmt(pr['jaccard']['alpha'])} hard7={fmt(pr['hard7_nominal']['alpha'])} "
              f"| acted hit={ac['hit_rate']:.3f} vs chance={ac['chance_baseline']:.3f}")
    print("low practice per-label alphas:",
          {k: round(v, 3) for k, v in low.items()})
    print("wrote agreement_report.json/.md")


if __name__ == "__main__":
    main()
