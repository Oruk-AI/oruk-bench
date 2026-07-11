"""Statistical machinery for the Oruk speech-emotion benchmark.

Provides cluster-aware bootstrap confidence intervals, paired model
comparisons, soft-label (multi-rater) metrics, prompt-ensemble summaries,
minimum-detectable-delta simulations, and the ``leaderboard`` CLI that
attaches CIs and adjacent-rank significance tests to public numbers.

See docs/STATISTICS.md for methodology and the reporting policy.

CLI usage::

    python -m oruk_bench.stats leaderboard --results-dir <dir> \
        --preds-dir <dir> --labels <npz> --out leaderboard_ci.json
    python -m oruk_bench.stats mdd --n 1000 5000 --base-acc 0.45 0.77
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import f1_score

MetricFn = Callable[[np.ndarray, np.ndarray], float]

__all__ = [
    "BootstrapResult",
    "PairedTestResult",
    "MinDetectableDeltaResult",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "soft_label_metrics",
    "prompt_ensemble_summary",
    "min_detectable_delta",
    "main",
]


# ---------------------------------------------------------------------------
# Metric resolution
# ---------------------------------------------------------------------------

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _make_macro_f1(labels: np.ndarray) -> MetricFn:
    """Macro-F1 with the class set frozen to ``labels``.

    Freezing matters under the bootstrap: a resample can drop a rare class
    entirely, and letting sklearn re-infer the label set would silently
    change the macro average's denominator between replicates.
    """

    def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        )

    return _macro_f1


def _resolve_metric(metric_fn: str | MetricFn, y_true: np.ndarray) -> MetricFn:
    if callable(metric_fn):
        return metric_fn
    if metric_fn == "accuracy":
        return _accuracy
    if metric_fn == "macro_f1":
        return _make_macro_f1(np.unique(y_true))
    raise ValueError(
        f"Unknown metric {metric_fn!r}; use 'accuracy', 'macro_f1', or a callable."
    )


# ---------------------------------------------------------------------------
# Cluster-aware bootstrap
# ---------------------------------------------------------------------------

def _group_indices(clusters: np.ndarray) -> list[np.ndarray]:
    """Item indices grouped by cluster id (any hashable/sortable dtype)."""
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    # boundaries where the cluster id changes
    change = np.flatnonzero(sorted_c[1:] != sorted_c[:-1]) + 1
    return np.split(order, change)


def _cluster_resample_indices(
    groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    picks = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[g] for g in picks])


@dataclass
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    alpha: float
    n_boot: int
    n_clusters: int
    replicates: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("replicates")
        return d


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    clusters: Sequence | None = None,
    metric_fn: str | MetricFn = "accuracy",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BootstrapResult:
    """Cluster-aware percentile-bootstrap CI for a classification metric.

    Clusters (e.g. source corpus or speaker) are resampled with replacement;
    all items belonging to a sampled cluster enter the replicate together.
    This respects within-cluster correlation, which an item-level bootstrap
    would ignore (producing overconfident intervals).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n,)
        Ground-truth and predicted labels (ints or strings).
    clusters : array-like of shape (n,), optional
        Cluster id per item. ``None`` falls back to an item-level (iid)
        bootstrap, i.e. every item is its own cluster.
    metric_fn : "accuracy" | "macro_f1" | callable(y_true, y_pred) -> float
    n_boot : number of bootstrap replicates.
    alpha : 1 - confidence level (0.05 -> 95% CI).
    seed : RNG seed for reproducibility.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if clusters is None:
        clusters = np.arange(len(y_true))
    clusters = np.asarray(clusters)
    if clusters.shape[0] != y_true.shape[0]:
        raise ValueError("clusters must have one entry per item")

    fn = _resolve_metric(metric_fn, y_true)
    point = float(fn(y_true, y_pred))

    groups = _group_indices(clusters)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _cluster_resample_indices(groups, rng)
        reps[i] = fn(y_true[idx], y_pred[idx])

    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(
        point=point,
        ci_low=float(lo),
        ci_high=float(hi),
        alpha=alpha,
        n_boot=n_boot,
        n_clusters=len(groups),
        replicates=reps,
    )


# ---------------------------------------------------------------------------
# Paired bootstrap test
# ---------------------------------------------------------------------------

@dataclass
class PairedTestResult:
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    metric_a: float
    metric_b: float
    alpha: float
    n_boot: int
    n_clusters: int
    replicates: np.ndarray = field(repr=False)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("replicates")
        d["significant"] = self.significant
        return d


def paired_bootstrap_test(
    y_true: Sequence,
    pred_a: Sequence,
    pred_b: Sequence,
    clusters: Sequence | None = None,
    metric_fn: str | MetricFn = "accuracy",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> PairedTestResult:
    """Paired cluster bootstrap comparing two models on identical items.

    Each replicate resamples clusters once and evaluates *both* models on the
    same resample, so shared item difficulty cancels out of the difference.
    Returns delta = metric(A) - metric(B), a percentile CI for the delta, and
    a two-sided sign-flip p-value: the (add-one smoothed, doubled) fraction of
    replicates whose delta sign contradicts the observed sign. This is what
    powers "is rank i really above rank i+1" on the leaderboard.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    if not (y_true.shape == pred_a.shape == pred_b.shape):
        raise ValueError("y_true, pred_a, pred_b must have identical shapes")
    if clusters is None:
        clusters = np.arange(len(y_true))
    clusters = np.asarray(clusters)

    fn = _resolve_metric(metric_fn, y_true)
    m_a = float(fn(y_true, pred_a))
    m_b = float(fn(y_true, pred_b))
    delta = m_a - m_b

    groups = _group_indices(clusters)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _cluster_resample_indices(groups, rng)
        reps[i] = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])

    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided sign-flip p-value with add-one smoothing (never exactly 0).
    n_le = int(np.sum(reps <= 0))
    n_ge = int(np.sum(reps >= 0))
    p = 2.0 * (min(n_le, n_ge) + 1) / (n_boot + 1)
    return PairedTestResult(
        delta=delta,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(min(p, 1.0)),
        metric_a=m_a,
        metric_b=m_b,
        alpha=alpha,
        n_boot=n_boot,
        n_clusters=len(groups),
        replicates=reps,
    )


# ---------------------------------------------------------------------------
# Soft-label (multi-rater) metrics
# ---------------------------------------------------------------------------

def soft_label_metrics(
    pred_probs_or_onehot: Sequence,
    soft_targets: Sequence,
    labels: Sequence | None = None,
    eps: float = 1e-12,
) -> dict:
    """Metrics against multi-rater soft label distributions.

    Parameters
    ----------
    pred_probs_or_onehot :
        Either (n, k) predicted probabilities / one-hot rows, or a (n,) array
        of hard predictions. Hard integer predictions are taken as column
        indices; hard string predictions require ``labels`` (the class name
        of each soft-target column) to map them to columns.
    soft_targets : (n, k)
        Per-item rater label distributions (rows are normalized if needed).
    labels : optional sequence of k class names for the soft-target columns.

    Returns dict with:
      - ``cross_entropy``: mean -sum(target * log(pred)); predictions are
        clipped to ``eps`` so one-hot (hard) predictions stay finite.
      - ``js_divergence``: mean Jensen-Shannon divergence (base 2, in [0, 1]).
      - ``soft_accuracy``: mean probability mass the raters assigned to the
        model's argmax class.
    """
    soft = np.asarray(soft_targets, dtype=float)
    if soft.ndim != 2:
        raise ValueError("soft_targets must be 2-D (n_items, n_classes)")
    n, k = soft.shape
    row_sums = soft.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("every soft_targets row must have positive mass")
    soft = soft / row_sums

    pred = np.asarray(pred_probs_or_onehot)
    if pred.ndim == 1:
        if pred.shape[0] != n:
            raise ValueError("hard predictions must have one entry per item")
        if np.issubdtype(pred.dtype, np.integer):
            cols = pred
        else:
            if labels is None:
                raise ValueError(
                    "string hard predictions require `labels` naming the "
                    "soft-target columns"
                )
            lookup = {lab: i for i, lab in enumerate(labels)}
            try:
                cols = np.array([lookup[p] for p in pred])
            except KeyError as exc:
                raise ValueError(f"prediction {exc} not found in labels") from exc
        if cols.min() < 0 or cols.max() >= k:
            raise ValueError("hard prediction index out of range")
        P = np.zeros((n, k))
        P[np.arange(n), cols] = 1.0
    else:
        if pred.shape != (n, k):
            raise ValueError(f"pred shape {pred.shape} != soft_targets shape {(n, k)}")
        P = pred.astype(float)
        p_sums = P.sum(axis=1, keepdims=True)
        if np.any(p_sums <= 0):
            raise ValueError("every prediction row must have positive mass")
        P = P / p_sums

    cross_entropy = float(np.mean(-np.sum(soft * np.log(np.clip(P, eps, 1.0)), axis=1)))

    def _kl_base2(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        ratio = np.divide(p, np.clip(q, eps, None))
        terms = np.where(p > 0, p * np.log2(np.clip(ratio, eps, None)), 0.0)
        return terms.sum(axis=1)

    m = 0.5 * (P + soft)
    jsd = 0.5 * _kl_base2(P, m) + 0.5 * _kl_base2(soft, m)
    js_divergence = float(np.mean(jsd))

    soft_accuracy = float(np.mean(soft[np.arange(n), P.argmax(axis=1)]))

    return {
        "cross_entropy": cross_entropy,
        "js_divergence": js_divergence,
        "soft_accuracy": soft_accuracy,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Prompt-ensemble summary (audio LLMs)
# ---------------------------------------------------------------------------

def prompt_ensemble_summary(
    list_of_pred_arrays: Sequence[Sequence],
    y_true: Sequence,
    metric_fn: str | MetricFn = "accuracy",
) -> dict:
    """Summary of one model's scores across k prompt variants.

    Returns per-prompt metric values plus mean, sample std (ddof=1), min,
    max, and spread (max - min). The spread is our headline prompt-
    sensitivity number for audio-LLM rows on the leaderboard.
    """
    y_true = np.asarray(y_true)
    if len(list_of_pred_arrays) == 0:
        raise ValueError("need at least one prediction array")
    fn = _resolve_metric(metric_fn, y_true)
    per_prompt = []
    for i, preds in enumerate(list_of_pred_arrays):
        preds = np.asarray(preds)
        if preds.shape != y_true.shape:
            raise ValueError(f"prompt {i}: shape {preds.shape} != y_true {y_true.shape}")
        per_prompt.append(float(fn(y_true, preds)))
    arr = np.array(per_prompt)
    return {
        "k": len(per_prompt),
        "per_prompt": per_prompt,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "spread": float(arr.max() - arr.min()),
    }


# ---------------------------------------------------------------------------
# Minimum detectable delta (power simulation)
# ---------------------------------------------------------------------------

@dataclass
class MinDetectableDeltaResult:
    n_items: int
    base_acc: float
    delta: float
    power: float
    alpha: float
    n_boot: int
    n_sim: int

    def to_dict(self) -> dict:
        return asdict(self)


def _paired_rejection_rate(
    n_items: int,
    base_acc: float,
    delta: float,
    n_sim: int,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    """Fraction of simulated paired evals where the 95% CI excludes zero.

    Simulates two models with independent per-item correctness at accuracies
    ``base_acc`` and ``base_acc + delta`` on the same items, then bootstraps
    the mean paired difference. The per-item difference takes values in
    {-1, 0, +1}, so an exact iid bootstrap of its mean reduces to a
    multinomial draw over those three cells, which keeps this fast enough to
    binary-search over deltas.
    """
    acc_b = min(base_acc + delta, 1.0)
    q_lo, q_hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    rejected = 0
    for _ in range(n_sim):
        a = rng.random(n_items) < base_acc
        b = rng.random(n_items) < acc_b
        d = b.astype(np.int8) - a.astype(np.int8)
        counts = np.bincount(d + 1, minlength=3).astype(float)
        reps = rng.multinomial(n_items, counts / n_items, size=n_boot)
        means = (reps[:, 2] - reps[:, 0]) / n_items
        lo, hi = np.percentile(means, [q_lo, q_hi])
        if lo > 0 or hi < 0:
            rejected += 1
    return rejected / n_sim


def min_detectable_delta(
    n_items: int,
    base_acc: float,
    n_boot: int = 1000,
    n_sim: int = 200,
    power: float = 0.8,
    alpha: float = 0.05,
    seed: int = 0,
    tol: float = 5e-4,
) -> MinDetectableDeltaResult:
    """Smallest accuracy delta detectable at (1 - alpha) confidence.

    Simulation-based: binary-searches for the smallest delta at which a
    paired comparison of two models on the same ``n_items`` clips rejects
    "no difference" (bootstrap CI excludes zero) in at least ``power`` of
    simulated evals. Per-item correctness of the two models is simulated as
    independent, which is *conservative*: real models' errors correlate
    positively, shrinking the variance of the paired difference, so real
    detectable deltas are at or below the returned value. Items are iid in
    this simulation; clustering (speaker/corpus) reduces the effective n and
    pushes the real-world threshold up — which is exactly why the leaderboard
    CIs use the cluster bootstrap instead of this idealized bound.

    Used to justify the 5k closed-model subsample and per-cell reporting
    rules (see docs/STATISTICS.md).
    """
    if not 0.0 < base_acc < 1.0:
        raise ValueError("base_acc must be in (0, 1)")
    rng = np.random.default_rng(seed)

    lo, hi = 0.0, 0.05
    while (
        _paired_rejection_rate(n_items, base_acc, hi, n_sim, n_boot, alpha, rng) < power
    ):
        hi *= 2.0
        if hi > 1.0 - base_acc:
            hi = 1.0 - base_acc
            break
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        rate = _paired_rejection_rate(n_items, base_acc, mid, n_sim, n_boot, alpha, rng)
        if rate >= power:
            hi = mid
        else:
            lo = mid
    return MinDetectableDeltaResult(
        n_items=n_items,
        base_acc=base_acc,
        delta=round(hi, 4),
        power=power,
        alpha=alpha,
        n_boot=n_boot,
        n_sim=n_sim,
    )


# ---------------------------------------------------------------------------
# Leaderboard CLI
# ---------------------------------------------------------------------------
#
# Expected on-disk schema once per-clip predictions are exported from GCS:
#
#   --labels <npz>       keys: y_true (n,) labels; clusters (n,) cluster id
#                        per clip (source corpus or speaker; optional but
#                        strongly recommended); clip_ids (n,) optional.
#   --preds-dir <dir>    one <model-file-stem>.npz per model, keys: y_pred
#                        (n,) labels aligned with the labels file (or
#                        clip_ids (n,) for explicit alignment); probs (n, k)
#                        optional.
#   --results-dir <dir>  the existing per-model summary JSONs; every
#                        <stem>.json (excluding *.errors.json) must have a
#                        matching <stem>.npz in --preds-dir.

_PRED_KEYS = ("y_pred", "pred", "predictions")


def _model_stems(results_dir: Path) -> list[str]:
    return sorted(
        p.stem
        for p in results_dir.glob("*.json")
        if not p.name.endswith(".errors.json")
    )


def _load_labels(labels_path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(labels_path, allow_pickle=False) as data:
        if "y_true" not in data:
            raise SystemExit(
                f"ERROR: labels file {labels_path} has no 'y_true' array "
                f"(found keys: {sorted(data.files)})"
            )
        y_true = data["y_true"]
        clusters = data["clusters"] if "clusters" in data else None
        clip_ids = data["clip_ids"] if "clip_ids" in data else None
    return y_true, clusters, clip_ids


def _load_preds(preds_path: Path, n: int, label_clip_ids: np.ndarray | None) -> np.ndarray:
    with np.load(preds_path, allow_pickle=False) as data:
        key = next((k for k in _PRED_KEYS if k in data), None)
        if key is None:
            raise SystemExit(
                f"ERROR: {preds_path} has none of {_PRED_KEYS} "
                f"(found keys: {sorted(data.files)})"
            )
        y_pred = data[key]
        pred_clip_ids = data["clip_ids"] if "clip_ids" in data else None
    if pred_clip_ids is not None and label_clip_ids is not None:
        pos = {cid: i for i, cid in enumerate(pred_clip_ids)}
        try:
            order = np.array([pos[cid] for cid in label_clip_ids])
        except KeyError as exc:
            raise SystemExit(
                f"ERROR: {preds_path} is missing predictions for clip id {exc}"
            ) from exc
        y_pred = y_pred[order]
    if y_pred.shape[0] != n:
        raise SystemExit(
            f"ERROR: {preds_path} has {y_pred.shape[0]} predictions but the "
            f"labels file has {n} clips (and no clip_ids to align on)"
        )
    return y_pred


def _check_leaderboard_inputs(
    results_dir: Path, preds_dir: Path, labels_path: Path
) -> list[str]:
    """Return a list of human-readable problems; empty means ready to run."""
    missing: list[str] = []
    if not results_dir.is_dir():
        missing.append(f"results dir not found: {results_dir}")
        return missing
    stems = _model_stems(results_dir)
    if not stems:
        missing.append(f"no result JSONs (*.json) found in: {results_dir}")
    if not labels_path.is_file():
        missing.append(f"labels npz not found: {labels_path}")
    if not preds_dir.is_dir():
        missing.append(f"preds dir not found: {preds_dir}")
        missing.extend(f"missing per-clip predictions: {preds_dir / (s + '.npz')}" for s in stems)
    else:
        for s in stems:
            p = preds_dir / (s + ".npz")
            if not p.is_file():
                missing.append(f"missing per-clip predictions: {p}")
    return missing


def _cmd_leaderboard(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    preds_dir = Path(args.preds_dir)
    labels_path = Path(args.labels)

    missing = _check_leaderboard_inputs(results_dir, preds_dir, labels_path)
    if missing:
        lines = "\n".join(f"  - {m}" for m in missing)
        raise SystemExit(
            "Cannot build leaderboard yet; the following inputs are missing:\n"
            f"{lines}\n"
            "Per-clip predictions live in GCS as .npz — export them plus the "
            "ground-truth labels npz (keys: y_true, clusters[, clip_ids]) and "
            "re-run this exact command."
        )

    y_true, clusters, clip_ids = _load_labels(labels_path)
    n = len(y_true)
    if clusters is None:
        print(
            "WARNING: labels file has no 'clusters' array; falling back to an "
            "item-level bootstrap. CIs will be too narrow if clips share "
            "speakers/corpora.",
            file=sys.stderr,
        )

    stems = _model_stems(results_dir)
    metric = args.metric
    rows = []
    preds_by_stem: dict[str, np.ndarray] = {}
    for stem in stems:
        y_pred = _load_preds(preds_dir / (stem + ".npz"), n, clip_ids)
        preds_by_stem[stem] = y_pred
        with open(results_dir / (stem + ".json"), encoding="utf-8") as f:
            meta = json.load(f)
        row: dict = {"name": meta.get("name", stem), "file_stem": stem, "n": n}
        for m in ("accuracy", "macro_f1"):
            res = bootstrap_ci(
                y_true, y_pred, clusters, m,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
            )
            row[m] = res.to_dict()
        rows.append(row)
        print(
            f"  {row['name']}: {metric}={row[metric]['point']:.4f} "
            f"[{row[metric]['ci_low']:.4f}, {row[metric]['ci_high']:.4f}]"
        )

    rows.sort(key=lambda r: r[metric]["point"], reverse=True)
    adjacent = []
    for upper, lower in zip(rows, rows[1:]):
        test = paired_bootstrap_test(
            y_true,
            preds_by_stem[upper["file_stem"]],
            preds_by_stem[lower["file_stem"]],
            clusters,
            metric,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )
        adjacent.append(
            {"upper": upper["name"], "lower": lower["name"], **test.to_dict()}
        )

    # Display ranks: adjacent pairs not separable at (1 - alpha) share a rank.
    display_rank = 1
    rows[0]["rank"] = 1
    rows[0]["display_rank"] = 1
    for i, test in enumerate(adjacent):
        rows[i + 1]["rank"] = i + 2
        if not test["significant"]:
            rows[i + 1]["display_rank"] = rows[i]["display_rank"]
        else:
            display_rank = i + 2
            rows[i + 1]["display_rank"] = display_rank
    for row in rows:
        row["tied_with_prev"] = row["rank"] > 1 and (
            row["display_rank"] == rows[row["rank"] - 2]["display_rank"]
        )

    out = {
        "metric": metric,
        "alpha": args.alpha,
        "n_boot": args.n_boot,
        "n_items": n,
        "clustered": clusters is not None,
        "n_clusters": int(len(np.unique(clusters))) if clusters is not None else n,
        "labels_file": str(labels_path),
        "models": rows,
        "adjacent_tests": adjacent,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path} ({len(rows)} models, {len(adjacent)} adjacent tests)")


def _cmd_mdd(args: argparse.Namespace) -> None:
    results = []
    print(f"{'n_items':>8}  {'base_acc':>8}  {'min detectable delta':>20}")
    for n in args.n:
        for acc in args.base_acc:
            r = min_detectable_delta(
                n, acc,
                n_boot=args.n_boot, n_sim=args.n_sim,
                power=args.power, alpha=args.alpha, seed=args.seed,
            )
            results.append(r.to_dict())
            print(f"{n:>8}  {acc:>8.2f}  {r.delta:>20.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out}")


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> None:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(
        prog="python -m oruk_bench.stats",
        description="Statistics CLI for the Oruk speech-emotion benchmark.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    lb = sub.add_parser(
        "leaderboard",
        help="Emit leaderboard JSON with bootstrap CIs and adjacent-rank paired tests.",
    )
    lb.add_argument("--results-dir", required=True, help="Dir of per-model summary JSONs")
    lb.add_argument("--preds-dir", required=True, help="Dir of per-clip prediction .npz files")
    lb.add_argument("--labels", required=True, help="Ground-truth labels .npz (y_true, clusters)")
    lb.add_argument("--out", required=True, help="Output leaderboard JSON path")
    lb.add_argument("--metric", default="macro_f1", choices=["accuracy", "macro_f1"])
    lb.add_argument("--n-boot", type=int, default=2000)
    lb.add_argument("--alpha", type=float, default=0.05)
    lb.add_argument("--seed", type=int, default=0)
    lb.set_defaults(func=_cmd_leaderboard)

    mdd = sub.add_parser("mdd", help="Min-detectable-delta table via power simulation.")
    mdd.add_argument("--n", type=int, nargs="+", required=True)
    mdd.add_argument("--base-acc", type=float, nargs="+", required=True)
    mdd.add_argument("--n-boot", type=int, default=1000)
    mdd.add_argument("--n-sim", type=int, default=200)
    mdd.add_argument("--power", type=float, default=0.8)
    mdd.add_argument("--alpha", type=float, default=0.05)
    mdd.add_argument("--seed", type=int, default=0)
    mdd.add_argument("--out", default=None)
    mdd.set_defaults(func=_cmd_mdd)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
