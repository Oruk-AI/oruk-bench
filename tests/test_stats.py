"""Tests for oruk_bench.stats on synthetic data with known properties."""

import json

import numpy as np
import pytest

from oruk_bench import stats

N_CLASSES = 7
LABELS = np.arange(N_CLASSES)


def _preds_from_correct(y_true: np.ndarray, correct: np.ndarray, rng) -> np.ndarray:
    """Predictions that match y_true where `correct`, else a random wrong class."""
    wrong = (y_true + rng.integers(1, N_CLASSES, size=len(y_true))) % N_CLASSES
    return np.where(correct, y_true, wrong)


def _clustered_dataset(rng, n_clusters=30, cluster_size=20, base_acc=0.7, effect=0.15):
    """Clustered eval where the *expected* accuracy is exactly `base_acc`.

    Half the clusters run `effect` hot, half run `effect` cold, so the truth
    is known analytically while items within a cluster stay correlated.
    """
    accs = np.repeat([base_acc + effect, base_acc - effect], n_clusters // 2)
    clusters = np.repeat(np.arange(n_clusters), cluster_size)
    y_true = rng.integers(0, N_CLASSES, size=n_clusters * cluster_size)
    correct = rng.random(len(y_true)) < np.repeat(accs, cluster_size)
    y_pred = _preds_from_correct(y_true, correct, rng)
    return y_true, y_pred, clusters


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_point_estimate_matches_metric(self):
        rng = np.random.default_rng(0)
        y_true, y_pred, clusters = _clustered_dataset(rng)
        res = stats.bootstrap_ci(y_true, y_pred, clusters, "accuracy", n_boot=200, seed=1)
        assert res.point == pytest.approx(np.mean(y_true == y_pred))
        assert res.ci_low <= res.point <= res.ci_high
        assert res.n_clusters == 30

    def test_coverage_of_true_accuracy(self):
        """95% cluster-bootstrap CI should cover the true accuracy ~95% of the time."""
        rng = np.random.default_rng(42)
        true_acc, n_sims, covered = 0.7, 60, 0
        for _ in range(n_sims):
            y_true, y_pred, clusters = _clustered_dataset(rng, base_acc=true_acc)
            res = stats.bootstrap_ci(
                y_true, y_pred, clusters, "accuracy", n_boot=400, seed=rng.integers(1 << 30)
            )
            if res.ci_low <= true_acc <= res.ci_high:
                covered += 1
        # allow binomial noise around the nominal 95%
        assert covered / n_sims >= 0.85

    def test_cluster_ci_wider_than_iid_on_clustered_data(self):
        """Ignoring clusters must give overconfident (narrower) intervals."""
        rng = np.random.default_rng(7)
        y_true, y_pred, clusters = _clustered_dataset(rng, n_clusters=40, cluster_size=50)
        clustered = stats.bootstrap_ci(y_true, y_pred, clusters, "accuracy", n_boot=600, seed=3)
        iid = stats.bootstrap_ci(y_true, y_pred, None, "accuracy", n_boot=600, seed=3)
        assert (clustered.ci_high - clustered.ci_low) > (iid.ci_high - iid.ci_low)

    def test_macro_f1_metric_and_custom_callable(self):
        rng = np.random.default_rng(2)
        y_true, y_pred, clusters = _clustered_dataset(rng)
        res = stats.bootstrap_ci(y_true, y_pred, clusters, "macro_f1", n_boot=100, seed=4)
        assert 0.0 < res.point < 1.0
        custom = stats.bootstrap_ci(
            y_true, y_pred, clusters,
            metric_fn=lambda t, p: float(np.mean(t == p)),
            n_boot=100, seed=4,
        )
        assert custom.point == pytest.approx(np.mean(y_true == y_pred))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            stats.bootstrap_ci([0, 1], [0, 1, 2])
        with pytest.raises(ValueError):
            stats.bootstrap_ci([0, 1], [0, 1], clusters=[0])


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------

def _paired_models(rng, n=5000, cluster_size=50, acc_a=0.74, gap=0.03):
    """Two correlated models on identical items with a planted accuracy gap."""
    y_true = rng.integers(0, N_CLASSES, size=n)
    clusters = np.arange(n) // cluster_size
    correct_a = rng.random(n) < acc_a
    # B mostly agrees with A (correlated errors), but has a net +gap accuracy:
    # keep A's correct items with p=0.97, then fix enough of A's errors.
    p_keep = 0.97
    acc_b = acc_a + gap
    p_fix = (acc_b - acc_a * p_keep) / (1 - acc_a)
    correct_b = np.where(correct_a, rng.random(n) < p_keep, rng.random(n) < p_fix)
    pred_a = _preds_from_correct(y_true, correct_a, rng)
    pred_b = _preds_from_correct(y_true, correct_b, rng)
    return y_true, pred_a, pred_b, clusters


class TestPairedBootstrapTest:
    def test_detects_planted_3pct_gap(self):
        rng = np.random.default_rng(11)
        y_true, pred_a, pred_b, clusters = _paired_models(rng, gap=0.03)
        res = stats.paired_bootstrap_test(
            y_true, pred_b, pred_a, clusters, "accuracy", n_boot=1000, seed=5
        )
        assert res.delta == pytest.approx(0.03, abs=0.015)
        assert res.p_value < 0.05
        assert res.significant
        assert res.ci_low > 0  # CI excludes zero
        assert res.metric_a > res.metric_b

    def test_no_gap_is_not_significant(self):
        rng = np.random.default_rng(13)
        y_true, pred_a, pred_b, clusters = _paired_models(rng, gap=0.0)
        res = stats.paired_bootstrap_test(
            y_true, pred_a, pred_b, clusters, "accuracy", n_boot=1000, seed=6
        )
        assert res.p_value > 0.05
        assert res.ci_low < 0 < res.ci_high

    def test_gap_detected_with_macro_f1(self):
        rng = np.random.default_rng(17)
        y_true, pred_a, pred_b, clusters = _paired_models(rng, gap=0.05)
        res = stats.paired_bootstrap_test(
            y_true, pred_b, pred_a, clusters, "macro_f1", n_boot=500, seed=7
        )
        assert res.delta > 0
        assert res.p_value < 0.05

    def test_p_value_never_zero(self):
        rng = np.random.default_rng(19)
        y_true, pred_a, pred_b, clusters = _paired_models(rng, gap=0.15)
        res = stats.paired_bootstrap_test(
            y_true, pred_b, pred_a, clusters, "accuracy", n_boot=200, seed=8
        )
        assert res.p_value >= 2.0 / 201


# ---------------------------------------------------------------------------
# soft_label_metrics
# ---------------------------------------------------------------------------

class TestSoftLabelMetrics:
    def test_perfect_soft_prediction(self):
        targets = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
        out = stats.soft_label_metrics(targets.copy(), targets)
        assert out["js_divergence"] == pytest.approx(0.0, abs=1e-9)
        entropy = float(np.mean(-np.sum(targets * np.log(targets), axis=1)))
        assert out["cross_entropy"] == pytest.approx(entropy)
        # argmax classes hold 0.5 and 0.8 of rater mass
        assert out["soft_accuracy"] == pytest.approx(0.65)

    def test_hard_int_preds_equal_onehot(self):
        targets = np.array([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
        hard = np.array([0, 1, 0])
        onehot = np.eye(2)[hard]
        a = stats.soft_label_metrics(hard, targets)
        b = stats.soft_label_metrics(onehot, targets)
        for k in ("cross_entropy", "js_divergence", "soft_accuracy"):
            assert a[k] == pytest.approx(b[k])
        assert a["soft_accuracy"] == pytest.approx((0.6 + 0.8 + 0.5) / 3)

    def test_string_hard_preds_with_labels(self):
        targets = np.array([[0.9, 0.1], [0.3, 0.7]])
        out = stats.soft_label_metrics(
            np.array(["anger", "neutral"]), targets, labels=["anger", "neutral"]
        )
        assert out["soft_accuracy"] == pytest.approx((0.9 + 0.7) / 2)

    def test_string_preds_without_labels_raise(self):
        with pytest.raises(ValueError, match="labels"):
            stats.soft_label_metrics(np.array(["anger"]), np.array([[1.0, 0.0]]))

    def test_onehot_pred_has_finite_cross_entropy(self):
        targets = np.array([[0.5, 0.5]])
        out = stats.soft_label_metrics(np.array([[1.0, 0.0]]), targets)
        assert np.isfinite(out["cross_entropy"])
        assert 0.0 <= out["js_divergence"] <= 1.0

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            stats.soft_label_metrics(np.zeros((2, 3)), np.zeros((2, 4)) + 0.25)
        with pytest.raises(ValueError):
            stats.soft_label_metrics(np.array([5]), np.array([[0.5, 0.5]]))


# ---------------------------------------------------------------------------
# prompt_ensemble_summary
# ---------------------------------------------------------------------------

class TestPromptEnsembleSummary:
    def test_known_accuracies(self):
        y_true = np.zeros(8, dtype=int)
        prompts = [
            np.zeros(8, dtype=int),                      # acc 1.0
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),          # acc 0.5
            np.array([0, 0, 0, 0, 0, 0, 1, 1]),          # acc 0.75
        ]
        out = stats.prompt_ensemble_summary(prompts, y_true, "accuracy")
        assert out["per_prompt"] == [1.0, 0.5, 0.75]
        assert out["mean"] == pytest.approx(0.75)
        assert out["std"] == pytest.approx(np.std([1.0, 0.5, 0.75], ddof=1))
        assert out["min"] == 0.5
        assert out["max"] == 1.0
        assert out["spread"] == pytest.approx(0.5)
        assert out["k"] == 3

    def test_single_prompt_has_zero_std(self):
        y_true = np.array([0, 1, 2])
        out = stats.prompt_ensemble_summary([y_true.copy()], y_true)
        assert out["std"] == 0.0
        assert out["spread"] == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stats.prompt_ensemble_summary([], np.array([0, 1]))


# ---------------------------------------------------------------------------
# min_detectable_delta
# ---------------------------------------------------------------------------

class TestMinDetectableDelta:
    def test_shrinks_with_sample_size(self):
        small = stats.min_detectable_delta(2000, 0.7, n_boot=300, n_sim=60, seed=1)
        large = stats.min_detectable_delta(20000, 0.7, n_boot=300, n_sim=60, seed=1)
        assert large.delta < small.delta

    def test_magnitude_matches_analytic_approximation(self):
        # Independent-errors paired MDD ~ (z_{1-a/2} + z_power) * sqrt(2p(1-p)/n)
        res = stats.min_detectable_delta(5000, 0.7, n_boot=400, n_sim=100, seed=2)
        approx = 2.8 * np.sqrt(2 * 0.7 * 0.3 / 5000)
        assert 0.6 * approx < res.delta < 1.6 * approx

    def test_invalid_base_acc_raises(self):
        with pytest.raises(ValueError):
            stats.min_detectable_delta(1000, 1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_result_json(path, name, acc):
    path.write_text(
        json.dumps({"name": name, "n": 100, "accuracy": acc, "macro_f1": acc}),
        encoding="utf-8",
    )


class TestLeaderboardCLI:
    def test_missing_inputs_listed_exactly(self, tmp_path, capsys):
        results = tmp_path / "results"
        results.mkdir()
        _write_result_json(results / "model-a.json", "model-a", 0.7)
        _write_result_json(results / "model-b.json", "model-b", 0.6)
        (results / "model-b.errors.json").write_text("{}", encoding="utf-8")
        preds = tmp_path / "preds"
        preds.mkdir()
        np.savez(preds / "model-a.npz", y_pred=np.zeros(3, dtype=int))
        labels = tmp_path / "labels.npz"  # intentionally absent

        with pytest.raises(SystemExit) as exc:
            stats.main([
                "leaderboard",
                "--results-dir", str(results),
                "--preds-dir", str(preds),
                "--labels", str(labels),
                "--out", str(tmp_path / "lb.json"),
            ])
        msg = str(exc.value)
        assert str(labels) in msg
        assert str(preds / "model-b.npz") in msg
        assert "model-a.npz" not in msg          # present -> not reported
        assert "model-b.errors" not in msg       # errors sidecars ignored

    def test_end_to_end_with_synthetic_preds(self, tmp_path):
        rng = np.random.default_rng(23)
        n, cluster_size = 2000, 40
        y_true = rng.integers(0, N_CLASSES, size=n)
        clusters = np.arange(n) // cluster_size
        clip_ids = np.array([f"clip_{i:05d}" for i in range(n)])

        correct_good = rng.random(n) < 0.85
        correct_bad = rng.random(n) < 0.55
        pred_good = _preds_from_correct(y_true, correct_good, rng)
        pred_bad = _preds_from_correct(y_true, correct_bad, rng)

        results = tmp_path / "results"
        preds = tmp_path / "preds"
        results.mkdir()
        preds.mkdir()
        _write_result_json(results / "good-model.json", "good-model", 0.85)
        _write_result_json(results / "bad-model.json", "bad-model", 0.55)
        np.savez(preds / "good-model.npz", y_pred=pred_good, clip_ids=clip_ids)
        # store the bad model's rows shuffled to exercise clip_id alignment
        perm = rng.permutation(n)
        np.savez(preds / "bad-model.npz", y_pred=pred_bad[perm], clip_ids=clip_ids[perm])
        labels = tmp_path / "labels.npz"
        np.savez(labels, y_true=y_true, clusters=clusters, clip_ids=clip_ids)
        out = tmp_path / "leaderboard_ci.json"

        stats.main([
            "leaderboard",
            "--results-dir", str(results),
            "--preds-dir", str(preds),
            "--labels", str(labels),
            "--out", str(out),
            "--metric", "accuracy",
            "--n-boot", "400",
        ])

        lb = json.loads(out.read_text(encoding="utf-8"))
        assert [m["name"] for m in lb["models"]] == ["good-model", "bad-model"]
        top = lb["models"][0]
        assert top["accuracy"]["ci_low"] <= top["accuracy"]["point"] <= top["accuracy"]["ci_high"]
        assert top["accuracy"]["point"] == pytest.approx(np.mean(y_true == pred_good))
        assert lb["clustered"] is True
        assert lb["n_clusters"] == n // cluster_size
        [test] = lb["adjacent_tests"]
        assert test["upper"] == "good-model"
        assert test["significant"] is True
        assert lb["models"][1]["tied_with_prev"] is False

    def test_tied_models_share_display_rank(self, tmp_path):
        rng = np.random.default_rng(29)
        n = 400
        y_true = rng.integers(0, N_CLASSES, size=n)
        correct = rng.random(n) < 0.7
        pred_a = _preds_from_correct(y_true, correct, rng)
        # nearly identical model: flip a handful of items
        pred_b = pred_a.copy()
        flip = rng.choice(n, size=4, replace=False)
        pred_b[flip] = (pred_b[flip] + 1) % N_CLASSES

        results = tmp_path / "results"
        preds = tmp_path / "preds"
        results.mkdir()
        preds.mkdir()
        _write_result_json(results / "twin-a.json", "twin-a", 0.7)
        _write_result_json(results / "twin-b.json", "twin-b", 0.7)
        np.savez(preds / "twin-a.npz", y_pred=pred_a)
        np.savez(preds / "twin-b.npz", y_pred=pred_b)
        labels = tmp_path / "labels.npz"
        np.savez(labels, y_true=y_true, clusters=np.arange(n) // 20)
        out = tmp_path / "lb.json"

        stats.main([
            "leaderboard",
            "--results-dir", str(results),
            "--preds-dir", str(preds),
            "--labels", str(labels),
            "--out", str(out),
            "--metric", "accuracy",
            "--n-boot", "400",
        ])
        lb = json.loads(out.read_text(encoding="utf-8"))
        assert lb["adjacent_tests"][0]["significant"] is False
        assert lb["models"][0]["display_rank"] == lb["models"][1]["display_rank"]
        assert lb["models"][1]["tied_with_prev"] is True

    def test_mdd_subcommand(self, tmp_path, capsys):
        out = tmp_path / "mdd.json"
        stats.main([
            "mdd", "--n", "1000", "--base-acc", "0.7",
            "--n-boot", "200", "--n-sim", "40", "--out", str(out),
        ])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data[0]["n_items"] == 1000
        assert 0.0 < data[0]["delta"] < 0.2
