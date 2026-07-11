"""Protocol tests on the synthetic fixture (no model downloads, no real data)."""

import numpy as np
import pytest
from make_fixture import make_fixture

from oruk_bench.core import (
    LABELS,
    MAX_SECONDS,
    TARGET_SR,
    clip_audio,
    load_eval,
    norm_label,
    score,
    stratified_subsample,
)

N_CLIPS = 24


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    gold = make_fixture(d, n_clips=N_CLIPS, n_shards=2, seed=0)
    return d, gold


def test_load_eval(fixture_dir):
    d, gold = fixture_dir
    tables, index, labels, langs, sources = load_eval(d)
    assert len(index) == N_CLIPS
    assert labels.tolist() == gold
    assert set(langs) <= {"en", "de", "th", "unknown"}
    assert set(sources) == {"synthetic-0", "synthetic-1"}


def test_load_eval_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval(tmp_path / "nope")


def test_clip_audio(fixture_dir):
    d, _ = fixture_dir
    tables, index, *_ = load_eval(d)
    for i in range(N_CLIPS):
        y = clip_audio(tables, index, i)
        assert y.dtype == np.float32
        assert y.ndim == 1
        assert 0 < len(y) <= int(MAX_SECONDS * TARGET_SR)


def test_norm_label_aliases():
    assert norm_label("Angry") == "anger"
    assert norm_label("hap") == "happiness"
    assert norm_label("joy") == "happiness"
    assert norm_label("<|SURPRISED|>") == "surprise"
    assert norm_label("suprised") == "surprise"  # known upstream typo
    assert norm_label("calm") == "neutral"
    assert norm_label("emotion2vec/angry") == "anger"
    assert norm_label("bogus") is None


def test_score_perfect():
    y = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    langs = np.array(["en"] * len(y), dtype=object)
    sources = np.array(["synthetic"] * len(y), dtype=object)
    out = score(y, y.copy(), langs, sources, list(LABELS))
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0
    assert out["n"] == len(y)
    assert set(out["per_class"]) == set(LABELS)
    assert "subset_macro_f1" not in out  # full label coverage -> no subset block


def test_score_supported_subset():
    y = np.array([0, 1, 2, 6, 0, 1, 2, 6])
    preds = np.array([0, 1, 2, 6, 0, 1, 2, 0])
    langs = np.array(["en"] * len(y), dtype=object)
    sources = np.array(["synthetic"] * len(y), dtype=object)
    supported = ["anger", "happiness", "sadness", "neutral"]
    out = score(y, preds, langs, sources, supported)
    assert out["supported_labels"] == supported
    assert out["subset_n"] == len(y)  # all gold labels fall in the supported set
    assert 0.0 < out["subset_macro_f1"] <= 1.0
    assert out["subset_accuracy"] == 7 / 8


def test_score_on_fixture(fixture_dir):
    d, _ = fixture_dir
    tables, index, labels, langs, sources = load_eval(d)
    rng = np.random.default_rng(1)
    preds = rng.integers(0, len(LABELS), size=len(labels))
    out = score(labels, preds, langs, sources, list(LABELS))
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert sum(v["support"] for v in out["per_class"].values()) == len(labels)


def test_stratified_subsample_deterministic():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, len(LABELS), size=1000)
    a = stratified_subsample(labels, 100, seed=0)
    b = stratified_subsample(labels, 100, seed=0)
    assert np.array_equal(a, b)
    assert len(a) == len(np.unique(a))  # no replacement
    # roughly proportional per class
    counts = np.bincount(labels[a], minlength=len(LABELS))
    assert counts.min() >= 1
    assert abs(len(a) - 100) <= len(LABELS)
