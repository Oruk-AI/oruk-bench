"""CLI tests: score path and model listing only (no model downloads)."""

import json

import numpy as np
import pytest

from oruk_bench.cli import main


def test_score_json_labels(tmp_path, capsys):
    preds = ["anger", "happy", "sad", "neutral"]
    labels = ["anger", "happiness", "sadness", "surprise"]
    p = tmp_path / "preds.json"
    g = tmp_path / "labels.json"
    p.write_text(json.dumps(preds), encoding="utf-8")
    g.write_text(json.dumps(labels), encoding="utf-8")
    out_path = tmp_path / "score.json"

    main(["score", "--preds", str(p), "--labels", str(g), "--out", str(out_path)])

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["n"] == 4
    assert result["accuracy"] == 0.75  # aliases normalized; surprise missed
    printed = json.loads(capsys.readouterr().out)
    assert printed["accuracy"] == result["accuracy"]


def test_score_npz(tmp_path, capsys):
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    p = tmp_path / "preds.npz"
    g = tmp_path / "labels.npz"
    np.savez(p, preds=y)
    np.savez(g, labels=y)

    main(["score", "--preds", str(p), "--labels", str(g)])

    result = json.loads(capsys.readouterr().out)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_score_retains_headline_when_optional_ci_fails(tmp_path, capsys, monkeypatch):
    from oruk_bench import stats

    def unavailable_ci(*args, **kwargs):
        raise RuntimeError("synthetic-private-diagnostic-must-not-be-logged")

    monkeypatch.setattr(stats, "bootstrap_ci", unavailable_ci)
    p = tmp_path / "preds.json"
    g = tmp_path / "labels.json"
    p.write_text(json.dumps(["anger", "neutral"]), encoding="utf-8")
    g.write_text(json.dumps(["anger", "surprise"]), encoding="utf-8")

    main(["score", "--preds", str(p), "--labels", str(g)])

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert "macro_f1_ci95" not in result
    assert "synthetic-private-diagnostic" not in captured.out + captured.err


def test_score_length_mismatch(tmp_path):
    p = tmp_path / "preds.json"
    g = tmp_path / "labels.json"
    p.write_text(json.dumps(["anger", "anger"]), encoding="utf-8")
    g.write_text(json.dumps(["anger"]), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["score", "--preds", str(p), "--labels", str(g)])


def test_score_bad_label(tmp_path):
    p = tmp_path / "preds.json"
    g = tmp_path / "labels.json"
    p.write_text(json.dumps(["blissful"]), encoding="utf-8")
    g.write_text(json.dumps(["anger"]), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["score", "--preds", str(p), "--labels", str(g)])


def test_list_models(capsys):
    main(["list-models"])
    out = capsys.readouterr().out
    assert "emotion2vec-plus-large" in out
    assert "voxtral-mini-3b" in out
    assert "gemini-" in out


def test_eval_unknown_model(tmp_path):
    # unknown model exits with a helpful error after data loads
    from make_fixture import make_fixture

    make_fixture(tmp_path, n_clips=8, n_shards=1)
    with pytest.raises(SystemExit):
        main(["eval", "--model", "not-a-model", "--data-dir", str(tmp_path)])
