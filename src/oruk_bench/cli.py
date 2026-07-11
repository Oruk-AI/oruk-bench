"""oruk-bench command-line interface.

Subcommands:
  eval        run a registered model over user-provided eval shards
  score       score a predictions file against gold labels
  list-models list every registered model across all arms
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .core import LABELS, clip_audio, load_eval, norm_label, score, stratified_subsample


def _load_label_array(path):
    """Load a label array from .json (list of label strings/ints), .npz
    (first of keys preds/labels), or plain text (one label per line).
    String labels are normalized through the shared alias map."""
    path = Path(path)
    if path.suffix == ".npz":
        blob = np.load(path)
        for key in ("preds", "labels"):
            if key in blob:
                return np.asarray(blob[key], dtype=np.int64)
        raise SystemExit(f"{path}: expected a 'preds' or 'labels' array")
    # utf-8-sig tolerates the BOM that Windows tools often prepend
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        raw = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()
               if ln.strip()]
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, (int, np.integer)):
            out.append(int(item))
            continue
        lab = norm_label(item)
        if lab is None:
            raise SystemExit(f"{path}: unrecognized label {item!r} at position {i}")
        out.append(LABELS.index(lab))
    return np.asarray(out, dtype=np.int64)


def cmd_score(args):
    preds = _load_label_array(args.preds)
    y = _load_label_array(args.labels)
    if len(preds) != len(y):
        raise SystemExit(f"length mismatch: {len(preds)} preds vs {len(y)} labels")
    langs = np.array(["unknown"] * len(y), dtype=object)
    sources = np.array(["unknown"] * len(y), dtype=object)
    result = score(y, preds, langs, sources, list(LABELS))
    result.pop("per_language", None)
    result.pop("per_source", None)
    try:
        # Optional statistical extras (bootstrap CIs); skipped if stats is absent.
        from sklearn.metrics import f1_score

        from .stats import bootstrap_ci

        def protocol_macro_f1(yt, yp):
            # same metric as the headline score: macro over all 7 classes
            return f1_score(yt, yp, labels=range(len(LABELS)), average="macro",
                            zero_division=0)

        ci = bootstrap_ci(y, preds, metric_fn=protocol_macro_f1, seed=0)
        result["macro_f1_ci95"] = ci.to_dict() if hasattr(ci, "to_dict") else ci
    except Exception:
        pass
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def cmd_list_models(args):
    from .adapters.audio_llm import REGISTRY as AUDIO_LLM_REGISTRY
    from .adapters.open_models import MODELS as OPEN_MODELS

    print("open-source SER models (full eval set; extra: [open]):")
    for cfg in OPEN_MODELS:
        print(f"  {cfg['name']:32s} {cfg['adapter']:14s} {cfg['model_id']}")
    print()
    print("open-weight audio-LLMs (5k stratified subsample; extra: [audiollm]):")
    for name, (model_id, _) in sorted(AUDIO_LLM_REGISTRY.items()):
        print(f"  {name:32s} {'audio_llm':14s} {model_id}")
    print()
    print("closed API models (5k stratified subsample; extra: [api]):")
    print("  gemini-*      via Vertex AI (GOOGLE_CLOUD_PROJECT + gcloud auth)")
    print("  gpt-audio-*   via OpenAI (OPENAI_API_KEY)")
    print("  claude-*      via Anthropic, transcript-only (ANTHROPIC_API_KEY + OPENAI_API_KEY)")


def cmd_eval(args):
    from .adapters.audio_llm import REGISTRY as AUDIO_LLM_REGISTRY
    from .adapters.open_models import get_model_cfg

    model = args.model
    open_cfg = get_model_cfg(model)

    if open_cfg is not None:
        from .adapters.open_models import run_eval

        run_eval(open_cfg, args.data_dir, device=args.device, out_dir=args.out_dir,
                 max_examples=args.max_examples)
        return

    # every other arm runs on the fixed stratified subsample
    tables, index, labels, langs, sources = load_eval(args.data_dir)
    sub_idx = stratified_subsample(labels, args.subsample)
    if args.max_examples:
        sub_idx = sub_idx[np.linspace(0, len(sub_idx) - 1, args.max_examples, dtype=int)]
    y, sl, ss = labels[sub_idx], langs[sub_idx], sources[sub_idx]

    def clip_fn(i):
        return clip_audio(tables, index, i)

    if model in AUDIO_LLM_REGISTRY:
        from .adapters.audio_llm import run_eval as run_audio_llm

        run_audio_llm(model, sub_idx, clip_fn, y, sl, ss, args.out_dir, device=args.device)
    elif model.startswith("gemini-"):
        from .adapters.gemini import get_project, run_model

        project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT") or get_project()
        run_model(model, project, sub_idx, clip_fn, y, sl, ss, args.out_dir,
                  workers=args.workers)
    elif model.startswith(("gpt-", "openai/")):
        from .adapters.openai_api import run_openai_model

        run_openai_model(model.removeprefix("openai/"), sub_idx, clip_fn, y, sl, ss,
                         args.out_dir, workers=args.workers)
    elif model.startswith(("claude-", "anthropic/")):
        from .adapters.openai_api import run_anthropic_model

        run_anthropic_model(model.removeprefix("anthropic/"), sub_idx, clip_fn, y, sl, ss,
                            args.out_dir, workers=args.workers)
    else:
        raise SystemExit(
            f"unknown model {model!r}; run `oruk-bench list-models` to see what's registered")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oruk-bench",
                                 description="Oruk AI speech-emotion benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="run a registered model over eval shards")
    ev.add_argument("--model", required=True, help="model name (see list-models)")
    ev.add_argument("--data-dir", required=True,
                    help="directory of parquet shards (audio_flac/label/language/source_id)")
    ev.add_argument("--device", default="cuda:0")
    ev.add_argument("--out-dir", default="bench_results")
    ev.add_argument("--max-examples", type=int, default=0, help="debug: cap clips")
    ev.add_argument("--subsample", type=int, default=5000,
                    help="stratified subsample size for API/audio-LLM arms")
    ev.add_argument("--workers", type=int, default=8, help="API request concurrency")
    ev.add_argument("--project", default="", help="Google Cloud project (Gemini arm)")
    ev.set_defaults(fn=cmd_eval)

    sc = sub.add_parser("score", help="score predictions against gold labels")
    sc.add_argument("--preds", required=True,
                    help=".json list of labels, .npz with 'preds', or one label per line")
    sc.add_argument("--labels", required=True,
                    help=".json list of labels, .npz with 'labels', or one label per line")
    sc.add_argument("--out", default="", help="also write the score JSON here")
    sc.set_defaults(fn=cmd_score)

    lm = sub.add_parser("list-models", help="list registered models")
    lm.set_defaults(fn=cmd_list_models)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
