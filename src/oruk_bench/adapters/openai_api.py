"""Closed-source arm 2: OpenAI audio models + Anthropic (transcript-only).

Same protocol and 5,000-clip stratified subsample (seed 0) as the Gemini arm.

- OpenAI: native audio input via Chat Completions (gpt-audio-1.5,
  gpt-audio-mini, gpt-4o-audio-preview), text-only output, same expert prompt,
  single-word answer parsed with the shared label normalizer.
- Anthropic: Claude has NO audio modality, so clips are transcribed once with
  gpt-4o-mini-transcribe (cached) and Claude classifies the TRANSCRIPT ONLY --
  a text-only run with no prosody, flagged as such in results.

Keys are read from OPENAI_API_KEY / ANTHROPIC_API_KEY env vars.
Requires the ``[api]`` extra (requests).
"""

import base64
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..core import LABELS, PROMPT, norm_label, score, wav_bytes

WORD_RE = re.compile(r"[a-z]+")

OPENAI_PROMPT = PROMPT + "\n\nRespond with ONLY the single label word, nothing else."
CLAUDE_PROMPT = """You are an expert emotion annotator. Below is the TRANSCRIPT of a
speech clip from a multilingual corpus (you cannot hear the audio, so judge from
the words alone; the transcript may be in any language and may be short or noisy).

Classify the speaker's most likely emotion as exactly one of:
anger, happiness, sadness, fear, disgust, surprise, neutral.

Rules:
- Judge word choice, content, exclamations and punctuation.
- If several emotions are plausible, pick the most likely one.
- Only pick neutral when nothing suggests another emotion.
- Respond with ONLY the single label word, nothing else.

Transcript:
"""


def _openai_key():
    return os.environ.get("OPENAI_API_KEY", "")


def _anthropic_key():
    return os.environ.get("ANTHROPIC_API_KEY", "")


def parse_label(text):
    for w in WORD_RE.findall(str(text).lower()):
        lab = norm_label(w)
        if lab is not None:
            return LABELS.index(lab)
    return None


def post_retry(url, headers, payload, max_retries=6, timeout=120):
    import requests

    last = "?"
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504, 529):
                return None, last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt * 2))
    return None, last


def openai_classify(model, audio):
    payload = {
        "model": model,
        "modalities": ["text"],
        "temperature": 0,
        "max_completion_tokens": 16,
        "messages": [
            {"role": "system",
             "content": ("You are a speech-emotion annotation system with audio input. "
                         "You listen to clips and label them. You always answer with "
                         "exactly one label word and never refuse.")},
            {"role": "user",
             "content": [
                 {"type": "text", "text": OPENAI_PROMPT},
                 {"type": "input_audio",
                  "input_audio": {"data": base64.b64encode(wav_bytes(audio)).decode(),
                                  "format": "wav"}},
             ]},
        ],
    }
    last = "?"
    for _ in range(3):  # extra loop: audio models stochastically refuse
        out, err = post_retry("https://api.openai.com/v1/chat/completions",
                              {"Authorization": f"Bearer {_openai_key()}"}, payload)
        if out is None:
            return None, err
        text = out["choices"][0]["message"].get("content") or ""
        pred = parse_label(text)
        if pred is not None:
            return pred, None
        last = f"unparsed: {text[:80]!r}"
    return None, last


def transcribe(audio):
    import requests

    files = {"file": ("clip.wav", wav_bytes(audio), "audio/wav")}
    data = {"model": "gpt-4o-mini-transcribe"}
    last = "?"
    for attempt in range(6):
        try:
            r = requests.post("https://api.openai.com/v1/audio/transcriptions",
                              headers={"Authorization": f"Bearer {_openai_key()}"},
                              files=files, data=data, timeout=120)
            if r.status_code == 200:
                return r.json().get("text", ""), None
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                return None, last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt * 2))
    return None, last


def claude_classify(model, transcript):
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": CLAUDE_PROMPT + (transcript or "(empty)")}],
    }
    out, err = post_retry("https://api.anthropic.com/v1/messages",
                          {"x-api-key": _anthropic_key(),
                           "anthropic-version": "2023-06-01"}, payload)
    if out is None:
        return None, err
    text = "".join(b.get("text", "") for b in out.get("content", []))
    pred = parse_label(text)
    return (pred, None) if pred is not None else (None, f"unparsed: {text[:80]!r}")


def run_threaded(name, fn, items, out_dir, workers, extra_meta, y, sl, ss, sub_idx):
    """fn(i) -> (pred_or_None, err). Saves resumable npz + scored json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    n = len(items)
    preds = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    if npz_path.exists():
        blob = np.load(npz_path)
        if len(blob["preds"]) == n:
            preds, done = blob["preds"].copy(), blob["done"].copy()
            print(f"[resume] {name}: {done.sum()}/{n}", flush=True)
    todo = [i for i in range(n) if not done[i]]
    errors, t0, completed = [], time.time(), 0
    lock = threading.Lock()

    def work(i):
        return i, *fn(i)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, i) for i in todo]
        for fut in as_completed(futs):
            i, pred, err = fut.result()
            with lock:
                completed += 1
                if pred is not None:
                    preds[i], done[i] = pred, True
                else:
                    preds[i], done[i] = LABELS.index("neutral"), True
                    errors.append({"i": int(i), "err": err})
                if completed % 200 == 0:
                    rate = completed / (time.time() - t0)
                    print(f"  {name}: {completed}/{len(todo)} ({rate:.1f}/s, "
                          f"eta {(len(todo) - completed) / max(rate, 1e-9) / 60:.0f} min, "
                          f"errors={len(errors)})", flush=True)
                    np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)

    np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)
    result = {
        "name": name,
        "subsample_n": n,
        "api_errors_defaulted_to_neutral": len(errors),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        **extra_meta,
        **score(y, preds, sl, ss, list(LABELS)),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if errors:
        (out_dir / f"{name}.errors.json").write_text(
            json.dumps(errors[:100], indent=2), encoding="utf-8")
    print(f"API_MODEL_DONE {name} acc={result['accuracy']:.4f} "
          f"macro_f1={result['macro_f1']:.4f} errors={len(errors)}", flush=True)
    return result


def run_openai_model(model, sub_idx, clip_fn, y, sl, ss, out_dir, workers=8):
    """Run one OpenAI audio model over the subsample."""
    if not _openai_key():
        raise SystemExit("set OPENAI_API_KEY to use the OpenAI arm")
    n = len(sub_idx)
    return run_threaded(
        f"openai_{model}",
        lambda i, m=model: openai_classify(m, clip_fn(int(sub_idx[i]))),
        range(n), out_dir, workers,
        {"model_id": model, "adapter": "openai_audio", "modality": "audio"},
        y, sl, ss, sub_idx)


def run_anthropic_model(model, sub_idx, clip_fn, y, sl, ss, out_dir, workers=8):
    """Run one Anthropic model (transcript-only) over the subsample. Clips are
    transcribed once with gpt-4o-mini-transcribe and cached in out_dir."""
    if not _anthropic_key():
        raise SystemExit("set ANTHROPIC_API_KEY to use the Anthropic arm")
    if not _openai_key():
        raise SystemExit("set OPENAI_API_KEY too (transcription pass)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(sub_idx)
    tx_path = out_dir / f"transcripts_{n}.json"
    if tx_path.exists():
        transcripts = json.loads(tx_path.read_text(encoding="utf-8"))
    else:
        transcripts = [None] * n
    missing = [i for i in range(n) if not transcripts[i]]
    if missing:
        print(f"=== transcribing {len(missing)} clips (gpt-4o-mini-transcribe) ===", flush=True)
        lock = threading.Lock()
        t0, cnt = time.time(), 0

        def tx_work(i):
            text, err = transcribe(clip_fn(int(sub_idx[i])))
            return i, text if text is not None else ""

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(tx_work, i) for i in missing]
            for fut in as_completed(futs):
                i, text = fut.result()
                with lock:
                    transcripts[i] = text
                    cnt += 1
                    if cnt % 500 == 0:
                        rate = cnt / (time.time() - t0)
                        print(f"  tx {cnt}/{len(missing)} ({rate:.1f}/s)", flush=True)
                        tx_path.write_text(json.dumps(transcripts), encoding="utf-8")
        tx_path.write_text(json.dumps(transcripts), encoding="utf-8")
    print("transcripts ready", flush=True)

    return run_threaded(
        f"anthropic_{model}",
        lambda i, m=model: claude_classify(m, transcripts[i]),
        range(n), out_dir, workers,
        {"model_id": model, "adapter": "anthropic_text",
         "modality": "transcript-only (Claude has no audio input)"},
        y, sl, ss, sub_idx)
