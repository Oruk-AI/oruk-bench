"""Closed-source arm: Gemini models via Vertex AI.

Same protocol as the open arm (same eval shards, 16 s truncation, same 7-label
space and scoring code), but run on a fixed stratified subsample (default 5,000
clips, seed 0) to keep API cost/time sane.

Prompting gives the model its best chance: expert-annotator framing, short
label definitions, instruction to use both prosody and content in any language,
JSON-schema-constrained single-label output, thinking budget where supported.

Auth: GCE metadata token when on a VM, else ``gcloud auth print-access-token``.
Set the Google Cloud project via ``GOOGLE_CLOUD_PROJECT`` or ``--project``.

Uses only the Python stdlib for HTTP (no extra install needed).
"""

import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..core import LABELS, PROMPT, norm_label, score, wav_bytes

# API models cannot emit a class outside the schema; retriable failures default
# to neutral and are counted in `api_errors_defaulted_to_neutral`.
SCHEMA = {"type": "OBJECT",
          "properties": {"emotion": {"type": "STRING", "enum": list(LABELS)}},
          "required": ["emotion"]}

# thinking budgets: lite thinks off; flash/pro get a modest budget
THINKING_BUDGETS = {"gemini-2.5-flash-lite": 0, "gemini-2.5-flash": 512,
                    "gemini-2.5-pro": 512}


def get_project():
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise SystemExit("set GOOGLE_CLOUD_PROJECT (or pass --project) to use the Gemini arm")
    return project


class TokenSource:
    def __init__(self):
        self._token = None
        self._expiry = 0.0
        self._lock = threading.Lock()

    def _fetch(self):
        try:
            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/"
                "service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                d = json.load(r)
            return d["access_token"], time.time() + d["expires_in"]
        except Exception:
            tok = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, shell=sys.platform == "win32",
            ).stdout.strip()
            return tok, time.time() + 1800

    def get(self):
        with self._lock:
            if self._token is None or time.time() > self._expiry - 300:
                self._token, self._expiry = self._fetch()
            return self._token


def classify(model, audio, tokens, project, thinking_budget, max_retries=6):
    gen_config = {
        "temperature": 0,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
        "responseSchema": SCHEMA,
    }
    # gemini-3 uses thinkingLevel and cannot fully disable thinking;
    # gemini-2.5 takes an explicit token budget.
    if model.startswith("gemini-3"):
        gen_config["thinkingConfig"] = {"thinkingLevel": "LOW"}
    else:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": "audio/wav",
                                "data": base64.b64encode(wav_bytes(audio)).decode()}},
                {"text": PROMPT},
            ],
        }],
        "generationConfig": gen_config,
    }
    url = (f"https://aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/global/publishers/google/models/{model}:generateContent")
    data = json.dumps(body).encode()
    last_err = "?"
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {tokens.get()}",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            text = out["candidates"][0]["content"]["parts"][0]["text"]
            lab = norm_label(json.loads(text)["emotion"])
            if lab is not None:
                return LABELS.index(lab), None
            last_err = f"unmapped label {text!r}"
        except urllib.error.HTTPError as e:
            code = e.code
            last_err = f"HTTP {code}: {e.read().decode()[:200]}"
            if code not in (429, 500, 503, 504):
                break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** attempt * 2))
    return None, last_err


def run_model(model, project, sub_idx, clip_fn, y, langs, sources, out_dir, workers=8,
              thinking_budget=None):
    """Run one Gemini model over the subsample. ``clip_fn(i)`` returns the
    decoded waveform for global example index ``i``. Resumable via .npz."""
    if thinking_budget is None:
        thinking_budget = THINKING_BUDGETS.get(model, 0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"gemini_{model}.json"
    npz_path = out_dir / f"gemini_{model}.npz"
    n = len(sub_idx)
    preds = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    if npz_path.exists():  # resume partial run
        blob = np.load(npz_path)
        if len(blob["preds"]) == n:
            preds, done = blob["preds"].copy(), blob["done"].copy()
            print(f"[resume] {done.sum()}/{n} already done", flush=True)

    tokens = TokenSource()
    todo = [i for i in range(n) if not done[i]]
    errors, t0 = [], time.time()
    completed = 0
    lock = threading.Lock()

    def work(i):
        audio = clip_fn(int(sub_idx[i]))
        return i, *classify(model, audio, tokens, project, thinking_budget)

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
                    print(f"  {completed}/{len(todo)} ({rate:.1f} clips/s, "
                          f"eta {(len(todo) - completed) / max(rate, 1e-9) / 60:.0f} min, "
                          f"errors={len(errors)})", flush=True)
                    np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)

    np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)
    result = {
        "name": f"gemini-{model}",
        "model_id": model,
        "adapter": "gemini_vertex",
        "modality": "audio",
        "thinking_budget": thinking_budget,
        "subsample_n": n,
        "api_errors_defaulted_to_neutral": len(errors),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        **score(y, preds, langs, sources, list(LABELS)),
    }
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if errors:
        (out_dir / f"gemini_{model}.errors.json").write_text(
            json.dumps(errors[:100], indent=2), encoding="utf-8")
    print(f"GEMINI_MODEL_DONE {model} acc={result['accuracy']:.4f} "
          f"macro_f1={result['macro_f1']:.4f} errors={len(errors)}", flush=True)
    return result
