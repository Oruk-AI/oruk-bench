"""Open-weight audio-LLM arm: zero-shot prompted classification.

Local audio-language models on the same fixed 5,000-clip stratified subsample
(seed 0) used for the closed-source arm. Same 16 s truncation, same 7-label
space, same scoring code.

Each model family gets a small adapter that builds the model's chat format with
the audio clip plus a constrained prompt listing all 7 options, generates a few
tokens, and parses a single label. Unparsed outputs default to neutral and are
counted, mirroring the API arm.

Run one model per process (VRAM + dependency hygiene). Requires the
``[audiollm]`` extra: torch, recent transformers, accelerate.
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ..core import LABELS, PROMPT, TARGET_SR, norm_label, score

WORD_RE = re.compile(r"[a-zA-Z]+")

AUDIO_LLM_PROMPT = PROMPT.replace(
    "Answer with the single best label.",
    "Answer with exactly ONE word from: anger, happiness, sadness, fear, disgust, "
    "surprise, neutral. No other text.")


def parse_label(text):
    """Last resort scan: first alias hit wins, scanning from the end (models
    often reason first and answer last)."""
    words = WORD_RE.findall(text.lower())
    for w in reversed(words):
        lab = norm_label(w)
        if lab is not None:
            return lab
    return None


# ------------------------------------------------------------------ adapters


class QwenOmniAdapter:
    """Qwen/Qwen2.5-Omni-7B and fine-tunes (ddwang2000/EmotionThinker).
    Thinker-only load: text output, no talker weights."""

    def __init__(self, model_id, device):
        import torch
        from transformers import (Qwen2_5OmniProcessor,
                                  Qwen2_5OmniThinkerForConditionalGeneration)

        self.torch = torch
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            attn_implementation="sdpa")
        self.model.eval()
        self.device = device

    def classify(self, audio):
        conv = [
            {"role": "system", "content": [{"type": "text", "text":
                "You are a precise audio annotation system."}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": AUDIO_LLM_PROMPT},
            ]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True).to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)


class VoxtralAdapter:
    """mistralai/Voxtral-Mini-3B-2507 via VoxtralForConditionalGeneration."""

    def __init__(self, model_id, device):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device
        self.model_id = model_id

    def classify(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, TARGET_SR)
            path = f.name
        try:
            conv = [{"role": "user", "content": [
                {"type": "audio", "path": path},
                {"type": "text", "text": AUDIO_LLM_PROMPT},
            ]}]
            inputs = self.processor.apply_chat_template(conv, return_tensors="pt")
            inputs = inputs.to(self.device, dtype=self.torch.bfloat16)
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            return self.processor.decode(gen, skip_special_tokens=True)
        finally:
            os.unlink(path)


class AudioFlamingoAdapter:
    """nvidia/audio-flamingo-3-hf via AudioFlamingo3ForConditionalGeneration."""

    def __init__(self, model_id, device):
        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device

    def classify(self, audio):
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": AUDIO_LLM_PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", sampling_rate=TARGET_SR)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)


class MeralionAdapter:
    """MERaLiON/MERaLiON-2-10B via AutoModelForSpeechSeq2Seq (trust_remote_code)."""

    PROMPT_TMPL = ("Instruction: {instr}\n"
                   "Follow the text instruction based on the following audio: <SpeechHere>")

    def __init__(self, model_id, device):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map=device)
        self.model.eval()
        self.device = device

    def classify(self, audio):
        chat = [{"role": "user", "content": self.PROMPT_TMPL.format(instr=AUDIO_LLM_PROMPT)}]
        text = self.processor.tokenizer.apply_chat_template(
            conversation=chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        for k in list(inputs):
            if inputs[k].dtype == self.torch.float32:
                inputs[k] = inputs[k].to(self.torch.bfloat16)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True)


REGISTRY = {
    "qwen2.5-omni-7b": ("Qwen/Qwen2.5-Omni-7B", QwenOmniAdapter),
    "emotionthinker": ("ddwang2000/EmotionThinker", QwenOmniAdapter),
    "voxtral-mini-3b": ("mistralai/Voxtral-Mini-3B-2507", VoxtralAdapter),
    "audio-flamingo-3": ("nvidia/audio-flamingo-3-hf", AudioFlamingoAdapter),
    "meralion-2-10b": ("MERaLiON/MERaLiON-2-10B", MeralionAdapter),
}


def run_eval(model_name, sub_idx, clip_fn, y, sl, ss, out_dir, device="cuda:0"):
    """Run one registered audio-LLM over the subsample. Resumable via .npz."""
    model_id, cls = REGISTRY[model_name]
    n = len(sub_idx)
    print(f"audio-llm bench: {model_name} ({model_id}) on {n} clips", flush=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"audiollm_{model_name}.npz"
    preds = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    if npz_path.exists():
        blob = np.load(npz_path)
        if len(blob["preds"]) == n:
            preds, done = blob["preds"].copy(), blob["done"].copy()
            print(f"[resume] {done.sum()}/{n} already done", flush=True)

    adapter = cls(model_id, device)
    errors, t0 = [], time.time()
    n_done_at_start = int(done.sum())
    for i in range(n):
        if done[i]:
            continue
        audio = clip_fn(int(sub_idx[i]))
        try:
            text = adapter.classify(audio)
            lab = parse_label(text)
        except Exception as e:
            lab, text = None, f"EXC {type(e).__name__}: {e}"
        if lab is None:
            errors.append({"i": int(i), "out": str(text)[:120]})
            preds[i] = LABELS.index("neutral")
        else:
            preds[i] = LABELS.index(lab)
        done[i] = True
        completed = int(done.sum()) - n_done_at_start
        if completed % 100 == 0:
            rate = completed / max(time.time() - t0, 1e-9)
            print(f"  {completed} done ({rate:.2f} clips/s, "
                  f"eta {(n - int(done.sum())) / max(rate, 1e-9) / 60:.0f} min, "
                  f"unparsed={len(errors)})", flush=True)
            np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)

    np.savez(npz_path, preds=preds, done=done, sub_idx=sub_idx)
    result = {
        "name": f"audiollm-{model_name}",
        "model_id": model_id,
        "adapter": "audio_llm_prompted",
        "modality": "audio",
        "subsample_n": n,
        "api_errors_defaulted_to_neutral": len(errors),
        "elapsed_min": round((time.time() - t0) / 60, 1),
        **score(y, preds, sl, ss, list(LABELS)),
    }
    (out_dir / f"audiollm_{model_name}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    if errors:
        (out_dir / f"audiollm_{model_name}.errors.json").write_text(
            json.dumps(errors[:100], indent=2), encoding="utf-8")
    print(f"AUDIOLLM_MODEL_DONE {model_name} acc={result['accuracy']:.4f} "
          f"macro_f1={result['macro_f1']:.4f} unparsed={len(errors)}", flush=True)
    return result
