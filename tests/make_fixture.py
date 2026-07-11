"""Generate a tiny synthetic eval fixture (sine waves + random labels).

Produces parquet shards in the exact schema the benchmark expects
(audio_flac / label / language / source_id) so the test suite and CI can
exercise data loading and scoring without the real (undistributed) eval set.

Usage:
  python tests/make_fixture.py --out-dir fixture_shards
"""

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR = 16000
N_LABELS = 7  # matches oruk_bench.core.LABELS
FIXTURE_LANGS = ["en", "de", "th", "unknown"]


def sine_clip(freq, seconds, sr=TARGET_SR, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def flac_bytes(audio, sr=TARGET_SR):
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="FLAC")
    return buf.getvalue()


def make_fixture(out_dir, n_clips=24, n_shards=2, seed=0):
    """Write ``n_clips`` synthetic examples across ``n_shards`` parquet shards.

    Returns the list of gold label indices, in deterministic shard order.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [int(rng.integers(0, N_LABELS)) for _ in range(n_clips)]
    rows = []
    for i, label in enumerate(labels):
        freq = 200 + 60 * label + float(rng.uniform(-10, 10))
        seconds = float(rng.uniform(0.5, 2.0))
        rows.append({
            "audio_flac": flac_bytes(sine_clip(freq, seconds)),
            "label": label,
            "language": FIXTURE_LANGS[i % len(FIXTURE_LANGS)],
            "source_id": f"synthetic-{i % 2}",
        })

    per_shard = (n_clips + n_shards - 1) // n_shards
    for s in range(n_shards):
        chunk = rows[s * per_shard: (s + 1) * per_shard]
        if not chunk:
            continue
        table = pa.table({
            "audio_flac": pa.array([r["audio_flac"] for r in chunk], type=pa.binary()),
            "label": pa.array([r["label"] for r in chunk], type=pa.int64()),
            "language": pa.array([r["language"] for r in chunk], type=pa.string()),
            "source_id": pa.array([r["source_id"] for r in chunk], type=pa.string()),
        })
        pq.write_table(table, out_dir / f"shard_{s:03d}.parquet")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="fixture_shards")
    ap.add_argument("--n-clips", type=int, default=24)
    ap.add_argument("--n-shards", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    labels = make_fixture(args.out_dir, args.n_clips, args.n_shards, args.seed)
    print(f"wrote {len(labels)} clips to {args.out_dir}")


if __name__ == "__main__":
    main()
