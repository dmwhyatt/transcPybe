"""Separated stems -> what the network sees, and what it should say, frame by frame.

    python -m boundary.features STEMS.jsonl OUT_DIR [--device auto] [--crepe-model full]

One ``.npz`` per stem, on transcPybe's 10 ms f0 grid (frame k centred at k*10 ms):

- ``mel``  (T, 80) float16  log-mel of the stem at 16 kHz, n_fft 1024
- ``aux``  (T, 5)  float16  transcPybe's own f0 track: pitch relative to C4 in
  octaves (0 where unpitched), a pitched flag, periodicity, voicing, and level
  in dB relative to the stem's 95th percentile, divided by 40
- ``y``    (T, 3)  float16  targets: onset, offset, note sounding (see targets())

The f0 features come from ``transcpybe.f0.estimate_f0`` with the shipped
settings, so at inference the network is given exactly what the decoder has.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import FEATURE_SR, HOP_S, items as manifest

N_MELS = 80
TARGET_WIDTH = 2      # an onset or offset target spreads 2 frames (20 ms) either side, tapering


def targets(notes: Sequence[Sequence[float]], n_frames: int, width: int = TARGET_WIDTH) -> np.ndarray:
    """(T, 3): onset, offset, sounding. Onset and offset are triangles peaking at 1 on
    the labelled frame, so being one frame out costs a little rather than everything."""
    y = np.zeros((n_frames, 3), dtype=np.float32)
    ramp = 1.0 - np.abs(np.arange(-width, width + 1)) / (width + 1)
    for on, off, _ in notes:
        a, b = int(round(on / HOP_S)), int(round(off / HOP_S))
        if b <= a or a >= n_frames:
            continue
        y[max(a, 0):min(b, n_frames), 2] = 1.0
        for col, c in ((0, a), (1, b)):
            lo, hi = max(c - width, 0), min(c + width + 1, n_frames)
            if lo < hi:
                y[lo:hi, col] = np.maximum(y[lo:hi, col], ramp[lo - (c - width):hi - (c - width)])
    return y


def frame_features(audio: np.ndarray, track) -> tuple:
    """(mel, aux) for 16 kHz mono ``audio`` and its transcPybe ``F0Track``, trimmed to the same frames."""
    import librosa

    hop = int(round(HOP_S * FEATURE_SR))
    mel = librosa.feature.melspectrogram(y=audio, sr=FEATURE_SR, n_fft=1024, hop_length=hop,
                                         n_mels=N_MELS, fmin=40, fmax=8000)
    mel = np.log(mel + 1e-6).T
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)          # per recording: gain-invariant
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    pitched = np.isfinite(f0) & (f0 > 0)
    rel = np.zeros_like(f0)
    rel[pitched] = np.log2(f0[pitched] / 261.63)
    level = np.asarray(track.loudness_db, dtype=np.float64)
    level = (level - np.percentile(level, 95)) / 40.0
    aux = np.stack([rel, pitched.astype(float), track.periodicity, track.voicing, level], axis=1)
    n = min(len(mel), len(aux))
    return mel[:n].astype(np.float32), aux[:n].astype(np.float32)


def f0_config(device: str = "auto", crepe_model: str = "full", backend: Optional[str] = None):
    from transcpybe.config import TranscriptionConfig
    from transcpybe.core import resolve_config_device

    cfg = TranscriptionConfig(device=device, crepe_model=crepe_model)
    if backend:
        cfg = dataclasses.replace(cfg, f0_backend=backend)
    return resolve_config_device(cfg)[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--crepe-model", default="full")
    ap.add_argument("--f0-backend", default=None, help="override, e.g. pyin for quick tests")
    args = ap.parse_args(argv)

    from transcpybe.audio import load_audio
    from transcpybe.f0 import estimate_f0
    from transcpybe.progress import null_reporter

    cfg = f0_config(args.device, args.crepe_model, args.f0_backend)
    rows = manifest.read(args.stems)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(rows):
        path = args.out_dir / f"{r['mix_id']}.npz"
        if not path.exists():
            audio, _ = load_audio(r["stem"], FEATURE_SR)
            track = estimate_f0(audio, cfg, null_reporter())
            mel, aux = frame_features(audio, track)
            y = targets(r["notes"], len(mel))
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, mel=mel.astype(np.float16), aux=aux.astype(np.float16), y=y.astype(np.float16))
            tmp.replace(path)
        out.append(dict(r, features=str(path)))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    manifest.write(args.out_dir / "features.jsonl", out)
    print(f"{len(out)} feature files -> {args.out_dir}")


if __name__ == "__main__":
    main()
