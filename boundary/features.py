"""Separated stems -> what the network sees, and what it should say, frame by frame.

    python -m boundary.features STEMS.jsonl OUT_DIR [--device auto] [--crepe-model full]

One ``.npz`` per stem, on transcPybe's 10 ms f0 grid (frame k centred at k*10 ms):

- ``mel``  (T, 80) float16  log-mel of the stem at 16 kHz, n_fft 1024, floored
  80 dB below the recording's loudest bin, then standardised per recording
- ``aux``  (T, 5)  float16  transcPybe's own f0 track: pitch relative to C4 in
  octaves (0 where unpitched), a pitched flag, periodicity, voicing, and level
  in dB relative to the stem's 95th percentile, divided by 40 and clipped to
  [-2, 0.5]
- ``y``    (T, 3)  float16  targets: onset, offset, note sounding (see targets())
- ``meta`` str             what the file was made from (stem, notes, f0
  settings, feature version), checked before an existing file is reused

The two floors keep silence from producing values a real stem never does:
exact digital silence would give a level of -200 dB and a log-mel of -14, and a
network can learn those as a shortcut that never fires on separated audio.

The f0 features come from ``transcpybe.f0.estimate_f0`` with the shipped
settings, so at inference the network is given exactly what the decoder has.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import FEATURE_SR, HOP_S, items as manifest

FEATURE_VERSION = 2
N_MELS = 80
MEL_TOP_DB = 80.0
LEVEL_RANGE = (-2.0, 0.5)     # 80 dB below the 95th percentile, 20 dB above it
TARGET_WIDTH = 2              # an onset or offset target spreads 2 frames (20 ms) either side, tapering


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
    mel = librosa.power_to_db(mel, ref=np.max, top_db=MEL_TOP_DB).T
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)          # per recording: gain-invariant
    f0 = np.asarray(track.f0_hz, dtype=np.float64)
    pitched = np.isfinite(f0) & (f0 > 0)
    rel = np.zeros_like(f0)
    rel[pitched] = np.log2(f0[pitched] / 261.63)
    level = np.asarray(track.loudness_db, dtype=np.float64)
    level = np.clip((level - np.percentile(level, 95)) / 40.0, *LEVEL_RANGE)
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


def _meta(row: dict, cfg) -> str:
    return json.dumps(dict(version=FEATURE_VERSION, stem=row["stem"], notes=row["notes"],
                           f0_backend=cfg.f0_backend, crepe_model=cfg.crepe_model), sort_keys=True)


def _current(path: Path, meta: str) -> bool:
    """An existing feature file may be reused only if it was made from this stem, these notes and these settings."""
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            return "meta" in z.files and str(z["meta"]) == meta
    except Exception:
        return False


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
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out, made = [], 0
    for i, r in enumerate(rows):
        path = out_dir / f"{r['mix_id']}.npz"
        meta = _meta(r, cfg)
        if not _current(path, meta):
            audio, _ = load_audio(r["stem"], FEATURE_SR)
            track = estimate_f0(audio, cfg, null_reporter())
            mel, aux = frame_features(audio, track)
            y = targets(r["notes"], len(mel))
            tmp = path.with_name(path.stem + ".part.npz")
            np.savez(tmp, mel=mel.astype(np.float16), aux=aux.astype(np.float16), y=y.astype(np.float16),
                     meta=np.array(meta))
            tmp.replace(path)
            made += 1
        out.append(dict(r, features=str(path)))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
    manifest.write(out_dir / "features.jsonl", out)
    print(f"{len(out)} feature files -> {out_dir} ({made} computed, {len(out) - made} reused)")


if __name__ == "__main__":
    main()
