"""The trained network's onset posterior as a transcPybe ``OnsetTrack``.

    python -m boundary.infer CHECKPOINT AUDIO... --out-dir DIR [--threshold 0.5]

writes ``DIR/<name>.json`` with the onset posterior and the picked onsets, which
``onset_track`` turns back into an OnsetTrack for ``transcpybe.core.decode_notes``.
The network's frames are the f0 frames, so its onsets sit on the f0 grid
already: an onset off that grid stops the same-pitch merge seeing the split it made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import FEATURE_SR, HOP_S

MIN_GAP_FRAMES = 5      # two onsets closer than 50 ms are one onset


def pick_peaks(p: np.ndarray, threshold: float = 0.5, min_gap: int = MIN_GAP_FRAMES) -> np.ndarray:
    """Frames that are local maxima of ``p`` above ``threshold``, strongest first when two are too close."""
    if len(p) < 3:
        return np.zeros(0, dtype=int)
    cand = np.flatnonzero((p[1:-1] >= threshold) & (p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:])) + 1
    keep = []
    for f in cand[np.argsort(-p[cand])]:
        if all(abs(f - k) >= min_gap for k in keep):
            keep.append(int(f))
    return np.array(sorted(keep), dtype=int)


def posteriors(model, mel: np.ndarray, aux: np.ndarray, device="cpu") -> np.ndarray:
    """(T, 3) probabilities: onset, offset, sounding."""
    import torch

    with torch.no_grad():
        m = torch.from_numpy(np.asarray(mel, np.float32))[None].to(device)
        a = torch.from_numpy(np.asarray(aux, np.float32))[None].to(device)
        return torch.sigmoid(model(m, a))[0].float().cpu().numpy()


def onset_track(prob_onset: np.ndarray, threshold: float = 0.5):
    """An OnsetTrack whose envelope is the onset posterior and whose strengths are its peak values."""
    from transcpybe.onsets import OnsetTrack

    times = np.arange(len(prob_onset)) * HOP_S
    frames = pick_peaks(prob_onset, threshold)
    return OnsetTrack(times[frames], prob_onset[frames].astype(float), prob_onset.astype(float), times)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("audio", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--f0-backend", default=None, help="override, e.g. pyin for quick tests")
    args = ap.parse_args(argv)

    from transcpybe.audio import load_audio
    from transcpybe.f0 import estimate_f0
    from transcpybe.progress import null_reporter

    from .features import f0_config, frame_features
    from .model import load

    cfg = f0_config(args.device, backend=args.f0_backend)
    model = load(args.checkpoint, "cpu" if cfg.device == "auto" else cfg.device)
    dev = next(model.parameters()).device
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.audio:
        audio, _ = load_audio(path, FEATURE_SR)
        mel, aux = frame_features(audio, estimate_f0(audio, cfg, null_reporter()))
        p = posteriors(model, mel, aux, dev)
        frames = pick_peaks(p[:, 0], args.threshold)
        json.dump(dict(hop_s=HOP_S, threshold=args.threshold, onsets=(frames * HOP_S).round(3).tolist(),
                       onset_prob=p[:, 0].round(4).tolist(), offset_prob=p[:, 1].round(4).tolist(),
                       sounding_prob=p[:, 2].round(4).tolist()),
                  open(args.out_dir / f"{path.stem}.json", "w"))
        print(f"{path.name}: {len(frames)} onsets")


if __name__ == "__main__":
    main()
