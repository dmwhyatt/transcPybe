"""The trained network's onset posterior as a transcPybe ``OnsetTrack``.

    python -m boundary.infer CHECKPOINT AUDIO... --out-dir DIR [--threshold 0.5]

writes ``DIR/<name>.json`` with the onset posterior and the picked onsets, which
``onset_track`` turns back into an OnsetTrack for ``transcpybe.core.decode_notes``.
The network's frames are the f0 frames, so its onsets sit on the f0 grid
already: an onset off that grid stops the same-pitch merge seeing the split it made.

transcPybe's decoder does two things with an onset. It **splits** a note at one
strong enough to pass its bar, and it **snaps** a nearby note start to any
onset. The decoder already puts a boundary wherever the pitch changes, so a
network onset there is only needed for timing. Fed every peak as a split, the
first pilots made half as many notes again as there were (Molina: 3,397 against
2,269). With ``split_policy="same-pitch"`` a peak may split a note only where
the pitch track cannot explain a boundary: voiced and level on both sides. It
must also reach ``split_threshold``. Every other peak is passed with a NaN
strength, which the decoder never splits on but still snaps to.

``"same-pitch"`` asks for voicing within 20-80 ms of the peak. A note sung again
after an unvoiced consonant has a gap exactly there, so on RWC-P that rule
blocked the re-sung splits the network exists to make: same-pitch merges went
back from 6.0% to 8.0%. ``"across-gap"`` compares the last voiced pitch before
the peak with the first voiced pitch after it, looking up to 150 ms each way,
so a consonant between two syllables on one pitch still counts as level. A
peak with no voiced pitch within 150 ms on one side starts a phrase, and only
snaps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import FEATURE_SR, HOP_S

MIN_GAP_FRAMES = 5      # two onsets closer than 50 ms are one onset
LEVEL_SIDE_FRAMES = (2, 8)  # pitch is compared over 20-80 ms either side of a peak
LEVEL_SEMITONES = 0.5       # level: the two sides' median pitches within half a semitone
GAP_FRAMES = 15             # across-gap: look up to 150 ms each way for voiced pitch
SIDE_VOICED = 6             # ...and compare the nearest 6 voiced frames on each side (at least 3)


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
    """(T, n_out) probabilities: onset, offset, sounding, and re-sung onset where the network has it."""
    import torch

    with torch.no_grad():
        m = torch.from_numpy(np.asarray(mel, np.float32))[None].to(device)
        a = torch.from_numpy(np.asarray(aux, np.float32))[None].to(device)
        return torch.sigmoid(model(m, a))[0].float().cpu().numpy()


def pitch_level_across(track, frame: int, voicing_threshold: float = 0.3) -> bool:
    """True where the pitch is voiced and level on both sides of ``frame``: no pitch change explains a boundary there."""
    a, b = LEVEL_SIDE_FRAMES
    f0, v = np.asarray(track.f0_hz, float), np.asarray(track.voicing, float)
    sides = []
    for lo, hi in ((frame - b, frame - a), (frame + a, frame + b)):
        lo, hi = max(lo, 0), min(hi, len(f0))
        ok = np.isfinite(f0[lo:hi]) & (f0[lo:hi] > 0) & (v[lo:hi] >= voicing_threshold)
        if hi - lo < 3 or ok.mean() < 0.6:
            return False
        sides.append(np.median(12 * np.log2(f0[lo:hi][ok] / 440.0)))
    return abs(sides[0] - sides[1]) < LEVEL_SEMITONES


def pitch_level_across_gap(track, frame: int, voicing_threshold: float = 0.3) -> bool:
    """True where the nearest voiced pitch before ``frame`` and after it (each within 150 ms) agree to half a semitone."""
    f0, v = np.asarray(track.f0_hz, float), np.asarray(track.voicing, float)
    voiced = np.isfinite(f0) & (f0 > 0) & (v >= voicing_threshold)
    before = [i for i in range(frame - 1, max(frame - GAP_FRAMES, 0) - 1, -1) if voiced[i]][:SIDE_VOICED]
    after = [i for i in range(frame + 1, min(frame + GAP_FRAMES, len(f0) - 1) + 1) if voiced[i]][:SIDE_VOICED]
    if len(before) < 3 or len(after) < 3:
        return False
    semis = lambda idx: float(np.median(12 * np.log2(f0[idx] / 440.0)))
    return abs(semis(before) - semis(after)) < LEVEL_SEMITONES


def onset_track(prob_onset: np.ndarray, threshold: float = 0.5, track=None,
                split_policy: str = "all", split_threshold: float | None = None, split_only_times=None,
                sounding=None):
    """An OnsetTrack whose envelope is the onset posterior and whose strengths are its peak values.

    ``split_policy="all"`` lets every peak split a note; ``"same-pitch"`` (which
    needs the f0 ``track``) lets a peak split only where the pitch is level
    across it and it reaches ``split_threshold`` (default: ``threshold``);
    ``"none"`` lets no peak split. Peaks that may not split get a NaN strength:
    the decoder still snaps note starts to them.

    ``split_only_times`` (seconds) are cuts from elsewhere, such as the fourth
    output's re-sung onsets: the decoder cuts the note each falls in and lets
    none of them move a note start. With ``"none"``, they are the only cuts.

    ``sounding`` is the network's sounding output, on the same frames: the
    decoder's ``sustain_merge_above`` rejoins touching notes where it stays high.
    """
    from transcpybe.onsets import OnsetTrack

    if split_policy not in ("all", "same-pitch", "across-gap", "none"):
        raise ValueError(f"split_policy must be 'all', 'same-pitch', 'across-gap' or 'none', not {split_policy!r}")
    times = np.arange(len(prob_onset)) * HOP_S
    frames = pick_peaks(prob_onset, threshold)
    strengths = prob_onset[frames].astype(float)
    bar = threshold if split_threshold is None else split_threshold
    if split_policy in ("same-pitch", "across-gap"):
        if track is None:
            raise ValueError(f"split_policy={split_policy!r} needs the f0 track")
        level = pitch_level_across if split_policy == "same-pitch" else pitch_level_across_gap
        may_split = np.array([strengths[i] >= bar and level(track, int(f))
                              for i, f in enumerate(frames)], dtype=bool)
        strengths[~may_split] = np.nan
    elif split_policy == "none":
        strengths[:] = np.nan
    elif bar > threshold:
        strengths[strengths < bar] = np.nan
    return OnsetTrack(times[frames], strengths, prob_onset.astype(float), times,
                      split_only_times=None if split_only_times is None else np.sort(np.asarray(split_only_times, float)),
                      sounding=sounding)


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
