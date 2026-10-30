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
LOW_FRAMES = 3              # close_gaps: the sounding output must stay low this long for the voice to have stopped


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


def averaged_posteriors(models, mel: np.ndarray, aux: np.ndarray, device="cpu") -> np.ndarray:
    """The mean of several networks' ``posteriors``, per frame and output, before any peak picking."""
    return np.mean([posteriors(model, mel, aux, device) for model in models], axis=0)


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
                sounding=None, end_times=None):
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
    ``end_times`` (seconds) are note ends, such as the offset output's peaks:
    with ``end_snap_window_s`` the decoder moves each note's end to the nearest.
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
                      sounding=sounding, end_times=end_times)


PAIR_WINDOW_S = 0.25      # a vowel takes an articulation onset at most this far before it (sources.MAX_CONSONANT_S)
LONE_GAP_S = 0.15         # lone="drop" or "snap": a lone peak this near a paired start is that note's second peak


def landmark_onsets(post: np.ndarray, fraction: float, threshold: float = 0.5, window: float = PAIR_WINDOW_S,
                    lone: str = "all", peak_gap: int = MIN_GAP_FRAMES):
    """Note starts from a landmark network's two onset outputs (``train --onset-labels landmarks``): column 0, where
    a syllable starts sounding, and column 4, where its vowel starts. Returns (times in seconds, strengths).

    Each vowel peak takes the nearest articulation peak before it within ``window``, unless another vowel peak lies
    between them, and gives an onset ``fraction`` of the way from the articulation to the vowel, with the larger
    of the two peaks as its strength. A vowel peak with no articulation peak gives an onset at the vowel; an
    articulation peak with no vowel peak within ``window`` after it gives one at the articulation. Onsets closer than
    50 ms keep the stronger, as ``pick_peaks`` does.

    Every onset is rounded to the nearest frame and given as that frame's time, exactly as ``onset_track`` gives a
    peak's: the decoder works on the f0 grid, and an onset between frames stops the same-pitch merge seeing the split
    it made. Round 2's first landmark decodes (27-28 October) interpolated between frames, and merged re-sung notes
    about twice as often as the single-onset network (ROUND2-AMENDMENT-1.md).

    ``lone`` sets what happens to a peak with no partner (HANDOVER-PLAN.md). ``"all"`` treats it as a paired start. With
    ``"drop"``, one within ``LONE_GAP_S`` of a paired start is removed, since snapping takes the nearest onset whatever
    its strength, so it could steal that note's start. ``"snap"`` does the same and gives the remaining lone peaks a NaN
    strength: the decoder still snaps note starts to them, but they never cut a note. ``peak_gap`` spaces each output's
    peaks in frames, for networks with wide onset targets (``train --onset-width``).
    """
    if lone not in ("all", "drop", "snap"):
        raise ValueError(f"lone must be 'all', 'drop' or 'snap', not {lone!r}")
    if post.shape[1] < 5:
        raise ValueError("landmark_onsets needs a network with the vowel output (five outputs)")
    grid = np.arange(len(post)) * HOP_S
    a_fr, v_fr = pick_peaks(post[:, 0], threshold, peak_gap), pick_peaks(post[:, 4], threshold, peak_gap)
    win = window / HOP_S
    used, out = set(), []                           # (frame, strength, paired)
    for k, vf in enumerate(v_fr):
        prev_v = v_fr[k - 1] if k else -1
        cands = [i for i, af in enumerate(a_fr) if i not in used and vf - win - 1e-9 <= af < vf and af > prev_v]
        if cands:
            i = max(cands, key=lambda j: a_fr[j])
            used.add(i)
            frame = int(np.floor(a_fr[i] + fraction * (vf - a_fr[i]) + 0.5))
            out.append((frame, max(post[a_fr[i], 0], post[vf, 4]), True))
        else:
            out.append((int(vf), post[vf, 4], False))
    for i, af in enumerate(a_fr):
        if i not in used and not np.any((v_fr > af) & (v_fr <= af + win + 1e-9)):
            out.append((int(af), post[af, 0], False))
    if lone != "all":
        paired = np.array([f for f, _, p in out if p], dtype=float)
        reach = LONE_GAP_S / HOP_S + 1e-9
        out = [o for o in out if o[2] or not paired.size or np.abs(paired - o[0]).min() > reach]
    keep = []                                       # 50 ms apart at least, the stronger kept
    for f, s, p in sorted(out, key=lambda x: (-x[1], x[0])):
        if all(abs(f - k[0]) >= MIN_GAP_FRAMES for k in keep):
            keep.append((f, s, p))
    keep.sort()
    return (grid[[f for f, _, _ in keep]],
            np.array([float(s) if p or lone != "snap" else np.nan for _, s, p in keep]))


def landmark_track(post: np.ndarray, fraction: float, threshold: float = 0.5, sounding=None, end_times=None,
                   lone: str = "all", peak_gap: int = MIN_GAP_FRAMES):
    """An OnsetTrack from a landmark network (``landmark_onsets``), for the decoder as ``onset_track`` with
    ``split_policy="all"``: every onset may cut, unless ``lone`` says otherwise. Its envelope is the larger of the
    two onset outputs."""
    from transcpybe.onsets import OnsetTrack

    times, strengths = landmark_onsets(post, fraction, threshold, lone=lone, peak_gap=peak_gap)
    envelope = np.maximum(post[:, 0], post[:, 4]).astype(float)
    return OnsetTrack(times, strengths, envelope, np.arange(len(envelope)) * HOP_S, split_only_times=None,
                      sounding=sounding, end_times=end_times)


def close_gaps(notes, sounding: np.ndarray, gap_s: float, threshold: float) -> list:
    """Run a note on towards the next where the voice never stopped (ENDS-GAP-PLAN.md).

    A note followed by another within ``gap_s`` is extended until ``sounding`` stays below ``threshold`` for
    ``LOW_FRAMES`` frames, or to the next note's start if it never does. A note before a longer gap is left alone, and
    no note gets shorter. ``notes`` are transcPybe ``Note``s in time order, and ``sounding`` is on the network's frames.
    """
    import dataclasses

    out = list(notes)
    for i, (note, following) in enumerate(zip(notes, notes[1:])):
        if not 0 < following.onset - note.offset <= gap_s + 1e-9:
            continue
        first = int(round(note.offset / HOP_S))
        low = np.asarray(sounding[first:int(round(following.onset / HOP_S))]) < threshold
        stop = next((j for j in range(len(low) - LOW_FRAMES + 1) if low[j:j + LOW_FRAMES].all()), None)
        end = following.onset if stop is None else (first + stop) * HOP_S
        if end > note.offset:
            out[i] = dataclasses.replace(note, offset=float(end))
    return out


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
