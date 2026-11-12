"""The networks' outputs as note starts: an ``OnsetTrack`` for transcPybe's note decoder.

Each network marks two onsets per syllable: column 0, where the syllable starts sounding (its articulation, often a
consonant), and column 4, where its vowel starts. Annotators differ on which of the two a note starts at, so a note
start is put a fraction *f* of the way from one to the other (``landmark_track``).

Copied from the training repository's ``boundary/infer.py`` (boundary 56d97f6): ``pick_peaks``, ``posteriors``,
``averaged_posteriors``, ``landmark_onsets`` and ``landmark_track``, unchanged.
"""
from __future__ import annotations

import numpy as np

from .features import HOP_S

MIN_GAP_FRAMES = 5      # two onsets closer than 50 ms are one onset
PAIR_WINDOW_S = 0.25    # a vowel takes an articulation onset at most this far before it
LONE_GAP_S = 0.15       # lone="drop" or "snap": a lone peak this near a paired start is that note's second peak


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
    """(T, n_out) probabilities, one column per output."""
    import torch

    with torch.no_grad():
        m = torch.from_numpy(np.asarray(mel, np.float32))[None].to(device)
        a = torch.from_numpy(np.asarray(aux, np.float32))[None].to(device)
        return torch.sigmoid(model(m, a))[0].float().cpu().numpy()


def averaged_posteriors(models, mel: np.ndarray, aux: np.ndarray, device="cpu") -> np.ndarray:
    """The mean of several networks' ``posteriors``, per frame and output, before any peak picking."""
    return np.mean([posteriors(model, mel, aux, device) for model in models], axis=0)


def landmark_onsets(post: np.ndarray, fraction: float, threshold: float = 0.5, window: float = PAIR_WINDOW_S,
                    lone: str = "all", peak_gap: int = MIN_GAP_FRAMES, lone_articulation: str = "peak"):
    """Note starts from the networks' two onset outputs: column 0, where a syllable starts sounding, and column 4,
    where its vowel starts. Returns (times in seconds, strengths).

    Each vowel peak takes the nearest articulation peak before it within ``window``, unless another vowel peak lies
    between them, and gives an onset ``fraction`` of the way from the articulation to the vowel, with the larger
    of the two peaks as its strength. A vowel peak with no articulation peak gives an onset at the vowel; an
    articulation peak with no vowel peak within ``window`` after it gives one at the articulation. Onsets closer than
    50 ms keep the stronger, as ``pick_peaks`` does.

    Every onset is rounded to the nearest frame and given as that frame's time: the decoder works on the f0 grid, and
    an onset between frames stops the same-pitch merge seeing the split it made.

    ``lone`` sets what happens to a peak with no partner. ``"all"`` treats it as a paired start. With ``"drop"``, one
    within ``LONE_GAP_S`` of a paired start is removed, since snapping takes the nearest onset whatever its strength,
    so it could steal that note's start. ``"snap"`` does the same and gives the remaining lone peaks a NaN strength:
    the decoder still snaps note starts to them, but they never cut a note. ``peak_gap`` spaces each output's peaks
    in frames.

    ``lone_articulation="shift"`` moves a lone articulation's start ``fraction`` of the recording's median paired gap
    (vowel minus articulation) after its peak, where a paired start would be. Whether ``lone`` drops it is still
    decided at the peak; the move comes after that and before the 50 ms thinning. With no pair in the recording it
    stays at the peak. Neither preset uses it.
    """
    times, strengths, _ = _landmark_starts(post, fraction, threshold, window, lone, peak_gap, lone_articulation)
    return times, strengths


def _landmark_starts(post, fraction, threshold, window, lone, peak_gap, lone_articulation):
    """``landmark_onsets``' times and strengths, and which of them are paired starts."""
    if lone not in ("all", "drop", "snap"):
        raise ValueError(f"lone must be 'all', 'drop' or 'snap', not {lone!r}")
    if lone_articulation not in ("peak", "shift"):
        raise ValueError(f"lone_articulation must be 'peak' or 'shift', not {lone_articulation!r}")
    if post.shape[1] < 5:
        raise ValueError("landmark_onsets needs a network with the vowel output (five outputs)")
    grid = np.arange(len(post)) * HOP_S
    a_fr, v_fr = pick_peaks(post[:, 0], threshold, peak_gap), pick_peaks(post[:, 4], threshold, peak_gap)
    win = window / HOP_S
    used, out, gaps = set(), [], []                 # out: (frame, strength, paired, lone articulation)
    for k, vf in enumerate(v_fr):
        prev_v = v_fr[k - 1] if k else -1
        cands = [i for i, af in enumerate(a_fr) if i not in used and vf - win - 1e-9 <= af < vf and af > prev_v]
        if cands:
            i = max(cands, key=lambda j: a_fr[j])
            used.add(i)
            frame = int(np.floor(a_fr[i] + fraction * (vf - a_fr[i]) + 0.5))
            out.append((frame, max(post[a_fr[i], 0], post[vf, 4]), True, False))
            gaps.append(vf - a_fr[i])
        else:
            out.append((int(vf), post[vf, 4], False, False))
    for i, af in enumerate(a_fr):
        if i not in used and not np.any((v_fr > af) & (v_fr <= af + win + 1e-9)):
            out.append((int(af), post[af, 0], False, True))
    if lone != "all":
        paired = np.array([f for f, _, p, _ in out if p], dtype=float)
        reach = LONE_GAP_S / HOP_S + 1e-9
        out = [o for o in out if o[2] or not paired.size or np.abs(paired - o[0]).min() > reach]
    if lone_articulation == "shift" and gaps:
        move = fraction * float(np.median(gaps))
        out = [(min(int(np.floor(f + move + 0.5)), len(post) - 1), s, p, la) if la else (f, s, p, la)
               for f, s, p, la in out]
    keep = []                                       # 50 ms apart at least, the stronger kept
    for f, s, p, _ in sorted(out, key=lambda x: (-x[1], x[0])):
        if all(abs(f - k[0]) >= MIN_GAP_FRAMES for k in keep):
            keep.append((f, s, p))
    keep.sort()
    return (grid[[f for f, _, _ in keep]],
            np.array([float(s) if p or lone != "snap" else np.nan for _, s, p in keep]),
            np.array([p for _, _, p in keep], dtype=bool))


def landmark_track(post: np.ndarray, fraction: float, threshold: float = 0.5, sounding=None, end_times=None,
                   lone: str = "all", peak_gap: int = MIN_GAP_FRAMES, lone_articulation: str = "peak",
                   protect: bool = False):
    """An OnsetTrack from the networks' outputs (``landmark_onsets``), for the note decoder: every onset may cut a
    note, unless ``lone`` says otherwise. Its envelope is the larger of the two onset outputs.

    ``sounding`` is the networks' sounding output, on the same frames: the decoder's ``sustain_merge_above`` rejoins
    touching notes where it stays high. With ``protect``, the paired starts are also the track's
    ``protected_times``: with ``landmark_min_note_s`` a note starting at one escapes the short-note rules. Neither
    preset uses ``end_times`` or ``protect``."""
    from ..onsets import OnsetTrack

    times, strengths, paired = _landmark_starts(post, fraction, threshold, PAIR_WINDOW_S, lone, peak_gap,
                                                lone_articulation)
    envelope = np.maximum(post[:, 0], post[:, 4]).astype(float)
    extra = {"protected_times": times[paired]} if protect else {}
    return OnsetTrack(times, strengths, envelope, np.arange(len(envelope)) * HOP_S, split_only_times=None,
                      sounding=sounding, end_times=end_times, **extra)
