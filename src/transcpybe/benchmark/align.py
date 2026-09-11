"""Warp a notated reference onto the recording it describes.

A melody notated at one nominal tempo drifts against a performance that
breathes. BiMMuDa's do, by a median 0.6 s from the start of a song to its end,
which is twelve times the onset tolerance: every timed metric then measures the
mismatch instead of the transcription.

Dynamic time warping recovers the correspondence, but what it is aligned *to*
decides whether the result means anything. Aligning a reference to our own
transcription would absorb whatever timing errors that transcription has, and
every configuration would then score well. Aligning it to the **f0 contour**
does not: f0 is estimated upstream of every parameter worth tuning, so the
alignment is computed once per recording and is identical for every
configuration compared against it.

The warp is constrained rather than free. An unconstrained path can align
anything to anything, so the tempo ratio is held near one and the path is kept
inside a band around the diagonal. Alignments that do not fit are reported as
failures rather than used, because a reference warped onto the wrong part of a
song scores like a broken transcriber and looks like nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metrics import RefNote

# How far the path may stray from the diagonal. BiMMuDa's references already
# carry a start offset, so the correction needed is a tempo curve, not a search
# of the whole song.
BAND_S = 12.0
# Frames are coarser than the f0 hop: a melody's shape is what is being matched,
# and a finer grid costs quadratic time for no extra information.
FRAME_S = 0.05
# Above this median cost the alignment is not trusted. A semitone of median
# disagreement over a whole song is already a poor fit.
MAX_MEDIAN_COST = 3.0
MIN_COVERAGE = 0.5
# A performance is not played at half or double the notated tempo, so a warp
# whose local rate leaves this range is finding coincidences, not a tempo.
MIN_SLOPE, MAX_SLOPE = 0.7, 1.45
MIN_STEADY_FRACTION = 0.85
SLOPE_WINDOW_S = 5.0
MIN_WARPED_NOTE_S = 0.02


@dataclass
class Alignment:
    """The warp from reference time to recording time, and how much to trust it."""

    ok: bool
    reason: str = ""
    median_cost: float = 0.0
    coverage: float = 0.0
    slope_spread: float = 0.0
    """How much the warp's local rate varies. A tempo wanders; a false
    alignment lurches."""
    steady_fraction: float = 0.0
    """The share of the warp running at a plausible rate."""
    max_shift_s: float = 0.0
    """The largest correction the warp applies anywhere in the recording.

    Measured across the whole path, not at its ends: dynamic time warping pins
    the first and last frames to each other, so a shift measured at the extremes
    is zero however much the middle moved."""
    _source: Optional[np.ndarray] = None
    _target: Optional[np.ndarray] = None

    def warp_time(self, seconds: float) -> float:
        if self._source is None or self._target is None:
            return seconds
        return float(np.interp(seconds, self._source, self._target))

    def warp_notes(self, notes: Sequence[RefNote]) -> List[RefNote]:
        """Move a reference into the recording's clock."""
        if not self.ok:
            return list(notes)
        warped = []
        for start, end, pitch in notes:
            # A warp that compresses hard can bring a short note's ends
            # together, and a zero-length note is not a note.
            new_start = self.warp_time(start)
            new_end = max(self.warp_time(end), new_start + MIN_WARPED_NOTE_S)
            warped.append((new_start, new_end, pitch))
        return warped


def _reference_contour(notes: Sequence[RefNote], times: np.ndarray) -> np.ndarray:
    """The reference as a pitch per frame, NaN where nothing is sounding."""
    contour = np.full(times.shape[0], np.nan)
    for start, end, pitch in notes:
        contour[(times >= start) & (times < end)] = float(pitch)
    return contour


def _audio_contour(f0_hz: np.ndarray, voicing: np.ndarray, times: np.ndarray,
                   frame_times: np.ndarray, threshold: float) -> np.ndarray:
    """The recording's f0 as semitones per frame, on the coarse grid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        semitones = 69.0 + 12.0 * np.log2(np.where(f0_hz > 0, f0_hz, np.nan) / 440.0)
    semitones = np.where(voicing >= threshold, semitones, np.nan)
    index = np.clip(np.searchsorted(frame_times, times), 0, semitones.shape[0] - 1)
    return semitones[index]


def _cost_matrix(reference: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Distance between every pair of frames, by pitch class.

    Pitch class rather than absolute pitch: an octave error in either the
    transcription or the reference should not stop two frames matching, and the
    melody's shape is what carries the alignment.
    """
    ref = reference[:, None]
    aud = audio[None, :]
    difference = np.abs(ref - aud) % 12.0
    difference = np.minimum(difference, 12.0 - difference)
    # A frame that is silent on one side is neither a match nor a contradiction.
    # Charging a middling cost lets the path cross rests without being drawn to
    # them, which a zero cost would do.
    return np.where(np.isnan(difference), 2.0, difference)


def _warp_path(reference: np.ndarray, audio: np.ndarray, *,
               band_s: float, frame_s: float):
    """The warping path between two contours, inside a band around the diagonal.

    librosa's implementation is compiled; a pure Python one costs half a minute
    per song, which over a corpus is the difference between doing this and not.
    """
    import librosa

    cost = _cost_matrix(reference, audio)
    n, m = cost.shape
    band = max(band_s / frame_s, 1.0) / max(n, m)
    try:
        _, path = librosa.sequence.dtw(C=cost, global_constraints=True,
                                       band_rad=float(min(band, 0.5)))
    except Exception:
        return None
    if path is None or len(path) == 0:
        return None
    path = np.asarray(path)[::-1]
    return path[:, 0], path[:, 1]


def _local_slopes(source: np.ndarray, target: np.ndarray,
                  window_s: float = SLOPE_WINDOW_S) -> np.ndarray:
    """How fast the warp advances through the recording, window by window.

    One number per window rather than per frame: frame to frame a warping path
    steps either diagonally or sideways, so its instantaneous slope is only ever
    zero or one and says nothing.
    """
    if source.size < 4:
        return np.zeros(0)
    edges = np.arange(source[0], source[-1], window_s)
    if edges.size < 2:
        return np.zeros(0)
    positions = np.interp(edges, source, target)
    return np.diff(positions) / window_s


def align_reference(
    notes: Sequence[RefNote],
    f0_hz: np.ndarray,
    voicing: np.ndarray,
    frame_times: np.ndarray,
    *,
    voicing_threshold: float = 0.5,
    band_s: float = BAND_S,
    frame_s: float = FRAME_S,
) -> Alignment:
    """Find the warp taking a notated reference onto a recording."""
    if len(notes) < 4 or frame_times.size < 4:
        return Alignment(False, "too few notes or frames")

    duration = max(float(frame_times[-1]), max(end for _, end, _ in notes))
    times = np.arange(0.0, duration + frame_s, frame_s)
    if times.size < 8:
        return Alignment(False, "recording too short to align")

    reference = _reference_contour(notes, times)
    audio = _audio_contour(f0_hz, voicing, times, frame_times, voicing_threshold)
    if np.isnan(reference).all() or np.isnan(audio).all():
        return Alignment(False, "one side has no pitched frames")

    path = _warp_path(reference, audio, band_s=band_s, frame_s=frame_s)
    if path is None:
        return Alignment(False, "no path inside the band")
    path_i, path_j = path

    # Cost is judged only where both sides are sounding. A rest matches a rest
    # at no cost and tells you nothing, and counting those drags the median
    # towards whatever the silence penalty happens to be.
    pairs = np.abs(reference[path_i] - audio[path_j]) % 12.0
    pairs = np.minimum(pairs, 12.0 - pairs)
    informative = pairs[np.isfinite(pairs)]
    if informative.size < 20:
        return Alignment(False, "too few frames sounding on both sides")

    source = times[path_i]
    target = times[path_j]
    unique_source, index = np.unique(source, return_index=True)
    alignment = Alignment(
        ok=True,
        median_cost=float(np.median(informative)),
        coverage=float(np.mean(~np.isnan(reference))),
        _source=unique_source,
        _target=target[index],
    )
    alignment.max_shift_s = float(np.max(np.abs(alignment._target
                                                 - alignment._source)))

    # A cost threshold cannot tell a true alignment from a false one: warping
    # is free to stretch, and a long recording holds plenty of frames at every
    # pitch, so a wrong melody also finds a cheap path. What separates them is
    # the shape of the warp. A real one runs at close to the recording's tempo
    # throughout -- local slope near one, drifting slowly. A false one races
    # back and forth to reach whichever frames happen to match.
    slopes = _local_slopes(alignment._source, alignment._target)
    alignment.slope_spread = float(np.std(slopes)) if slopes.size else 9.9
    steady = (float(np.mean((slopes > MIN_SLOPE) & (slopes < MAX_SLOPE)))
              if slopes.size else 0.0)
    alignment.steady_fraction = steady
    if steady < MIN_STEADY_FRACTION:
        return Alignment(
            False,
            f"the warp is not a tempo: only {steady:.0%} of it runs at a plausible "
            f"rate", alignment.median_cost, alignment.coverage,
            alignment.slope_spread, steady, alignment.max_shift_s)

    if alignment.median_cost > MAX_MEDIAN_COST:
        return Alignment(False, f"poor fit (median cost {alignment.median_cost:.2f} "
                                f"semitones)", alignment.median_cost,
                         alignment.coverage, alignment.slope_spread,
                         alignment.steady_fraction, alignment.max_shift_s)
    if alignment.coverage < MIN_COVERAGE:
        return Alignment(False, f"reference covers only {alignment.coverage:.0%} of the "
                                "recording", alignment.median_cost, alignment.coverage,
                         alignment.slope_spread, alignment.steady_fraction,
                         alignment.max_shift_s)
    return alignment
