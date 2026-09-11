"""Note segmentation by Viterbi decoding over a pitch-state grid.

The state space is one state per semitone across the vocal range, plus a single
unvoiced state. Emissions score how well each frame's measured f0 matches a
given semitone; transitions make staying in a note cheap and changing note
expensive. Decoding the maximum-likelihood path therefore turns a jittery,
continuous f0 contour into piecewise-constant note events.

This follows the note-tracking design of the Tony melody transcription system
(Mauch, Cannam, Bittner, Fazekas, Salamon, Dai, Bello & Dixon, "Computer-aided
melody note transcription using the Tony software", 2015), simplified to one
state per semitone rather than Tony's sub-semitone grid with explicit
attack/sustain sub-states. Vibrato is absorbed by the emission width and the
self-transition bias instead; note starts are then sharpened against
spectral-flux onsets (see ``onsets.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import TranscriptionConfig
from .f0 import F0Track, hz_to_midi

_EPS = 1e-9
_NEG_INF = -1e30


@dataclass
class Segment:
    """A contiguous run of frames decoded to the same pitch state."""

    start_frame: int
    end_frame: int      # exclusive
    pitch: int          # integer MIDI note number

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


def pitch_range(config: TranscriptionConfig) -> Tuple[int, int]:
    """Inclusive integer MIDI range implied by the configured frequency range."""
    lo = int(np.floor(hz_to_midi(np.array([config.fmin_hz]))[0]))
    hi = int(np.ceil(hz_to_midi(np.array([config.fmax_hz]))[0]))
    lo = max(0, lo)
    hi = min(127, hi)
    if hi <= lo:
        raise ValueError(f"Empty pitch range: MIDI {lo}..{hi}")
    return lo, hi


def build_transition_matrix(n_pitches: int, config: TranscriptionConfig) -> np.ndarray:
    """Log transition matrix of shape (S, S) with S = n_pitches + 1.

    State 0 is unvoiced; states 1..n_pitches are ascending semitones.
    """
    size = n_pitches + 1
    log_a = np.full((size, size), _NEG_INF, dtype=np.float64)

    # --- from unvoiced ---
    log_a[0, 0] = np.log(config.p_unvoiced_self)
    # Entering a note from silence is uniform over pitch: silence carries no
    # information about what pitch comes next.
    log_a[0, 1:] = np.log(1.0 - config.p_unvoiced_self) - np.log(n_pitches)

    # --- from a pitch state ---
    p_change = 1.0 - config.p_self - config.p_voiced_to_unvoiced
    if p_change <= 0:
        raise ValueError("p_self + p_voiced_to_unvoiced must be < 1")

    pitches = np.arange(n_pitches)
    interval = np.abs(pitches[:, None] - pitches[None, :]).astype(np.float64)

    # Melodic interval prior: small steps are likelier than large leaps, which
    # discourages the octave jumps that pitch trackers occasionally produce.
    weights = np.exp(-interval / config.interval_tau_semitones)
    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(row_sums, _EPS)

    with np.errstate(divide="ignore"):
        log_change = np.log(p_change) + np.log(np.maximum(weights, _EPS))
    np.fill_diagonal(log_change, np.log(config.p_self))

    log_a[1:, 1:] = log_change
    log_a[1:, 0] = np.log(config.p_voiced_to_unvoiced)

    return log_a


def build_emission_matrix(
    track: F0Track, pitch_lo: int, n_pitches: int, config: TranscriptionConfig
) -> np.ndarray:
    """Log emission matrix of shape (T, S).

    A voiced state scores ``log(voicing)`` discounted by the squared distance in
    semitones between the measured pitch and that state. The unvoiced state
    scores ``log(1 - voicing)``. Because the Gaussian term peaks at zero, a
    perfectly matching pitch state and the unvoiced state trade off exactly at
    voicing = 0.5, which makes ``unvoiced_offset`` an interpretable dial.
    """
    measured = track.midi_float                      # (T,), NaN when unvoiced
    voicing = np.clip(track.voicing, 0.0, 1.0)

    log_voiced = np.log(voicing + _EPS)
    log_unvoiced = np.log(1.0 - voicing + _EPS) + config.unvoiced_offset

    pitches = pitch_lo + np.arange(n_pitches, dtype=np.float64)
    # Frames with no f0 get a flat (uninformative) pitch profile; their very low
    # log_voiced term already rules the voiced states out.
    filled = np.where(np.isfinite(measured), measured, pitches[0])
    deviation = (filled[:, None] - pitches[None, :]) / config.pitch_sigma_semitones
    gaussian = -0.5 * deviation ** 2
    gaussian = np.where(np.isfinite(measured)[:, None], gaussian, 0.0)

    emission = np.empty((measured.shape[0], n_pitches + 1), dtype=np.float64)
    emission[:, 0] = log_unvoiced
    emission[:, 1:] = log_voiced[:, None] + gaussian
    return emission


def viterbi(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Standard Viterbi decode. Returns the (T,) array of state indices."""
    n_frames, n_states = log_emission.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int32)

    # Start in silence with probability 1/2, or in any pitch with the rest
    # spread uniformly -- most recordings begin before the first note.
    log_init = np.empty(n_states, dtype=np.float64)
    log_init[0] = np.log(0.5)
    log_init[1:] = np.log(0.5) - np.log(n_states - 1)

    delta = log_init + log_emission[0]
    backpointers = np.zeros((n_frames, n_states), dtype=np.int32)

    for t in range(1, n_frames):
        scores = delta[:, None] + log_transition      # (S_prev, S_next)
        best_prev = np.argmax(scores, axis=0)
        delta = scores[best_prev, np.arange(n_states)] + log_emission[t]
        backpointers[t] = best_prev

        # Renormalise to keep the running log-likelihood away from underflow on
        # long recordings; this does not change the argmax path.
        delta -= delta.max()

    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(delta))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return path


def path_to_segments(path: np.ndarray, pitch_lo: int) -> List[Segment]:
    """Group a decoded state path into contiguous voiced segments."""
    segments: List[Segment] = []
    if path.size == 0:
        return segments

    # Boundaries wherever the state index changes.
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [path.size]))

    for start, end in zip(starts, ends):
        state = int(path[start])
        if state == 0:          # unvoiced
            continue
        segments.append(
            Segment(start_frame=int(start), end_frame=int(end), pitch=pitch_lo + state - 1)
        )
    return segments


def segment_notes(track: F0Track, config: TranscriptionConfig) -> List[Segment]:
    """Decode an f0 track into pitch segments."""
    pitch_lo, pitch_hi = pitch_range(config)
    n_pitches = pitch_hi - pitch_lo + 1

    log_emission = build_emission_matrix(track, pitch_lo, n_pitches, config)
    log_transition = build_transition_matrix(n_pitches, config)
    path = viterbi(log_emission, log_transition)
    return path_to_segments(path, pitch_lo)
