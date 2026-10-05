"""Note segmentation by Viterbi decoding over a pitch-state grid.

The state space is a grid of pitches across the vocal range, a fraction of a
semitone apart, plus a single unvoiced state. Emissions score how well each
frame's measured f0 matches a given pitch; transitions make staying in a note
cheap and changing note expensive. Decoding the maximum-likelihood path therefore turns a jittery,
continuous f0 contour into piecewise-constant note events.

This follows the note-tracking design of the Tony melody transcription system
(Mauch, Cannam, Bittner, Fazekas, Salamon, Dai, Bello & Dixon, "Computer-aided
melody note transcription using the Tony software", 2015): a sub-semitone grid
of pitch states (three per semitone by default, as in Tony), simplified by
leaving out Tony's explicit attack/sustain sub-states. Vibrato is absorbed by
the emission width and the self-transition bias instead; note starts are then
sharpened against spectral-flux onsets (see ``onsets.py``).

The sub-semitone grid matters more than it looks. With one state per semitone
a note sung near the midpoint of two semitones -- ordinary for a singer with no
reference pitch, or with a quarter tone of intonation scatter -- is equally well
explained by both, and its vibrato flips the decoder between them several times
a second. On a finer grid such a note has a state of its own to sit on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    # Where to start measuring the pitch, if not at the start: an attack
    # later than the note's start, which snapping found but did not move to.
    measure_from: Optional[int] = None

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


def build_transition_matrix(n_pitches: int, config: TranscriptionConfig,
                            steps: int = 1) -> np.ndarray:
    """Log transition matrix of shape (S, S) with S = n_pitches + 1.

    State 0 is unvoiced; states 1..n_pitches are ascending pitches, ``steps``
    to the semitone.

    With ``steps`` above one the grid is finer than the musical question being
    asked, and the transition model is written to keep it that way. Which
    *semitone* comes next carries the interval prior exactly as on a one-state
    grid; *where within* that semitone the new note sits is not something a
    prior has an opinion on, so every placement is priced like the semitone
    itself rather than sharing its probability. Sharing it would make a note
    change ``log(steps)`` dearer than on the coarse grid, and the decoder
    measurably stickier: it would sooner hold a compromise state between two
    notes 0.9 semitones apart than pay to change. Rows then sum to more than
    one, which Viterbi -- a maximisation, not a sum -- does not mind. Changes
    smaller than ``min_note_interval_semitones`` are forbidden, so intonation
    drift inside a note cannot split it.
    """
    size = n_pitches + 1
    log_a = np.full((size, size), _NEG_INF, dtype=np.float64)
    steps = max(int(steps), 1)
    placement = np.log(steps)

    # --- from unvoiced ---
    log_a[0, 0] = np.log(config.p_unvoiced_self)
    # Entering a note from silence is uniform over pitch: silence carries no
    # information about what pitch comes next.
    log_a[0, 1:] = (np.log(1.0 - config.p_unvoiced_self) - np.log(n_pitches)
                    + placement)

    # --- from a pitch state ---
    p_change = 1.0 - config.p_self - config.p_voiced_to_unvoiced
    if p_change <= 0:
        raise ValueError("p_self + p_voiced_to_unvoiced must be < 1")

    pitches = np.arange(n_pitches)
    interval = np.abs(pitches[:, None] - pitches[None, :]).astype(np.float64) / steps

    # Melodic interval prior: small steps are likelier than large leaps, which
    # discourages the octave jumps that pitch trackers occasionally produce.
    weights = np.exp(-interval / config.interval_tau_semitones)
    np.fill_diagonal(weights, 0.0)
    if steps > 1:
        weights[interval < config.min_note_interval_semitones - 1e-9] = 0.0
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / np.maximum(row_sums, _EPS)

    with np.errstate(divide="ignore"):
        log_change = np.log(p_change) + np.log(np.maximum(weights, _EPS)) + placement
    log_change[weights <= 0] = _NEG_INF
    np.fill_diagonal(log_change, np.log(config.p_self))

    log_a[1:, 1:] = log_change
    log_a[1:, 0] = np.log(config.p_voiced_to_unvoiced)

    return log_a


def build_emission_matrix(
    track: F0Track, pitch_lo: int, n_pitches: int, config: TranscriptionConfig,
    tuning_offset: float = 0.0, steps: int = 1,
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

    # The state centres move with the recording's own pitch centre. Leaving
    # them on the A440 grid when the performance is not on it puts every frame
    # the same distance off its correct state, which both weakens that state's
    # emission and narrows the margin over its neighbour -- so a performance
    # near the quarter-tone boundary does not merely round the wrong way, it
    # decodes unstably, flipping between two states inside one held note.
    pitches = (pitch_lo + np.arange(n_pitches, dtype=np.float64) / max(int(steps), 1)
               + tuning_offset)
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


def path_to_segments(path: np.ndarray, pitch_lo: int, steps: int = 1) -> List[Segment]:
    """Group a decoded state path into contiguous voiced segments.

    A segment is labelled with the semitone its state lies nearest. That label
    only steers the stages between here and pitch measurement, which re-reads
    every note's pitch from its own frames."""
    steps = max(int(steps), 1)
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
        pitch = pitch_lo + int(np.floor((state - 1) / steps + 0.5))
        segments.append(Segment(start_frame=int(start), end_frame=int(end), pitch=pitch))
    return segments


def segment_notes(track: F0Track, config: TranscriptionConfig,
                  tuning_offset: float = 0.0) -> List[Segment]:
    """Decode an f0 track into pitch segments.

    ``tuning_offset`` shifts the pitch states onto the recording's own centre;
    see ``f0.estimate_tuning_offset``. It moves where the states sit, not which
    states exist, so a segment's integer pitch is unchanged in meaning.
    """
    pitch_lo, pitch_hi = pitch_range(config)
    steps = config.pitch_states_per_semitone
    n_pitches = (pitch_hi - pitch_lo) * steps + 1

    log_emission = build_emission_matrix(track, pitch_lo, n_pitches, config,
                                         tuning_offset, steps)
    log_transition = build_transition_matrix(n_pitches, config, steps)
    path = viterbi(log_emission, log_transition)
    return path_to_segments(path, pitch_lo, steps)
