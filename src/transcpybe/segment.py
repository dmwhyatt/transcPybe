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

The finer grid is for the decoder's benefit only. Segments are still cut where
the *semitone* changes, not where the state does, so a held note that drifts
from one of its semitone's states to another remains one note (see
``path_to_segments``).
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
    """A contiguous run of voiced frames decoded to the same semitone."""

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


def _log_initial(n_states: int) -> np.ndarray:
    """Start in silence with probability 1/2, or in any pitch with the rest
    spread uniformly -- most recordings begin before the first note."""
    log_init = np.empty(n_states, dtype=np.float64)
    log_init[0] = np.log(0.5)
    log_init[1:] = np.log(0.5) - np.log(n_states - 1)
    return log_init


def viterbi(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """Standard Viterbi decode. Returns the (T,) array of state indices."""
    n_frames, n_states = log_emission.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int32)

    delta = _log_initial(n_states) + log_emission[0]
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


@dataclass
class ChangeCost:
    """The note changes of a transition matrix, as a distance cost.

    ``build_transition_matrix`` prices a change from pitch state ``i`` to
    ``j`` as ``row[i] - slope * |i - j|`` when the two are at least ``band``
    states apart, and forbids it when they are closer. Each frame's best
    change into every state is then a one-dimensional distance transform --
    a running maximum up the states and another down them -- rather than a
    maximum over every pair of states.
    """

    row: np.ndarray     # (n_pitches,) the per-row part of the cost
    slope: float        # cost per state of distance
    band: int           # nearest pitch state a change may reach


def change_cost(log_transition: np.ndarray) -> Optional[ChangeCost]:
    """Read a transition matrix's change cost from the matrix itself.

    Returns None unless the matrix has exactly that shape -- one cost for
    staying on a pitch, one for leaving it to silence, one for entering any
    pitch from silence, and changes priced as above -- so a decode that uses
    the result cannot disagree with the matrix it was read from. A matrix built
    with its weights on the ``_EPS`` floor, or any other matrix, is decoded
    densely instead.
    """
    block = log_transition[1:, 1:]
    n = block.shape[0]
    stay = np.diag(block)
    if (n < 2 or not np.all(stay == stay[0])
            or not np.all(log_transition[0, 1:] == log_transition[0, 1])
            or not np.all(log_transition[1:, 0] == log_transition[1, 0])):
        return None

    allowed = block[0, 1:] > _NEG_INF / 2
    if not allowed.any():
        return None
    band = int(np.argmax(allowed)) + 1
    if band + 1 >= n:
        return None
    slope = float(block[0, band] - block[0, band + 1])

    states = np.arange(n)
    nearest = np.where(states + band < n, states + band, states - band)
    row = block[states, nearest] + slope * band

    distance = np.abs(states[:, None] - states[None, :])
    change = distance >= band
    priced = row[:, None] - slope * distance
    if not (np.allclose(priced[change], block[change], rtol=0.0, atol=1e-9)
            and np.all(block[(distance > 0) & ~change] <= _NEG_INF / 2)):
        return None
    return ChangeCost(row=row, slope=slope, band=band)


def viterbi_banded(log_emission: np.ndarray, log_transition: np.ndarray,
                   cost: ChangeCost) -> np.ndarray:
    """Viterbi decode for a transition matrix of the shape ``ChangeCost`` reads.

    The path ``viterbi`` finds, in time linear in the number of states rather
    than quadratic. Three states to the semitone tripled the states, and a
    dense decode is a maximum over every pair of them every frame, which made
    it the one stage of segmentation whose cost the finer grid multiplied:
    about three times slower. Here each frame's best change into every state
    is a running maximum, and only the scores are kept going forwards: the
    backtrace recovers each frame's predecessor from the matrix, for the one
    state the path passes through.

    The scores are summed in a different order from the dense decode's, so
    they can differ in the last bit. That only matters where two paths are
    exactly equally likely -- a contour passing exactly between two states,
    which a synthetic glide can do -- and then either path may come back.
    Both are equally the most likely one. On the realistic corpus's 160
    recordings, decoded at two tuning offsets, the paths are identical.
    """
    n_frames, n_states = log_emission.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.int32)

    band = cost.band
    ramp = cost.slope * np.arange(n_states - 1)
    from_below = cost.row + ramp
    from_above = cost.row - ramp
    stay_unvoiced = log_transition[0, 0]
    voice = log_transition[0, 1]
    unvoice = log_transition[1, 0]
    stay = log_transition[1, 1]

    scores = np.empty((n_frames, n_states), dtype=np.float64)
    scores[0] = _log_initial(n_states) + log_emission[0]
    for t in range(1, n_frames):
        previous, pitched = scores[t - 1], scores[t - 1, 1:]
        current = scores[t]
        best = current[1:]
        np.add(pitched, stay, out=best)
        np.maximum(best, previous[0] + voice, out=best)
        # A change up from at least ``band`` states below, then down from at
        # least ``band`` above.
        below = np.maximum.accumulate(pitched + from_below)
        np.maximum(best[band:], below[:-band] - ramp[band:], out=best[band:])
        above = np.maximum.accumulate((pitched + from_above)[::-1])[::-1]
        np.maximum(best[:-band], above[band:] + ramp[:-band], out=best[:-band])
        current[0] = max(previous[0] + stay_unvoiced, pitched.max() + unvoice)
        current += log_emission[t]
        current -= current.max()

    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(scores[-1]))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = int(np.argmax(scores[t - 1] + log_transition[:, path[t]]))
    return path


def path_to_segments(path: np.ndarray, pitch_lo: int, steps: int = 1) -> List[Segment]:
    """Group a decoded state path into contiguous voiced segments.

    A segment is labelled with the semitone its state lies nearest, and is cut
    where that semitone changes or voicing stops -- not wherever the state
    does. With more than one state per semitone the decoder may move between
    two states of the same semitone inside one held note: at three to the
    semitone, 60.67 to 61.33 is a change of two thirds, which
    ``min_note_interval_semitones`` allows, and both are 61. Cut there, the
    note became two contiguous segments of one pitch, and the boundary between
    them was then treated as a note start -- snapped onto any weak onset
    nearby, which merging afterwards read as a re-articulation. A genuine
    re-articulation on one semitone is no business of the decoder's either
    way: it has no pitch change to see, and the onset stage splits it.

    The label only steers the stages between here and pitch measurement, which
    re-reads every note's pitch from its own frames."""
    steps = max(int(steps), 1)
    segments: List[Segment] = []
    if path.size == 0:
        return segments

    # Every frame's semitone, with -1 (no MIDI number) for unvoiced frames, so
    # a boundary falls wherever the semitone changes or voicing starts or stops.
    labels = np.where(path > 0,
                      pitch_lo + np.floor((path - 1) / steps + 0.5).astype(np.int64),
                      -1)
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [path.size]))

    for start, end in zip(starts, ends):
        pitch = int(labels[start])
        if pitch < 0:           # unvoiced
            continue
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
    cost = change_cost(log_transition)
    if cost is None:
        path = viterbi(log_emission, log_transition)
    else:
        path = viterbi_banded(log_emission, log_transition, cost)
    return path_to_segments(path, pitch_lo, steps)
