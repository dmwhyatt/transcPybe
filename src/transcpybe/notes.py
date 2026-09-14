"""Turning decoded segments into final MIDI notes.

Two jobs live here: measuring each segment's pitch from the f0 contour, and
cleaning up the note list afterwards.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .config import TranscriptionConfig
from .f0 import F0Track
from .onsets import OnsetTrack
from .segment import Segment


@dataclass
class Note:
    """A transcribed note event."""

    onset: float          # seconds
    offset: float         # seconds
    pitch: int            # integer MIDI note number, 0-127
    velocity: int         # MIDI velocity, 1-127
    confidence: float     # mean voicing probability across the measured frames
    loudness_db: float    # 75th-percentile frame level, in dBFS
    pitch_range_semitones: float  # f0 spread across measured frames (diagnostic)
    pitch_hz: float       # median measured f0 in Hz
    pitch_midi_float: float   # unrounded measured pitch, for tuning analysis
    hmm_pitch: int        # pitch the Viterbi decoder assigned, before remeasuring
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Weighted quantile of ``values``, with ``quantile`` in [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")

    values, weights = values[finite], weights[finite]
    order = np.argsort(values)
    values, weights = values[order], weights[order]

    cumulative = np.cumsum(weights)
    cutoff = cumulative[-1] * float(np.clip(quantile, 0.0, 1.0))
    index = int(np.searchsorted(cumulative, cutoff))
    return float(values[min(index, values.shape[0] - 1)])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``values``.

    Preferred over a weighted mean because a handful of octave-error frames
    cannot drag the result: the median only cares about which side of the
    midpoint the weight sits on.
    """
    return weighted_quantile(values, weights, 0.5)


def _measure_frames(seg: Segment, track: F0Track, config: TranscriptionConfig) -> np.ndarray:
    """Frame indices to use when measuring a segment's pitch.

    The head of the note is skipped, because singers commonly scoop into a
    pitch and because the consonant that starts a syllable produces unreliable
    f0. Falls back progressively so that a measurement is always possible.
    """
    hop_s = config.hop_length / config.sample_rate
    max_trim = int(round(config.attack_trim_max_s / hop_s))
    trim = min(int(seg.n_frames * config.attack_trim_frac), max_trim)
    trim = min(trim, max(seg.n_frames - 1, 0))

    indices = np.arange(seg.start_frame + trim, seg.end_frame)
    if indices.size == 0:
        indices = np.arange(seg.start_frame, seg.end_frame)

    confident = indices[track.voicing[indices] >= config.voicing_threshold]
    if confident.size > 0:
        return confident

    # No confident frame survived the trim: relax, then relax again.
    all_frames = np.arange(seg.start_frame, seg.end_frame)
    confident = all_frames[track.voicing[all_frames] >= config.voicing_threshold]
    return confident if confident.size > 0 else all_frames


def _note_level_db(loudness_db: np.ndarray) -> float:
    """A note's representative level: the 75th percentile of its frame levels.

    A percentile rather than the mean, so that the quiet attack and release
    tails do not drag a fully sung note's level down.
    """
    finite = loudness_db[np.isfinite(loudness_db)]
    return float(np.percentile(finite, 75)) if finite.size else float("nan")


def _velocity(note_db: float, reference_db: float, config: TranscriptionConfig) -> int:
    """Map a note's loudness onto the configured velocity range."""
    if not config.velocity_from_loudness or not np.isfinite(note_db):
        return int(np.clip(config.velocity_constant, 1, 127))

    # Spread the configured dynamic range across the velocity range.
    fraction = (note_db - (reference_db - config.loudness_dynamic_range_db)) / \
               max(config.loudness_dynamic_range_db, 1e-6)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    velocity = config.velocity_min + fraction * (config.velocity_max - config.velocity_min)
    return int(np.clip(round(velocity), 1, 127))


def segments_to_notes(
    segments: List[Segment], track: F0Track, config: TranscriptionConfig
) -> List[Note]:
    """Measure pitch, timing, confidence and velocity for each segment."""
    hop_s = config.hop_length / config.sample_rate
    n_frames = track.n_frames

    finite_loudness = track.loudness_db[np.isfinite(track.loudness_db)]
    reference_db = (
        float(np.percentile(finite_loudness, config.loudness_percentile))
        if finite_loudness.size else -20.0
    )

    notes: List[Note] = []
    for seg in segments:
        indices = _measure_frames(seg, track, config)
        if indices.size == 0:
            continue

        measured = weighted_median(track.midi_float[indices], track.voicing[indices])
        if not np.isfinite(measured):
            # No usable f0 anywhere in the segment; trust the decoder's state.
            measured = float(seg.pitch)

        pitch = int(np.clip(round(measured), 0, 127))
        voiced = track.voicing[indices]
        hz = track.f0_hz[indices]
        hz_valid = hz[np.isfinite(hz)]
        note_db = _note_level_db(track.loudness_db[indices])
        measured_pitches = track.midi_float[indices]
        measured_pitches = measured_pitches[np.isfinite(measured_pitches)]
        spread = (
            float(np.percentile(measured_pitches, 95) - np.percentile(measured_pitches, 5))
            if measured_pitches.size >= 3 else 0.0
        )

        notes.append(
            Note(
                onset=seg.start_frame * hop_s,
                offset=min(seg.end_frame, n_frames) * hop_s,
                pitch=pitch,
                velocity=_velocity(note_db, reference_db, config),
                confidence=float(np.mean(voiced)) if voiced.size else 0.0,
                loudness_db=note_db,
                pitch_range_semitones=spread,
                pitch_hz=float(np.median(hz_valid)) if hz_valid.size else float("nan"),
                pitch_midi_float=float(measured),
                hmm_pitch=int(seg.pitch),
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
            )
        )
    return notes


def merge_same_pitch(
    notes: List[Note],
    track: F0Track,
    config: TranscriptionConfig,
    onsets: Optional[OnsetTrack] = None,
) -> List[Note]:
    """Rejoin consecutive notes of equal pitch separated by only a brief gap.

    A short unvoiced dropout mid-note (a breathy patch, a consonant) otherwise
    splits one sung note into two. A detected onset in the gap is taken as
    evidence that the singer really did re-articulate, and blocks the merge.
    """
    if len(notes) < 2:
        return notes

    onset_times = onsets.times if onsets is not None else np.zeros(0)

    merged: List[Note] = [notes[0]]
    for note in notes[1:]:
        previous = merged[-1]
        gap = note.onset - previous.offset

        rearticulated = bool(
            onset_times.size
            and np.any(
                (onset_times > previous.offset - 1e-9)
                & (onset_times < note.onset + 1e-9)
            )
        )

        if (
            note.pitch == previous.pitch
            and 0 <= gap <= config.merge_same_pitch_gap_s
            and not rearticulated
        ):
            combined = Segment(
                start_frame=previous.start_frame,
                end_frame=note.end_frame,
                pitch=previous.hmm_pitch,
            )
            remeasured = segments_to_notes([combined], track, config)
            if remeasured:
                merged[-1] = remeasured[0]
                continue

        merged.append(note)

    return merged


def trim_note_releases(
    notes: List[Note], track: F0Track, config: TranscriptionConfig
) -> List[Note]:
    """Pull each note's end back to where the note's own level collapses.

    This is the only stage that touches a note's end. Everything else --
    fragment absorption, glide backtracking, onset splitting, start snapping --
    refines note starts, and the end was simply wherever the decoder left the
    pitch state. The adaptive gate holds that state open until a frame is
    ``loudness_dynamic_range_db`` below the whole performance, which is a good
    threshold for "is anyone singing" and much too generous for "is this note
    still sounding". A note with a half-second release is reported half a
    second long.

    So: find the note's sustained level, walk back from its end while frames
    sit below that level by ``release_drop_db``, and end the note there. The
    threshold is per-note because a quiet note's release is quiet too, and it
    is small because what it has to find is where the level *departed* from
    sustain rather than where it became inaudible. A threshold set further down
    the ramp lands at a fixed fraction of the release instead of at its top, so
    the error it leaves grows with the release and long ones overshoot whatever
    value is chosen.

    Two things this deliberately does not do. It never lengthens a note, only
    shortens one, so a decoder that already ended a note early is left alone.
    And it does not re-measure pitch: the release frames were already the
    least confident in the note and the weighted median had all but discarded
    them, so the measurement stands and only the reported end moves.
    """
    if not config.trim_note_releases or not notes:
        return notes

    hop_s = config.hop_length / config.sample_rate
    min_frames = max(1, int(round(config.min_note_duration_s / hop_s)))

    trimmed: List[Note] = []
    for note in notes:
        start = note.start_frame
        end = min(note.end_frame, track.n_frames)
        levels = track.loudness_db[start:end]
        finite = levels[np.isfinite(levels)]
        if finite.size < 3 or end - start <= min_frames:
            trimmed.append(note)
            continue

        floor = float(np.percentile(finite, 75)) - config.release_drop_db
        # Never take more than release_min_fraction of the note: a note that
        # fades throughout sits below its own 75th percentile for most of its
        # length, and is a diminuendo rather than a release.
        keep_at_least = max(min_frames,
                            int(round((end - start) * config.release_min_fraction)))
        limit = start + keep_at_least

        cut = end
        while cut - 1 >= limit:
            level = levels[cut - 1 - start]
            if not np.isfinite(level) or level >= floor:
                break
            cut -= 1

        if cut < end:
            trimmed.append(dataclasses.replace(
                note, offset=cut * hop_s, end_frame=cut))
        else:
            trimmed.append(note)

    return trimmed


def drop_short_notes(notes: List[Note], config: TranscriptionConfig) -> List[Note]:
    """Discard notes too short to be a deliberate sung pitch."""
    return [n for n in notes if n.duration >= config.min_note_duration_s]


def drop_quiet_notes(notes: List[Note], config: TranscriptionConfig) -> List[Note]:
    """Discard notes far quieter than the rest of the performance.

    This is the defence against the dominant failure mode on source-separated
    input. Separation leaves pitched residue -- bleed from other instruments,
    reverb tails, artefacts -- in passages where the singer is silent, and a
    neural pitch tracker reports it with high confidence, because it genuinely
    is periodic. Periodicity therefore cannot tell the two apart; level can, by
    a wide margin, and aggregating level over a whole note is far more reliable
    than testing frames one at a time.

    The reference is taken across notes rather than across frames so that a
    sparse vocal with long rests is judged against its own sung notes.
    """
    if not config.reject_quiet_notes or len(notes) < 2:
        return notes

    levels = np.array([n.loudness_db for n in notes], dtype=np.float64)
    durations = np.array([max(n.duration, 1e-6) for n in notes], dtype=np.float64)
    if np.isfinite(levels).sum() < 2:
        return notes

    reference = weighted_quantile(
        levels, durations, config.note_level_reference_percentile / 100.0
    )
    if not np.isfinite(reference):
        return notes

    threshold = reference - config.note_level_floor_db
    return [n for n in notes if not np.isfinite(n.loudness_db) or n.loudness_db >= threshold]


def postprocess_notes(
    notes: List[Note],
    track: F0Track,
    config: TranscriptionConfig,
    onsets: Optional[OnsetTrack] = None,
) -> List[Note]:
    """Merge fragments first, then trim releases, then drop what is too short.

    Order matters twice over. Merging can rescue a fragment that would not
    survive the duration filter on its own, so it goes first -- and it must see
    untrimmed ends, because the gap it measures is the decoder's silence
    between two notes, not the silence a release trim would manufacture.
    Trimming then goes before the duration filter rather than after it, so that
    a note which is only long enough because of its release is judged on the
    length it is actually reported with.
    """
    notes = merge_same_pitch(notes, track, config, onsets)
    notes = trim_note_releases(notes, track, config)
    notes = drop_short_notes(notes, config)
    notes = drop_quiet_notes(notes, config)
    return sorted(notes, key=lambda n: (n.onset, n.pitch))
