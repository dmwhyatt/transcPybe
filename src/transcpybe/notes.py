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
    # The attack the pitch was measured from, when that is later than the
    # start (see Segment.measure_from); None when measured from the start.
    # Kept so that re-measuring the note, after a merge, measures it the same
    # way.
    measure_from_frame: Optional[int] = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Pipeline bookkeeping rather than something the note reports: the
        # written formats stay as they were.
        data.pop("measure_from_frame")
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


def _attack_frame(seg: Segment) -> Optional[int]:
    """The segment's ``measure_from``, if it lies inside the segment."""
    if seg.measure_from is not None and seg.start_frame < seg.measure_from < seg.end_frame:
        return int(seg.measure_from)
    return None


def _measure_frames(seg: Segment, track: F0Track, config: TranscriptionConfig) -> np.ndarray:
    """Frame indices to use when measuring a segment's pitch.

    The head of the note is skipped, because singers commonly scoop into a
    pitch and because the consonant that starts a syllable produces unreliable
    f0. Falls back progressively so that a measurement is always possible.
    """
    hop_s = config.hop_length / config.sample_rate
    attack = _attack_frame(seg)
    start = seg.start_frame if attack is None else attack
    n_frames = seg.end_frame - start
    max_trim = int(round(config.attack_trim_max_s / hop_s))
    trim = min(int(n_frames * config.attack_trim_frac), max_trim)
    trim = min(trim, max(n_frames - 1, 0))

    indices = np.arange(start + trim, seg.end_frame)
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
    segments: List[Segment], track: F0Track, config: TranscriptionConfig,
    tuning_offset: float = 0.0,
) -> List[Note]:
    """Measure pitch, timing, confidence and velocity for each segment.

    ``tuning_offset`` is the recording's own pitch centre, in semitones away
    from the A440 grid (see ``estimate_tuning_offset``). With
    ``pitch_reference='tuned'`` rounding happens relative to it, so a
    performance sung consistently a third of a semitone flat reports the notes
    it meant rather than the ones it was nearest to; with the default 'sung'
    each note is the nearest semitone to what was sung, which is what a
    reference annotated in sung pitch scores against. ``pitch_midi_float``
    stays as measured, in absolute MIDI, so the deviation a caller reads off it
    remains the real distance from A440.
    """
    grid = tuning_offset if config.pitch_reference == "tuned" else 0.0
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

        pitch = int(np.clip(round(measured - grid), 0, 127))
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
                measure_from_frame=_attack_frame(seg),
            )
        )
    return notes


def estimate_tuning_offset(
    notes: List[Note], config: TranscriptionConfig
) -> float:
    """How far this performance's pitch centre sits from the A440 semitone grid.

    Returned in semitones, in [-0.5, 0.5]; zero means leave the grid alone.

    Both the decoder's pitch states and the rounding that produces an integer
    MIDI number assume a note's centre lies on an integer. That is an
    assumption about the *recording*: a phone take sung with no reference
    pitch, or against an instrument tuned by ear, sits wherever it sits. Offset
    the whole performance by more than a quarter tone and every note is nearer
    the wrong semitone than the right one -- the transcription comes out a
    semitone adrift while the singing was perfectly in tune with itself.

    Measured across notes rather than across frames, which is the whole
    difficulty. A vibrato sweeps its frames across the entire deviation range
    several times a second, so a frame-weighted estimate is dominated by
    material that carries no tuning information at all: measured per frame,
    the suite's vibrato case reports a 49-cent offset where the true one is
    zero, which is the most damaging answer available. Nor can the sweep be
    filtered out by holding on to frames whose pitch is momentarily still --
    the still points of a vibrato are the extremes of its swing, so that
    selects precisely the frames furthest from the centre. Averaging inside
    each note first collapses the swing to the pitch the note was aiming at,
    and only then is there something to take a centre of.

    Weighted by duration and confidence, because a held, confident note says
    where the singer thinks a pitch is and a 60 ms fragment does not.

    The average is circular, not arithmetic: these deviations live on a circle,
    where a performance 49 cents sharp and one 49 cents flat are nearly the
    same performance sitting at opposite ends of [-0.5, 0.5]. Averaging those
    linearly returns zero, the one answer that is certainly wrong. The circular
    mean also yields a concentration R in [0, 1] for free, and that is what
    decides whether to act: it asks whether this performance has a pitch centre
    at all. Singing scattered evenly around the circle has none -- intonation
    that wanders by a quarter tone note to note is not an offset and must not
    be corrected as one -- and its mean angle is noise that would shift every
    note by an arbitrary amount.
    """
    if not config.estimate_tuning or len(notes) < config.tuning_min_notes:
        return 0.0

    values, weights = [], []
    for note in notes:
        if not np.isfinite(note.pitch_midi_float):
            continue
        weight = max(note.duration, 0.0) * max(note.confidence, 0.0)
        if weight <= 0.0:
            continue
        values.append(note.pitch_midi_float)
        weights.append(weight)

    if len(values) < config.tuning_min_notes:
        return 0.0

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0

    # Deviation from the nearest semitone, mapped onto a full turn so that the
    # wrap at +/-0.5 is a continuity rather than a cliff.
    theta = 2.0 * np.pi * (values - np.round(values))
    cosine = float(np.sum(weights * np.cos(theta)))
    sine = float(np.sum(weights * np.sin(theta)))

    concentration = float(np.hypot(cosine, sine) / total)
    if concentration < config.tuning_min_concentration:
        return 0.0

    offset = float(np.arctan2(sine, cosine) / (2.0 * np.pi))
    # Near the far side of the circle the reading is ambiguous rather than
    # merely uncertain, and the ambiguity resolves the wrong way: see
    # tuning_max_offset_semitones.
    if abs(offset) > config.tuning_max_offset_semitones:
        return 0.0
    return offset


def merge_same_pitch(
    notes: List[Note],
    track: F0Track,
    config: TranscriptionConfig,
    onsets: Optional[OnsetTrack] = None,
    tuning_offset: float = 0.0,
) -> List[Note]:
    """Rejoin consecutive notes of equal pitch separated by only a brief gap.

    A short unvoiced dropout mid-note (a breathy patch, a consonant) otherwise
    splits one sung note into two. A detected onset in the gap is taken as
    evidence that the singer really did re-articulate, and blocks the merge.
    So does an attack the second note carries just after its start
    (``measure_from_frame``): a start the decoder put on breath noise before
    the attack stays there when snapping only pulls starts earlier, and the
    attack that would have been in the gap is inside the note.
    """
    if len(notes) < 2:
        return notes

    onset_times = onsets.times if onsets is not None else np.zeros(0)

    merged: List[Note] = [notes[0]]
    for note in notes[1:]:
        previous = merged[-1]
        gap = note.onset - previous.offset

        # The attack snapping recorded after the start is the one that, snapped
        # both ways, would have become the start and so been found in the gap.
        rearticulated = note.measure_from_frame is not None or bool(
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
            # The merged note starts where the previous one did, so its pitch
            # is measured from wherever the previous one's was: from its
            # attack, when snapping left its start before one.
            combined = Segment(
                start_frame=previous.start_frame,
                end_frame=note.end_frame,
                pitch=previous.hmm_pitch,
                measure_from=previous.measure_from_frame,
            )
            # Measured on the same grid as every other note. Without the
            # offset, a merged note was rounded on A440 while its neighbours
            # were rounded on the recording's own grid.
            remeasured = segments_to_notes([combined], track, config, tuning_offset)
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
    sit more than ``release_drop_db`` below it, and end the note there. The
    threshold is per-note because a quiet note's release is quiet too. What it
    is after is where the level *departed* from sustain rather than where it
    became inaudible, but it cannot look for that just below sustain: a held
    note's level moves a long way without the note ending -- singers sag
    across a phrase and ripple with their vibrato, and reverberation beating
    against a moving pitch swings the level by 5 dB or more. A threshold
    inside that range cannot tell those from a release, and walks back
    through the note itself: at 1 dB it often ended notes half way through,
    and cost vocadito 0.16 COnPOff, worse on all forty recordings. The price
    of a deeper threshold is that it lands part of the way down the release
    rather than at its top, so a long release leaves more of itself in the
    note than a short one does. Where on that ramp a note's end belongs is an
    annotation convention, and the 15 dB default is the compromise across the
    conventions measured (see ``TranscriptionConfig.release_drop_db``).

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
    tuning_offset: float = 0.0,
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
    notes = merge_same_pitch(notes, track, config, onsets, tuning_offset)
    notes = trim_note_releases(notes, track, config)
    notes = drop_short_notes(notes, config)
    notes = drop_quiet_notes(notes, config)
    return sorted(notes, key=lambda n: (n.onset, n.pitch))
