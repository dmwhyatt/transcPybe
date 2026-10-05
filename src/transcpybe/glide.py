"""Handling portamento: glides between sung notes.

Singers slide between pitches rather than stepping between them, and that
breaks a pitch-state decoder in two distinct ways:

1. **Late onsets.** The decoder switches state only once the f0 has arrived at
   the new pitch, so a 150 ms glide puts the note start 150 ms late. Reference
   annotations place the onset where the singer *left* the previous note.
2. **Passing tones.** A glide across a wide interval lingers briefly in each
   semitone it crosses, and the decoder emits those as real notes.

Both are corrected here, after decoding and before pitch measurement, by
recognising the transition region between two note levels and attributing it to
the note being approached.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import TranscriptionConfig
from .f0 import F0Track
from .segment import Segment


def _smoothed_midi(track: F0Track, width: int = 3) -> np.ndarray:
    """Running median of the pitch contour, interpolated across unvoiced gaps.

    Smoothing matters here because the backtrack below walks frame by frame;
    raw jitter would stop the walk early. A median rather than a mean, because
    a mean smears a pitch step backwards: on an instantaneous step a five-frame
    mean starts moving two frames before the step does, and the walk follows it
    there, putting every legato onset 20 ms early -- two fifths of the onset
    tolerance spent before anything else has gone wrong. A median holds a step
    where it is and still removes single-frame jitter.
    """
    from scipy.ndimage import median_filter

    midi = track.midi_float
    valid = np.isfinite(midi)
    if not valid.any():
        return midi

    indices = np.arange(midi.shape[0])
    filled = np.interp(indices, indices[valid], midi[valid])
    smoothed = median_filter(filled, size=width, mode="nearest")
    return np.where(valid, smoothed, np.nan)


def absorb_passing_tones(
    segments: List[Segment], config: TranscriptionConfig
) -> List[Segment]:
    """Delete short segments that are merely a glide crossing a semitone.

    A segment qualifies only if it is brief, contiguous with both neighbours,
    strictly between their pitches, and shorter than both of them -- so a real
    short note in a melisma is not mistaken for a passing tone.
    """
    if len(segments) < 3:
        return segments

    hop_s = config.hop_length / config.sample_rate
    max_frames = int(round(config.max_passing_note_s / hop_s))
    # A glide can briefly drop below the voicing threshold, so neighbours are
    # allowed to be separated by a short gap rather than exactly contiguous --
    # but only a short one. A note with a rest on each side is not the trace
    # of a slide between its neighbours, however short it is.
    max_gap = int(round(config.passing_max_gap_s / hop_s))

    keep = [True] * len(segments)
    for i in range(1, len(segments) - 1):
        previous, current, following = segments[i - 1], segments[i], segments[i + 1]

        if not keep[i - 1]:
            continue
        if current.n_frames > max_frames:
            continue
        if not (0 <= current.start_frame - previous.end_frame <= max_gap):
            continue          # not a contiguous legato run
        if not (0 <= following.start_frame - current.end_frame <= max_gap):
            continue
        if not (min(previous.pitch, following.pitch) < current.pitch < max(previous.pitch, following.pitch)):
            continue          # not between the neighbours
        if current.n_frames >= previous.n_frames or current.n_frames >= following.n_frames:
            continue          # as long as a neighbour: treat it as a real note

        keep[i] = False
        if current.start_frame == previous.end_frame:
            # Contiguous: hand the frames to the preceding note, and let the
            # backtrack below move the boundary to where the glide began.
            segments[i - 1] = Segment(previous.start_frame, current.end_frame, previous.pitch)
        # Separated by a gap: simply drop the passing tone. Extending the
        # previous note across that gap would falsify its offset.

    return [seg for seg, alive in zip(segments, keep) if alive]


def backtrack_glide_onsets(
    segments: List[Segment], track: F0Track, config: TranscriptionConfig
) -> List[Segment]:
    """Move legato note starts back to the beginning of the pitch transition.

    From each boundary, walk backwards while the contour is still travelling
    between the two note levels, stopping once it has returned to the previous
    note's pitch or has stopped moving. Bounded by ``glide_max_backtrack_s`` so
    that a wobbling sustain cannot swallow the note before it.

    "The previous note's pitch" is the pitch it was sung at, measured from its
    frames, not the integer state the decoder gave it. The two differ by up to
    half a semitone, and when the difference points towards the next note the
    held part of the previous note sits inside the corridor between the two
    without ever coming within ``glide_edge_tolerance_semitones`` of the
    integer: the walk then marched back through the held note to its full
    300 ms limit. A legato phrase with alternate notes 40 cents sharp lost
    three onsets in four that way. The plateau test is the second guard: a
    glide moves, so ``glide_plateau_frames`` frames that hold one pitch mean
    the walk has left it, wherever the pitch they hold.

    The walk may cross a short unvoiced gap: a rapid glide drives the pitch
    tracker's confidence down, so the decoder often inserts spurious silence in
    the middle of exactly the region being traversed.
    """
    if len(segments) < 2:
        return segments

    hop_s = config.hop_length / config.sample_rate
    max_backtrack = int(round(config.glide_max_backtrack_s / hop_s))
    max_gap = int(round(config.glide_max_gap_s / hop_s))
    min_frames = max(1, int(round(config.min_note_duration_s / hop_s)))
    tolerance = config.glide_edge_tolerance_semitones
    plateau_frames = max(int(config.glide_plateau_frames), 2)
    smoothed = _smoothed_midi(track)
    measured = track.midi_float

    def pitch_of(seg: Segment) -> float:
        """The pitch a segment was sung at: the median of its frames."""
        frames = measured[seg.start_frame:seg.end_frame]
        finite = frames[np.isfinite(frames)]
        return float(np.median(finite)) if finite.size else float(seg.pitch)

    def level_of(seg: Segment) -> float:
        """Representative level of a segment, in dBFS."""
        frames = track.loudness_db[seg.start_frame:seg.end_frame]
        finite = frames[np.isfinite(frames)]
        return float(np.percentile(finite, 75)) if finite.size else float("nan")

    out = list(segments)
    for i in range(1, len(out)):
        previous, current = out[i - 1], out[i]

        gap = current.start_frame - previous.end_frame
        if not (0 <= gap <= max_gap):
            continue                                    # real silence: a real onset
        interval = abs(current.pitch - previous.pitch)
        if interval < 1:
            continue

        previous_pitch, current_pitch = pitch_of(previous), pitch_of(current)
        low = min(previous_pitch, current_pitch) - tolerance
        high = max(previous_pitch, current_pitch) + tolerance

        # A glide is sung, so it sits at roughly the level of the note it leads
        # into. Requiring that stops the walk from marching back through
        # separation residue, whose wandering pitch can drift through the
        # corridor between two notes and drag the onset far too early.
        current_level = level_of(current)
        min_level = (current_level - config.glide_max_level_drop_db
                     if np.isfinite(current_level) else -np.inf)

        limit = max(previous.start_frame + min_frames, current.start_frame - max_backtrack)
        boundary = current.start_frame
        # Walking backwards, a real glide retreats steadily towards the pitch it
        # came from. Noise between notes wanders instead, and that is the cue
        # which separates the two: without it, the walk happily marches back
        # through a breathy gap and reports the onset far too early.
        towards_previous = 1.0 if previous_pitch < current_pitch else -1.0
        last_pitch = smoothed[boundary] if boundary < smoothed.shape[0] else np.nan
        visited: List[float] = []
        while boundary - 1 >= limit:
            pitch = smoothed[boundary - 1]
            if not np.isfinite(pitch):
                break
            if np.isfinite(last_pitch):
                step = (last_pitch - pitch) * towards_previous
                if step < -config.glide_monotonic_slack_semitones:
                    break
            # Gate on audibility, not on pitch confidence. Pitch trackers lose
            # confidence precisely during a fast glide, which is the region
            # this walk exists to cross.
            if track.loudness_gate[boundary - 1] <= config.glide_min_audibility:
                break
            if track.periodicity[boundary - 1] < config.glide_min_periodicity:
                break
            if track.loudness_db[boundary - 1] < min_level:
                break
            if abs(pitch - previous_pitch) <= tolerance:
                break                                   # back at the previous note
            if not (low <= pitch <= high):
                break                                   # left the transition corridor
            visited.append(pitch)
            if (len(visited) >= plateau_frames
                    and np.ptp(visited[-plateau_frames:]) < config.glide_plateau_semitones):
                # Held, not gliding: give back the plateau's frames.
                boundary += plateau_frames - 1
                break
            last_pitch = pitch
            boundary -= 1

        if boundary != current.start_frame and config.glide_onset_position > 0:
            # The walk found where the glide left the previous note; place the
            # onset that fraction of the way back towards where it arrived.
            boundary += int(round(config.glide_onset_position
                                  * (current.start_frame - boundary)))

        if boundary != current.start_frame:
            # Moving a start earlier may only ever shorten the note before it.
            # Writing the boundary in unconditionally would stretch the previous
            # note forward across the rest it was resting through.
            previous_end = min(previous.end_frame, boundary)
            out[i - 1] = Segment(previous.start_frame, previous_end, previous.pitch)
            out[i] = Segment(boundary, current.end_frame, current.pitch)

    return [seg for seg in out if seg.n_frames > 0]


def handle_glides(
    segments: List[Segment], track: F0Track, config: TranscriptionConfig
) -> List[Segment]:
    """Absorb passing tones, then align legato onsets to the glide start."""
    if not config.absorb_glides:
        return segments
    segments = absorb_passing_tones(segments, config)
    segments = backtrack_glide_onsets(segments, track, config)
    return segments
