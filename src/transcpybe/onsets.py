"""Spectral-flux onset detection and note-boundary refinement.

The pitch HMM in ``segment.py`` can only see boundaries where the pitch
changes. Two things it therefore cannot do on its own:

1. Split repeated notes at the same pitch ("la-la-la" on one note), which have
   an energy attack but no pitch change.
2. Place a note start precisely. A singer's pitch stabilises tens of
   milliseconds after the consonant that begins the syllable, so the pitch
   contour's transition lags the audible attack.

Both are fixed here with an onset envelope, using the SuperFlux formulation of
Boeck & Widmer ("Maximum filter vibrato suppression for onset detection",
DAFx 2013): a max filter across frequency before the temporal difference, which
stops vibrato from being read as a stream of onsets. That property matters a
great deal for voice.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import TranscriptionConfig
from .segment import Segment


class OnsetTrack:
    """Detected onsets and the envelope they came from."""

    def __init__(self, times: np.ndarray, strengths: np.ndarray, envelope: np.ndarray,
                 envelope_times: np.ndarray) -> None:
        self.times = times
        self.strengths = strengths
        self.envelope = envelope
        self.envelope_times = envelope_times

    def __len__(self) -> int:
        return int(self.times.shape[0])


def detect_onsets(audio: np.ndarray, config: TranscriptionConfig) -> OnsetTrack:
    """Compute a SuperFlux onset envelope and pick peaks from it.

    Three of librosa's defaults are overridden here rather than inherited, and
    all three matter enough to name. ``n_fft`` is librosa's 2048, which is the
    46 ms SuperFlux assumes only at 44.1 kHz; at this package's 16 kHz it is a
    128 ms window, and an attack smeared across it cannot be placed inside the
    50 ms the scoring convention allows. ``aggregate`` was the median, which
    requires more than half the mel bands to rise together -- a bar broadband
    consonants clear and vowel onsets do not. And ``max_size``, which is the
    actual vibrato defence, was too narrow to be one.

    Those three interact, which is why they are set together: a sensitive
    envelope with a narrow max filter reads vibrato as a stream of onsets, and
    the median was hiding that by suppressing nearly every onset as well.
    """
    import librosa

    hop = config.hop_length
    aggregate = np.mean if config.onset_aggregate == "mean" else np.median

    envelope = librosa.onset.onset_strength(
        y=audio,
        sr=config.sample_rate,
        hop_length=hop,
        n_fft=config.onset_n_fft,
        lag=config.onset_lag,
        max_size=config.onset_max_size,
        aggregate=aggregate,
    )

    # A peak has to stand out from this recording's own noise floor. librosa's
    # default delta is an absolute margin, and this envelope's scale spans two
    # orders of magnitude across recordings, so an absolute one is far too
    # strict on a quiet take and far too loose on a noisy one -- on breathy
    # input it admitted fifty-six onsets where there were seven notes.
    delta = max(config.onset_delta_ratio * float(np.median(envelope)), 1e-6)

    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=config.sample_rate,
        hop_length=hop,
        backtrack=False,
        units="frames",
        delta=delta,
    )

    envelope_times = librosa.frames_to_time(
        np.arange(envelope.shape[0]), sr=config.sample_rate, hop_length=hop
    )
    times = envelope_times[frames] if frames.size else np.zeros(0, dtype=np.float64)
    strengths = envelope[frames] if frames.size else np.zeros(0, dtype=np.float64)

    return OnsetTrack(times, strengths, envelope, envelope_times)


def _dip_depth(values: np.ndarray, frame: int) -> float:
    """How far ``values`` fall around ``frame`` below their level either side.

    The valley is the quietest frame from 40 ms before to 20 ms after; each
    shoulder is the loudest frame 40-100 ms before, and 20-80 ms after. The
    depth is measured against the lower shoulder, so a note that simply gets
    louder or quieter across the onset does not count as a dip.
    """
    n = values.shape[0]

    def peak(a, b):
        window = values[max(a, 0):min(b, n)]
        window = window[np.isfinite(window)]
        return float(window.max()) if window.size else np.nan

    valley = values[max(frame - 4, 0):min(frame + 2, n)]
    valley = valley[np.isfinite(valley)]
    if not valley.size:
        return np.inf                    # silence at the cut: as deep as it gets
    shoulder = min(peak(frame - 10, frame - 4), peak(frame + 2, frame + 8))
    return shoulder - float(valley.min()) if np.isfinite(shoulder) else 0.0


def split_segments_on_onsets(
    segments: List[Segment],
    onsets: OnsetTrack,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
    loudness_db: np.ndarray | None = None,
    periodicity: np.ndarray | None = None,
) -> List[Segment]:
    """Split segments wherever a strong onset falls well inside them.

    An onset qualifies when it reaches ``onset_split_strength_ratio`` of the
    median detected onset, and when it leaves at least ``onset_split_min_gap_s``
    of segment on each side. The strength bar comes from the other onsets, not
    from the onset envelope: the envelope includes every frame between the
    peaks, and those frames are loud during continuous singing, so its upper
    percentiles sit above most real onsets and nothing splits.

    The gap guard remains, because an over-eager split shreds a sustained note.

    With ``onset_split_min_dip_db`` above zero and ``loudness_db`` given, an
    onset must also show that phonation was interrupted: a dip in level of at
    least that depth, or -- with ``onset_split_min_periodicity_dip`` above
    zero -- a dip in periodicity of at least that much. Singing a note again
    takes a consonant or a glottal onset, which either quietens the sound or
    roughens it with noise; a vowel change or vibrato does neither.
    """
    if len(onsets) == 0 or not segments:
        return segments

    strengths = onsets.strengths[np.isfinite(onsets.strengths)]
    if strengths.size == 0:
        return segments
    # The bar is set by the other onsets, so it is a typical onset for this
    # recording rather than a typical frame, and a handful of very loud ones
    # cannot drag it up.
    threshold = config.onset_split_strength_ratio * float(np.median(strengths))

    strong_times = onsets.times[onsets.strengths >= threshold]
    if strong_times.size == 0:
        return segments

    min_gap_frames = max(1, int(round(config.onset_split_min_gap_s
                                      * config.sample_rate / config.hop_length)))

    out: List[Segment] = []
    for seg in segments:
        if seg.n_frames < 2 * min_gap_frames:
            out.append(seg)
            continue

        seg_start_t = float(frame_times[seg.start_frame])
        seg_end_t = float(frame_times[min(seg.end_frame, frame_times.shape[0] - 1)])

        inside = strong_times[(strong_times > seg_start_t) & (strong_times < seg_end_t)]
        if inside.size == 0:
            out.append(seg)
            continue

        cut_frames = np.searchsorted(frame_times, inside)
        cut_frames = cut_frames[
            (cut_frames - seg.start_frame >= min_gap_frames)
            & (seg.end_frame - cut_frames >= min_gap_frames)
        ]
        if config.onset_split_min_dip_db > 0 and loudness_db is not None:
            use_periodicity = (config.onset_split_min_periodicity_dip > 0
                               and periodicity is not None)
            cut_frames = np.array([
                c for c in cut_frames
                if _dip_depth(loudness_db, int(c)) >= config.onset_split_min_dip_db
                or (use_periodicity and _dip_depth(periodicity, int(c))
                    >= config.onset_split_min_periodicity_dip)], dtype=int)
        if cut_frames.size == 0:
            out.append(seg)
            continue

        boundaries = [seg.start_frame, *sorted(set(int(c) for c in cut_frames)), seg.end_frame]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            out.append(Segment(start_frame=start, end_frame=end, pitch=seg.pitch))

    return out


def snap_segment_starts(
    segments: List[Segment],
    onsets: OnsetTrack,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
) -> List[Segment]:
    """Pull each segment start to a detected onset within ``onset_snap_window_s``.

    ``onset_snap_direction`` decides which side of the start an onset is
    looked for on, and only that side is searched: 'earlier' takes the nearest
    onset at or before the start, 'later' the nearest at or after it, and
    'both' the nearest on either side. Under 'earlier', when there is no onset
    in the window before the start that it can move to, but there is one
    after it, the start stays and that onset is recorded as the segment's
    ``measure_from`` instead.

    A start may move later, or earlier into silence, as long as the segment
    keeps at least one frame. Where the segment begins exactly where the one
    before it ends -- a legato transition -- the boundary is shared, and a
    start that moves earlier takes the previous segment's end with it, as long
    as that segment keeps ``min_note_duration_s``.

    That last case is the one that matters. The decoder places a legato
    boundary where the pitch has crossed over, which lags the attack that
    starts the new syllable: a pitch tracker holds the old pitch for a few
    frames, and a reverberant room holds it for longer. Snapping exists to pull
    such a start back onto the attack, and it used to be forbidden from doing
    so, because a start could never move before the previous segment's end --
    which, for contiguous segments, is the start itself.
    """
    if len(onsets) == 0 or not segments:
        return segments

    hop_s = config.hop_length / config.sample_rate
    window_frames = int(round(config.onset_snap_window_s / hop_s))
    if window_frames < 1:
        return segments
    min_frames = max(1, int(round(config.min_note_duration_s / hop_s)))

    onset_frames = np.searchsorted(frame_times, onsets.times)
    onset_frames = np.clip(onset_frames, 0, frame_times.shape[0] - 1)
    if onset_frames.size == 0:
        return segments

    direction = config.onset_snap_direction
    out: List[Segment] = []
    for seg in segments:
        offsets = onset_frames - seg.start_frame
        in_window = np.abs(offsets) <= window_frames
        # Each direction looks only on its own side of the start. Choosing the
        # nearest onset first and checking its side afterwards let a nearer
        # onset on the wrong side -- the vowel just after a consonant's attack
        # -- hide the attack the start should have snapped to.
        before = onset_frames[in_window & (offsets <= 0)]
        after = onset_frames[in_window & (offsets >= 0) & (onset_frames < seg.end_frame)]

        candidate: Optional[int] = None
        if direction == "earlier":
            if before.size:
                candidate = int(before.max())
        elif direction == "later":
            if after.size:
                candidate = int(after.min())
        else:
            nearest = int(np.argmin(np.abs(offsets)))
            if in_window[nearest] and onset_frames[nearest] < seg.end_frame:
                candidate = int(onset_frames[nearest])

        previous = out[-1] if out else None
        if candidate is not None and previous is not None and candidate < previous.end_frame:
            shared = previous.end_frame == seg.start_frame
            if shared and candidate - previous.start_frame >= min_frames:
                # Only the end moves: the attack snapping recorded for the
                # previous segment is still where its pitch settles. One the
                # shorter segment no longer contains is ignored when it is
                # measured.
                out[-1] = Segment(start_frame=previous.start_frame, end_frame=candidate,
                                  pitch=previous.pitch, measure_from=previous.measure_from)
            else:
                # The onset belongs to the previous note; this start cannot
                # move onto it.
                candidate = None

        if candidate is not None:
            out.append(Segment(start_frame=candidate, end_frame=seg.end_frame, pitch=seg.pitch))
        elif direction == "earlier" and after.size:
            # The start stays, but the attack is still where the pitch
            # settles: measure from there, as a start snapped to it would have
            # been.
            out.append(Segment(seg.start_frame, seg.end_frame, seg.pitch,
                               measure_from=int(after.min())))
        else:
            out.append(seg)

    return out


def attach_late_arrivals(
    segments: List[Segment],
    onsets: OnsetTrack,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
) -> List[Segment]:
    """Start a note at the articulation that began its syllable, not where its pitch arrived.

    A singer often begins a syllable on the old pitch -- or, after a rest, a
    little off the new one -- and reaches the note it is aimed at 60-200 ms
    later. Annotators give the syllable one note from its onset; a decoder
    that segments by pitch starts the note where the pitch arrives, and if an
    onset split has already cut the syllable's start off the note before, that
    piece is left as a note of its own at the wrong pitch.

    For a pitch change of at most three semitones that has no onset of its
    own, the latest strong onset between ``onset_snap_window_s`` and
    ``syllable_lead_s`` before it is taken to start the syllable. If that
    onset began the piece before the change, and the piece carries on the
    pitch of the note before it (or follows a rest), the piece is folded into
    the arriving note; if it falls inside the note before, the boundary moves
    back to it. Either way the arriving note's pitch is measured from where the
    pitch arrived. A pitch change with no onset before it in that window -- a
    melisma, sung on one vowel -- is left alone.
    """
    if config.syllable_lead_s <= 0 or len(onsets) == 0 or len(segments) < 2:
        return segments
    strengths = onsets.strengths[np.isfinite(onsets.strengths)]
    if strengths.size == 0:
        return segments
    threshold = config.onset_split_strength_ratio * float(np.median(strengths))
    strong = np.searchsorted(frame_times, onsets.times[onsets.strengths >= threshold])
    if strong.size == 0:
        return segments

    hop_s = config.hop_length / config.sample_rate
    max_lead = int(round(config.syllable_lead_s / hop_s))
    min_lead = max(1, int(round(config.onset_snap_window_s / hop_s)))
    min_frames = max(1, int(round(config.min_note_duration_s / hop_s)))
    touching = 3                                     # frames: a gap this small is legato

    out = list(segments)
    i = 1
    while i < len(out):
        before, after = out[i - 1], out[i]
        if (before.end_frame != after.start_frame or before.pitch == after.pitch
                or abs(after.pitch - before.pitch) > 3
                or np.any(np.abs(strong - after.start_frame) < min_lead)):
            i += 1
            continue
        leads = strong[(strong <= after.start_frame - min_lead)
                       & (strong >= after.start_frame - max_lead)
                       & (strong >= before.start_frame - 2)]
        if leads.size == 0:
            i += 1
            continue
        onset = int(leads.max())
        arrived = after.start_frame
        if onset <= before.start_frame + 2:
            earlier = out[i - 2] if i >= 2 else None
            gap = before.start_frame - earlier.end_frame if earlier is not None else None
            carries_on = earlier is not None and gap <= touching and earlier.pitch == before.pitch
            after_rest = earlier is None or gap > touching
            if carries_on or (after_rest and abs(after.pitch - before.pitch) <= 2):
                out[i] = Segment(before.start_frame, after.end_frame, after.pitch,
                                 measure_from=arrived)
                del out[i - 1]
                continue
        elif onset - before.start_frame >= min_frames:
            out[i - 1] = Segment(before.start_frame, onset, before.pitch,
                                 measure_from=before.measure_from)
            out[i] = Segment(onset, after.end_frame, after.pitch, measure_from=arrived)
        i += 1
    return out


def refine_segments(
    segments: List[Segment],
    audio: np.ndarray | None,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
    onsets: OnsetTrack | None = None,
    loudness_db: np.ndarray | None = None,
    periodicity: np.ndarray | None = None,
) -> tuple[List[Segment], OnsetTrack | None]:
    """Apply onset-based splitting then start-snapping.

    Pass ``onsets`` to reuse a detection already made for this audio. Detecting
    them is the only step here that needs the samples, so supplying them lets
    the whole of segmentation run from the f0 and onset tracks alone -- which is
    what makes sweeping the segmentation parameters cheap.
    """
    if not config.use_onsets:
        return segments, None

    if onsets is None:
        if audio is None:
            raise ValueError("refine_segments needs either audio or precomputed onsets")
        onsets = detect_onsets(audio, config)
    segments = split_segments_on_onsets(segments, onsets, frame_times, config,
                                        loudness_db, periodicity)
    segments = snap_segment_starts(segments, onsets, frame_times, config)
    return segments, onsets
