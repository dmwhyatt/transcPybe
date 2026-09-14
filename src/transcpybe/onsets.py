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

from typing import List

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


def split_segments_on_onsets(
    segments: List[Segment],
    onsets: OnsetTrack,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
) -> List[Segment]:
    """Split segments wherever a strong onset falls well inside them.

    An onset qualifies when it reaches ``onset_split_strength_ratio`` of the
    median detected onset, and when it leaves at least ``onset_split_min_gap_s``
    of segment on each side. The strength bar comes from the other onsets, not
    from the onset envelope: the envelope includes every frame between the
    peaks, and those frames are loud during continuous singing, so its upper
    percentiles sit above most real onsets and nothing splits.

    The gap guard remains, because an over-eager split shreds a sustained note.
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
    """Pull each segment start to a nearby detected onset.

    A start is only moved if the shift does not collide with the previous
    segment and does not shrink this segment below one frame.
    """
    if len(onsets) == 0 or not segments:
        return segments

    window_frames = int(round(config.onset_snap_window_s
                              * config.sample_rate / config.hop_length))
    if window_frames < 1:
        return segments

    onset_frames = np.searchsorted(frame_times, onsets.times)
    onset_frames = np.clip(onset_frames, 0, frame_times.shape[0] - 1)
    if onset_frames.size == 0:
        return segments

    out: List[Segment] = []
    previous_end = 0
    for seg in segments:
        distances = np.abs(onset_frames - seg.start_frame)
        nearest = int(np.argmin(distances))
        candidate = int(onset_frames[nearest])

        if (
            distances[nearest] <= window_frames
            and candidate >= previous_end
            and candidate < seg.end_frame
        ):
            seg = Segment(start_frame=candidate, end_frame=seg.end_frame, pitch=seg.pitch)

        out.append(seg)
        previous_end = seg.end_frame

    return out


def refine_segments(
    segments: List[Segment],
    audio: np.ndarray | None,
    frame_times: np.ndarray,
    config: TranscriptionConfig,
    onsets: OnsetTrack | None = None,
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
    segments = split_segments_on_onsets(segments, onsets, frame_times, config)
    segments = snap_segment_starts(segments, onsets, frame_times, config)
    return segments, onsets
