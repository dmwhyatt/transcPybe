"""Top-level transcription: audio in, notes out."""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .audio import load_audio
from .config import TranscriptionConfig
from .f0 import F0Track, estimate_f0
from .fragments import absorb_short_fragments
from .glide import handle_glides
from .notes import (Note, estimate_tuning_offset, postprocess_notes,
                    segments_to_notes, snap_note_ends)
from .onsets import (OnsetTrack, attach_late_arrivals, merge_sustained_boundaries, protected_frames,
                     refine_segments)
from .progress import ProgressReporter, format_duration, format_realtime_factor, null_reporter
from .segment import segment_notes

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Everything the transcriber produced, including intermediates.

    The intermediate tracks are kept so that callers can plot the f0 contour
    against the decoded notes, which is the fastest way to diagnose a bad
    transcription.
    """

    notes: List[Note]
    f0_track: F0Track
    onsets: Optional[OnsetTrack]
    config: TranscriptionConfig
    duration: float
    elapsed: float
    source: Optional[str] = None
    device: Optional[str] = None
    """The device CREPE actually ran on, after resolving 'auto'."""
    tuning_offset: float = 0.0
    """Semitones between this recording's pitch centre and the A440 grid.

    Zero when the estimate was switched off or refused for want of evidence, so
    it does not distinguish "on A440" from "not measured"; the log line says
    which. Kept because a large value is a fact about the recording worth
    seeing, and because a transcription that looks a semitone out is explained
    by it more often than by anything else."""

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    def summary(self) -> str:
        if not self.notes:
            return (
                f"No notes found in {self.duration:.1f}s of audio "
                f"({self.elapsed:.1f}s elapsed)."
            )

        pitches = np.array([n.pitch for n in self.notes])
        durations = np.array([n.duration for n in self.notes])
        voiced = float(durations.sum())
        return (
            f"{self.n_notes} notes over {self.duration:.1f}s "
            f"({self.elapsed:.1f}s elapsed, {self.elapsed / max(self.duration, 1e-9):.2f}x realtime)\n"
            f"  pitch range   : MIDI {pitches.min()}-{pitches.max()} "
            f"(median {int(np.median(pitches))})\n"
            f"  note duration : median {np.median(durations):.3f}s, "
            f"min {durations.min():.3f}s, max {durations.max():.3f}s\n"
            f"  voiced        : {voiced:.1f}s ({100 * voiced / max(self.duration, 1e-9):.1f}% of audio)"
        )


def resolve_config_device(config: TranscriptionConfig) -> tuple[TranscriptionConfig, Optional[str]]:
    """Return a copy of ``config`` with ``device`` resolved, plus a description.

    A copy rather than a mutation, so that the caller's config is untouched and
    the config recorded in the output JSON names the device that actually ran.
    """
    if config.f0_backend != "crepe":
        return config, None

    from .device import resolve_device

    info = resolve_device(config.device)
    return dataclasses.replace(config, device=info.device), info.describe()


def decode_notes(
    track: F0Track,
    config: TranscriptionConfig,
    *,
    audio: Optional[np.ndarray] = None,
    onsets: Optional[OnsetTrack] = None,
) -> tuple[List[Note], Optional[OnsetTrack], int, float]:
    """Everything after f0 estimation: tune, segment, clean up, and measure notes.

    Split out so it can be replayed from a cached f0 track and onset track
    without the audio. Estimating f0 dominates the cost and none of the
    segmentation parameters change it, so a parameter sweep that reuses the
    tracks runs in seconds per configuration rather than minutes.

    This is the same code path the full transcription takes, so a sweep cannot
    be measuring something the pipeline does not do.
    """
    notes, onsets, n_segments = _decode_pass(track, config, audio, onsets, 0.0)

    # Where the semitone grid sits can only be judged from notes, and notes can
    # only be had by assuming a grid -- so assume the standard one, look at what
    # came back, and decode again if the performance turns out not to be on it.
    # The second pass is the cheap half: estimating f0 dominates the cost and is
    # already done, and a negligible offset skips it entirely, which is nearly
    # every recording.
    tuning_offset = estimate_tuning_offset(notes, config)
    if abs(tuning_offset) >= config.tuning_min_offset_semitones:
        notes, onsets, n_segments = _decode_pass(
            track, config, audio, onsets, tuning_offset)
    else:
        tuning_offset = 0.0

    return notes, onsets, n_segments, tuning_offset


def _decode_pass(
    track: F0Track,
    config: TranscriptionConfig,
    audio: Optional[np.ndarray],
    onsets: Optional[OnsetTrack],
    tuning_offset: float,
) -> tuple[List[Note], Optional[OnsetTrack], int]:
    """One decode of the f0 track against a semitone grid at ``tuning_offset``.

    The onset track is passed back out so that a second pass reuses the one the
    first computed rather than re-running the detector over the audio: no
    searched or estimated parameter here can change it.
    """
    segments = segment_notes(track, config, tuning_offset)
    n_segments = len(segments)
    segments = absorb_short_fragments(segments, config, protected_frames(onsets, track.times, config))
    segments = handle_glides(segments, track, config)
    segments, onsets = refine_segments(segments, audio, track.times, config,
                                       onsets=onsets, loudness_db=track.loudness_db,
                                       periodicity=track.periodicity, voicing=track.voicing)
    if onsets is not None:
        segments = attach_late_arrivals(segments, onsets, track.times, config,
                                        track.loudness_db)
        segments = merge_sustained_boundaries(segments, onsets, track.times, config)
    notes = segments_to_notes(segments, track, config, tuning_offset)
    notes = postprocess_notes(notes, track, config, onsets, tuning_offset)
    notes = snap_note_ends(notes, onsets, config)
    return notes, onsets, n_segments


def transcribe_audio(
    audio: np.ndarray,
    config: Optional[TranscriptionConfig] = None,
    progress: Optional[ProgressReporter] = None,
) -> TranscriptionResult:
    """Transcribe a mono signal already at ``config.sample_rate``.

    The stages are: estimate f0 and voicing, decode notes with the pitch HMM,
    absorb edge fragments, correct glides, refine boundaries against
    spectral-flux onsets, measure each note's pitch, then clean up.

    Pass a ``ProgressReporter`` to see progress on long inputs; without one the
    call is silent apart from ordinary logging.
    """
    config = config or TranscriptionConfig()
    config.validate()

    started = time.time()
    duration = len(audio) / config.sample_rate

    config, device_description = resolve_config_device(config)
    progress = progress or null_reporter()
    if progress.audio_seconds is None:
        progress.audio_seconds = duration

    if device_description:
        progress.log(f"  device: {device_description}")
    progress.log(f"  transcribing {duration:.1f}s of audio, {config.f0_backend} backend")

    track = estimate_f0(audio, config, progress)
    voiced_fraction = 100.0 * float(np.mean(track.voicing >= config.voicing_threshold))
    progress.log(f"  {track.n_frames} frames, {voiced_fraction:.1f}% voiced")

    with progress.stage("note decoding", announce=False):
        notes, onsets, n_segments, tuning_offset = decode_notes(track, config, audio=audio)
    progress.log(f"  {n_segments} raw segments")
    if tuning_offset:
        progress.log(f"  tuning centre {100.0 * tuning_offset:+.0f} cents from A440")

    if onsets is not None:
        progress.log(f"  {len(onsets)} onsets detected")
    progress.log(f"  {len(notes)} notes after post-processing")

    elapsed = time.time() - started
    rate = format_realtime_factor(elapsed, duration)
    progress.log(f"  transcribed in {format_duration(elapsed)}"
                 + (f" ({rate})" if rate else ""))

    return TranscriptionResult(
        notes=notes,
        f0_track=track,
        onsets=onsets,
        config=config,
        duration=duration,
        elapsed=elapsed,
        device=config.device if config.f0_backend == "crepe" else None,
        tuning_offset=tuning_offset,
    )


def transcribe_file(
    path: str | Path,
    config: Optional[TranscriptionConfig] = None,
    progress: Optional[ProgressReporter] = None,
) -> TranscriptionResult:
    """Transcribe an audio file. Any librosa-readable format is accepted."""
    config = config or TranscriptionConfig()
    config.validate()

    path = Path(path)
    progress = progress or null_reporter()

    with progress.stage(f"loading {path.name}"):
        audio, _ = load_audio(path, config.sample_rate)

    result = transcribe_audio(audio, config, progress)
    result.source = str(path)
    return result
