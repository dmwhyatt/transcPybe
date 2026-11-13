"""Vocal melody to MIDI transcription.

Transcribes monophonic singing -- typically a source-separated lead vocal stem
-- into note events with integer MIDI pitches and floating-point onset times.
By default this is transcPybe v2's pop preset with pYIN pitch tracking; see
``TranscriptionConfig.preset`` and ``transcpybe.landmarks``.

    from transcpybe import transcribe_file, write_midi

    result = transcribe_file("lead-vocal.wav")
    write_midi(result.notes, "lead-vocal.mid", result.config)

Or from the command line::

    python -m transcpybe lead-vocal.wav -o out/
    python -m transcpybe stems/ -o midi/ --log-file run.log

See ``transcpybe/README.md`` for the algorithm and for credit to the
upstream projects this builds on.
"""

from .batch import BatchItem, BatchResult, find_audio_files, transcribe_batch
from .config import TranscriptionConfig
from .core import TranscriptionResult, transcribe_audio, transcribe_file
from .device import DeviceInfo, resolve_device
from .f0 import F0Track, estimate_f0, hz_to_midi, midi_to_hz
from .midi_io import format_note_table, note_name, write_csv, write_json, write_midi
from .notes import Note
from .onsets import OnsetTrack, detect_onsets
from .progress import ProgressReporter, null_reporter, setup_logging
from .segment import Segment, segment_notes

__all__ = [
    "TranscriptionConfig",
    "TranscriptionResult",
    "transcribe_audio",
    "transcribe_file",
    "transcribe_batch",
    "find_audio_files",
    "BatchItem",
    "BatchResult",
    "ProgressReporter",
    "null_reporter",
    "setup_logging",
    "DeviceInfo",
    "resolve_device",
    "F0Track",
    "estimate_f0",
    "hz_to_midi",
    "midi_to_hz",
    "Note",
    "Segment",
    "segment_notes",
    "OnsetTrack",
    "detect_onsets",
    "write_midi",
    "write_json",
    "write_csv",
    "format_note_table",
    "note_name",
]

__version__ = "2.1.0"
