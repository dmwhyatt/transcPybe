"""Letting a benchmark run this transcriber in process.

Registered through the ``transcription_benchmarking.transcribers`` entry point
group, so the benchmark can call this package without importing it directly and
without this package depending on the benchmark to be installed.

The dependency runs this way round on purpose. A benchmark that imports the
system it measures can only ever measure that one, so the benchmark knows
nothing about transcPybe and transcPybe reaches out to it, not the reverse.
Scoring lives in that package; this one transcribes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Tuple

from .config import TranscriptionConfig

Note = Tuple[float, float, float]


class TranscPybe:
    """transcPybe as something a benchmark can call.

    Constructed with no arguments by entry point discovery, which gives the
    default configuration. Build it directly to score a specific one:

        TranscPybe(TranscriptionConfig(f0_backend="pyin"))
    """

    name = "transcpybe"

    def __init__(self, config: TranscriptionConfig = None) -> None:
        self.config = config or TranscriptionConfig()
        self.config.validate()

    def transcribe(self, audio: Path) -> List[Note]:
        """A benchmark has a file; everything inside this package has samples."""
        from .audio import load_audio
        from .core import transcribe_audio

        samples, _ = load_audio(Path(audio), self.config.sample_rate)
        result = transcribe_audio(samples, self.config)
        return [(note.onset, note.offset, float(note.pitch)) for note in result.notes]

    def settings(self) -> Dict[str, object]:
        """What the benchmark records as provenance and keys its cache on.

        The whole configuration, not a chosen subset: a parameter left out here
        is one that can change silently between a cached run and a fresh one.
        """
        return dataclasses.asdict(self.config)
