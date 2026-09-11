"""Writing transcriptions out as MIDI, JSON and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import TranscriptionConfig
from .notes import Note


def write_midi(notes: Sequence[Note], path: str | Path, config: TranscriptionConfig) -> Path:
    """Write notes to a type-1 MIDI file.

    Note times are written as absolute seconds. ``config.midi_tempo`` only sets
    the tempo map a DAW uses to draw its bar grid; it does not quantise or
    otherwise move the notes.
    """
    import pretty_midi

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = pretty_midi.PrettyMIDI(initial_tempo=config.midi_tempo)
    instrument = pretty_midi.Instrument(program=config.midi_program, name="vocal melody")

    for note in notes:
        # pretty_midi silently drops zero-length notes; guarantee a tick of length.
        end = max(note.offset, note.onset + 1e-3)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, note.velocity))),
                pitch=int(max(0, min(127, note.pitch))),
                start=float(note.onset),
                end=float(end),
            )
        )

    midi.instruments.append(instrument)
    midi.write(str(path))
    return path


def write_json(
    notes: Sequence[Note],
    path: str | Path,
    config: TranscriptionConfig,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Write the note list plus the configuration that produced it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "notes": [note.to_dict() for note in notes],
        "n_notes": len(notes),
        "config": config.to_dict(),
    }
    if extra:
        payload.update(extra)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


_CSV_COLUMNS = [
    "onset", "offset", "duration", "pitch", "velocity",
    "confidence", "pitch_hz", "pitch_midi_float",
]


def write_csv(notes: Sequence[Note], path: str | Path) -> Path:
    """Write a flat note table, one row per note."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for note in notes:
            row = note.to_dict()
            writer.writerow({
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in row.items() if key in _CSV_COLUMNS
            })
    return path


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(pitch: int) -> str:
    """Scientific pitch notation for a MIDI note number (60 -> C4)."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def format_note_table(notes: Sequence[Note], max_rows: int | None = None) -> str:
    """Render notes as a fixed-width table for terminal output."""
    header = (
        f"{'#':>4}  {'onset':>8}  {'offset':>8}  {'dur':>7}  "
        f"{'pitch':>5}  {'name':>5}  {'cents':>6}  {'vel':>3}  {'conf':>5}"
    )
    lines = [header, "-" * len(header)]

    rows = notes if max_rows is None else notes[:max_rows]
    for index, note in enumerate(rows, start=1):
        cents = (note.pitch_midi_float - note.pitch) * 100.0
        cents_text = f"{cents:+6.0f}" if cents == cents else "   n/a"
        lines.append(
            f"{index:>4}  {note.onset:>8.3f}  {note.offset:>8.3f}  {note.duration:>7.3f}  "
            f"{note.pitch:>5d}  {note_name(note.pitch):>5}  {cents_text}  "
            f"{note.velocity:>3d}  {note.confidence:>5.2f}"
        )

    if max_rows is not None and len(notes) > max_rows:
        lines.append(f"... {len(notes) - max_rows} more notes")
    return "\n".join(lines)
