"""Writing transcriptions out as MIDI, JSON and CSV."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .config import TranscriptionConfig
from .notes import Note

PITCH_BEND_RANGE_SEMITONES = 2.0
"""The bend range measured pitch is written in: the General MIDI default.

No RPN is written to set it, because a player that follows General MIDI is
already there, and pretty_midi orders same-tick controller messages by number,
which would send the range before the RPN that selects it. A note sits at most
half a semitone from its integer pitch -- further only under
``pitch_reference='tuned'``, by at most the 0.4-semitone tuning limit -- so two
semitones always reach it."""

_BEND_STEPS = 8192


def _bend_value(note: Note) -> int:
    """The 14-bit bend that moves a note's integer pitch to its measured one."""
    deviation = note.pitch_midi_float - note.pitch
    if not math.isfinite(deviation):
        return 0
    value = int(round(deviation / PITCH_BEND_RANGE_SEMITONES * _BEND_STEPS))
    return max(-_BEND_STEPS, min(_BEND_STEPS - 1, value))


def write_midi(notes: Sequence[Note], path: str | Path, config: TranscriptionConfig) -> Path:
    """Write notes to a type-1 MIDI file.

    Note times are written as absolute seconds. ``config.midi_tempo`` only sets
    the tempo map a DAW uses to draw its bar grid; it does not quantise or
    otherwise move the notes.

    Unless ``config.twelve_tet`` is set, each note is preceded by a pitch bend
    to its measured pitch, at its onset tick. The line is monophonic, so one
    channel's bend serves every note, and pretty_midi writes a bend before a
    note-on that shares its tick.
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
        if not config.twelve_tet:
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=_bend_value(note), time=float(note.onset))
            )

    midi.instruments.append(instrument)
    midi.write(str(path))
    return path


def _bend_range(instrument) -> float:
    """The bend range a track sets with RPN 0, or the General MIDI default.

    Read from the first complete RPN 0 data entry, coarse (controller 6) plus
    fine (controller 38, in cents). A track that changes its range part-way
    through is rare enough that one value per track does.
    """
    rpn = [None, None]
    semitones = None
    for change in sorted(instrument.control_changes, key=lambda c: c.time):
        if change.number == 101:
            rpn[0] = change.value
        elif change.number == 100:
            rpn[1] = change.value
        elif rpn == [0, 0] and change.number == 6 and semitones is None:
            semitones = float(change.value)
        elif rpn == [0, 0] and change.number == 38 and semitones is not None:
            return semitones + change.value / 100.0
    return semitones if semitones is not None else PITCH_BEND_RANGE_SEMITONES


def read_midi_notes(path: str | Path) -> List[Tuple[float, float, float]]:
    """(onset, offset, pitch) for every note in a MIDI file, pitch as fractional MIDI.

    Each note takes the bend in force on its track when it starts, so a file
    written with measured pitch reads back as measured and a whole-semitone
    file reads back as whole semitones. Drum tracks are skipped. Notes from
    every track are returned together, sorted by onset.
    """
    import bisect

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    out: List[Tuple[float, float, float]] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        bends = sorted(instrument.pitch_bends, key=lambda b: b.time)
        times = [b.time for b in bends]
        scale = _bend_range(instrument) / _BEND_STEPS
        for note in instrument.notes:
            # A bend at the note's own tick counts: it was written for it.
            index = bisect.bisect_right(times, note.start + 1e-9) - 1
            bend = bends[index].pitch if index >= 0 else 0
            out.append((float(note.start), float(note.end), note.pitch + bend * scale))
    return sorted(out)


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
