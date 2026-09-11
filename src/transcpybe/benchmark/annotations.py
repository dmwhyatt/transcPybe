"""Reading reference note annotations from the formats benchmarks ship.

Note-level singing datasets agree on the content -- onset, offset, pitch -- and
on almost nothing else. vocadito ships CSV, MIR-ST500 ships JSON, others ship
tab-separated text or MIDI. Within CSV, the second column is sometimes the
offset and sometimes the duration, and pitch is sometimes MIDI and sometimes
Hz. Guessing wrong is not loud: a duration read as an offset gives notes that
all start at the right place and end almost immediately, which scores badly in
a way that looks like a poor transcriber rather than a misread file.

So every inference this module makes is recorded on the result and printed by
the runner. If the loader is wrong, that is visible before any score is
believed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# A reference note: onset seconds, offset seconds, integer MIDI pitch.
RefNote = Tuple[float, float, int]

ANNOTATION_SUFFIXES = (".csv", ".tsv", ".txt", ".mid", ".midi", ".json", ".jams")

# MIDI tops out at 127. A frequency column for singing sits mostly well above
# that, so the two are told apart by what fraction of values exceed it rather
# than by any single value: one corrupt row must not decide the units for a
# whole file.
MIDI_PLAUSIBLE_MAX = 127.0
# A MIDI column cannot legitimately hold any value above 127, so half of them
# doing so rules MIDI out entirely. A frequency column for singing clears 127 Hz
# (roughly C3) on most notes even for a low male voice, so this separates the
# two while leaving room for a handful of corrupt rows in an otherwise MIDI file.
HZ_FRACTION_THRESHOLD = 0.50
AMBIGUOUS_FRACTION = 0.05

# Counting values above 127 is not enough on its own. A bass singing around
# 90-200 Hz clears 127 on fewer than half his notes, so the count alone reads
# his frequencies as MIDI -- putting the reference two octaves above any human
# voice and scoring the transcriber at zero. So the two readings are compared
# on where they land instead: whichever puts more notes inside a range a person
# can actually sing is the right one, and the count only breaks a tie.
SUNG_MIDI_RANGE = (36.0, 88.0)   # bass low C to soprano high E, generously
PITCH_UNITS_MARGIN = 0.15

# An offset column exceeds its onset on essentially every row. A duration column
# does so only where the duration happens to exceed the elapsed time, which is
# possible for the first note or two and rare after that.
OFFSET_FRACTION_THRESHOLD = 0.90

# Datasets disagree about column order. Many write onset/offset/pitch, but
# vocadito and others write onset/pitch/duration, and reading the wrong column
# as pitch scores a working transcriber at exactly zero -- a failure that looks
# like a broken model rather than a misread file. So which column holds the
# pitch is decided from the data.
#
# A pitch column is recognised two ways, and both are needed. Its values fall in
# a musical range under one reading or the other; and it does *not* climb
# steadily the way a clock does. Range alone is not decisive, because elapsed
# seconds in a three-minute song spend part of the file inside the MIDI range.
PITCH_MIDI_RANGE = (21.0, 108.0)      # the piano, generously bracketing any voice
PITCH_HZ_RANGE = (55.0, 1760.0)       # A1 to A6
MONOTONE_FRACTION = 0.95
PITCH_COLUMN_MARGIN = 0.15            # how much better column 2 must look to win


@dataclass
class Annotation:
    """Reference notes plus a record of how they were read."""

    notes: List[RefNote]
    source: Path
    inferences: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    def describe(self) -> str:
        parts = [f"{self.n_notes} notes from {self.source.name}"]
        if self.inferences:
            parts.append("(" + ", ".join(f"{k}={v}" for k, v in self.inferences.items()) + ")")
        return " ".join(parts)


def _pitch_to_midi(values: np.ndarray, inferences: Dict[str, str]) -> np.ndarray:
    """Interpret a pitch column as MIDI numbers or as Hz, and say which.

    Decided on the proportion of values above the MIDI ceiling, not the maximum.
    A single corrupt row reading 200 in a column of MIDI numbers would otherwise
    reinterpret every pitch in the file as a frequency, shifting the whole
    reference down by octaves -- and the resulting score would look like a
    broken transcriber rather than a misread file.
    """
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        inferences["pitch_units"] = "midi (no usable values)"
        return values

    above = float(np.mean(finite > MIDI_PLAUSIBLE_MAX))
    as_midi = _fraction_singable(finite)
    as_hz = _fraction_singable(69.0 + 12.0 * np.log2(finite / 440.0))

    if as_hz > as_midi + PITCH_UNITS_MARGIN:
        inferences["pitch_units"] = (
            f"hz ({as_hz:.0%} of values are singable read as hz, "
            f"against {as_midi:.0%} read as midi)"
        )
        return _as_hz(values)

    if above >= HZ_FRACTION_THRESHOLD:
        # Neither reading looks singable, but a MIDI column cannot hold these
        # values at all, so frequencies remain the only reading left.
        inferences["pitch_units"] = f"hz ({above:.0%} of values above {MIDI_PLAUSIBLE_MAX:.0f})"
        return _as_hz(values)

    if above > AMBIGUOUS_FRACTION:
        # Neither reading is clean. Say so: read as MIDI, and the offending rows
        # are dropped as out of range further down with their own warning.
        inferences["pitch_units"] = f"midi ({above:.0%} of values above 127, treated as bad rows)"
    else:
        inferences["pitch_units"] = "midi"
    return values


def _fraction_singable(midi: np.ndarray) -> float:
    """How much of a column, read as MIDI, lands in a range a voice can sing."""
    finite = midi[np.isfinite(midi)]
    if finite.size == 0:
        return 0.0
    return float(np.mean((finite >= SUNG_MIDI_RANGE[0]) & (finite <= SUNG_MIDI_RANGE[1])))


def _as_hz(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values) & (values > 0)
    out[valid] = 69.0 + 12.0 * np.log2(values[valid] / 440.0)
    return out


def _second_column_to_offset(
    onsets: np.ndarray, second: np.ndarray, inferences: Dict[str, str]
) -> np.ndarray:
    """Decide whether the second column is an offset or a duration.

    Decided on the proportion of rows where the second column exceeds the first,
    ignoring rows where they are equal. Requiring it of *every* row would let one
    zero-length note reinterpret every duration in the file as an offset.
    """
    # Degenerate rows carry no evidence either way and are excluded from the
    # vote: a single zero-length note would otherwise decide the whole file.
    valid = np.isfinite(onsets) & np.isfinite(second) & (second != onsets)
    if not valid.any():
        inferences["second_column"] = "offset (no usable values)"
        return second

    fraction = float(np.mean(second[valid] > onsets[valid]))
    if fraction >= OFFSET_FRACTION_THRESHOLD:
        inferences["second_column"] = "offset"
        return second

    inferences["second_column"] = f"duration ({fraction:.0%} of rows exceeded their onset)"
    return onsets + second


def _pitch_plausibility(values: np.ndarray) -> float:
    """How much a column looks like pitches rather than times.

    Scored as the best fraction of values landing in a musical range under
    either reading, then discounted if the column rises steadily -- that is a
    clock, whatever range it happens to occupy.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    in_midi = float(np.mean((finite >= PITCH_MIDI_RANGE[0]) & (finite <= PITCH_MIDI_RANGE[1])))
    in_hz = float(np.mean((finite >= PITCH_HZ_RANGE[0]) & (finite <= PITCH_HZ_RANGE[1])))
    score = max(in_midi, in_hz)
    if finite.size > 2 and float(np.mean(np.diff(finite) >= 0)) >= MONOTONE_FRACTION:
        score *= 0.25
    return score


def _infer_pitch_column(clean: np.ndarray, inferences: Dict[str, str]) -> int:
    """Decide whether column 2 or column 3 holds the pitch.

    Column 3 is the common convention and keeps the benefit of the doubt: the
    second column has to look clearly more musical to displace it.
    """
    if clean.shape[0] == 0:
        inferences["columns"] = "onset, offset-or-duration, pitch (no usable rows)"
        return 2

    score_second = _pitch_plausibility(clean[:, 1])
    score_third = _pitch_plausibility(clean[:, 2])
    if score_second > score_third + PITCH_COLUMN_MARGIN:
        inferences["columns"] = (
            "onset, pitch, duration-or-offset "
            f"(column 2 scored {score_second:.0%} musical against column 3's {score_third:.0%})"
        )
        return 1
    inferences["columns"] = "onset, offset-or-duration, pitch"
    return 2


def _read_delimited(path: Path) -> Tuple[List[List[str]], Dict[str, str]]:
    """Read a CSV/TSV/TXT annotation, sniffing delimiter and header."""
    inferences: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return [], inferences

    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ","
    inferences["delimiter"] = {"\t": "tab", " ": "space", ",": "comma", ";": "semicolon"}.get(
        delimiter, repr(delimiter)
    )

    rows = [
        [cell.strip() for cell in line.split(delimiter) if cell.strip() != ""]
        for line in text.splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row]

    # A header is a first row whose first cell is not a number.
    if rows:
        try:
            float(rows[0][0])
        except (ValueError, IndexError):
            inferences["header"] = "yes"
            rows = rows[1:]
        else:
            inferences["header"] = "no"

    return rows, inferences


def load_delimited(path: Path) -> Annotation:
    """Load onset/offset-or-duration/pitch from a delimited text file."""
    rows, inferences = _read_delimited(path)
    warnings: List[str] = []

    usable = [row for row in rows if len(row) >= 3]
    if len(usable) != len(rows):
        warnings.append(f"{len(rows) - len(usable)} rows had fewer than 3 columns and were skipped")
    if not usable:
        return Annotation([], path, inferences, warnings + ["no usable rows"])

    columns = np.array([[_to_float(c) for c in row[:3]] for row in usable], dtype=float)

    # Infer units and column meaning from rows that parsed cleanly. An
    # unparseable row contributes NaN, which is not evidence about anything.
    parseable = np.isfinite(columns).all(axis=1)
    if not parseable.all():
        warnings.append(f"{int((~parseable).sum())} rows had unparseable numbers")
    clean = columns[parseable] if parseable.any() else columns

    pitch_index = _infer_pitch_column(clean, inferences)
    time_index = 2 if pitch_index == 1 else 1

    onsets, second, pitches = columns[:, 0], columns[:, time_index], columns[:, pitch_index]
    offsets = _second_column_to_offset_using(
        clean[:, 0], clean[:, time_index], onsets, second, inferences
    )
    midi = _pitch_to_midi_using(clean[:, pitch_index], pitches, inferences)

    return _assemble(onsets, offsets, midi, path, inferences, warnings)


def _second_column_to_offset_using(
    clean_onsets: np.ndarray, clean_second: np.ndarray,
    onsets: np.ndarray, second: np.ndarray, inferences: Dict[str, str],
) -> np.ndarray:
    """Infer from the clean rows, then apply the decision to every row."""
    probe: Dict[str, str] = {}
    _second_column_to_offset(clean_onsets, clean_second, probe)
    inferences.update(probe)
    if probe.get("second_column", "").startswith("duration"):
        return onsets + second
    return second


def _pitch_to_midi_using(
    clean_pitches: np.ndarray, pitches: np.ndarray, inferences: Dict[str, str]
) -> np.ndarray:
    """Infer units from the clean rows, then apply them to every row."""
    probe: Dict[str, str] = {}
    _pitch_to_midi(clean_pitches, probe)
    inferences.update(probe)
    if probe.get("pitch_units", "").startswith("hz"):
        return _as_hz(pitches)
    return pitches


def _to_float(cell: str) -> float:
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def load_midi(path: Path) -> Annotation:
    """Load reference notes from a MIDI file, merging all instrument tracks."""
    import pretty_midi

    inferences: Dict[str, str] = {"format": "midi"}
    warnings: List[str] = []

    midi = pretty_midi.PrettyMIDI(str(path))
    events = [
        (note.start, note.end, note.pitch)
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    if len(midi.instruments) > 1:
        inferences["tracks"] = f"{len(midi.instruments)} merged"

    events.sort()
    if not events:
        return Annotation([], path, inferences, warnings + ["no notes in file"])

    array = np.array(events, dtype=float)
    return _assemble(array[:, 0], array[:, 1], array[:, 2], path, inferences, warnings)


def load_json(path: Path) -> Annotation:
    """Load reference notes from JSON.

    Handles a bare list of triples, a list of objects with onset/offset/pitch
    keys under any of their common spellings, and a dict wrapping either.
    """
    inferences: Dict[str, str] = {"format": "json"}
    warnings: List[str] = []
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        for key in ("notes", "annotations", "data", "est_notes", "ref_notes"):
            if key in payload:
                inferences["key"] = key
                payload = payload[key]
                break

    rows = _rows_from_payload(payload, inferences, warnings)
    if rows is None:
        return Annotation([], path, inferences, warnings)

    array = np.array(rows, dtype=float)
    pitch_index = _infer_pitch_column(array, inferences)
    time_index = 2 if pitch_index == 1 else 1
    offsets = _second_column_to_offset(array[:, 0], array[:, time_index], inferences)
    midi = _pitch_to_midi(array[:, pitch_index], inferences)
    return _assemble(array[:, 0], offsets, midi, path, inferences, warnings)


def _rows_from_payload(
    payload: Any, inferences: Dict[str, str], warnings: List[str]
) -> Optional[List[List[float]]]:
    """Turn one item's notes into onset/second/third rows, whatever shape they came in.

    Three shapes appear in the wild and all reduce to the same three columns:
    a list of triples, a list of objects, and parallel arrays -- one list per
    field, which is how singing corpora built for synthesis usually store them.
    Which column holds the pitch, and whether the second is an offset or a
    duration, is then decided by the same inference used for delimited files.
    """
    # Parallel arrays: a dict of equal-length lists rather than a list of notes.
    if isinstance(payload, dict):
        arrays = {k: v for k, v in payload.items()
                  if isinstance(v, list) and v and all(
                      isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)}
        lengths = {len(v) for v in arrays.values()}
        if len(arrays) >= 2 and len(lengths) == 1:
            onset_key = _first_key(arrays, ("onset", "start", "start_time", "note_start",
                                            "time", "begin"))
            pitch_key = _first_key(arrays, ("note_midi", "pitch", "midi_pitch", "note",
                                            "midi", "note_pitch"))
            span_key = _first_key(arrays, ("offset", "end", "end_time", "note_end",
                                           "duration", "dur", "note_dur"))
            if pitch_key and span_key:
                inferences["shape"] = (
                    f"parallel arrays ({onset_key or 'cumulative'}/{span_key}/{pitch_key})"
                )
                spans = [float(x) for x in arrays[span_key]]
                pitches = [float(x) for x in arrays[pitch_key]]
                if onset_key:
                    onsets = [float(x) for x in arrays[onset_key]]
                else:
                    # No onset column: the durations are consecutive, so the
                    # onsets are their running total.
                    onsets, running = [], 0.0
                    for span in spans:
                        onsets.append(running)
                        running += span
                return [[o, s, p] for o, s, p in zip(onsets, spans, pitches)]

        for key in ("notes", "annotations", "data", "est_notes", "ref_notes"):
            if key in payload:
                inferences["key"] = key
                return _rows_from_payload(payload[key], inferences, warnings)

    if not isinstance(payload, list) or not payload:
        warnings.append("no note list found")
        return None

    rows: List[List[float]] = []
    if isinstance(payload[0], dict):
        onset_key = _first_key(payload[0], ("onset", "start", "start_time", "time", "begin"))
        pitch_key = _first_key(payload[0], ("pitch", "midi_pitch", "note", "midi"))
        offset_key = _first_key(payload[0], ("offset", "end", "end_time", "duration", "dur"))
        if not (onset_key and pitch_key and offset_key):
            warnings.append(f"could not find note keys in {sorted(payload[0])}")
            return None
        inferences["keys"] = f"{onset_key}/{offset_key}/{pitch_key}"
        rows = [[float(item[onset_key]), float(item[offset_key]), float(item[pitch_key])]
                for item in payload]
    else:
        inferences["shape"] = "list of triples"
        rows = [[float(v) for v in item[:3]] for item in payload if len(item) >= 3]

    if not rows:
        warnings.append("no usable rows")
        return None
    return rows


def _first_key(item: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    lowered = {k.lower(): k for k in item}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_jams(path: Path) -> Annotation:
    """Load reference notes from a JAMS file via its note_midi/note_hz namespace."""
    inferences: Dict[str, str] = {"format": "jams"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    warnings: List[str] = []

    for annotation in payload.get("annotations", []):
        namespace = annotation.get("namespace", "")
        if not namespace.startswith("note"):
            continue
        inferences["namespace"] = namespace
        rows = [
            [float(ob["time"]), float(ob["time"]) + float(ob["duration"]), float(ob["value"])]
            for ob in annotation.get("data", [])
        ]
        if not rows:
            continue
        array = np.array(rows, dtype=float)
        midi = (array[:, 2] if namespace == "note_midi"
                else _pitch_to_midi(array[:, 2], inferences))
        inferences.setdefault("pitch_units", "midi" if namespace == "note_midi" else "hz")
        return _assemble(array[:, 0], array[:, 1], midi, path, inferences, warnings)

    return Annotation([], path, inferences, warnings + ["no note annotation found"])


def _assemble(
    onsets: np.ndarray, offsets: np.ndarray, midi: np.ndarray,
    path: Path, inferences: Dict[str, str], warnings: List[str],
) -> Annotation:
    """Validate, clean and sort, reporting anything discarded."""
    usable = np.isfinite(onsets) & np.isfinite(offsets) & np.isfinite(midi)
    dropped_nan = int((~usable).sum())
    if dropped_nan:
        warnings.append(f"{dropped_nan} notes had unreadable values and were dropped")

    onsets, offsets, midi = onsets[usable], offsets[usable], midi[usable]

    positive = offsets > onsets
    dropped_empty = int((~positive).sum())
    if dropped_empty:
        warnings.append(f"{dropped_empty} notes had offset <= onset and were dropped")
    onsets, offsets, midi = onsets[positive], offsets[positive], midi[positive]

    in_range = (np.round(midi) >= 0) & (np.round(midi) <= 127)
    dropped_range = int((~in_range).sum())
    if dropped_range:
        warnings.append(f"{dropped_range} notes fell outside MIDI 0-127 and were dropped")
    onsets, offsets, midi = onsets[in_range], offsets[in_range], midi[in_range]

    # Reference pitches keep their fractional part. Rounding them to semitones
    # would displace every note by up to 50 cents -- the whole pitch tolerance --
    # and singers who drift from equal temperament would be scored as errors on
    # notes the transcriber got right. It would also stop these numbers being
    # comparable with published ones, which all score against the unrounded
    # reference. The integer constraint belongs on the output, not on the truth.
    order = np.argsort(onsets, kind="stable")
    notes = [
        (float(onsets[i]), float(offsets[i]), float(midi[i]))
        for i in order
    ]
    return Annotation(notes, path, inferences, warnings)


def load_annotation(path: str | Path) -> Annotation:
    """Load reference notes from any supported annotation file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".mid", ".midi"):
        return load_midi(path)
    if suffix == ".jams":
        return load_jams(path)
    if suffix == ".json":
        return load_json(path)
    if suffix in (".csv", ".tsv", ".txt"):
        return load_delimited(path)
    raise ValueError(
        f"Unsupported annotation format {suffix!r}. "
        f"Supported: {', '.join(ANNOTATION_SUFFIXES)}"
    )


def load_keyed(path: str | Path) -> Dict[str, Annotation]:
    """Load a single file holding annotations for many recordings, keyed by id.

    M4Singer and MIR-ST500 both ship one JSON for the whole corpus rather than
    a file per song, which per-item pairing cannot express. The values go
    through exactly the same shape and unit inference as a standalone file, so
    a keyed corpus is not a second code path with its own bugs.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} is not a mapping of id to notes")

    # A single item's own wrapper keys must not be mistaken for recording ids.
    if any(key in payload for key in ("notes", "annotations", "data")):
        raise ValueError(f"{path.name} looks like one recording, not a keyed corpus")

    out: Dict[str, Annotation] = {}
    for item_id, value in payload.items():
        inferences: Dict[str, str] = {"format": "json", "keyed": "yes"}
        warnings: List[str] = []
        rows = _rows_from_payload(value, inferences, warnings)
        if rows is None:
            out[str(item_id)] = Annotation([], path, inferences, warnings)
            continue
        array = np.array(rows, dtype=float)
        pitch_index = _infer_pitch_column(array, inferences)
        time_index = 2 if pitch_index == 1 else 1
        offsets = _second_column_to_offset(array[:, 0], array[:, time_index], inferences)
        midi = _pitch_to_midi(array[:, pitch_index], inferences)
        out[str(item_id)] = _assemble(array[:, 0], offsets, midi, path,
                                      inferences, warnings)
    return out
