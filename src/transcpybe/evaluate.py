"""Scoring transcriptions against ground truth with mir_eval.

Metrics follow the MIREX note-tracking conventions implemented by mir_eval
(Raffel, McFee, Humphrey, Salamon, Nieto, Liang & Ellis, "mir_eval: A
transparent implementation of common MIR metrics", ISMIR 2014):

* **Onset F-measure** - onsets matched within a 50 ms window, pitch ignored.
* **Note F-measure (onset + pitch)** - as above but the pitch must also match
  within a quarter tone. This is the headline number for this task.
* **Note F-measure (onset + pitch + offset)** - additionally requires the
  offset to land within 50 ms or 20% of the note's duration. Always lower,
  because note ends are genuinely ambiguous in singing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .notes import Note

GroundTruth = Sequence[Tuple[float, float, int]]


@dataclass
class Scores:
    """Transcription accuracy for one item."""

    onset_precision: float
    onset_recall: float
    onset_f1: float
    note_precision: float
    note_recall: float
    note_f1: float
    note_offset_f1: float
    n_reference: int
    n_estimated: int
    median_pitch_error: float   # signed, in semitones, over matched notes

    def to_dict(self) -> Dict[str, float]:
        from dataclasses import asdict
        return asdict(self)


def _to_intervals(items: Sequence[Tuple[float, float, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Split (onset, offset, pitch) tuples into mir_eval's interval/pitch arrays.

    mir_eval.transcription expects pitches in Hz, not MIDI numbers.
    """
    if not items:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[a, b] for a, b, _ in items], dtype=float)
    pitches_hz = np.array([440.0 * 2.0 ** ((p - 69) / 12.0) for _, _, p in items], dtype=float)
    return intervals, pitches_hz


def score(
    notes: Sequence[Note],
    ground_truth: GroundTruth,
    *,
    onset_tolerance: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
    measured_pitch: bool = False,
) -> Scores:
    """Score estimated notes against ground truth.

    Each note is scored at its integer ``pitch``, as a whole-semitone MIDI file
    would give it, unless ``measured_pitch`` is set: then at
    ``pitch_midi_float``, as a file carrying measured pitch gives it (see
    ``TranscriptionConfig.twelve_tet``). A note with no measured pitch keeps
    its integer one either way.
    """
    import mir_eval

    def pitch_of(note: Note) -> float:
        if measured_pitch and np.isfinite(note.pitch_midi_float):
            return float(note.pitch_midi_float)
        return note.pitch

    estimated = [(n.onset, n.offset, pitch_of(n)) for n in notes]
    ref_intervals, ref_hz = _to_intervals(list(ground_truth))
    est_intervals, est_hz = _to_intervals(estimated)

    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        zero = 0.0
        return Scores(zero, zero, zero, zero, zero, zero, zero,
                      len(ground_truth), len(estimated), float("nan"))

    onset_p, onset_r, onset_f = mir_eval.onset.f_measure(
        ref_intervals[:, 0], est_intervals[:, 0], window=onset_tolerance
    )

    note_p, note_r, note_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=None,
    )

    _, _, offset_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=0.2,
        offset_min_tolerance=onset_tolerance,
    )

    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=None,
    )
    if matching:
        errors = [estimated[e][2] - list(ground_truth)[r][2] for r, e in matching]
        median_error = float(np.median(errors))
    else:
        median_error = float("nan")

    return Scores(
        onset_precision=float(onset_p), onset_recall=float(onset_r), onset_f1=float(onset_f),
        note_precision=float(note_p), note_recall=float(note_r), note_f1=float(note_f),
        note_offset_f1=float(offset_f),
        n_reference=len(ground_truth), n_estimated=len(estimated),
        median_pitch_error=median_error,
    )


def format_scores(rows: List[Tuple[str, Scores]]) -> str:
    """Render a table of per-case scores with a mean row."""
    header = (
        f"{'case':<20} {'ref':>4} {'est':>4} {'onsetF':>7} "
        f"{'noteF':>7} {'noteP':>7} {'noteR':>7} {'+offF':>7}"
    )
    lines = [header, "-" * len(header)]
    for name, s in rows:
        lines.append(
            f"{name:<20} {s.n_reference:>4} {s.n_estimated:>4} {s.onset_f1:>7.3f} "
            f"{s.note_f1:>7.3f} {s.note_precision:>7.3f} {s.note_recall:>7.3f} "
            f"{s.note_offset_f1:>7.3f}"
        )
    if rows:
        lines.append("-" * len(header))
        mean = lambda fn: float(np.mean([fn(s) for _, s in rows]))
        lines.append(
            f"{'MEAN':<20} {'':>4} {'':>4} {mean(lambda s: s.onset_f1):>7.3f} "
            f"{mean(lambda s: s.note_f1):>7.3f} {mean(lambda s: s.note_precision):>7.3f} "
            f"{mean(lambda s: s.note_recall):>7.3f} {mean(lambda s: s.note_offset_f1):>7.3f}"
        )
    return "\n".join(lines)
