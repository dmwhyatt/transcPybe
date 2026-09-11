"""Running the transcriber over a benchmark and reporting what it scored.

A number is only worth quoting if it can be reproduced, so every result carries
what produced it: the full transcriber configuration, the random seed, the
versions of the packages that matter, the repository commit, and a digest of the
dataset files themselves. Change any of those and the digest or the version
string changes with it, which is the difference between a measurement and a
number someone once saw.
"""

from __future__ import annotations

import json
import hashlib
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TranscriptionConfig
from ..core import transcribe_file
from ..progress import ProgressReporter, format_duration, null_reporter
from .annotations import Annotation, load_annotation
from .dataset import Item, Pairing, dataset_digest
from .metrics import (
    PITCH_TOLERANCE, SHORT_NOTE_S, Aggregate, ItemScore, MissAnatomy,
    aggregate_scores, analyse_misses, score_item, score_sequence,
    sequence_chance_floor,
)

RELEVANT_PACKAGES = ("torch", "torchcrepe", "librosa", "mir_eval",
                     "pretty_midi", "numpy", "scipy")


def _package_versions() -> Dict[str, str]:
    import importlib.metadata as metadata

    versions: Dict[str, str] = {}
    for name in RELEVANT_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = "not installed"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=10, check=False,
        )
        commit = result.stdout.strip()
        if not commit:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


@dataclass
class Provenance:
    """Everything needed to reproduce a result."""

    timestamp: str
    git_commit: str
    dataset_digest: str
    n_items: int
    config: Dict
    packages: Dict[str, str]
    python: str
    platform: str
    time_offset: float = 0.0
    """Constant shift applied to whole estimated notes before scoring.
    Non-zero means the number is conditional on a stated assumption."""
    onset_shift: float = 0.0
    """Constant shift applied to estimated note starts only, leaving ends alone."""
    system: str = "transcpybe"
    """Which transcriber produced the estimates. Recorded so a saved result
    cannot later be compared against another without it being obvious which
    system each came from."""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    aggregate: Aggregate
    items: List[ItemScore]
    provenance: Provenance
    annotations: Dict[str, Dict] = field(default_factory=dict)
    elapsed: float = 0.0
    misses: List[MissAnatomy] = field(default_factory=list)
    sequences: List[Any] = field(default_factory=list)
    sequence_floor: Dict[str, float] = field(default_factory=dict)
    alignments: Dict[str, Any] = field(default_factory=dict)
    temperament_cents: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "provenance": self.provenance.to_dict(),
            "aggregate": {
                "n_items": self.aggregate.n_items,
                "n_reference_notes": self.aggregate.n_reference_notes,
                "n_estimated_notes": self.aggregate.n_estimated_notes,
                "macro": {k: asdict(v) for k, v in self.aggregate.macro.items()},
                "micro": self.aggregate.micro,
                "octave_error_rate": self.aggregate.octave_error_rate,
                "median_best_offset_s": self.aggregate.median_best_offset_s,
                "items_needing_alignment": self.aggregate.items_needing_alignment,
                "bootstrap_samples": self.aggregate.bootstrap_samples,
                "confidence": self.aggregate.confidence,
            },
            "items": [s.to_dict() for s in self.items],
            "annotations": self.annotations,
            "elapsed": self.elapsed,
        }


def _cached_notes(item: Item, cache_dir: Optional[Path],
                  config: TranscriptionConfig, progress) -> List:
    """Transcribe, reusing a cached result when the configuration matches.

    The cache is keyed on the configuration as well as the item, so changing a
    parameter and re-running cannot silently score the previous settings.
    """
    from ..notes import Note

    if cache_dir is None:
        return transcribe_file(item.audio, config, progress).notes

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Item ids carry directory names where stems collide, so they are not
    # filenames. Flattening keeps one file per recording in one directory,
    # which is what makes a cache easy to inspect and to delete.
    cache_path = cache_dir / f"{_cache_name(item.item_id)}.json"
    fingerprint = json.dumps(config.to_dict(), sort_keys=True)

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("fingerprint") == fingerprint:
                return [Note(**note) for note in payload["notes"]]
        except Exception:
            pass

    result = transcribe_file(item.audio, config, progress)
    cache_path.write_text(json.dumps({
        "fingerprint": fingerprint,
        "notes": [asdict_note(n) for n in result.notes],
    }))
    return result.notes


def _cache_name(item_id: str) -> str:
    """A filesystem-safe name for one item's cache entry."""
    # Unicode word characters are kept: M4Singer names its songs in Chinese, and
    # stripping those would map every song by one singer onto the same name --
    # the collision this is here to prevent. The digest is unconditional so the
    # guarantee does not depend on what the pattern happens to keep.
    safe = re.sub(r"[^\w.#-]+", "_", item_id, flags=re.UNICODE).strip("_")
    if safe == item_id and len(safe) <= 120:
        # Nothing was rewritten, so the name is the id, and ids are unique.
        # Flat datasets therefore keep readable cache files, and a cache built
        # before ids carried directories stays valid.
        return safe
    # Something was rewritten, so distinct ids could now share a name. The
    # digest restores the guarantee, and covers whatever the pattern strips --
    # M4Singer names its songs in Chinese, and dropping those would map every
    # song by one singer onto a single file.
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:100] or 'item'}_{digest}"


def asdict_note(note) -> Dict:
    from dataclasses import asdict as _asdict
    return _asdict(note)


def run_benchmark(
    pairing: Pairing,
    config: Optional[TranscriptionConfig] = None,
    *,
    cache_dir: Optional[Path] = None,
    progress: Optional[ProgressReporter] = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
    check_alignment: bool = True,
    estimates: Optional[Dict[str, Path]] = None,
    time_offset: float = 0.0,
    onset_shift: float = 0.0,
    align: bool = False,
    align_cache: Optional[Path] = None,
) -> BenchmarkResult:
    """Transcribe every item, score it, and aggregate with confidence intervals."""
    config = config or TranscriptionConfig()
    config.validate()
    progress = progress or null_reporter()

    started = time.time()
    scores: List[ItemScore] = []
    misses: List[MissAnatomy] = []
    alignments: Dict[str, Any] = {}
    stage_cache = None
    sequences: List[Any] = []
    sequence_pairs: List[Tuple[List, List]] = []
    deviations: List[float] = []
    annotation_notes: Dict[str, Dict] = {}

    progress.log(f"Benchmark: {len(pairing.items)} items")
    with progress.stage("benchmark", total=len(pairing.items)) as stage:
        for item in pairing.items:
            annotation: Annotation = load_annotation(item.annotation)
            annotation_notes[item.item_id] = {
                "n_notes": annotation.n_notes,
                "inferences": annotation.inferences,
                "warnings": annotation.warnings,
                "match": item.match,
            }
            if annotation.warnings:
                for warning in annotation.warnings:
                    progress.log(f"  {item.item_id}: {warning}")

            if align:
                # Warp the reference onto the recording before anything is
                # measured. The f0 it aligns to is the same for every
                # configuration, so the warp cannot absorb the differences
                # between them.
                from .align import align_reference
                from .tune import StageCache

                if stage_cache is None:
                    stage_cache = StageCache(align_cache or Path(".f0-cache"), config)
                f0_track, _ = stage_cache.get(item, config)
                warp = align_reference(annotation.notes, f0_track.f0_hz,
                                       f0_track.voicing, f0_track.times,
                                       voicing_threshold=config.voicing_threshold)
                alignments[item.item_id] = warp
                if not warp.ok:
                    # Scoring a reference that could not be placed would measure
                    # the failure to place it.
                    progress.log(f"  {item.item_id}: not aligned -- {warp.reason}")
                    stage.advance(1, detail=item.item_id)
                    continue
                annotation = replace(annotation, notes=warp.warp_notes(annotation.notes))

            if estimates is not None:
                # Another system's transcription, read through the same loader
                # as the references so both sides are treated identically.
                estimated = list(load_annotation(estimates[item.item_id]).notes)
            else:
                notes = _cached_notes(item, cache_dir, config, null_reporter())
                estimated = [(n.onset, n.offset, n.pitch) for n in notes]

            # Declared corrections, applied to every recording alike and recorded
            # in the provenance: assumptions, not a search for whichever shift
            # scores best. time_offset moves whole notes, for a clock that does
            # not match. onset_shift moves only note starts, for a reference that
            # calls something else the start -- moving the ends as well would
            # break the boundary the two sides already agree on.
            if time_offset:
                estimated = [(max(0.0, s0 + time_offset), max(0.0, e0 + time_offset), p)
                             for s0, e0, p in estimated]
            if onset_shift:
                estimated = [(max(0.0, min(s0 + onset_shift, e0 - MIN_NOTE_S)), e0, p)
                             for s0, e0, p in estimated]

            scores.append(score_item(
                item.item_id, annotation.notes, estimated,
                check_alignment=check_alignment,
            ))
            misses.append(analyse_misses(annotation.notes, estimated))
            sequences.append(score_sequence(annotation.notes, estimated))
            sequence_pairs.append((list(annotation.notes), list(estimated)))
            deviations.extend(
                abs(pitch - round(pitch)) * 100.0 for _, _, pitch in annotation.notes
            )
            stage.advance(1, detail=item.item_id)

    aggregate = aggregate_scores(scores, bootstrap_samples=bootstrap_samples, seed=seed)

    provenance = Provenance(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        git_commit=_git_commit(),
        dataset_digest=dataset_digest(pairing.items),
        n_items=len(pairing.items),
        config=config.to_dict() if estimates is None else {"estimates": "external"},
        packages=_package_versions(),
        python=platform.python_version(),
        platform=platform.platform(),
        system="transcpybe" if estimates is None else "external",
        time_offset=time_offset,
        onset_shift=onset_shift,
    )

    return BenchmarkResult(
        aggregate=aggregate, items=scores, provenance=provenance,
        annotations=annotation_notes, elapsed=time.time() - started,
        misses=misses,
        temperament_cents=float(np.median(deviations)) if deviations else 0.0,
        sequences=sequences,
        sequence_floor=sequence_chance_floor([r for r, _ in sequence_pairs],
                                             [e for _, e in sequence_pairs]),
        alignments=alignments,
    )


MIN_NOTE_S = 0.02


def _correction_banner(provenance: Provenance) -> str:
    """Say, above the table, that a number is conditional on an assumption."""
    parts = []
    if provenance.time_offset:
        parts.append(f"whole notes moved by {provenance.time_offset:+.3f}s")
    if provenance.onset_shift:
        parts.append(f"note starts moved by {provenance.onset_shift:+.3f}s")
    if not parts:
        return ""
    return ("SCORED WITH A STATED CORRECTION: " + ", ".join(parts) +
            ".\nThis is conditional on that assumption, not a plain score.\n")


def format_result(result: BenchmarkResult, worst: int = 10) -> str:
    """The headline table, the caveats, and the worst items."""
    aggregate = result.aggregate
    if aggregate.n_items == 0:
        return "No items were scored."

    confidence = int(aggregate.confidence * 100)
    lines = [
        (f"{aggregate.n_items} recordings from {aggregate.n_groups} performance(s), "
         if aggregate.n_groups else f"{aggregate.n_items} recordings, ")
        + f"{aggregate.n_reference_notes} reference notes, "
        f"{aggregate.n_estimated_notes} transcribed",
        "",
        _correction_banner(result.provenance),
        f"{'metric':<28} {'P':>6} {'R':>6} {'F':>6}   {confidence}% CI on F",
        "-" * 68,
    ]

    rows = (
        ("COn      (onset)", "con"),
        ("COnP     (onset+pitch)", "conp"),
        ("COnPOff  (+offset)", "conpoff"),
    )
    for label, key in rows:
        f_interval = aggregate.macro[f"{key}_f"]
        precision = aggregate.macro.get(f"{key}_p")
        recall = aggregate.macro.get(f"{key}_r")
        p_text = f"{precision.mean:.3f}" if precision else "-"
        r_text = f"{recall.mean:.3f}" if recall else "-"
        lines.append(
            f"{label:<28} {p_text:>6} {r_text:>6} {f_interval.mean:>6.3f}   "
            f"[{f_interval.low:.3f}-{f_interval.high:.3f}]"
        )

    lines += [
        "",
        f"Macro average over recordings; {confidence}% CI from "
        f"{aggregate.bootstrap_samples} bootstrap resamples of the "
        + ("performances (segments of one performance are not independent "
           "observations)." if aggregate.n_groups else "recordings."),
        f"Micro (pooled over all notes): COnP F = {aggregate.micro['conp_f']:.3f}",
        "",
        f"Octave errors: {aggregate.octave_error_rate:.1%} of notes whose onset was found",
        f"Median best time offset: {aggregate.median_best_offset_s:+.3f}s",
    ]

    # An integer output must land within 50 cents of the true pitch, and the
    # nearest semitone to a singer sitting d cents off equal temperament is
    # already d cents away. What is left is the slack the decoder has for its
    # own f0 error, which is why a dataset of untrained singers scores lower at
    # identical transcriber accuracy. Stating it separates the two.
    if result.temperament_cents:
        slack = PITCH_TOLERANCE - result.temperament_cents
        lines += [
            "",
            f"Reference notes sit a median {result.temperament_cents:.1f} cents from equal "
            "temperament.",
            f"  The nearest semitone is already that far out, leaving ~{slack:.1f} cents of the",
            "  50-cent tolerance for the decoder's own error. That is a property of singing",
            "  against an integer output, not a fault to be tuned away.",
        ]

    # Onsets landing but pitches never landing is not something a transcriber
    # does. It is what a misread pitch column looks like, and reporting the
    # zero without saying so invites weeks of tuning against a parsing bug.
    con_f = aggregate.macro["con_f"].mean
    conp_f = aggregate.macro["conp_f"].mean
    if con_f > 0.3 and conp_f < 0.1 * con_f:
        ref_median = np.median([s.ref_pitch_median for s in result.items])
        est_median = np.median([s.est_pitch_median for s in result.items])
        ref_lo = min(s.ref_pitch_lo for s in result.items)
        ref_hi = max(s.ref_pitch_hi for s in result.items)
        est_lo = min(s.est_pitch_lo for s in result.items)
        est_hi = max(s.est_pitch_hi for s in result.items)
        lines += [
            "",
            "=" * 68,
            "SUSPECT THE REFERENCE, NOT THE TRANSCRIBER.",
            f"  Onsets match ({con_f:.0%}) but pitches essentially never do "
            f"({conp_f:.1%}).",
            "  A transcriber that finds onsets gets some pitches right by chance;",
            "  this pattern is what a misread pitch column looks like.",
            "",
            f"  reference pitches: {ref_lo}-{ref_hi} (median {ref_median:.0f})",
            f"  transcribed pitches: {est_lo}-{est_hi} (median {est_median:.0f})",
            "",
            "  Sung notes sit around MIDI 40-84. If the reference range is far from",
            "  that, the wrong column was read as pitch. Check the annotation with",
            "  --dry-run, and look at the raw file: some datasets order columns",
            "  onset/pitch/duration rather than onset/offset/pitch.",
            "=" * 68,
        ]

    # Recall below precision means notes are being lost. Which fix applies
    # depends on how they were lost, so say how rather than leaving it to be
    # guessed at from the totals.
    if result.misses:
        total_missed = sum(m.total for m in result.misses)
        recall_gap = aggregate.macro["conp_p"].mean - aggregate.macro["conp_r"].mean
        if total_missed and recall_gap > 0.05:
            merged = sum(m.merged for m in result.misses)
            wrong_pitch = sum(m.wrong_pitch for m in result.misses)
            too_short = sum(m.too_short for m in result.misses)
            absent = sum(m.absent for m in result.misses)
            missed_median = np.median([m.missed_duration_median for m in result.misses
                                       if m.missed_duration_median > 0] or [0.0])
            found_median = np.median([m.found_duration_median for m in result.misses
                                      if m.found_duration_median > 0] or [0.0])
            lines += [
                "",
                f"Recall is the binding constraint (precision leads recall by "
                f"{recall_gap:.3f}). Of {total_missed} reference notes missed:",
                f"  {merged:>5} ({merged / total_missed:>4.0%})  ran together with a neighbour "
                "at the same pitch",
                f"  {wrong_pitch:>5} ({wrong_pitch / total_missed:>4.0%})  onset found, pitch "
                "outside tolerance",
                f"  {too_short:>5} ({too_short / total_missed:>4.0%})  shorter than "
                f"{SHORT_NOTE_S * 1000:.0f} ms",
                f"  {absent:>5} ({absent / total_missed:>4.0%})  nothing overlapped them at all",
                "",
                f"  Missed notes run {missed_median:.3f}s against {found_median:.3f}s for notes "
                "found.",
            ]
            wrong_dev = np.median([m.wrong_pitch_temperament_cents
                                   for m in result.misses
                                   if m.wrong_pitch_temperament_cents > 0] or [0.0])
            found_dev = np.median([m.found_temperament_cents for m in result.misses
                                   if m.found_temperament_cents > 0] or [0.0])
            if wrong_dev and found_dev:
                lines.append(
                    f"  Pitch misses sat {wrong_dev:.0f} cents from equal temperament "
                    f"against {found_dev:.0f} for hits."
                )
                if wrong_dev > found_dev + 8.0:
                    lines += [
                        "  They are the notes sung furthest between semitones, so an integer",
                        "  output cannot reach them however well the pitch is decoded. Only a",
                        "  fractional pitch, or a pitch bend, recovers these.",
                    ]

            if merged / total_missed > 0.25:
                lines += [
                    "  Re-articulated notes at one pitch are the largest share. Pitch",
                    "  cannot separate those -- only the onset detector can.",
                ]

    if result.alignments:
        ok = [a for a in result.alignments.values() if a.ok]
        failed = [(k, a) for k, a in result.alignments.items() if not a.ok]
        shifts = [a.max_shift_s for a in ok]
        lines += [
            "",
            f"References warped onto the recordings: {len(ok)} of "
            f"{len(result.alignments)} aligned.",
        ]
        if shifts:
            lines.append(
                f"  The warp moves notes by up to a median {float(np.median(shifts)):.2f}s, "
                "which is the drift\n  the reference carried.")
        if failed:
            lines += [
                f"  {len(failed)} could not be aligned and were not scored: "
                + ", ".join(k for k, _ in failed[:3])
                + (" ..." if len(failed) > 3 else ""),
                "  Scoring an unplaceable reference would measure the failure to place "
                "it.",
            ]

    # Reported always, and the only thing worth reading when the clock does not
    # match. A melody is a sequence, and whether its notes were found in the
    # right order does not depend on when they happened.
    if result.sequences:
        mean_f = float(np.mean([s.f for s in result.sequences]))
        mean_p = float(np.mean([s.precision for s in result.sequences]))
        mean_r = float(np.mean([s.recall for s in result.sequences]))
        mean_i = float(np.mean([s.interval_f for s in result.sequences]))
        floor = result.sequence_floor or {}
        lines += [
            "",
            "Note sequence, ignoring time:",
            f"  {'':<22}{'P':>7}{'R':>7}{'F':>7}",
            f"  {'pitches':<22}{mean_p:>7.3f}{mean_r:>7.3f}{mean_f:>7.3f}",
            f"  {'intervals':<22}{'':>7}{'':>7}{mean_i:>7.3f}"
            "   (unchanged by transposition)",
        ]
        if floor.get("n"):
            headroom = mean_f - floor["f"]
            lines += [
                "",
                f"  Against another song's melody the same metric scores "
                f"{floor['f']:.3f}, so this",
                f"  measure starts there, not at zero: the margin above chance is "
                f"{headroom:+.3f}.",
            ]
            if headroom < 0.1:
                lines.append(
                    "  That margin is too small to tune against -- the sequences are "
                    "barely\n  more like their own song than like any other.")

    # Checked before, and independently of, the misalignment guard below. A rate
    # mismatch does not trigger that guard at all: no constant shift improves the
    # score, so the gain it looks for stays near zero and the worst case of all
    # passes silently.
    drifts = np.array([abs(s.onset_drift) for s in result.items if s.n_delta_pairs])
    if drifts.size and float(np.median(drifts)) > 0.1:
        drifting = int((drifts > 0.1).sum())
        lines += [
            "",
            "=" * 68,
            "THE REFERENCE IS ON A DIFFERENT CLOCK RATE, NOT JUST OFFSET.",
            f"  On {drifting} of {drifts.size} recordings the timing error at the end "
            f"differs from",
            f"  the error at the start by a median {float(np.median(drifts)):.2f}s.",
            "",
            "  A constant offset cannot correct a difference in rate, so neither",
            "  --time-offset nor --onset-shift will help, and onset-based scores here",
            "  measure the mismatch rather than the transcription. This is what a",
            "  reference notated at a nominal tempo looks like against a recorded",
            "  performance: correct notes, in musical time rather than clock time.",
            "",
            "  Either align each recording first (dynamic time warping against the",
            "  audio), or score only what does not depend on absolute time -- the",
            "  sequence of pitches, and intervals between them.",
            "=" * 68,
        ]

    if aggregate.items_needing_alignment:
        shown = aggregate.items_needing_alignment[:5]
        lines += [
            "",
            f"WARNING: {len(aggregate.items_needing_alignment)} item(s) score materially "
            "better under a global time shift, which means the reference and the audio "
            "do not share a clock. Scores for these are not meaningful:",
            "  " + ", ".join(shown)
            + (f" ... and {len(aggregate.items_needing_alignment) - 5} more"
               if len(aggregate.items_needing_alignment) > 5 else ""),
        ]

        # A handful of misaligned recordings is a data problem in those files.
        # Many of them shifted the same way is a different claim, and it needs
        # the shifts to actually agree: after a correction the residue scatters
        # symmetrically about zero, and calling that a convention would assert a
        # direction the data does not have.
        share = len(aggregate.items_needing_alignment) / max(aggregate.n_items, 1)
        offsets = np.array([s.best_offset_s for s in result.items]) if result.items \
            else np.zeros(0)
        if share > 0.5 and offsets.size:
            median = float(np.median(offsets))
            low, high = (float(np.percentile(offsets, 25)),
                         float(np.percentile(offsets, 75)))
            agree = float(np.mean(np.sign(offsets) == np.sign(median))) if median else 0.0
            consistent = abs(median) >= 0.02 and agree >= 0.7

            starts = np.array([s.onset_delta_median for s in result.items
                               if s.n_delta_pairs])
            ends = np.array([s.end_delta_median for s in result.items if s.n_delta_pairs])
            start_median = float(np.median(starts)) if starts.size else 0.0
            end_median = float(np.median(ends)) if ends.size else 0.0

            if consistent:
                lines += [
                    "",
                    f"{share:.0%} of recordings are misaligned and {agree:.0%} of the "
                    f"shifts share a sign:",
                    f"  median {median:+.3f}s, half between {low:+.3f} and {high:+.3f}.",
                    f"  Measured separately, note starts differ by {start_median:+.3f}s "
                    f"and note ends by {end_median:+.3f}s",
                    "  (estimate minus reference, so a negative start means the estimate "
                    "begins earlier).",
                ]
                # Three different shapes, and only two of them are a shift.
                # Both boundaries moved the same way is a clock. Only the start
                # moved is a disagreement about where a note begins. The two
                # moved in opposite directions is neither: the notes are a
                # different length, and no translation makes them agree.
                meaningful = 0.015
                both_ways = (start_median < -meaningful < meaningful < end_median
                             or end_median < -meaningful < meaningful < start_median)
                if both_ways:
                    widening = end_median - start_median
                    lines += [
                        "",
                        f"  The starts and ends moved in opposite directions, so the "
                        f"estimated notes are",
                        f"  {abs(widening):.3f}s "
                        + ("longer" if widening > 0 else "shorter") +
                        " than the reference ones rather than shifted. No constant",
                        "  shift corrects a difference in length: the two sides disagree "
                        "about how much",
                        "  of the sound belongs to a note, not about when it happens.",
                        "",
                        f"  --onset-shift {-start_median:+.3f} would still align the "
                        "starts, which is what COn",
                        "  and COnP measure. COnPOff compares ends too and will not "
                        "recover.",
                    ]
                elif abs(end_median) < abs(start_median) / 2:
                    lines += [
                        "  The ends agree and the starts do not, so the two sides "
                        "disagree about where a",
                        "  note begins rather than about the clock -- an annotation "
                        "marking the syllable",
                        "  including its leading consonant, against a pitch tracker "
                        "marking the voiced",
                        "  onset, for one.",
                        "",
                        f"  Re-run with --onset-shift {-start_median:+.3f}, which moves "
                        "note starts and leaves",
                        "  ends alone. Moving whole notes would break the boundary the "
                        "two already share.",
                    ]
                else:
                    # Derived from the measured deltas rather than from the offset
                    # search, so every recommendation here uses one sign
                    # convention: shift by the negative of estimate-minus-reference.
                    shift = -(start_median + end_median) / 2.0
                    lines += [
                        "  Starts and ends are shifted alike, so this is a clock that "
                        "does not match.",
                        f"  Re-run with --time-offset {shift:+.3f} to score under that "
                        "assumption stated.",
                    ]
            else:
                lines += [
                    "",
                    f"{share:.0%} of recordings are misaligned, but the shifts do not "
                    f"agree in sign:",
                    f"  median {median:+.3f}s, half between {low:+.3f} and {high:+.3f}.",
                    "  Scatter about zero is not a convention and no constant corrects "
                    "it. Timing",
                    "  differs per note, which is what a reference aligned to something "
                    "other than",
                    "  pitch looks like once its average has been removed -- a "
                    "consonant's length",
                    "  varies with the consonant.",
                    "",
                    "  A 50 ms tolerance cannot be met against a reference that marks a "
                    "different",
                    "  event per note. This dataset measures agreement with its own "
                    "annotation",
                    "  convention as much as transcription accuracy.",
                ]

    ordered = sorted(result.items, key=lambda s: s.conp.f)[:worst]
    if ordered:
        lines += ["", f"Worst {len(ordered)} items by COnP F:",
                  f"  {'item':<26} {'ref':>5} {'est':>5} {'COn':>6} {'COnP':>6} "
                  f"{'offset':>7}  {'ref pitch':>11} {'est pitch':>11}"]
        for score in ordered:
            lines.append(
                f"  {score.item_id[:26]:<26} {score.n_reference:>5} {score.n_estimated:>5} "
                f"{score.con.f:>6.3f} {score.conp.f:>6.3f} {score.best_offset_s:>+7.3f}  "
                f"{score.ref_pitch_lo:>4}-{score.ref_pitch_hi:<6} "
                f"{score.est_pitch_lo:>4}-{score.est_pitch_hi:<6}"
            )

    lines += ["", f"Ran in {format_duration(result.elapsed)} | "
                  f"commit {result.provenance.git_commit} | "
                  f"dataset {result.provenance.dataset_digest}"]
    return "\n".join(lines)
