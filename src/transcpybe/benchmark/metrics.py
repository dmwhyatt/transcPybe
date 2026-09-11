"""Scoring transcriptions against reference annotations, with error bars.

Metrics follow the note-transcription conventions the singing-transcription
literature reports, so the numbers can sit beside published ones:

* **COn** - correct onset. The onset is matched within 50 ms; pitch ignored.
* **COnP** - correct onset and pitch. As above, and the pitch within 50 cents.
  This is the headline figure.
* **COnPOff** - correct onset, pitch and offset. As above, and the offset within
  50 ms or 20% of the reference note's duration, whichever is larger.

All three come from the same matching machinery, so they nest: a note counted
in COnPOff is counted in COnP and in COn.

A single number is not a measurement. Aggregates carry a bootstrap confidence
interval resampled over *recordings* rather than notes, because notes within one
recording are not independent -- a singer whose vibrato defeats the decoder
defeats it for the whole track, and treating those notes as independent samples
would report an interval several times too narrow.

Two diagnostics guard against believing a number that is really an artefact:
the **octave error rate**, and the **best global offset**, which catches
reference annotations that are misaligned or that the loader misread.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reference pitches are fractional MIDI; transcribed ones are integers. Scoring
# happens in cents either way, so the two mix freely.
RefNote = Tuple[float, float, float]

ONSET_TOLERANCE = 0.05        # seconds, the MIREX convention
PITCH_TOLERANCE = 50.0        # cents, a quarter tone
OFFSET_RATIO = 0.2
IGNORE_PITCH_TOLERANCE = 12000.0   # ten octaves: effectively no pitch constraint


@dataclass
class PRF:
    precision: float
    recall: float
    f: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ItemScore:
    """Scores for one recording."""

    item_id: str
    n_reference: int
    n_estimated: int
    con: PRF
    conp: PRF
    conpoff: PRF
    octave_errors: int = 0
    best_offset_s: float = 0.0
    best_offset_conp_f: float = 0.0
    # Summaries of both note sets. When onsets align but pitches do not, these
    # are what distinguish a misread reference from a failing transcriber.
    ref_pitch_lo: int = 0
    ref_pitch_hi: int = 0
    ref_pitch_median: float = 0.0
    est_pitch_lo: int = 0
    est_pitch_hi: int = 0
    est_pitch_median: float = 0.0
    ref_duration_median: float = 0.0
    est_duration_median: float = 0.0
    # Where the two sides disagree, measured separately at note starts and note
    # ends over onset-matched pairs. A clock offset moves both together; a
    # difference about what a note's start means moves only the first.
    onset_delta_median: float = 0.0
    end_delta_median: float = 0.0
    n_delta_pairs: int = 0
    # The same measured early and late in the recording. A constant offset is a
    # shift; an error that grows is a different clock *rate*, which is what a
    # score-quantised reference looks like against a real performance.
    onset_delta_early: float = 0.0
    onset_delta_late: float = 0.0

    @property
    def onset_drift(self) -> float:
        """How much the timing error grows from the start of the recording to the end."""
        return self.onset_delta_late - self.onset_delta_early

    @property
    def misalignment_gain(self) -> float:
        """How much COnP a global time shift would have bought.

        Large values mean the reference and the audio do not share a clock, and
        no transcriber score computed against it means anything.
        """
        return self.best_offset_conp_f - self.conp.f

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["misalignment_gain"] = self.misalignment_gain
        return data


@dataclass
class Interval:
    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}-{self.high:.3f}]"


@dataclass
class Aggregate:
    """Dataset-level results."""

    n_items: int
    n_reference_notes: int
    n_estimated_notes: int
    macro: Dict[str, Interval]     # mean over recordings, with bootstrap CI
    micro: Dict[str, float]        # pooled over all notes
    octave_error_rate: float
    median_best_offset_s: float
    items_needing_alignment: List[str] = field(default_factory=list)
    bootstrap_samples: int = 0
    confidence: float = 0.95
    n_groups: int = 0
    """Distinct performances behind the items, where items are segments of them.
    Zero when each item is its own recording and the two are the same thing."""


def _to_mir_eval(notes: Sequence[RefNote]) -> Tuple[np.ndarray, np.ndarray]:
    """mir_eval wants intervals in seconds and pitches in Hz."""
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[a, b] for a, b, _ in notes], dtype=float)
    hz = np.array([440.0 * 2.0 ** ((p - 69) / 12.0) for _, _, p in notes], dtype=float)
    return intervals, hz


def note_prf(reference, estimated, *, pitch_tolerance: float,
         offset_ratio: Optional[float], onset_tolerance: float) -> PRF:
    import mir_eval

    ref_intervals, ref_hz = _to_mir_eval(reference)
    est_intervals, est_hz = _to_mir_eval(estimated)
    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        return PRF(0.0, 0.0, 0.0)

    # mir_eval rejects negative times outright. Shifting both sides by a common
    # amount leaves every matching decision unchanged and keeps the failure out
    # of the middle of a scoring run.
    earliest = min(float(ref_intervals.min()), float(est_intervals.min()))
    if earliest < 0:
        ref_intervals = ref_intervals - earliest
        est_intervals = est_intervals - earliest

    precision, recall, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
        offset_ratio=offset_ratio,
        offset_min_tolerance=onset_tolerance,
    )
    return PRF(float(precision), float(recall), float(f))


def _count_octave_errors(reference, estimated, onset_tolerance: float) -> int:
    """Among notes whose onset was found, how many landed in the wrong octave.

    Matched with pitch ignored, so the pairing reflects timing alone; the pitch
    comparison then happens after the fact.
    """
    import mir_eval

    ref_intervals, ref_hz = _to_mir_eval(reference)
    est_intervals, est_hz = _to_mir_eval(estimated)
    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        return 0

    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=IGNORE_PITCH_TOLERANCE,
        offset_ratio=None,
    )
    errors = 0
    for ref_index, est_index in matching:
        # An octave error is a whole number of octaves out, so this one test
        # does need semitones rather than cents.
        ref_pitch = round(reference[ref_index][2])
        est_pitch = round(estimated[est_index][2])
        if ref_pitch != est_pitch and (est_pitch - ref_pitch) % 12 == 0:
            errors += 1
    return errors


def _best_global_offset(
    reference, estimated, *, search_s: float = 0.5, step_s: float = 0.01,
    onset_tolerance: float = ONSET_TOLERANCE,
) -> Tuple[float, float]:
    """Find the global time shift of the estimate that maximises COnP.

    Reference annotations are often offset from the audio by a constant, and a
    transcriber scored against one looks broken in a way no amount of tuning
    fixes. Reporting the shift makes that visible immediately.
    """
    if not reference or not estimated:
        return 0.0, 0.0

    # A negative shift would put note times before zero, which mir_eval rejects.
    # Padding both sides equally keeps every time positive and leaves the
    # relative alignment -- the only thing being measured -- untouched.
    pad = search_s + 1.0
    padded_reference = [(a + pad, b + pad, p) for a, b, p in reference]

    # Every offset within the onset tolerance of the true one scores identically,
    # so the maximum is a plateau rather than a point. Its centre is the best
    # estimate of the real shift; taking the first or the smallest tied value
    # instead reports an edge, understating or overstating the misalignment by
    # up to the tolerance.
    offsets = np.arange(-search_s, search_s + step_s / 2, step_s)
    scores = np.array([
        note_prf(padded_reference,
             [(a + pad + offset, b + pad + offset, p) for a, b, p in estimated],
             pitch_tolerance=PITCH_TOLERANCE, offset_ratio=None,
             onset_tolerance=onset_tolerance).f
        for offset in offsets
    ])
    best_f = float(scores.max())
    plateau = offsets[scores >= best_f - 1e-12]
    return float(np.median(plateau)), best_f


def score_item(
    item_id: str,
    reference: Sequence[RefNote],
    estimated: Sequence[RefNote],
    *,
    onset_tolerance: float = ONSET_TOLERANCE,
    pitch_tolerance: float = PITCH_TOLERANCE,
    offset_ratio: float = OFFSET_RATIO,
    check_alignment: bool = True,
) -> ItemScore:
    """Score one recording against its reference."""
    reference, estimated = list(reference), list(estimated)

    con = note_prf(reference, estimated, pitch_tolerance=IGNORE_PITCH_TOLERANCE,
               offset_ratio=None, onset_tolerance=onset_tolerance)
    conp = note_prf(reference, estimated, pitch_tolerance=pitch_tolerance,
                offset_ratio=None, onset_tolerance=onset_tolerance)
    conpoff = note_prf(reference, estimated, pitch_tolerance=pitch_tolerance,
                   offset_ratio=offset_ratio, onset_tolerance=onset_tolerance)

    best_offset, best_f = (0.0, conp.f)
    if check_alignment:
        best_offset, best_f = _best_global_offset(
            reference, estimated, onset_tolerance=onset_tolerance
        )

    return ItemScore(
        item_id=item_id,
        n_reference=len(reference),
        n_estimated=len(estimated),
        con=con, conp=conp, conpoff=conpoff,
        octave_errors=_count_octave_errors(reference, estimated, onset_tolerance),
        best_offset_s=best_offset,
        best_offset_conp_f=best_f,
        **_note_summary(reference, "ref"),
        **_note_summary(estimated, "est"),
        **_boundary_deltas(reference, estimated, onset_tolerance),
    )


def _boundary_deltas(
    reference: Sequence[RefNote], estimated: Sequence[RefNote], onset_tolerance: float
) -> Dict[str, float]:
    """How far the estimate sits from the reference at note starts and at ends.

    Matched on onset with pitch ignored and a tolerance wide enough to pair
    notes that a strict match would drop, since the pairs that fail the
    tolerance are exactly the ones carrying the disagreement.
    """
    import mir_eval

    empty = {"onset_delta_median": 0.0, "end_delta_median": 0.0, "n_delta_pairs": 0}
    if not reference or not estimated:
        return empty

    ref_intervals, ref_hz = _to_mir_eval(reference)
    est_intervals, est_hz = _to_mir_eval(estimated)
    matching = mir_eval.transcription.match_notes(
        ref_intervals, ref_hz, est_intervals, est_hz,
        onset_tolerance=max(onset_tolerance * 6.0, 0.3),
        pitch_tolerance=IGNORE_PITCH_TOLERANCE, offset_ratio=None,
    )
    if not matching:
        return empty

    starts = [estimated[e][0] - reference[r][0] for r, e in matching]
    ends = [estimated[e][1] - reference[r][1] for r, e in matching]

    early, late = _end_to_end_offsets(reference, estimated)
    return {
        "onset_delta_median": float(np.median(starts)),
        "end_delta_median": float(np.median(ends)),
        "n_delta_pairs": len(matching),
        "onset_delta_early": early,
        "onset_delta_late": late,
    }


# A reference on a different clock rate drifts by seconds over a song, far
# beyond any matching window, so the two ends are aligned independently with a
# search wide enough to find them wherever they have gone.
DRIFT_SEARCH_S = 3.0
DRIFT_STEP_S = 0.02
# Music repeats, so a search window that is a large fraction of the recording
# finds a shift that lines one phrase up with the next and calls it drift. The
# window is capped against the span, and a recording too short or too sparse to
# identify an alignment is not measured at all rather than measured wrongly.
DRIFT_MIN_SPAN_S = 20.0
DRIFT_MIN_NOTES_PER_END = 8


def _end_to_end_offsets(
    reference: Sequence[RefNote], estimated: Sequence[RefNote]
) -> Tuple[float, float]:
    """The best constant shift for the first third of a recording, and the last.

    Equal values mean one clock offset, which a constant corrects. Different
    values mean the two sides run at different rates -- a score quantised to a
    nominal tempo against a performance that breathes -- and no constant does.
    """
    span_start = min(start for start, _, _ in reference)
    span_end = max(start for start, _, _ in reference)
    span = span_end - span_start
    if span < DRIFT_MIN_SPAN_S:
        return 0.0, 0.0
    third = span / 3.0
    search = min(DRIFT_SEARCH_S, span / 8.0)

    def window(notes, low, high):
        return [n for n in notes if low <= n[0] <= high]

    out = []
    for low, high in ((span_start, span_start + third), (span_end - third, span_end)):
        # The estimate is searched over a wider window than the reference, so a
        # note that has drifted a long way is still available to match.
        ref_part = window(reference, low, high)
        est_part = window(estimated, low - search, high + search)
        if (len(ref_part) < DRIFT_MIN_NOTES_PER_END
                or len(est_part) < DRIFT_MIN_NOTES_PER_END):
            out.append(0.0)
            continue
        shift, _ = _best_global_offset(ref_part, est_part, search_s=search,
                                       step_s=DRIFT_STEP_S)
        out.append(-shift)   # report where the estimate sits, not how to move it
    return out[0], out[1]


def _note_summary(notes: Sequence[RefNote], prefix: str) -> Dict[str, float]:
    """Pitch and duration summaries, for telling bad data from bad output."""
    if not notes:
        return {f"{prefix}_pitch_lo": 0, f"{prefix}_pitch_hi": 0,
                f"{prefix}_pitch_median": 0.0, f"{prefix}_duration_median": 0.0}
    pitches = np.array([p for _, _, p in notes])
    durations = np.array([b - a for a, b, _ in notes])
    return {
        f"{prefix}_pitch_lo": int(round(float(pitches.min()))),
        f"{prefix}_pitch_hi": int(round(float(pitches.max()))),
        f"{prefix}_pitch_median": float(np.median(pitches)),
        f"{prefix}_duration_median": float(np.median(durations)),
    }


def _bootstrap_interval(
    values: np.ndarray, samples: int, confidence: float, rng: np.random.Generator,
    groups: Optional[Sequence[int]] = None,
) -> Interval:
    """Percentile bootstrap over recordings, or over groups of them.

    Corpora cut into segments break one performance into many rows, and those
    rows are not independent observations: a singer whose vibrato defeats the
    decoder defeats it for every segment of that song. Resampling segments would
    treat each as fresh evidence and report an interval several times too narrow
    -- the same error as resampling notes instead of recordings, one level up.
    Where segments carry a group the resampling is over whole groups, which is
    the cluster bootstrap.
    """
    mean = float(np.mean(values)) if values.size else 0.0
    if values.size < 2 or samples < 1:
        return Interval(mean, mean, mean)

    if groups is not None:
        members: Dict[int, List[int]] = {}
        for index, group in enumerate(groups):
            members.setdefault(group, []).append(index)
        buckets = list(members.values())
        if len(buckets) < 2:
            return Interval(mean, mean, mean)
        draws = rng.integers(0, len(buckets), size=(samples, len(buckets)))
        means = np.array([
            values[np.concatenate([buckets[b] for b in row])].mean() for row in draws
        ])
    else:
        draws = rng.integers(0, values.size, size=(samples, values.size))
        means = values[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return Interval(
        mean=mean,
        low=float(np.percentile(means, 100 * tail)),
        high=float(np.percentile(means, 100 * (1.0 - tail))),
    )


def aggregate_scores(
    scores: Sequence[ItemScore],
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    alignment_threshold: float = 0.05,
) -> Aggregate:
    """Combine per-recording scores into a dataset result with error bars.

    The headline is the macro average -- the mean over recordings -- which is
    what the singing-transcription literature reports, and which weights a short
    song the same as a long one. The micro average, pooling every note, is
    reported alongside because the two diverge when recordings differ greatly in
    length, and a reader is entitled to see both.
    """
    scores = list(scores)
    if not scores:
        return Aggregate(0, 0, 0, {}, {}, 0.0, 0.0, [], bootstrap_samples, confidence)

    rng = np.random.default_rng(seed)

    # An id of the form "singer#song/segment" names the performance the segment
    # came from. Where several segments share one, the interval is resampled
    # over performances rather than segments.
    group_names = [s.item_id.rsplit("/", 1)[0] if "/" in s.item_id else s.item_id
                   for s in scores]
    distinct = sorted(set(group_names))
    index_of = {name: i for i, name in enumerate(distinct)}
    groups = ([index_of[n] for n in group_names]
              if len(distinct) < len(scores) else None)

    macro: Dict[str, Interval] = {}
    for name in ("con_f", "con_p", "con_r",
                 "conp_f", "conp_p", "conp_r",
                 "conpoff_f", "conpoff_p", "conpoff_r"):
        macro[name] = _bootstrap_interval(
            np.array([_extract(s, name) for s in scores], dtype=float),
            bootstrap_samples, confidence, rng, groups=groups,
        )

    micro = _micro_average(scores)

    matched_onsets = sum(round(s.con.recall * s.n_reference) for s in scores)
    octave_errors = sum(s.octave_errors for s in scores)

    return Aggregate(
        n_items=len(scores),
        n_reference_notes=sum(s.n_reference for s in scores),
        n_estimated_notes=sum(s.n_estimated for s in scores),
        macro=macro,
        micro=micro,
        n_groups=len(distinct) if groups is not None else 0,
        octave_error_rate=(octave_errors / matched_onsets) if matched_onsets else 0.0,
        median_best_offset_s=float(np.median([s.best_offset_s for s in scores])),
        items_needing_alignment=[
            s.item_id for s in scores if s.misalignment_gain > alignment_threshold
        ],
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )


def _extract(score: ItemScore, name: str) -> float:
    metric, field_name = name.rsplit("_", 1)
    prf: PRF = getattr(score, metric)
    return {"f": prf.f, "p": prf.precision, "r": prf.recall}[field_name]


def _micro_average(scores: Sequence[ItemScore]) -> Dict[str, float]:
    """Pool matches over every note in the dataset rather than averaging songs."""
    out: Dict[str, float] = {}
    for metric in ("con", "conp", "conpoff"):
        true_positives = 0.0
        n_ref = 0
        n_est = 0
        for score in scores:
            prf: PRF = getattr(score, metric)
            true_positives += prf.recall * score.n_reference
            n_ref += score.n_reference
            n_est += score.n_estimated
        precision = true_positives / n_est if n_est else 0.0
        recall = true_positives / n_ref if n_ref else 0.0
        f = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[f"{metric}_p"] = precision
        out[f"{metric}_r"] = recall
        out[f"{metric}_f"] = f
    return out


# A reference note shorter than this is at the edge of what a note-level
# annotation and a frame-level decoder can be expected to agree on at all.
SHORT_NOTE_S = 0.12


@dataclass
class MissAnatomy:
    """Why reference notes were missed, which is not the same as how many.

    Recall below precision means notes are being lost, but the fix depends
    entirely on how. Two notes run together needs a segmentation change; a note
    found at the wrong pitch needs a decoding change; a note nothing overlaps at
    all is usually one that was never voiced loudly enough to survive. Reporting
    only the total leaves the choice to guesswork.
    """

    merged: int = 0          # ran together with a neighbour at the same pitch
    wrong_pitch: int = 0     # onset found, pitch outside tolerance
    too_short: int = 0       # shorter than a decoder can resolve
    absent: int = 0          # nothing overlaps it
    missed_duration_median: float = 0.0
    found_duration_median: float = 0.0
    # How far from equal temperament the notes sat, in cents. An integer output
    # must spend this before its own error counts, so if the pitch misses sit
    # further out than the hits, the singers' intonation is the cause and no
    # change to the decoder recovers them.
    wrong_pitch_temperament_cents: float = 0.0
    found_temperament_cents: float = 0.0

    @property
    def total(self) -> int:
        return self.merged + self.wrong_pitch + self.too_short + self.absent

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["total"] = self.total
        return data


def analyse_misses(
    reference: Sequence[RefNote], estimated: Sequence[RefNote],
    *, onset_tolerance: float = ONSET_TOLERANCE,
    pitch_tolerance: float = PITCH_TOLERANCE,
) -> MissAnatomy:
    """Classify every reference note the transcription failed to match."""
    import mir_eval

    anatomy = MissAnatomy()
    if not reference or not estimated:
        anatomy.absent = len(reference)
        return anatomy

    ref_intervals, ref_hz = _to_mir_eval(reference)
    est_intervals, est_hz = _to_mir_eval(estimated)

    def matched_reference_indices(tolerance: float) -> set:
        return {ref for ref, _ in mir_eval.transcription.match_notes(
            ref_intervals, ref_hz, est_intervals, est_hz,
            onset_tolerance=onset_tolerance, pitch_tolerance=tolerance,
            offset_ratio=None)}

    found = matched_reference_indices(pitch_tolerance)
    onset_found = matched_reference_indices(IGNORE_PITCH_TOLERANCE)

    durations = np.array([b - a for a, b, _ in reference])
    missed_mask = np.array([i not in found for i in range(len(reference))])
    if missed_mask.any():
        anatomy.missed_duration_median = float(np.median(durations[missed_mask]))
    if (~missed_mask).any():
        anatomy.found_duration_median = float(np.median(durations[~missed_mask]))

    def off_temperament(p: float) -> float:
        return abs(p - round(p)) * 100.0

    hit_deviations = [off_temperament(p) for i, (_, _, p) in enumerate(reference)
                      if i in found]
    wrong_deviations: List[float] = []

    for index, (start, end, pitch) in enumerate(reference):
        if index in found:
            continue
        if index in onset_found:
            anatomy.wrong_pitch += 1
            wrong_deviations.append(off_temperament(pitch))
            continue
        # A note swallowed by a neighbour: some estimated note at the same pitch
        # spans this one's onset without starting near it. That is one estimate
        # covering two reference notes -- a re-articulation the decoder did not
        # split, which no amount of pitch accuracy will recover.
        swallowed = any(
            est_start < start - onset_tolerance and est_end > start
            and round(est_pitch) == round(pitch)
            for est_start, est_end, est_pitch in estimated
        )
        if swallowed:
            anatomy.merged += 1
        elif end - start < SHORT_NOTE_S:
            anatomy.too_short += 1
        else:
            anatomy.absent += 1

    if wrong_deviations:
        anatomy.wrong_pitch_temperament_cents = float(np.median(wrong_deviations))
    if hit_deviations:
        anatomy.found_temperament_cents = float(np.median(hit_deviations))
    return anatomy


@dataclass
class Comparison:
    """Whether one run beats another, tested on the recordings they share.

    Two runs over the same dataset are paired data, and comparing their separate
    confidence intervals throws that away: per-recording difficulty is the
    largest source of variance, and it is common to both runs. Overlapping
    intervals routinely hide a real, consistent improvement, and non-overlapping
    ones can appear where a couple of recordings moved a lot. Resampling the
    *differences* removes the shared difficulty and answers the question that is
    actually being asked.
    """

    metric: str
    n_items: int
    mean_a: float
    mean_b: float
    delta: float
    low: float
    high: float
    n_better: int
    n_worse: int

    @property
    def significant(self) -> bool:
        """Whether the interval for the difference excludes no change."""
        return self.low > 0.0 or self.high < 0.0

    def describe(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (f"{self.metric}: {self.mean_a:.3f} -> {self.mean_b:.3f} "
                f"({self.delta:+.3f}, 95% CI [{self.low:+.3f}, {self.high:+.3f}], "
                f"{verdict}); better on {self.n_better}, worse on {self.n_worse}")


def compare_runs(
    items_a: Dict[str, ItemScore], items_b: Dict[str, ItemScore],
    metric: str = "conp_f", *, bootstrap_samples: int = 2000, seed: int = 0,
    confidence: float = 0.95,
) -> Optional[Comparison]:
    """Paired bootstrap over the recordings two runs have in common."""
    shared = sorted(set(items_a) & set(items_b))
    if not shared:
        return None

    def value(score: ItemScore) -> float:
        family, field = metric.rsplit("_", 1)
        return getattr(getattr(score, family), {"f": "f", "p": "precision",
                                                "r": "recall"}[field])

    a = np.array([value(items_a[k]) for k in shared])
    b = np.array([value(items_b[k]) for k in shared])
    differences = b - a

    rng = np.random.default_rng(seed)
    means = np.array([
        differences[rng.integers(0, differences.size, differences.size)].mean()
        for _ in range(bootstrap_samples)
    ])
    tail = (1.0 - confidence) / 2.0 * 100.0

    return Comparison(
        metric=metric, n_items=len(shared),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        delta=float(differences.mean()),
        low=float(np.percentile(means, tail)),
        high=float(np.percentile(means, 100.0 - tail)),
        n_better=int((differences > 0).sum()),
        n_worse=int((differences < 0).sum()),
    )


# ---------------------------------------------------------------------------
# Time-free comparison.
#
# A reference notated at a nominal tempo drifts against the performance it
# describes, so nothing measured against a clock means anything. The notes are
# still right, and in the right order, and that much can be compared: a melody
# is a sequence, and getting its notes and their order right is most of what a
# transcription is for.
# ---------------------------------------------------------------------------


@dataclass
class SequenceScore:
    """How much of a melody's note sequence was recovered, ignoring time."""

    n_reference: int
    n_estimated: int
    matched: int
    precision: float
    recall: float
    f: float
    interval_f: float = 0.0
    """The same on the sequence of intervals, which is unchanged by
    transposition -- so a transcription an octave out still scores."""

    def to_dict(self) -> Dict:
        return asdict(self)


def _aligned_matches(left: Sequence[int], right: Sequence[int]) -> int:
    """Length of the best order-preserving correspondence between two sequences.

    Order-preserving is the point: a melody's notes come in an order, and a
    transcription that finds the same pitches scattered differently has not
    found the melody. Repeated notes are counted as repeated, so two notes run
    into one is a miss here exactly as it is under the timed metrics.
    """
    if not left or not right:
        return 0
    import difflib

    # autojunk discards elements it considers too popular, which in a melody is
    # the tonic. Disabling it is what keeps this from silently ignoring the
    # most common note in the tune.
    matcher = difflib.SequenceMatcher(None, list(left), list(right), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def score_sequence(
    reference: Sequence[RefNote], estimated: Sequence[RefNote], *,
    beta: float = 1.0,
) -> SequenceScore:
    """Compare two melodies as sequences of notes, without reference to time."""
    ref_pitches = [int(round(p)) for _, _, p in sorted(reference)]
    est_pitches = [int(round(p)) for _, _, p in sorted(estimated)]

    matched = _aligned_matches(ref_pitches, est_pitches)
    precision = matched / len(est_pitches) if est_pitches else 0.0
    recall = matched / len(ref_pitches) if ref_pitches else 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    f = (1.0 + b2) * precision * recall / denominator if denominator > 0 else 0.0

    ref_intervals = [b - a for a, b in zip(ref_pitches, ref_pitches[1:])]
    est_intervals = [b - a for a, b in zip(est_pitches, est_pitches[1:])]
    interval_matched = _aligned_matches(ref_intervals, est_intervals)
    interval_p = interval_matched / len(est_intervals) if est_intervals else 0.0
    interval_r = interval_matched / len(ref_intervals) if ref_intervals else 0.0
    interval_denominator = b2 * interval_p + interval_r
    interval_f = ((1.0 + b2) * interval_p * interval_r / interval_denominator
                  if interval_denominator > 0 else 0.0)

    return SequenceScore(
        n_reference=len(ref_pitches), n_estimated=len(est_pitches), matched=matched,
        precision=precision, recall=recall, f=f, interval_f=interval_f,
    )


def sequence_chance_floor(
    references: Sequence[Sequence[RefNote]], estimates: Sequence[Sequence[RefNote]],
    *, beta: float = 1.0,
) -> Dict[str, float]:
    """What this metric scores when the transcription belongs to another song.

    Two pop melodies in the same key share most of their notes, so comparing
    sequences has a floor well above zero: on a synthetic pair of unrelated
    tunes it is around 0.45. A score is only worth reading against that floor,
    and the floor depends on the corpus, so it is measured rather than assumed
    -- each transcription against the next song's reference.
    """
    if len(references) < 2:
        return {"f": 0.0, "interval_f": 0.0, "n": 0}

    scores, intervals = [], []
    for index, estimate in enumerate(estimates):
        wrong = references[(index + 1) % len(references)]
        if not wrong or not estimate:
            continue
        score = score_sequence(wrong, estimate, beta=beta)
        scores.append(score.f)
        intervals.append(score.interval_f)

    if not scores:
        return {"f": 0.0, "interval_f": 0.0, "n": 0}
    return {"f": float(np.mean(scores)), "interval_f": float(np.mean(intervals)),
            "n": len(scores)}
