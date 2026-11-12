"""Fit the transcriber's parameters to a dataset, and measure the fit honestly.

The transcriber has no learned weights; it has parameters. Fitting them is a
search, and a search over enough configurations will find one that flatters
whatever it was scored on. The machinery here exists mostly to stop that:

* Recordings are split once, by recording and stratified, into fit, select and
  report sets. The search runs on fit, the winner is chosen on select, and the
  report set is scored exactly once at the end.
* The score is F-beta with beta above one, weighting recall. Against a reference
  that lists only a song's main melody, anything else the singer sang is a false
  positive that no parameter can fix, so plain F rewards emitting fewer notes --
  a config that transcribes less looks better while being worse.
* Estimating f0 is most of the cost and no parameter searched here changes it,
  so each recording's f0 and onset tracks are computed once and replayed. The
  replay is the same code path the transcriber takes, so a sweep cannot measure
  something the pipeline does not do.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from transcription_benchmarking.dataset import Item
from transcription_benchmarking.metrics import (
    OFFSET_RATIO, ONSET_TOLERANCE, PITCH_TOLERANCE, RefNote, note_prf,
    score_sequence)

from .config import TranscriptionConfig

# Parameters that change the f0 or onset tracks, and so invalidate a cached one.
# Everything else runs downstream of them and can be swept for free.
UPSTREAM_FIELDS = (
    "sample_rate", "hop_ms", "f0_backend", "crepe_model", "device", "seed",
    "fmin_hz", "fmax_hz", "crepe_batch_size", "onset_max_size", "onset_lag",
    "onset_n_fft", "onset_aggregate", "onset_delta_ratio",
    "use_onsets",
)

DEFAULT_BETA = 1.5


def fbeta(precision: float, recall: float, beta: float = DEFAULT_BETA) -> float:
    """F with recall weighted beta squared times as heavily as precision."""
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    return 0.0 if denominator <= 0 else (1.0 + b2) * precision * recall / denominator


@dataclass
class Split:
    """Which recordings are used to fit, to select, and to report."""

    fit: List[str] = field(default_factory=list)
    select: List[str] = field(default_factory=list)
    report: List[str] = field(default_factory=list)
    stratified_by: str = ""
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"fit": self.fit, "select": self.select, "report": self.report,
                "stratified_by": self.stratified_by, "seed": self.seed}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Split":
        return cls(fit=list(payload["fit"]), select=list(payload["select"]),
                   report=list(payload["report"]),
                   stratified_by=payload.get("stratified_by", ""),
                   seed=int(payload.get("seed", 0)))


def make_split(
    item_ids: Sequence[str],
    groups: Optional[Dict[str, str]] = None,
    *,
    fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
    stratified_by: str = "",
) -> Split:
    """Divide recordings into fit, select and report, stratified within groups.

    Stratifying matters when the corpus spans decades: 1950s and 2020s
    recordings differ in almost every way that affects a pitch tracker, and an
    unstratified draw can put most of one era on one side of the split and
    measure era rather than parameters.
    """
    by_group: Dict[str, List[str]] = {}
    for item_id in item_ids:
        by_group.setdefault((groups or {}).get(item_id, ""), []).append(item_id)

    rng = random.Random(seed)
    split = Split(stratified_by=stratified_by, seed=seed)
    for group in sorted(by_group):
        members = sorted(by_group[group])
        rng.shuffle(members)
        n = len(members)
        n_fit = round(n * fractions[0])
        n_select = round(n * fractions[1])
        # Small groups must not spend every member on fitting, leaving nothing
        # to select or report with.
        if n >= 3:
            n_fit = min(n_fit, n - 2)
            n_select = min(max(n_select, 1), n - n_fit - 1)
        split.fit += members[:n_fit]
        split.select += members[n_fit:n_fit + n_select]
        split.report += members[n_fit + n_select:]
    return split


def save_track(path: Path, track, onsets) -> None:
    """Store one recording's f0 and onset tracks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"f0_{name}": getattr(track, name) for name in
              ("times", "f0_hz", "periodicity", "loudness_db", "loudness_gate", "voicing")}
    if onsets is not None:
        arrays.update({f"on_{name}": getattr(onsets, name) for name in
                       ("times", "strengths", "envelope", "envelope_times")})
    np.savez_compressed(path, has_onsets=np.array([onsets is not None]), **arrays)


def load_track(path: Path):
    """Read back what save_track wrote."""
    from .f0 import F0Track
    from .onsets import OnsetTrack

    with np.load(path) as data:
        track = F0Track(*(data[f"f0_{name}"] for name in
                          ("times", "f0_hz", "periodicity", "loudness_db",
                           "loudness_gate", "voicing")))
        onsets = None
        if bool(data["has_onsets"][0]):
            onsets = OnsetTrack(*(data[f"on_{name}"] for name in
                                  ("times", "strengths", "envelope", "envelope_times")))
    return track, onsets


def upstream_fingerprint(config: TranscriptionConfig) -> str:
    """Identify a cached track by only what could have changed it."""
    values = {name: getattr(config, name, None) for name in UPSTREAM_FIELDS}
    return json.dumps(values, sort_keys=True, default=str)


# The parameters the diagnostics point at, with the ranges worth exploring.
# Deliberately few: a search over all fifty-odd fields would find a
# configuration that fits the fit set and nothing else.
DEFAULT_SPACE: Dict[str, Tuple[float, float]] = {
    "onset_split_strength_ratio": (0.15, 0.80),   # merged repeated notes
    "onset_split_min_gap_s": (0.03, 0.12),
    "p_self": (0.965, 0.995),                     # short notes
    "min_note_duration_s": (0.030, 0.120),
    "note_level_floor_db": (15.0, 45.0),          # notes dropped as residue
    "pitch_sigma_semitones": (0.35, 0.80),
}

# Only worth searching against a reference whose note ends are meant seriously
# -- pass it explicitly along with --objective offset. It is not in
# DEFAULT_SPACE because an onset objective cannot see it, and an inert
# parameter in a random search spends trials to learn nothing.
RELEASE_SPACE: Dict[str, Tuple[float, float]] = {
    "release_drop_db": (3.0, 30.0),
}


def cache_name(item_id: str) -> str:
    """A filesystem-safe name for one recording's cached tracks.

    The digest is hashlib rather than the builtin hash(): Python salts string
    hashing per interpreter, so a name built from hash() is different on every
    run. The cache then never finds what the last run wrote, silently
    recomputes everything, and writes it under another name nobody will look
    for either -- a cache that only ever costs.
    """
    safe = "".join(c if c.isalnum() or c in "._-#" else "_" for c in item_id)
    # Unconditional, because the sanitiser above maps distinct ids onto one
    # name: a corpus laid out one directory per song names every segment
    # 0000, and M4Singer writes its titles in Chinese, which it strips
    # entirely.
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:100] or 'item'}_{digest}.npz"


class StageCache:
    """f0 and onset tracks per recording, computed once and replayed."""

    def __init__(self, directory: Path, config: TranscriptionConfig) -> None:
        self.directory = Path(directory)
        self.fingerprint = upstream_fingerprint(config)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stamp = self.directory / "upstream.json"
        # A cache built with a different f0 backend or hop is not this cache.
        # Silently reusing it would sweep parameters against the wrong tracks.
        if self._stamp.exists() and self._stamp.read_text() != self.fingerprint:
            for stale in self.directory.glob("*.npz"):
                stale.unlink()
        self._stamp.write_text(self.fingerprint)
        self._memory: Dict[str, Any] = {}

    def get(self, item: Item, config: TranscriptionConfig):
        if item.item_id in self._memory:
            return self._memory[item.item_id]

        path = self.directory / cache_name(item.item_id)
        if path.exists():
            tracks = load_track(path)
        else:
            from .audio import load_audio
            from .core import resolve_config_device
            from .f0 import estimate_f0
            from .onsets import detect_onsets

            audio, _ = load_audio(item.audio, config.sample_rate)
            resolved, _ = resolve_config_device(config)
            track = estimate_f0(audio, resolved)
            onsets = detect_onsets(audio, resolved) if resolved.use_onsets else None
            save_track(path, track, onsets)
            tracks = (track, onsets)

        self._memory[item.item_id] = tracks
        return tracks


def score_config(
    config: TranscriptionConfig,
    items: Sequence[Item],
    references: Dict[str, Sequence[RefNote]],
    cache: StageCache,
    *,
    beta: float = DEFAULT_BETA,
    onset_tolerance: float = ONSET_TOLERANCE,
    objective: str = "onset",
) -> Dict[str, float]:
    """Mean F-beta over recordings for one configuration.

    ``objective`` selects what is compared. "onset" needs the reference to share
    a clock with the audio. "sequence" compares the notes in order and ignores
    time entirely, which is the only honest choice against a reference notated
    at a nominal tempo -- and it still sees the defect that matters most here,
    since two notes run into one shortens the sequence. "offset" is "onset"
    with the note end required as well, and exists because without it nothing
    in this package could optimise a note end: both other objectives discard
    offsets entirely, so a search could not have improved one even by accident.
    """
    from .core import decode_notes

    scores, precisions, recalls = [], [], []
    per_item: Dict[str, float] = {}
    for item in items:
        reference = references.get(item.item_id)
        if not reference:
            continue
        track, onsets = cache.get(item, config)
        notes, _, _, _ = decode_notes(track, config, onsets=onsets)
        estimated = [(n.onset, n.offset, n.pitch) for n in notes]
        if objective == "sequence":
            sequence = score_sequence(reference, estimated, beta=beta)
            scores.append(sequence.f)
            precisions.append(sequence.precision)
            recalls.append(sequence.recall)
            per_item[item.item_id] = sequence.f
        else:
            prf = note_prf(reference, estimated, pitch_tolerance=PITCH_TOLERANCE,
                       offset_ratio=(OFFSET_RATIO if objective == "offset" else None),
                       onset_tolerance=onset_tolerance)
            scores.append(fbeta(prf.precision, prf.recall, beta))
            precisions.append(prf.precision)
            recalls.append(prf.recall)
            per_item[item.item_id] = fbeta(prf.precision, prf.recall, beta)

    if not scores:
        return {"fbeta": 0.0, "precision": 0.0, "recall": 0.0, "n": 0, "per_item": {}}
    return {"fbeta": float(np.mean(scores)), "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)), "n": len(scores), "per_item": per_item}


def sample_config(
    base: TranscriptionConfig, space: Dict[str, Tuple[float, float]], rng: random.Random
) -> TranscriptionConfig:
    """Draw one configuration, keeping the integer fields integral."""
    import dataclasses

    changes = {}
    for name, (low, high) in space.items():
        current = getattr(base, name)
        value = rng.uniform(low, high)
        changes[name] = int(round(value)) if isinstance(current, int) else value
    return dataclasses.replace(base, **changes)


def random_search(
    base: TranscriptionConfig,
    fit_items: Sequence[Item],
    select_items: Sequence[Item],
    references: Dict[str, Sequence[RefNote]],
    cache: StageCache,
    *,
    space: Optional[Dict[str, Tuple[float, float]]] = None,
    trials: int = 60,
    beta: float = DEFAULT_BETA,
    seed: int = 0,
    objective: str = "onset",
    on_trial: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    """Search on the fit set, choose on the select set, and say what it cost.

    Random rather than grid: with six parameters a grid spends most of its
    budget resolving dimensions that do not matter, and random search covers
    each dimension at every trial.

    The winner is chosen by the select score, not the fit score. The best of
    many fit scores is biased upward by the search itself -- that is what a
    search does -- and reporting it would be reporting the bias.
    """
    space = space or DEFAULT_SPACE
    rng = random.Random(seed)

    baseline_fit = score_config(base, fit_items, references, cache, beta=beta,
                                objective=objective)
    baseline_select = score_config(base, select_items, references, cache, beta=beta,
                                   objective=objective)

    trials_log: List[Dict[str, Any]] = []
    best = {"config": base, "select": baseline_select, "fit": baseline_fit,
            "trial": 0, "changes": {}}

    for trial in range(1, trials + 1):
        candidate = sample_config(base, space, rng)
        fit_score = score_config(candidate, fit_items, references, cache,
                                 beta=beta, objective=objective)
        # Only configurations that lead on the fit set are spent on the select
        # set, so selection sees far fewer candidates and is biased far less.
        select_score = None
        if fit_score["fbeta"] >= baseline_fit["fbeta"]:
            select_score = score_config(candidate, select_items, references,
                                        cache, beta=beta, objective=objective)
            if select_score["fbeta"] > best["select"]["fbeta"]:
                best = {"config": candidate, "select": select_score, "fit": fit_score,
                        "trial": trial,
                        "changes": {k: getattr(candidate, k) for k in space}}
        trials_log.append({"trial": trial, "fit": fit_score["fbeta"],
                           "select": select_score["fbeta"] if select_score else None,
                           "params": {k: getattr(candidate, k) for k in space}})
        if on_trial:
            on_trial(trial, fit_score)

    return {"best": best, "baseline_fit": baseline_fit,
            "baseline_select": baseline_select, "trials": trials_log,
            "beta": beta, "objective": objective, "space": {k: list(v) for k, v in space.items()}}


def paired_delta(
    baseline: Dict[str, float], tuned: Dict[str, float], *,
    samples: int = 2000, seed: int = 0, confidence: float = 0.95,
) -> Dict[str, Any]:
    """Bootstrap the per-recording difference, not the two means separately.

    Each recording is scored under both configurations, so the difficulty of
    the recording -- which both share -- cancels. Two separate intervals do not
    answer whether a change helped; the interval on the difference does.
    """
    shared = sorted(set(baseline) & set(tuned))
    deltas = np.array([tuned[k] - baseline[k] for k in shared], dtype=float)
    if deltas.size < 2:
        mean = float(deltas.mean()) if deltas.size else 0.0
        return {"mean": mean, "low": mean, "high": mean, "n": int(deltas.size),
                "better": 0, "worse": 0}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean": float(deltas.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "n": int(deltas.size),
        "better": int((deltas > 0).sum()),
        "worse": int((deltas < 0).sum()),
    }


def cross_validate(
    base: TranscriptionConfig,
    items: Sequence[Item],
    references: Dict[str, Sequence[RefNote]],
    cache: StageCache,
    *,
    folds: int = 5,
    space: Optional[Dict[str, Tuple[float, float]]] = None,
    trials: int = 60,
    beta: float = DEFAULT_BETA,
    seed: int = 0,
    objective: str = "onset",
    select_fraction: float = 0.25,
    on_fold: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Tune and report on every recording, none of them in the same fold.

    A single three-way split of a small corpus leaves too few recordings to
    report on: forty items split 60/20/20 report on eight, and eight cannot
    distinguish any improvement a search of this kind produces. Cross
    validation scores all forty, each by a search that never saw it, so the
    estimate rests on the whole corpus instead of a fifth of it.

    The search inside each fold still selects on data it did not fit on, so the
    winner is not chosen by the number being reported. What this buys is power,
    not a licence to select on the report set.

    It answers "is there anything here", not "here is the tuned model". Every
    recording has been fitted on in some fold, so the corpus is spent: a number
    from it is not an estimate of performance on unseen material, and the
    per-fold configurations are five answers rather than one.
    """
    scorable = [item for item in items if references.get(item.item_id)]
    if len(scorable) < folds * 3:
        raise ValueError(
            f"{len(scorable)} recordings cannot support {folds} folds; each fold "
            "needs enough left over to fit and select on")

    ordered = sorted(scorable, key=lambda item: item.item_id)
    random.Random(seed).shuffle(ordered)
    parts = [ordered[k::folds] for k in range(folds)]

    baseline_by_item: Dict[str, float] = {}
    tuned_by_item: Dict[str, float] = {}
    per_fold: List[Dict[str, Any]] = []

    for k in range(folds):
        report = parts[k]
        rest = [item for j, part in enumerate(parts) if j != k for item in part]
        n_select = max(1, round(len(rest) * select_fraction))
        select, fit = rest[:n_select], rest[n_select:]

        # A different seed per fold: five searches down the same random path
        # would agree with each other about nothing but the path.
        result = random_search(base, fit, select, references, cache, space=space,
                               trials=trials, beta=beta, seed=seed + k,
                               objective=objective)
        best = result["best"]
        before = score_config(base, report, references, cache, beta=beta,
                              objective=objective)
        after = score_config(best["config"], report, references, cache, beta=beta,
                             objective=objective)
        baseline_by_item.update(before["per_item"])
        tuned_by_item.update(after["per_item"])

        fold = {"fold": k, "n_report": before["n"], "changes": best["changes"],
                "baseline": before["fbeta"], "tuned": after["fbeta"]}
        per_fold.append(fold)
        if on_fold:
            on_fold(k, fold)

    return {
        "folds": per_fold,
        "kept_default": sum(1 for fold in per_fold if not fold["changes"]),
        "baseline": float(np.mean(list(baseline_by_item.values()))),
        "tuned": float(np.mean(list(tuned_by_item.values()))),
        "delta": paired_delta(baseline_by_item, tuned_by_item, seed=seed),
        "stability": parameter_stability(base, per_fold, space or DEFAULT_SPACE),
        "per_item": {"baseline": baseline_by_item, "tuned": tuned_by_item},
    }


def parameter_stability(
    base: TranscriptionConfig, per_fold: Sequence[Dict[str, Any]],
    space: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """How much the folds agreed about each parameter.

    Five folds choosing one value are evidence; five choosing five scattered
    values are the search fitting noise, and in any single run the two look
    identical -- one chosen number either way.

    Three things separate them, and all three are needed.

    ``moved`` is how many folds changed the parameter at all. A fold whose
    search beat nothing keeps the defaults and contributes no value, and
    counting only the folds that spoke turns one fold's opinion into a
    unanimous one: a single value has a range of zero and agrees with itself
    perfectly. That is the most flattering possible reading of the weakest
    possible evidence, so abstentions are counted here rather than dropped.

    ``spread`` is how much of the searched range the choices cover, near 1
    meaning the folds used all of it -- what drawing at random looks like. It
    is undefined below two values, and reported as such rather than as zero.

    ``agreement`` is how many of the folds that moved it went the same way;
    near 0.5 they disagree about even the direction. A parameter can agree on
    direction and still be scattered, so a table showing only this would call
    that stable.
    """
    stability: Dict[str, Dict[str, float]] = {}
    n_folds = len(per_fold)
    names = sorted({name for fold in per_fold for name in fold["changes"]})
    for name in names:
        values = [float(fold["changes"][name]) for fold in per_fold
                  if name in fold["changes"]]
        if not values:
            continue
        low, high = min(values), max(values)
        current = float(getattr(base, name, 0.0))
        searched = (space or {}).get(name)
        width = (searched[1] - searched[0]) if searched else 0.0
        stability[name] = {
            "median": float(np.median(values)),
            "low": low,
            "high": high,
            "baseline": current,
            "moved": len(values),
            "folds": n_folds,
            "spread": ((high - low) / width
                       if width > 0 and len(values) > 1 else float("nan")),
            "agreement": max(
                sum(1 for v in values if v > current),
                sum(1 for v in values if v < current),
            ) / len(values),
        }
    return stability
