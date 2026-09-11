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

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TranscriptionConfig
from .dataset import Item
from .metrics import (ONSET_TOLERANCE, PITCH_TOLERANCE, RefNote, note_prf,
                      score_sequence)

# Parameters that change the f0 or onset tracks, and so invalidate a cached one.
# Everything else runs downstream of them and can be swept for free.
UPSTREAM_FIELDS = (
    "sample_rate", "hop_ms", "f0_backend", "crepe_model", "device", "seed",
    "fmin_hz", "fmax_hz", "crepe_batch_size", "onset_max_size", "onset_lag",
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
    from ..f0 import F0Track
    from ..onsets import OnsetTrack

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

        safe = "".join(c if c.isalnum() or c in "._-#" else "_" for c in item.item_id)
        path = self.directory / f"{safe[:100]}_{abs(hash(item.item_id)) % 10**8}.npz"
        if path.exists():
            tracks = load_track(path)
        else:
            from ..audio import load_audio
            from ..core import resolve_config_device
            from ..f0 import estimate_f0
            from ..onsets import detect_onsets

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
    since two notes run into one shortens the sequence.
    """
    from ..core import decode_notes

    scores, precisions, recalls = [], [], []
    for item in items:
        reference = references.get(item.item_id)
        if not reference:
            continue
        track, onsets = cache.get(item, config)
        notes, _, _ = decode_notes(track, config, onsets=onsets)
        estimated = [(n.onset, n.offset, n.pitch) for n in notes]
        if objective == "sequence":
            sequence = score_sequence(reference, estimated, beta=beta)
            scores.append(sequence.f)
            precisions.append(sequence.precision)
            recalls.append(sequence.recall)
        else:
            prf = note_prf(reference, estimated, pitch_tolerance=PITCH_TOLERANCE,
                       offset_ratio=None, onset_tolerance=onset_tolerance)
            scores.append(fbeta(prf.precision, prf.recall, beta))
            precisions.append(prf.precision)
            recalls.append(prf.recall)

    if not scores:
        return {"fbeta": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}
    return {"fbeta": float(np.mean(scores)), "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)), "n": len(scores)}


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
