"""Another pitch tracker's confidence put on CREPE full's scale, for the note decoder.

The note decoder makes notes only where a track's voicing (periodicity, times a loudness gate) is at least 0.3, so a
tracker whose confidence runs lower than CREPE's loses notes whatever the networks say. PESTO's and pYIN's do: on
vocadito, PESTO passed 84% of the sung frames at 0.3 to CREPE full's 99%.

Each of the two has a voicing map: a quantile map from its periodicity to CREPE full's, fitted on the training
data's validation singers, never on a set we score on. It is monotone, so the tracker's ranking of frames is kept,
and after it the share of frames above any threshold is CREPE's. The decoder gets the mapped track; the networks get
the tracker's own.

Copied from the training repository's ``boundary/voicing_map.py`` (boundary 56d97f6): ``map_periodicity`` and
``apply``, unchanged. The maps themselves are ``weights/voicing-*.json``, as that module's ``fit`` wrote them.
"""
from __future__ import annotations

import dataclasses

import numpy as np


def map_periodicity(p: np.ndarray, m: dict) -> np.ndarray:
    return np.clip(np.interp(p, m["src"], m["dst"]), 0.0, 1.0)


def apply(track, m: dict):
    """``track`` (a transcPybe ``F0Track``) with its periodicity mapped and its voicing rebuilt from it as
    ``transcpybe.f0.estimate_f0`` builds it: periodicity times the loudness gate, zero where there is no pitch."""
    p = map_periodicity(np.asarray(track.periodicity, dtype=np.float64), m)
    voicing = np.where(np.isfinite(track.f0_hz), p * track.loudness_gate, 0.0)
    return dataclasses.replace(track, periodicity=p, voicing=np.clip(voicing, 0.0, 1.0))
