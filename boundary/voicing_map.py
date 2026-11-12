"""Another pitch tracker's confidence put on CREPE full's scale, for transcPybe's note decoder.

    python -m boundary.voicing_map CREPE.jsonl OTHER.jsonl OUT.json [--split valid]

The note decoder makes notes only where a track's voicing (periodicity, times a loudness gate) is high, so a tracker
whose confidence runs lower than CREPE's loses notes whatever the boundary network says. On 4 November PESTO passed
84% of vocadito's sung frames at 0.3 to CREPE's 99%, and lost notes accordingly.

The two manifests hold the same mixes (``swap_features``), one made with CREPE full and one with OTHER, so every
frame has both trackers' periodicity. This fits a quantile map from OTHER's periodicity to CREPE's over the frames of
the ``--split`` rows: monotone, so OTHER's ranking of frames is kept, and after it the share of frames above any
threshold is CREPE's. It is fitted on training data only, never on a set we score on. ``apply`` uses it on a track.

OUT.json holds the map and, on the same frames, the share of frames whose voicing is at least 0.3: CREPE's,
OTHER's, and OTHER's after the map.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from . import items as manifest

PERIODICITY, VOICING = 2, 3          # aux columns (features.frame_features)
LEVELS = 201
THRESHOLD = 0.3                      # transcPybe's voicing_threshold


def fit(other: np.ndarray, crepe: np.ndarray, levels: int = LEVELS) -> dict:
    """The quantile map from ``other`` to ``crepe`` (same frames), as increasing knots for ``np.interp``."""
    q = np.linspace(0.0, 1.0, levels)
    src, dst = np.quantile(other, q), np.quantile(crepe, q)
    keep = np.concatenate([[True], np.diff(src) > 1e-9])   # np.interp wants increasing knots; ties keep the first
    return dict(src=[float(x) for x in src[keep]], dst=[float(x) for x in dst[keep]])


def map_periodicity(p: np.ndarray, m: dict) -> np.ndarray:
    return np.clip(np.interp(p, m["src"], m["dst"]), 0.0, 1.0)


def apply(track, m: dict):
    """``track`` (a transcPybe ``F0Track``) with its periodicity mapped and its voicing rebuilt from it as
    ``transcpybe.f0.estimate_f0`` builds it: periodicity times the loudness gate, zero where there is no pitch."""
    p = map_periodicity(np.asarray(track.periodicity, dtype=np.float64), m)
    voicing = np.where(np.isfinite(track.f0_hz), p * track.loudness_gate, 0.0)
    return dataclasses.replace(track, periodicity=p, voicing=np.clip(voicing, 0.0, 1.0))


def columns(rows: list, cols: tuple) -> list:
    """Each row's aux columns, from its shard or feature file."""
    from .pack import read

    shards, out = {}, []
    for r in rows:
        if "shard" in r:
            z = read(r["shard"], r["index"], shards)
        else:
            with np.load(r["features"]) as f:
                z = {"aux": f["aux"]}
        out.append(z["aux"][:, cols].astype(np.float64))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crepe", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", default="valid", help="rows to fit on (default: validation singers)")
    args = ap.parse_args(argv)

    crepe = {r["mix_id"]: r for r in manifest.split(manifest.read(args.crepe), args.split)}
    other = [r for r in manifest.split(manifest.read(args.other), args.split) if r["mix_id"] in crepe]
    if not other:
        raise SystemExit(f"no {args.split} rows of {args.other} are in {args.crepe}")
    c = columns([crepe[r["mix_id"]] for r in other], (PERIODICITY, VOICING))
    o = columns(other, (PERIODICITY, VOICING))
    bad = [r["mix_id"] for r, a, b in zip(other, c, o) if len(a) != len(b)]
    if bad:
        raise SystemExit(f"{len(bad)} rows have different frame counts in the two manifests, e.g. {bad[0]}")
    c, o = np.concatenate(c), np.concatenate(o)
    m = fit(o[:, 0], c[:, 0])
    gate = np.where(o[:, 0] > 1e-6, o[:, 1] / np.maximum(o[:, 0], 1e-6), 0.0)   # voicing = periodicity x gate
    mapped = map_periodicity(o[:, 0], m) * np.clip(gate, 0.0, 1.0)
    report = dict(crepe=str(args.crepe.resolve()), other=str(args.other.resolve()), split=args.split,
                  recordings=len(other), frames=int(len(c)), threshold=THRESHOLD,
                  voiced_share=dict(crepe=float(np.mean(c[:, 1] >= THRESHOLD)), other=float(np.mean(o[:, 1] >= THRESHOLD)),
                                    other_mapped=float(np.mean(mapped >= THRESHOLD))),
                  map=m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    v = report["voiced_share"]
    print(f"{len(other)} {args.split} recordings, {len(c)} frames: voiced at {THRESHOLD}: CREPE {v['crepe']:.3f}, "
          f"other {v['other']:.3f}, other mapped {v['other_mapped']:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
