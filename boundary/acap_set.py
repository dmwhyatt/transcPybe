"""Feature files and a manifest for a labelled set of recordings that has only notes, for fine-tuning
(FINETUNE-PLAN.md).

    from boundary.acap_set import build
    manifest = build(rows, out_dir, hold_back=4, seed=0)
    python -m boundary.train MANIFEST --onset-labels reference-starts --init CHECKPOINT --distill-articulation ...

Each row is a dict: ``id``; ``wav``, the recording; ``f0``, a pickled transcPybe ``F0Track`` of that audio at 16 kHz,
the one the decoder uses; and ``notes``, ``[onset, offset, MIDI pitch]`` in seconds. The features are made exactly as
at inference (``frame_features`` on the 16 kHz audio and that track), so the network trains on what it will be given.
The stored targets are ``features.targets`` of the notes: onset, offset and sounding. ``train`` adds the re-sung and
vowel targets as it loads. ``hold_back`` recordings, drawn with ``seed``, form the ``valid`` split, which ``train``
logs; the rest are ``train``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import FEATURE_SR, items as manifest
from .features import frame_features, targets


def build(rows, out_dir, hold_back: int = 4, seed: int = 0) -> Path:
    """Write ``out_dir/features/<id>.npz`` and ``out_dir/manifest.jsonl``; returns the manifest's path."""
    from transcpybe.audio import load_audio

    out_dir = Path(out_dir)
    if not 0 <= hold_back < len(rows):
        raise ValueError(f"hold_back must leave something to train on: {hold_back} of {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("recording ids must be unique")
    held = set(np.random.default_rng(seed).permutation(sorted(ids))[:hold_back].tolist())
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    items = []
    for r in rows:
        audio, _ = load_audio(r["wav"], FEATURE_SR)
        with open(r["f0"], "rb") as f:
            track = pickle.load(f)
        mel, aux = frame_features(audio, track)
        notes = [[float(on), float(off), float(p)] for on, off, p in sorted(r["notes"])]
        path = out_dir / "features" / f"{r['id']}.npz"
        np.savez(path, mel=mel, aux=aux, y=targets(notes, len(mel)))
        items.append(dict(mix_id=r["id"], split="valid" if r["id"] in held else "train", notes=notes,
                          features=str(path.resolve()), wav=str(r["wav"]), f0=str(r["f0"])))
    path = out_dir / "manifest.jsonl"
    manifest.write(path, items)
    return path
