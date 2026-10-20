"""Decode the RWC-P development half with a trained network's onsets, for scoring against configuration B.

    PYTHONPATH=<this repo>:<spurious-notes tree>/src \
        python -m boundary.rwc_dev CHECKPOINT --work ~/transcription-benchmarking/orchestrator/bench-work-rwc-20261019 \
            --name net-r1 [--thresholds 0.3 0.5 0.7]

Runs on musix, in the RWC work directory's own replay set-up: the cached f0
tracks in ``dev/f0cache/sp`` (spurious-notes tree c7d3ab2) and the frozen
configuration B in ``dev/frozen.json``. For each threshold it writes two runs
to ``dev/runs/sp/<name>-<variant>-t<threshold>/midi``:

- ``B``:       configuration B with the network's onsets in place of
  transcPybe's detector (B's voicing-break and timbre splits still added);
- ``B-plain``: the same, without the voicing-break and timbre splits, which
  asks whether the network replaces those heuristics rather than adds to them.

Then score with the work directory's own scorer, against B's replay:

    python dev/score.py dev sp-f-delta0.6 sp-f-delta0.6=dev/runs/sp/f-delta0.6/midi \
        <name>-B-t0.5=dev/runs/sp/<name>-B-t0.5/midi ...

Development half only. The test half is never decoded here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

from . import FEATURE_SR


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from transcpybe.audio import load_audio
    from transcpybe.config import TranscriptionConfig
    from transcpybe.core import decode_notes
    from transcpybe.midi_io import write_midi

    from .features import frame_features
    from .infer import onset_track, posteriors
    from .model import load

    W = args.work
    songs = json.load(open(W / "dev" / "split.json"))["dev"]
    b = json.load(open(W / "dev" / "frozen.json"))["B-full"]
    variants = {"B": b, "B-plain": dict(b, timbre_split_ratio=0.0, split_on_voicing_breaks=False)}
    model = load(args.checkpoint, args.device)
    runs = {}
    for v, ov in variants.items():
        for t in args.thresholds:
            name = f"{args.name}-{v}-t{t}"
            runs[name] = (dataclasses.replace(TranscriptionConfig(device="cpu"), **ov), t)
            (W / "dev/runs/sp" / name / "midi").mkdir(parents=True, exist_ok=True)
            json.dump(dict(ov, checkpoint=str(args.checkpoint.resolve()), threshold=t),
                      open(W / "dev/runs/sp" / name / "overrides.json", "w"), indent=1)
    for i, s in enumerate(songs):
        audio, _ = load_audio(W / "stems_flat" / f"{s}.wav", FEATURE_SR)
        track = pickle.load(open(W / "dev/f0cache/sp" / f"{s}.pkl", "rb"))
        mel, aux = frame_features(audio, track)
        p = posteriors(model, mel, aux, args.device)[:, 0]
        for name, (cfg, t) in runs.items():
            notes, _, _, _ = decode_notes(track, cfg, audio=audio, onsets=onset_track(p, t))
            write_midi(notes, W / "dev/runs/sp" / name / "midi" / f"{s}.mid", cfg)
        print(f"{i + 1}/{len(songs)} {s}", flush=True)
    print("score with:\n  python dev/score.py dev sp-f-delta0.6 sp-f-delta0.6=dev/runs/sp/f-delta0.6/midi "
          + " ".join(f"{n}=dev/runs/sp/{n}/midi" for n in runs))


if __name__ == "__main__":
    main()
