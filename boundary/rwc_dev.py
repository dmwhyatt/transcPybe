"""Decode the RWC-P development half with a trained network's onsets, for scoring against configuration B.

    PYTHONPATH=<this repo>:<spurious-notes tree>/src \
        python -m boundary.rwc_dev CHECKPOINT --work ~/transcription-benchmarking/orchestrator/bench-work-rwc-20261019 \
            --name net-r1 [--thresholds 0.3 0.5 0.7] [--split-policies all same-pitch] [--split-thresholds 0.7 0.9]

Runs on musix, in the RWC work directory's own replay set-up: the cached f0
tracks in ``dev/f0cache/sp`` (spurious-notes tree c7d3ab2) and the frozen
configuration B in ``dev/frozen.json``. For each threshold it writes two runs
to ``dev/runs/sp/<name>-<variant>-t<threshold>/midi``:

- ``B``:       configuration B with the network's onsets in place of
  transcPybe's detector (B's voicing-break and timbre splits still added);
- ``B-plain``: the same, without the voicing-break and timbre splits, which
  asks whether the network replaces those heuristics rather than adds to them.

With ``--split-policies`` / ``--split-thresholds`` (see ``infer.onset_track``),
every combination is decoded as well, named ``<name>-<variant>-t<threshold>``
for the default (every peak splits) and ``...-sp<split threshold>`` for
same-pitch-only splitting.

Three policies follow the decoder plan (decoder-plan.md, 26 October). In each,
the onset output's peaks only move note starts and never cut a note:

- ``resung``: the fourth output's peaks at each ``--split-thresholds`` value cut
  notes (``...-rs<s>``); it needs a network with four outputs;
- ``snap``: nothing cuts (``...-snap``);
- ``oracle``: the reference's re-sung note starts cut (``...-oracle``), from
  ``refs/notes``; development-only, it selects nothing.

The sounding plan (sounding-plan.md, 26 October) adds ``own``: transcPybe's own
onset detector, as configuration B uses it, with only the network's sounding
output added (``...-own``; ``--thresholds`` does not apply). Every hand-over
carries the sounding output, which the decoder uses only when
``sustain_merge_above`` is set (``--config-overrides sustain_merge_above=0.98``).
The ends plan (ends-plan.md, 26 October) adds the offset output's peaks at 0.5
as note end times, on every hand-over too; the decoder moves note ends to them
only when ``end_snap_window_s`` is set.

``--variants`` limits the decoder configurations (the plan uses ``B-plain`` only).
``--config-overrides KEY=VALUE ...`` sets further decoder options on top of the
variant (values read as JSON where they parse, else as text), and ``--suffix``
is appended to every run name, as the pitch plan does (``...-p1``).

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

import numpy as np

from . import FEATURE_SR

OFFSET_THRESHOLD = 0.5      # the ends plan: offset-output peaks at this level are note end times


def parse_overrides(items) -> dict:
    """``KEY=VALUE`` strings as a dict; a value is JSON where it parses (true, 0.05), else text."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--config-overrides wants KEY=VALUE, not {item!r}")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--split-policies", nargs="+", default=["all"],
                    choices=("all", "same-pitch", "across-gap", "resung", "snap", "oracle", "own"))
    ap.add_argument("--variants", nargs="+", default=["B", "B-plain"], choices=("B", "B-plain"))
    ap.add_argument("--split-thresholds", type=float, nargs="+", default=[None],
                    help="for same-pitch splitting: the bar a peak must reach to split (default: the peak threshold)")
    ap.add_argument("--config-overrides", nargs="+", default=[], metavar="KEY=VALUE",
                    help="further decoder options, applied on top of the variant")
    ap.add_argument("--suffix", default="", help="appended to every run name")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    extra = parse_overrides(args.config_overrides)

    from transcpybe.audio import load_audio
    from transcpybe.config import TranscriptionConfig
    from transcpybe.core import decode_notes
    from transcpybe.midi_io import write_midi
    from transcpybe.onsets import OnsetTrack, detect_onsets

    from . import HOP_S
    from .features import frame_features
    from .infer import onset_track, pick_peaks, posteriors
    from .model import load
    from .train import resung

    W = args.work
    songs = json.load(open(W / "dev" / "split.json"))["dev"]
    b = json.load(open(W / "dev" / "frozen.json"))["B-full"]
    variants = {"B": b, "B-plain": dict(b, timbre_split_ratio=0.0, split_on_voicing_breaks=False)}
    variants = {v: dict(variants[v], **extra) for v in args.variants}
    model = load(args.checkpoint, args.device)
    if "resung" in args.split_policies:
        if model.cfg.n_out < 4:
            raise SystemExit(f"{args.checkpoint} has no re-sung output (n_out={model.cfg.n_out})")
        if None in args.split_thresholds:
            raise SystemExit("the resung policy needs --split-thresholds: the fourth output's peak thresholds")
    runs = {}
    for v, ov in variants.items():
        for t in args.thresholds:
            for pol in args.split_policies:
                for st in (args.split_thresholds if pol in ("same-pitch", "across-gap", "resung") else [None]):
                    tag = {"all": "", "same-pitch": "-sp", "across-gap": "-xg", "resung": "-rs",
                           "snap": "-snap", "oracle": "-oracle", "own": "-own"}[pol]
                    if pol == "own":
                        name = f"{args.name}-{v}-own"
                    elif pol in ("snap", "oracle"):
                        name = f"{args.name}-{v}-t{t}{tag}"
                    else:
                        name = f"{args.name}-{v}-t{t}" + ("" if pol == "all" else f"{tag}{st if st is not None else t}")
                    name += args.suffix
                    if name in runs:                # own: one run whatever the threshold
                        continue
                    runs[name] = (dataclasses.replace(TranscriptionConfig(device="cpu"), **ov), t, pol, st)
                    (W / "dev/runs/sp" / name / "midi").mkdir(parents=True, exist_ok=True)
                    with open(W / "dev/runs/sp" / name / "overrides.json", "w") as f:
                        json.dump(dict(ov, checkpoint=str(args.checkpoint.resolve()), threshold=t,
                                       split_policy=pol, split_threshold=st), f, indent=1)
    for i, s in enumerate(songs):
        audio, _ = load_audio(W / "stems_flat" / f"{s}.wav", FEATURE_SR)
        track = pickle.load(open(W / "dev/f0cache/sp" / f"{s}.pkl", "rb"))
        mel, aux = frame_features(audio, track)
        post = posteriors(model, mel, aux, args.device)
        p, sounding = post[:, 0], post[:, 2]
        ends = pick_peaks(post[:, 1], OFFSET_THRESHOLD) * HOP_S
        ref = W / "refs/notes" / f"{s}.txt"
        ref_resung = resung(np.loadtxt(ref, ndmin=2).tolist()) if ref.exists() else []
        for name, (cfg, t, pol, st) in runs.items():
            if pol == "own":
                own = detect_onsets(audio, cfg)
                tr = OnsetTrack(own.times, own.strengths, own.envelope, own.envelope_times,
                                sounding=np.interp(own.envelope_times, np.arange(len(sounding)) * HOP_S, sounding),
                                end_times=ends)
            elif pol == "resung":
                tr = onset_track(p, t, split_policy="none", split_only_times=pick_peaks(post[:, 3], st) * HOP_S,
                                 sounding=sounding, end_times=ends)
            elif pol == "snap":
                tr = onset_track(p, t, split_policy="none", sounding=sounding, end_times=ends)
            elif pol == "oracle":
                tr = onset_track(p, t, split_policy="none", split_only_times=ref_resung, sounding=sounding,
                                 end_times=ends)
            else:
                tr = onset_track(p, t, track=track, split_policy=pol, split_threshold=st, sounding=sounding,
                                 end_times=ends)
            notes, _, _, _ = decode_notes(track, cfg, audio=audio, onsets=tr)
            write_midi(notes, W / "dev/runs/sp" / name / "midi" / f"{s}.mid", cfg)
        print(f"{i + 1}/{len(songs)} {s}", flush=True)
    print("score with:\n  python dev/score.py dev sp-f-delta0.6 sp-f-delta0.6=dev/runs/sp/f-delta0.6/midi "
          + " ".join(f"{n}=dev/runs/sp/{n}/midi" for n in runs))


if __name__ == "__main__":
    main()
