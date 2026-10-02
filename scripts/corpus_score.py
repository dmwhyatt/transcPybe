"""Score the transcriber on the realistic corpus, or compare two such runs.

    # score whatever transcpybe is importable, writing per-recording scores;
    # the recordings always come from this checkout
    python scripts/corpus_score.py --out branch.json
    # the same for another checkout, or with a setting changed
    PYTHONPATH=../transcpybe-main/src python scripts/corpus_score.py --out main.json
    python scripts/corpus_score.py --out notrim.json --set trim_note_releases=False
    # the paired comparison: development set, then held-out set
    python scripts/corpus_score.py --compare main.json branch.json

Every recording is scored at the 50 ms / 50 cent / 20% tolerances mir_eval and
the benchmark package use, under each annotation convention the corpus offers,
so a change that only helps under one of them is visible as such. The
comparison is a paired bootstrap over recordings, as the benchmark's is.

CREPE output is cached by the audio's hash, so the first run costs a few
minutes of CPU per recording-minute and every later one, of any checkout,
takes seconds. The held-out set is scored every time; look at it once, after
choosing.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CONVENTIONS = {
    "release 0.25": dict(release_point=0.25),
    "release 0.5": dict(release_point=0.5),
    "release 1.0": dict(release_point=1.0),
    "gaps closed": dict(release_point=0.5, close_gaps=True),
    "intended pitch": dict(release_point=0.5, intended_pitch=True),
}


def cache_f0(directory: Path) -> None:
    """Replay CREPE and pYIN from disk, keyed by the samples and the settings."""
    import transcpybe.f0 as f0

    directory.mkdir(parents=True, exist_ok=True)

    def cached(name, estimate):
        def wrapper(audio, config, progress=None):
            key = hashlib.sha1(np.ascontiguousarray(audio).tobytes())
            key.update(repr((name, config.crepe_model, config.hop_length, config.sample_rate,
                             config.fmin_hz, config.fmax_hz, config.seed)).encode())
            path = directory / f"{name}-{key.hexdigest()[:20]}.npz"
            if path.exists():
                stored = np.load(path)
                return stored["f0"], stored["p"]
            f0_hz, periodicity = estimate(audio, config, progress)
            np.savez(path, f0=f0_hz, p=periodicity)
            return f0_hz, periodicity
        return wrapper

    f0._estimate_crepe = cached("crepe", f0._estimate_crepe)
    f0._estimate_pyin = cached("pyin", f0._estimate_pyin)


def note_scores(reference, estimated):
    """COn, COnP and COnPOff F-measures for one recording."""
    import mir_eval

    if not reference or not estimated:
        return [0.0, 0.0, 0.0]
    hz = lambda p: 440.0 * 2.0 ** ((np.asarray(p, dtype=float) - 69.0) / 12.0)
    ref_i = np.array([[a, b] for a, b, _ in reference])
    est_i = np.array([[a, b] for a, b, _ in estimated])
    ref_p, est_p = hz([p for *_, p in reference]), hz([p for *_, p in estimated])
    out = []
    for pitch, offset in ((1e5, None), (50.0, None), (50.0, 0.2)):
        _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, pitch_tolerance=pitch,
            offset_ratio=offset, offset_min_tolerance=0.05)
        out.append(float(f))
    return out


def load_corpus():
    """The corpus from this checkout, whichever transcriber is being scored.

    Loaded by path rather than imported, so that scoring an older checkout --
    one that predates the corpus -- still renders the same recordings.
    """
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "src" / "transcpybe" / "corpus.py"
    spec = importlib.util.spec_from_file_location("transcpybe_corpus", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def score(out: Path, overrides, cache: Path) -> None:
    import transcpybe
    from transcpybe import TranscriptionConfig, transcribe_audio

    corpus = load_corpus()
    DEVELOPMENT_SEEDS, HELD_OUT_SEEDS = corpus.DEVELOPMENT_SEEDS, corpus.HELD_OUT_SEEDS
    make_recording = corpus.make_recording

    cache_f0(cache)
    config = dataclasses.replace(TranscriptionConfig(), **overrides)
    result = {"transcpybe": transcpybe.__file__, "overrides": {k: repr(v) for k, v in overrides.items()}}
    for split, seeds in (("development", DEVELOPMENT_SEEDS), ("held-out", HELD_OUT_SEEDS)):
        rows = {name: [] for name in CONVENTIONS}
        for seed in seeds:
            recording = make_recording(seed)
            notes = transcribe_audio(recording.audio, config).notes
            estimated = [(n.onset, n.offset, float(n.pitch)) for n in notes]
            for name, convention in CONVENTIONS.items():
                rows[name].append(note_scores(recording.reference(**convention), estimated))
            print(f"  {recording.name}: {len(notes)} notes", file=sys.stderr, flush=True)
        result[split] = rows
    out.write_text(json.dumps(result))


def compare(before: Path, after: Path, n_boot: int = 2000) -> None:
    a, b = json.loads(before.read_text()), json.loads(after.read_text())
    rng = np.random.default_rng(0)
    for split in ("development", "held-out"):
        print(f"\n{split}: mean change, 95% interval, recordings better/worse")
        for name in CONVENTIONS:
            x, y = np.array(a[split][name]), np.array(b[split][name])
            cells = []
            for j, metric in enumerate(("COn", "COnP", "COnPOff")):
                d = y[:, j] - x[:, j]
                boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                cells.append(f"{metric} {x[:, j].mean():.3f}->{y[:, j].mean():.3f} "
                             f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
                             f"{int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}")
            print(f"  {name:15s} " + " | ".join(cells))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="Write per-recording scores here")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override a TranscriptionConfig field (Python literal)")
    parser.add_argument("--cache", type=Path, default=Path(".corpus-cache"),
                        help="Where CREPE output is kept between runs")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.out:
        parser.error("give --out to score, or --compare BEFORE AFTER")
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = ast.literal_eval(value)
    score(args.out, overrides, args.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
