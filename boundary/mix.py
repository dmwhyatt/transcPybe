"""Put each labelled vocal over an accompaniment, as a pop mix would have it.

    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp ROOT --accomp-depth 2 [--variants 1] [--workers 8]
    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp synthetic          # tests only
    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp none               # dry baseline: no mix at all

Writes ``OUT_DIR/mixes/<mix_id>.flac`` (44.1 kHz stereo, what the separator
expects) and ``OUT_DIR/mixes.jsonl``, each line the vocal's item plus how it
was mixed, with the notes moved by the silence put in front.

Everything is drawn from a generator seeded by (seed, item, variant), so a mix
can be rebuilt exactly from its manifest line. The ranges are assumptions about
commercial pop, not measurements:

- vocal-to-accompaniment level -6 to +4 dB, by RMS over the sung frames;
- 0.3-2 s of accompaniment alone before the vocal and 0.3-1.5 s after, so the
  network sees what bleed sounds like where nobody is singing;
- reverb on the vocal in 40% of mixes (RT60 0.2-1.2 s, 10-35% wet);
- 10% with no accompaniment at all, so a dry take is not out of distribution.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import MIX_SR, accomp, items as manifest

VAR_DB = (-6.0, 4.0)
PAD_BEFORE = (0.3, 2.0)
PAD_AFTER = (0.3, 1.5)
P_REVERB, RT60, WET = 0.4, (0.2, 1.2), (0.10, 0.35)
P_DRY = 0.10
PEAK = 0.89


def _reverb(x: np.ndarray, rng: np.random.Generator, sr: int):
    from scipy.signal import fftconvolve

    rt60, wet = float(rng.uniform(*RT60)), float(rng.uniform(*WET))
    n = int(rt60 * sr)
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)    # -60 dB at rt60
    rir /= np.sqrt(np.sum(rir ** 2))
    tail = fftconvolve(x, rir)[: len(x)]
    return ((1 - wet) * x + wet * tail).astype(np.float32), dict(rt60=round(rt60, 3), wet=round(wet, 3))


def _active_rms(x: np.ndarray, sr: int) -> float:
    """RMS over the 20 ms frames within 40 dB of the loudest: the sung part, not the gaps."""
    hop = int(0.02 * sr)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    e = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return float(np.sqrt(np.mean(e[e > e.max() * 10 ** (-40 / 20)] ** 2)))


def render(item: dict, index: int, variant: int, seed: int, accs, out_dir: Path, keep_dry: bool) -> dict:
    import librosa
    import soundfile as sf

    rng = np.random.default_rng([seed, index, variant])
    mix_id = f"{index:06d}-{variant}"
    vocal, _ = librosa.load(item["audio"], sr=MIX_SR, mono=True)
    pre, post = float(rng.uniform(*PAD_BEFORE)), float(rng.uniform(*PAD_AFTER))
    vocal = np.concatenate([np.zeros(int(pre * MIX_SR)), vocal, np.zeros(int(post * MIX_SR))]).astype(np.float32)
    rec = dict(item, mix_id=mix_id, variant=variant, pad_before=round(pre, 4),
               notes=[[round(a + pre, 4), round(b + pre, 4), p] for a, b, p in item["notes"]])
    if rng.random() < P_REVERB:
        vocal, rec["reverb"] = _reverb(vocal, rng, MIX_SR)
    vocal *= 0.1 / max(_active_rms(vocal, MIX_SR), 1e-6)
    if accs == "none":                 # the dry baseline: the vocal alone, written once, as its own 'mix'
        path = out_dir / "mixes" / f"{mix_id}.dry.flac"
        gain = min(1.0, PEAK / max(float(np.abs(vocal).max()), 1e-6))
        sf.write(path, vocal * gain, MIX_SR, subtype="PCM_24")
        return dict(rec, accomp=None, var_db=None, mix=str(path), dry=str(path), gain=round(gain, 4))
    if accs == "synthetic":
        acc, rec["accomp"] = accomp.synthetic(rng, len(vocal), MIX_SR), "synthetic"
    else:
        a, start = accomp.choose(accs, rng, len(vocal) / MIX_SR)
        acc = accomp.load_segment(a, start, len(vocal) / MIX_SR, MIX_SR)
        rec["accomp"], rec["accomp_start"] = a.id, round(start, 3)
    if rng.random() < P_DRY:
        acc[:] = 0
        rec["var_db"] = None
    else:
        var = float(rng.uniform(*VAR_DB))
        acc *= 0.1 * 10 ** (-var / 20) / max(float(np.sqrt(np.mean(acc ** 2))), 1e-6)
        rec["var_db"] = round(var, 2)
    mix = acc + vocal[None, :]
    gain = min(1.0, PEAK / max(float(np.abs(mix).max()), 1e-6))
    path = out_dir / "mixes" / f"{mix_id}.flac"
    sf.write(path, (mix * gain).T, MIX_SR, subtype="PCM_24")
    rec["mix"], rec["gain"] = str(path), round(gain, 4)
    if keep_dry:                       # the vocal as it is in the mix: the 'no separator' ablation
        dry = out_dir / "mixes" / f"{mix_id}.dry.flac"
        sf.write(dry, vocal * gain, MIX_SR, subtype="PCM_24")
        rec["dry"] = str(dry)
    return rec


def _job(args):
    return render(*args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocals", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--accomp", required=True, help="root of multitrack songs, 'none' for dry vocals, or 'synthetic' for tests")
    ap.add_argument("--accomp-depth", type=int, default=2, help="2 for MUSDB18-HQ, 1 for MoisesDB")
    ap.add_argument("--variants", type=int, default=1, help="mixes per vocal, each with its own draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-dry", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    vocals = manifest.read(args.vocals)
    accs = args.accomp if args.accomp in ("synthetic", "none") else accomp.scan(Path(args.accomp), args.accomp_depth)
    if not accs:
        raise SystemExit(f"no accompaniment found under {args.accomp}")
    (args.out_dir / "mixes").mkdir(parents=True, exist_ok=True)
    jobs = [(it, i, v, args.seed, accs, args.out_dir, args.keep_dry) for i, it in enumerate(vocals) for v in range(args.variants)]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(_job, jobs, chunksize=8)
    else:
        rows = [_job(j) for j in jobs]
    n = manifest.write(args.out_dir / "mixes.jsonl", rows)
    print(f"{n} mixes -> {args.out_dir / 'mixes'}")


if __name__ == "__main__":
    main()
