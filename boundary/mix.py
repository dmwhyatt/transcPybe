"""Put each labelled vocal over an accompaniment, as a pop mix would have it.

    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp ROOT --accomp-depth 2 [--variants 1] [--workers 8]
    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp synthetic          # tests only
    python -m boundary.mix VOCALS.jsonl OUT_DIR --accomp none               # dry baseline: no mix at all

Writes ``OUT_DIR/mixes/<mix_id>.flac`` (44.1 kHz stereo, what the separator
expects), a ``<mix_id>.json`` beside each with how it was made, and
``OUT_DIR/mixes.jsonl``: each line the vocal's item plus how it was mixed, with
the notes moved by the silence put in front.

Everything is drawn from a generator seeded by (seed, item, variant), so a mix
can be rebuilt exactly from its manifest line. The mix id ends in a digest of
everything that decides the mix (the item, its notes, the seed, the variant,
the accompaniment set and the ranges below), so a changed input gets a new id
and no later stage can mistake an old stem or feature file for it. A mix whose
files and record already exist is not rendered again, which makes a job that
ran out of time resumable.

The ranges are assumptions about commercial pop, not measurements:

- vocal-to-accompaniment level -6 to +4 dB, by RMS over the sung frames;
- 0.3-2 s of accompaniment alone before the vocal and 0.3-1.5 s after, so the
  network sees what bleed sounds like where nobody is singing;
- reverb on the vocal in 40% of mixes (RT60 0.2-1.2 s, 10-35% wet);
- a noise floor 55-75 dB under the voice, so no frame is digital silence:
  a separated stem never is, and features of exact zeros are values the
  network would never meet in use;
- 10% with no accompaniment at all, so a dry take is not out of distribution;
- with ``--p-mp3 P``, that share of mixes goes through an MP3 round trip at a
  bitrate drawn from ``--mp3-kbps`` before it is written, as a lossy source
  would reach the separator. The encoder can delay the audio (1,105 samples at
  low bitrates), so the decoded mix is realigned to the original by
  cross-correlation and the labels stay true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from . import MIX_SR, accomp, items as manifest
from .progress import Progress

MIX_VERSION = 2       # bump when a change here alters what an (item, seed, variant) renders
VAR_DB = (-6.0, 4.0)
PAD_BEFORE = (0.3, 2.0)
PAD_AFTER = (0.3, 1.5)
P_REVERB, RT60, WET = 0.4, (0.2, 1.2), (0.10, 0.35)
NOISE_FLOOR_DB = (-75.0, -55.0)
P_DRY = 0.10
PEAK = 0.89


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def setup_key(accs, var_db=VAR_DB, mp3=(0.0, (96, 320))) -> str:
    """What every mix in a run shares and depends on: the accompaniment set and the ranges."""
    acc = accs if isinstance(accs, str) else [(a.id, a.files) for a in accs]
    key = dict(version=MIX_VERSION, accomp=acc, var=tuple(var_db), pads=(PAD_BEFORE, PAD_AFTER),
               reverb=(P_REVERB, RT60, WET), floor=NOISE_FLOOR_DB, dry=P_DRY, peak=PEAK)
    if mp3[0] > 0:                     # only then, so mixes made before this option keep their ids
        key["mp3"] = [mp3[0], list(mp3[1])]
    return digest(key)


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


def _mp3(x: np.ndarray, kbps: float):
    """(n, 2) audio after an MP3 round trip at about ``kbps``, realigned to ``x``; and the delay removed, in samples."""
    import io
    import soundfile as sf

    peak = max(float(np.abs(x).max()), 1e-9)
    buf = io.BytesIO()
    sf.write(buf, x / peak * 0.98, MIX_SR, format="MP3", bitrate_mode="CONSTANT",
             compression_level=float(np.clip((320.0 - kbps) / 314.0, 0.0, 1.0)))   # libsndfile: 0 -> 320 kbps, 0.5 -> ~160
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float32", always_2d=True)
    n = min(len(x), len(y), 10 * MIX_SR)
    a, b = x[:n].sum(1), y[:n].sum(1)
    cc = np.fft.irfft(np.fft.rfft(a, 2 * n).conj() * np.fft.rfft(b, 2 * n))
    lag = int(np.argmax(np.r_[cc[-3000:], cc[:3001]]) - 3000)
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros((-lag, 2), np.float32), y])
    y = y[:len(x)] if len(y) >= len(x) else np.concatenate([y, np.zeros((len(x) - len(y), 2), np.float32)])
    return y * peak / 0.98, lag


def _write(path: Path, data: np.ndarray) -> None:
    """Write through a temporary name, so a killed job never leaves a truncated file under the real one."""
    import soundfile as sf

    tmp = path.with_name(path.name + ".part")
    sf.write(tmp, data, MIX_SR, subtype="PCM_24", format="FLAC")
    os.replace(tmp, path)


def render(item: dict, index: int, variant: int, seed: int, accs, out_dir: Path, keep_dry: bool,
           setup: str, var_db=VAR_DB, mp3=(0.0, (96, 320))) -> dict:
    import librosa

    mix_id = f"{index:06d}-{variant}-{digest([item['id'], item['audio'], item['notes'], seed, variant, setup])}"
    (out_dir / "mixes").mkdir(parents=True, exist_ok=True)
    record = out_dir / "mixes" / f"{mix_id}.json"
    if record.exists():
        rec = json.loads(record.read_text(encoding="utf-8"))
        if Path(rec["mix"]).exists() and (not keep_dry or Path(rec.get("dry", "")).exists()):
            return rec

    rng = np.random.default_rng([seed, index, variant])
    vocal, _ = librosa.load(item["audio"], sr=MIX_SR, mono=True)
    pre, post = float(rng.uniform(*PAD_BEFORE)), float(rng.uniform(*PAD_AFTER))
    vocal = np.concatenate([np.zeros(int(pre * MIX_SR)), vocal, np.zeros(int(post * MIX_SR))]).astype(np.float32)
    rec = dict(item, mix_id=mix_id, variant=variant, setup=setup, pad_before=round(pre, 4),
               notes=[[round(a + pre, 4), round(b + pre, 4), p] for a, b, p in item["notes"]])
    if rng.random() < P_REVERB:
        vocal, rec["reverb"] = _reverb(vocal, rng, MIX_SR)
    vocal *= 0.1 / max(_active_rms(vocal, MIX_SR), 1e-6)
    floor = float(rng.uniform(*NOISE_FLOOR_DB))
    vocal += (0.1 * 10 ** (floor / 20) * rng.standard_normal(len(vocal))).astype(np.float32)
    rec["noise_floor_db"] = round(floor, 1)

    if accs == "none":                 # the dry baseline: the vocal alone, written once, as its own 'mix'
        gain = min(1.0, PEAK / max(float(np.abs(vocal).max()), 1e-6))
        path = out_dir / "mixes" / f"{mix_id}.dry.flac"
        _write(path, vocal * gain)
        rec.update(accomp=None, var_db=None, mix=str(path), dry=str(path), gain=round(gain, 4))
    else:
        if rng.random() < P_DRY:       # drawn before any accompaniment is read: a dry mix reads none
            acc = np.zeros((2, len(vocal)), np.float32)
            rec["accomp"], rec["var_db"] = None, None
        else:
            if accs == "synthetic":
                acc, rec["accomp"] = accomp.synthetic(rng, len(vocal), MIX_SR), "synthetic"
            else:
                a, start = accomp.choose(accs, rng, len(vocal) / MIX_SR)
                acc = accomp.load_segment(a, start, len(vocal) / MIX_SR, MIX_SR)
                rec["accomp"], rec["accomp_start"] = a.id, round(start, 3)
            var = float(rng.uniform(*var_db))
            acc *= 0.1 * 10 ** (-var / 20) / max(float(np.sqrt(np.mean(acc ** 2))), 1e-6)
            rec["var_db"] = round(var, 2)
        mix = acc + vocal[None, :]
        gain = min(1.0, PEAK / max(float(np.abs(mix).max()), 1e-6))
        path = out_dir / "mixes" / f"{mix_id}.flac"
        if mp3[0] > 0 and rng.random() < mp3[0]:
            kbps = float(rng.uniform(*mp3[1]))
            coded, lag = _mp3((mix * gain).T.astype(np.float32), kbps)
            rec["mp3_kbps"], rec["mp3_realigned_samples"] = round(kbps), lag
            _write(path, coded)
        else:
            _write(path, (mix * gain).T)
        rec["mix"], rec["gain"] = str(path), round(gain, 4)
        if keep_dry:                   # the vocal as it is in the mix: the 'no separator' ablation
            dry = out_dir / "mixes" / f"{mix_id}.dry.flac"
            _write(dry, vocal * gain)
            rec["dry"] = str(dry)
    tmp = record.with_name(record.name + ".part")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, record)
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
    ap.add_argument("--p-mp3", type=float, default=0.0, help="share of mixes put through an MP3 round trip before writing")
    ap.add_argument("--mp3-kbps", type=float, nargs=2, default=[96, 320], metavar=("LO", "HI"))
    ap.add_argument("--var-db", type=float, nargs=2, default=list(VAR_DB), metavar=("LO", "HI"),
                    help="vocal-to-accompaniment level range in dB; e.g. 20 35 leaves only faint bleed, a stand-in for separation")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)

    out_dir = args.out_dir.resolve()          # manifests carry absolute paths: later stages may run elsewhere
    vocals = manifest.read(args.vocals)
    accs = args.accomp if args.accomp in ("synthetic", "none") else accomp.scan(Path(args.accomp), args.accomp_depth)
    if not accs:
        raise SystemExit(f"no accompaniment found under {args.accomp}")
    mp3 = (args.p_mp3, tuple(args.mp3_kbps))
    setup = setup_key(accs, args.var_db, mp3)
    (out_dir / "mixes").mkdir(parents=True, exist_ok=True)
    jobs = [(it, i, v, args.seed, accs, out_dir, args.keep_dry, setup, tuple(args.var_db), mp3)
            for i, it in enumerate(vocals) for v in range(args.variants)]
    prog, rows = Progress(len(jobs), "mix"), []
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for row in pool.imap(_job, jobs, chunksize=8):
                rows.append(row)
                prog.tick()
    else:
        for j in jobs:
            rows.append(_job(j))
            prog.tick()
    n = manifest.write(out_dir / "mixes.jsonl", rows)
    print(f"{n} mixes -> {out_dir / 'mixes'} (setup {setup})")


if __name__ == "__main__":
    main()
