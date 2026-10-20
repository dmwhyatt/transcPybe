"""Accompaniments to put under the vocals: real multitrack stems, or a stand-in for tests.

A song directory's accompaniment is every audio file under it except the
vocals and any premixed mixture, summed. That covers MUSDB18-HQ
(``bass.wav drums.wav other.wav vocals.wav mixture.wav``) and MoisesDB (one
folder per instrument, with the singing under ``vocals/``) alike. Backing vocals
are left out with the lead: the separator is asked for the lead vocal, and
backing singers in the accompaniment would be a second label-free voice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg"}


@dataclass
class Accompaniment:
    id: str
    files: List[str]
    duration: float


def scan(root: Path, depth: int) -> List[Accompaniment]:
    """Every song directory ``depth`` levels below ``root`` with at least one non-vocal stem.

    MUSDB18-HQ is ``root/{train,test}/<song>/`` (depth 2); MoisesDB is
    ``root/<song>/<instrument>/`` (depth 1).
    """
    import soundfile as sf

    out = []
    for song in sorted(p for p in root.glob("/".join("*" * depth)) if p.is_dir()):
        files = [str(p) for p in sorted(song.rglob("*")) if p.suffix.lower() in AUDIO_EXT
                 and "vocal" not in str(p.relative_to(song)).lower() and not p.stem.lower().startswith("mixture")]
        if files:
            out.append(Accompaniment(str(song.relative_to(root)), files, min(sf.info(f).duration for f in files)))
    return out


def load_segment(acc: Accompaniment, start: float, duration: float, sr: int) -> np.ndarray:
    """``duration`` seconds of the summed stems from ``start``, stereo at ``sr``, looping if short."""
    import librosa
    import soundfile as sf

    total = np.zeros((2, int(round(duration * sr))), dtype=np.float32)
    for f in acc.files:
        info = sf.info(f)
        a = int(start * info.samplerate) % max(1, info.frames)
        n = int(np.ceil(duration * info.samplerate)) + 1
        x, _ = sf.read(f, start=a, frames=n, dtype="float32", always_2d=True)
        while x.shape[0] < n:                      # loop a stem shorter than the segment
            more, _ = sf.read(f, start=0, frames=n - x.shape[0], dtype="float32", always_2d=True)
            if not len(more):
                break
            x = np.concatenate([x, more])
        x = x.T if x.shape[1] == 2 else np.repeat(x.T[:1], 2, axis=0)
        if info.samplerate != sr:
            x = librosa.resample(x, orig_sr=info.samplerate, target_sr=sr)
        m = min(total.shape[1], x.shape[1])
        total[:, :m] += x[:, :m]
    return total


def synthetic(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    """A stand-in accompaniment for tests: a sustained triad over kick and snare at 120 bpm.

    Not for training. It exists so the mixing and separation stages can be run
    end to end before any multitrack data is on disk.
    """
    t = np.arange(n) / sr
    root = 110.0 * 2 ** (rng.integers(0, 12) / 12)
    pad = sum(np.sin(2 * np.pi * root * r * k * t) / k
              for r in (1.0, 2 ** (4 / 12), 2 ** (7 / 12)) for k in range(1, 6))
    beats = np.zeros(n)
    for i, b in enumerate(np.arange(0, n / sr, 0.5)):
        a = int(b * sr)
        m = min(n - a, int(0.15 * sr))
        env = np.exp(-np.arange(m) / (0.03 * sr))
        hit = np.sin(2 * np.pi * 55 * np.arange(m) / sr) if i % 2 == 0 else rng.standard_normal(m) * 0.6
        beats[a:a + m] += hit * env
    x = 0.08 * pad / 3 + 0.3 * beats
    return np.stack([x, x * 0.9]).astype(np.float32)


def choose(accs: Sequence[Accompaniment], rng: np.random.Generator, duration: float):
    acc = accs[int(rng.integers(len(accs)))]
    start = float(rng.uniform(0, max(0.0, acc.duration - duration)))
    return acc, start
