"""Labelled singing -> a manifest of vocals and their notes.

    python -m boundary.sources m4singer ROOT OUT.jsonl [--onsets consonant|vowel]
    python -m boundary.sources corpus OUT.jsonl --audio-dir DIR --n 2000 [--syllabic]

Every source says which onset convention its notes follow, because annotators
differ on it and the target the network learns is whatever convention it is
given: M4Singer's MIDI starts a syllable at its consonant, and its TextGrids
allow the vowel reading too.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

from . import items as manifest

# Held out for validation, by singer, so the network is never scored on a voice
# it trained on: one of each voice type, about a seventh of M4Singer's songs.
M4SINGER_VALID = ("Alto-3", "Bass-3", "Soprano-2", "Tenor-4")

# Mandarin initials as M4Singer spells them: the consonant a syllable may start on.
PINYIN_INITIALS = frozenset("b p m f d t n l g k h j q x zh ch sh r z c s y w".split())


def textgrid_tiers(path: Path) -> Dict[str, List[tuple]]:
    """Interval tiers of a Praat TextGrid (long text format) as ``{name: [(xmin, xmax, text)]}``.

    M4Singer's tiers are words then phones; both are named "None", so a repeated
    name gets its position appended.
    """
    tiers: Dict[str, List[tuple]] = {}
    current = None
    xmin = xmax = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("name ="):
            name = line.split("=", 1)[1].strip().strip('"')
            if name in tiers:
                name = f"{name}{len(tiers) + 1}"
            current = tiers.setdefault(name, [])
        elif current is not None and re.match(r"xmin = ", line):
            xmin = float(line.split("=")[1])
        elif current is not None and re.match(r"xmax = ", line):
            xmax = float(line.split("=")[1])
        elif current is not None and line.startswith("text ="):
            current.append((xmin, xmax, line.split("=", 1)[1].strip().strip('"')))
    return tiers


def _vowel_onsets(notes: List[list], phones: Sequence[tuple]) -> List[list]:
    """Move each note that starts on a consonant to where its vowel starts.

    A note keeps its end; a consonant so long it would leave the note under
    30 ms is left alone rather than producing a note with no length.
    """
    out = []
    for on, off, pitch in notes:
        for p0, p1, ph in phones:
            if abs(p0 - on) < 0.006 and ph in PINYIN_INITIALS and p1 < off - 0.03:
                on = p1
                break
        out.append([on, off, pitch])
    return out


def m4singer(root: Path, onsets: str = "consonant",
             valid_singers: Sequence[str] = M4SINGER_VALID) -> Iterator[dict]:
    import pretty_midi

    for song in sorted(p for p in root.iterdir() if p.is_dir() and "#" in p.name):
        singer = song.name.split("#")[0]
        for wav in sorted(song.glob("*.wav")):
            mid = wav.with_suffix(".mid")
            if not mid.exists():
                continue
            pm = pretty_midi.PrettyMIDI(str(mid))
            notes = sorted([round(n.start, 4), round(n.end, 4), int(n.pitch)]
                           for inst in pm.instruments for n in inst.notes)
            if not notes:
                continue
            if onsets == "vowel":
                tiers = list(textgrid_tiers(wav.with_suffix(".TextGrid")).values())
                if len(tiers) < 2:
                    continue
                notes = _vowel_onsets(notes, tiers[1])
            yield dict(id=f"m4singer/{song.name}/{wav.stem}", source="m4singer", audio=str(wav),
                       notes=notes, onsets=onsets, singer=singer,
                       split="valid" if singer in valid_singers else "train")


def corpus(audio_dir: Path, n: int, start_seed: int = 0, syllabic: bool = True,
           valid_every: int = 10) -> Iterator[dict]:
    """transcPybe's own synthetic corpus, rendered to ``audio_dir``.

    Needs ``transcpybe.corpus``, which is on the claude/spurious-notes branch
    and not yet on main.

    Its audio is synthesised at 16 kHz, so it has nothing above 8 kHz: fine for
    training on dry vocals, but a separator trained at 44.1 kHz will treat the
    missing top octave as something to explain. Mix it with that in mind.
    """
    import soundfile as sf
    from transcpybe.corpus import SR, make_recording

    audio_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(start_seed, start_seed + n):
        rec = make_recording(seed, syllabic=syllabic)
        wav = audio_dir / f"corpus-{seed:06d}.wav"
        if not wav.exists():
            sf.write(wav, rec.audio, SR, subtype="FLOAT")
        notes = [[round(a, 4), round(b, 4), round(float(p), 3)] for a, b, p in rec.reference()]
        yield dict(id=f"corpus/{seed:06d}", source="corpus", audio=str(wav), notes=notes,
                   onsets="corpus-default", syllabic=syllabic,
                   split="valid" if seed % valid_every == 0 else "train")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    m = sub.add_parser("m4singer")
    m.add_argument("root", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--onsets", choices=("consonant", "vowel"), default="consonant")
    c = sub.add_parser("corpus")
    c.add_argument("out", type=Path)
    c.add_argument("--audio-dir", type=Path, required=True)
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--start-seed", type=int, default=0)
    c.add_argument("--plain", action="store_true", help="without the syllabic pitch movement")
    args = ap.parse_args(argv)
    if args.source == "m4singer":
        rows = m4singer(args.root, args.onsets)
    else:
        rows = corpus(args.audio_dir, args.n, args.start_seed, syllabic=not args.plain)
    n = manifest.write(args.out, rows)
    print(f"{n} items -> {args.out}")


if __name__ == "__main__":
    main()
