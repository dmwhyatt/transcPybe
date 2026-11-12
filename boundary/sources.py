"""Labelled singing -> a manifest of vocals and their notes.

    python -m boundary.sources m4singer ROOT OUT.jsonl [--onsets consonant|vowel|between --fraction 0.5]
    python -m boundary.sources gtsinger ROOT OUT.jsonl [--onsets ...] [--languages English Chinese ...]
    python -m boundary.sources corpus OUT.jsonl --audio-dir DIR --n 2000 [--plain]

Every source says which onset convention its notes follow, because annotators
differ on it and the target the network learns is whatever convention it is
given. M4Singer's MIDI and GTSinger's notes both start a syllable at its
consonant, and their phoneme timings allow the vowel and half-way readings too.

With ``--onsets vowel`` or ``between``, each row also carries ``landmarks``: for
every note, ``[articulation, vowel, opens_on_vowel]`` in the source's own
timing (round 2's plan, ``ROUND2-PLAN.md``). The notes themselves are exactly
as before, so a manifest built now matches one built earlier, field for field,
apart from the new one.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence

from . import items as manifest

# Held out for validation, by singer, so the network is never scored on a voice
# it trained on: one of each voice type, about an eighth of M4Singer's clips.
# Many songs are sung by several singers, so every other singer's recording of
# a held-out singer's song is left out of training too ("excluded"): otherwise
# validation would be scored on melodies, lyrics and rhythms it trained on.
M4SINGER_VALID = ("Alto-3", "Bass-3", "Soprano-2", "Tenor-4")

# Mandarin initials as M4Singer spells them: the consonant a syllable may start on.
PINYIN_INITIALS = frozenset("b p m f d t n l g k h j q x zh ch sh r z c s y w".split())

# GTSinger (20 singers, 9 languages), held out the same way. One English voice, because English pop is the
# target and M4Singer has none, and one Korean voice, because Korean is the only other language with three
# singers. That leaves every held-out language with a voice of each sex in training.
GTSINGER_VALID = ("EN-Alto-2", "KO-Soprano-2")
GTSINGER_SILENCE = frozenset({"<SP>", "<AP>"})
# IPA vowel letters, for the languages GTSinger transcribes in IPA (all but English and Chinese).
IPA_VOWELS = frozenset("aeiouyæøœɐɑɒɔəɛɜɤɨɪɯɵɶʉʊʌʏ")


def gtsinger_is_vowel(phone: str, language: str) -> bool:
    """Whether a GTSinger phone is a vowel. English is ARPAbet, where vowels carry a stress digit; Chinese is
    pinyin initials and finals; the rest are IPA, with diacritics and length marks stripped before the test."""
    if phone in GTSINGER_SILENCE or not phone:
        return False
    if language == "English":
        return phone[-1].isdigit()
    if language == "Chinese":
        return phone not in PINYIN_INITIALS
    base = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    return base[:1] in IPA_VOWELS


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


def _vowel_onsets(notes: List[list], phones: Sequence[tuple], fraction: float = 1.0) -> List[list]:
    """Move each note that starts on a consonant ``fraction`` of the way to where its vowel starts.

    1 is the vowel; 0.5 is half way, which is where the dry pilots placed
    Molina's and vocadito's annotators: consonant labels put notes 40-70 ms early,
    vowel labels 30-35 ms late. A note keeps its end; a consonant so long it
    would leave the note under 30 ms is left alone rather than producing a note
    with no length.
    """
    out = []
    for on, off, pitch in notes:
        for p0, p1, ph in phones:
            moved = round(p0 + fraction * (p1 - p0), 4)
            if abs(p0 - on) < 0.006 and ph in PINYIN_INITIALS and moved < off - 0.03:
                on = moved
                break
        out.append([on, off, pitch])
    return out


SILENT_PHONES = frozenset({"<SP>", "<AP>", "sil", "sp", ""})


def _m4singer_landmarks(notes: List[list], phones: Sequence[tuple]) -> List[list]:
    """Each note's ``[articulation, vowel, opens_on_vowel]``, from M4Singer's phone tier, for the round-2 labels.

    ``notes`` are M4Singer's own (consonant-start) notes. The articulation onset
    is the note's labelled start. Where the note starts on a pinyin initial whose
    vowel begins at least 30 ms before the note ends, the vowel onset is that
    initial's end, the same phone ``_vowel_onsets`` moves towards. Otherwise the
    two coincide. ``opens_on_vowel`` marks a note whose first phone, at its start,
    is a final rather than an initial or silence; a note inside a syllable (a
    second note on one vowel) has no phone starting with it and is not marked.
    """
    out = []
    for on, off, _ in notes:
        first = next(((p0, p1, ph) for p0, p1, ph in phones if abs(p0 - on) < 0.006), None)
        if first is not None and first[2] in PINYIN_INITIALS and first[1] < off - 0.03:
            out.append([on, round(first[1], 4), False])
        else:
            out.append([on, on, first is not None and first[2] not in PINYIN_INITIALS
                        and first[2] not in SILENT_PHONES])
    return out


MAX_CONSONANT_S = 0.25   # longer than this before the vowel: a held, pitched consonant, where the note starts


def _first_vowel_onsets(notes: List[list], phones: Sequence[tuple], is_vowel: Callable[[str], bool],
                        fraction: float, untrusted: Sequence[tuple] = (),
                        max_consonant: float = MAX_CONSONANT_S) -> List[list]:
    """Move each note that starts on a consonant ``fraction`` of the way to the start of its first vowel.

    Unlike ``_vowel_onsets``, the consonant may be a cluster (English "street" is
    S T R before the vowel), so the distance is from the note's start to the first
    vowel at or after it. A note keeps its start when:
    - it starts on a vowel, or between phones (a second note within a syllable);
    - the move would leave it under 30 ms;
    - the consonant lasts longer than ``max_consonant``. Singers hold pitched
      consonants: in GTSinger, "word" opens on 685 ms of W and Japanese "む" on
      700 ms of m, and the note audibly starts there. Every Mandarin consonant
      in a sample stayed under 0.25 s;
    - its start lies in one of the ``untrusted`` (start, end) spans, whose phone
      timing is not to be relied on.
    """
    out = []
    for on, off, pitch in notes:
        trusted = not any(a - 0.006 <= on < b for a, b in untrusted)
        if trusted and any(abs(p0 - on) < 0.006 for p0, _, _ in phones):
            vowel = next((p0 for p0, _, ph in phones if on - 0.006 <= p0 < off and is_vowel(ph)), None)
            if vowel is not None and on + 0.006 < vowel <= on + max_consonant:
                moved = round(on + fraction * (vowel - on), 4)
                if moved < off - 0.03:
                    on = moved
        out.append([on, off, pitch])
    return out


def _gtsinger_landmarks(notes: List[list], phones: Sequence[tuple], is_vowel: Callable[[str], bool],
                        untrusted: Sequence[tuple] = (), max_consonant: float = MAX_CONSONANT_S) -> List[list]:
    """Each note's ``[articulation, vowel, opens_on_vowel]`` for GTSinger, by ``_first_vowel_onsets``'s rules.

    ``notes`` are GTSinger's own (consonant-start) notes. Where that function
    would move a note (a trusted start on a phone, a vowel after it within
    ``max_consonant`` and at least 30 ms before the note ends), the vowel onset
    is that vowel's start; otherwise it is the note's start. ``opens_on_vowel``
    marks a trusted note whose first phone, at its start, is a vowel.
    """
    out = []
    for on, off, _ in notes:
        trusted = not any(a - 0.006 <= on < b for a, b in untrusted)
        first = next((ph for p0, _, ph in phones if abs(p0 - on) < 0.006), None) if trusted else None
        vowel = None
        if first is not None:
            vowel = next((p0 for p0, _, ph in phones if on - 0.006 <= p0 < off and is_vowel(ph)), None)
        if vowel is not None and on + 0.006 < vowel <= on + max_consonant and vowel < off - 0.03:
            out.append([on, round(vowel, 4), False])
        else:
            out.append([on, on, first is not None and is_vowel(first)])
    return out


def _evenly_divided(word: dict) -> bool:
    """Whether a GTSinger word's phones look divided evenly rather than aligned: two neighbours of equal length.
    In a sample, no English or Mandarin word did, but 11-27% of neighbouring phones within Japanese, Korean
    and Spanish words had identical lengths (Spanish "brille": b, ɾ, i, ʝ at 178 ms each)."""
    lengths = [e - s for s, e in zip(word["ph_start"], word["ph_end"])]
    return any(abs(a - b) < 0.0015 for a, b in zip(lengths, lengths[1:]))


def gtsinger(root: Path, onsets: str = "consonant", valid_singers: Sequence[str] = GTSINGER_VALID,
             fraction: float = 0.5, languages: Sequence[str] = ()) -> Iterator[dict]:
    """GTSinger as downloaded from Hugging Face: ROOT/<Language>/<Singer>/<Technique>/<Song>/<Group>/NNNN.{wav,json}.

    Each song is sung with a technique (``<Technique>_Group``) and plainly
    (``Control_Group``); both are singing and both are used. ``Paired_Speech_Group``
    is the lyrics spoken, not sung, and is skipped. Notes come from the JSON
    (``note``, ``note_start``, ``note_end``, MIDI pitch, 0 for rests); phones from
    ``ph``, ``ph_start``, ``ph_end``. Notes start at the syllable's consonant, as
    in M4Singer. A song title held out with a validation singer is matched within
    its language, ignoring case.
    """
    root = root.resolve()
    clips = sorted(p for p in root.glob("*/*/*/*/*/*.json")
                   if not p.parent.name.startswith("Paired_Speech")
                   and (not languages or p.relative_to(root).parts[0] in languages))
    key = lambda p: (p.relative_to(root).parts[0], p.relative_to(root).parts[3].casefold())
    held_out = {key(p) for p in clips if p.relative_to(root).parts[1] in valid_singers}
    if not clips:
        raise SystemExit(f"no GTSinger annotations under {root}: expected <Language>/<Singer>/<Technique>/<Song>/"
                         "<Group>/NNNN.json")
    missing = 0
    for path in clips:
        language, singer, technique, title, group = path.relative_to(root).parts[:5]
        wav = path.with_suffix(".wav")
        if not wav.exists():
            missing += 1
            continue
        words = json.loads(path.read_text(encoding="utf-8"))
        notes = sorted([round(s, 4), round(e, 4), int(n)] for w in words
                       for n, s, e in zip(w["note"], w["note_start"], w["note_end"]) if n and e > s)
        if not notes:
            continue
        extra = {}
        if onsets in ("vowel", "between"):
            phones = [(s, e, ph) for w in words for ph, s, e in zip(w["ph"], w["ph_start"], w["ph_end"])]
            untrusted = [(w["start_time"], w["end_time"]) for w in words if _evenly_divided(w)]
            is_vowel = lambda ph: gtsinger_is_vowel(ph, language)
            extra["landmarks"] = _gtsinger_landmarks(notes, phones, is_vowel, untrusted)
            notes = _first_vowel_onsets(notes, phones, is_vowel, 1.0 if onsets == "vowel" else fraction, untrusted)
        split = "valid" if singer in valid_singers else "excluded" if key(path) in held_out else "train"
        yield dict(id=f"gtsinger/{language}/{singer}/{technique}/{title}/{group}/{path.stem}", source="gtsinger",
                   audio=str(wav), notes=notes, onsets=onsets if onsets != "between" else f"between-{fraction}",
                   singer=singer, title=title, language=language, technique=group.removesuffix("_Group"),
                   split=split, **extra)
    if missing:
        # an annotation without its audio is a broken download, not a clip to skip quietly (22 Oct: every .wav missing)
        raise SystemExit(f"{missing} of {len(clips)} GTSinger clips under {root} have a .json but no .wav: "
                         "finish the download first")


def m4singer(root: Path, onsets: str = "consonant",
             valid_singers: Sequence[str] = M4SINGER_VALID, fraction: float = 0.5) -> Iterator[dict]:
    import pretty_midi

    root = root.resolve()
    songs = sorted(p for p in root.iterdir() if p.is_dir() and "#" in p.name)
    held_out = {s.name.split("#", 1)[1] for s in songs if s.name.split("#", 1)[0] in valid_singers}
    for song in songs:
        singer, title = song.name.split("#", 1)
        split = "valid" if singer in valid_singers else "excluded" if title in held_out else "train"
        for wav in sorted(song.glob("*.wav")):
            mid = wav.with_suffix(".mid")
            if not mid.exists():
                continue
            pm = pretty_midi.PrettyMIDI(str(mid))
            notes = sorted([round(n.start, 4), round(n.end, 4), int(n.pitch)]
                           for inst in pm.instruments for n in inst.notes)
            if not notes:
                continue
            extra = {}
            if onsets in ("vowel", "between"):
                tiers = list(textgrid_tiers(wav.with_suffix(".TextGrid")).values())
                if len(tiers) < 2:
                    continue
                extra["landmarks"] = _m4singer_landmarks(notes, tiers[1])
                notes = _vowel_onsets(notes, tiers[1], 1.0 if onsets == "vowel" else fraction)
            yield dict(id=f"m4singer/{song.name}/{wav.stem}", source="m4singer", audio=str(wav),
                       notes=notes, onsets=onsets if onsets != "between" else f"between-{fraction}",
                       singer=singer, title=title, split=split, **extra)


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

    audio_dir = audio_dir.resolve()
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
    m.add_argument("--onsets", choices=("consonant", "vowel", "between"), default="consonant")
    m.add_argument("--fraction", type=float, default=0.5, help="for --onsets between: how far from consonant to vowel")
    g = sub.add_parser("gtsinger")
    g.add_argument("root", type=Path, help="the folder holding the language folders")
    g.add_argument("out", type=Path)
    g.add_argument("--onsets", choices=("consonant", "vowel", "between"), default="consonant")
    g.add_argument("--fraction", type=float, default=0.5)
    g.add_argument("--languages", nargs="*", default=(), help="only these language folders (default: all)")
    c = sub.add_parser("corpus")
    c.add_argument("out", type=Path)
    c.add_argument("--audio-dir", type=Path, required=True)
    c.add_argument("--n", type=int, default=2000)
    c.add_argument("--start-seed", type=int, default=0)
    c.add_argument("--plain", action="store_true", help="without the syllabic pitch movement")
    args = ap.parse_args(argv)
    if args.source == "m4singer":
        rows = m4singer(args.root, args.onsets, fraction=args.fraction)
    elif args.source == "gtsinger":
        rows = gtsinger(args.root, args.onsets, fraction=args.fraction, languages=args.languages)
    else:
        rows = corpus(args.audio_dir, args.n, args.start_seed, syllabic=not args.plain)
    counts: Dict[str, int] = {}

    def counted(rows):
        for r in rows:
            counts[r["split"]] = counts.get(r["split"], 0) + 1
            yield r

    n = manifest.write(args.out, counted(rows))
    print(f"{n} items -> {args.out}: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
