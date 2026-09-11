"""Report what a dataset directory actually contains.

Every dataset in this project has arrived in a shape different from the one its
paper describes. vocadito writes onset/pitch/duration where the convention is
onset/offset/pitch; its low voices store frequencies that a value count reads as
MIDI; M4Singer has no file under the name its paper implies. Each was found by
scoring a real run and working backwards from a number that made no sense.

Reading the directory first is cheaper. This makes no assumptions about layout
or naming: it counts what is there, previews one example of each kind, and says
what the loader makes of it. What the file turns out to be is then a fact rather
than an inference from a paper.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif", ".opus"}
PREVIEW_LINES = 6
PREVIEW_BYTES = 4000
MAX_EXAMPLES = 8


def _read_text(path: Path) -> Optional[str]:
    """Read a text file, trying the encodings annotation files actually use.

    Praat writes TextGrids as UTF-16 as often as UTF-8, and a mis-decoded file
    looks like a corrupt one rather than an encoding problem.
    """
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as handle:
                text = handle.read(PREVIEW_BYTES)
            if "\x00" not in text:
                return text
        except (UnicodeDecodeError, UnicodeError, OSError):
            continue
    return None


def _describe_json(path: Path) -> List[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return [f"unreadable as JSON: {exc}"]

    lines = []
    if isinstance(payload, dict):
        keys = list(payload)
        lines.append(f"object with {len(keys)} keys, first: {keys[:5]}")
        if keys:
            value = payload[keys[0]]
            lines.append(f"first value is {type(value).__name__}")
            if isinstance(value, dict):
                lines.append(f"  its keys: {list(value)[:10]}")
                for k, v in list(value.items())[:4]:
                    if isinstance(v, list):
                        lines.append(f"  {k}: list of {len(v)}, e.g. {v[:5]}")
            elif isinstance(value, list) and value:
                lines.append(f"  list of {len(value)}, first entry: {value[0]}")
    elif isinstance(payload, list):
        lines.append(f"array of {len(payload)}; first entry: "
                     f"{payload[0] if payload else 'empty'}")
    return lines


def _describe_textgrid(path: Path) -> List[str]:
    """Name the tiers, which is what says whether notes are in here at all."""
    text = _read_text(path)
    if text is None:
        return ["could not decode as text in utf-8, utf-16 or latin-1"]

    tiers = [line.split("=", 1)[1].strip().strip('"')
             for line in text.splitlines()
             if line.strip().startswith("name =")]
    lines = [f"tiers: {tiers}" if tiers else "no tier names found in the first 4 KB"]
    for line in text.splitlines()[:4]:
        if line.strip():
            lines.append(f"  {line.strip()[:90]}")
    return lines


def _describe_midi(path: Path) -> List[str]:
    try:
        import pretty_midi

        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        return [f"unreadable as MIDI: {exc}"]
    total = sum(len(i.notes) for i in midi.instruments)
    lines = [f"{len(midi.instruments)} instrument(s), {total} notes"]
    for instrument in midi.instruments[:2]:
        if instrument.notes:
            note = instrument.notes[0]
            lines.append(f"  first note: pitch {note.pitch}, "
                         f"{note.start:.3f}-{note.end:.3f}s")
    return lines


def _describe_text(path: Path) -> List[str]:
    text = _read_text(path)
    if text is None:
        return ["could not decode as text"]
    return [f"  {line[:100]}" for line in text.splitlines()[:PREVIEW_LINES] if line.strip()]


def _try_loader(path: Path) -> List[str]:
    """What the benchmark's own loader makes of the file."""
    from .annotations import load_annotation, load_keyed

    lines = []
    try:
        annotation = load_annotation(path)
        if annotation.notes:
            pitches = [p for _, _, p in annotation.notes]
            lines.append(f"loader: {annotation.n_notes} notes, "
                         f"pitch {round(min(pitches))}-{round(max(pitches))}, "
                         f"first onset {annotation.notes[0][0]:.2f}s")
            lines.append(f"  inferred: " + ", ".join(
                f"{k}={v}" for k, v in annotation.inferences.items()))
        else:
            lines.append(f"loader: no notes read"
                         + (f" ({'; '.join(annotation.warnings)})"
                            if annotation.warnings else ""))
    except Exception as exc:
        lines.append(f"loader: failed -- {type(exc).__name__}: {exc}")

    if path.suffix.lower() == ".json":
        try:
            keyed = load_keyed(path)
            with_notes = sum(1 for a in keyed.values() if a.notes)
            lines.append(f"as a keyed corpus: {len(keyed)} ids, {with_notes} with notes")
        except Exception as exc:
            lines.append(f"as a keyed corpus: no -- {exc}")
    return lines


def inspect_directory(root: Path, max_depth: int = 4) -> str:
    """Census the directory, then preview one example of each kind of file."""
    root = Path(root)
    if not root.exists():
        return f"error: no such directory: {root}"

    by_suffix: Counter = Counter()
    examples: Dict[str, Path] = {}
    depths: Counter = Counter()
    total = 0
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        total += 1
        suffix = path.suffix.lower() or "(no extension)"
        by_suffix[suffix] += 1
        depths[depth] += 1
        examples.setdefault(suffix, path)

    lines = [f"\n{root}", f"{total} files"]

    audio = sum(n for s, n in by_suffix.items() if s in AUDIO_SUFFIXES)
    lines.append(f"{audio} audio, {total - audio} other\n")

    lines.append("By type:")
    for suffix, count in by_suffix.most_common(20):
        marker = " (audio)" if suffix in AUDIO_SUFFIXES else ""
        lines.append(f"  {suffix:<16} {count:>6}{marker}   e.g. "
                     f"{examples[suffix].relative_to(root)}")

    # Layout matters: annotations one directory per recording pair differently
    # from a flat directory, and both differ from one file for the corpus.
    lines.append("\nDepth of files below the root:")
    for depth in sorted(depths):
        lines.append(f"  {depth} level(s): {depths[depth]} files")

    lines.append("\n" + "=" * 70)
    lines.append("Previews (one example per type, audio skipped)")
    lines.append("=" * 70)

    shown = 0
    for suffix, _ in by_suffix.most_common():
        if suffix in AUDIO_SUFFIXES or shown >= MAX_EXAMPLES:
            continue
        path = examples[suffix]
        shown += 1
        lines.append(f"\n{path.relative_to(root)}  ({by_suffix[suffix]} of this type)")
        if suffix == ".json":
            body = _describe_json(path)
        elif suffix == ".textgrid":
            body = _describe_textgrid(path)
        elif suffix in (".mid", ".midi"):
            body = _describe_midi(path)
        else:
            body = _describe_text(path)
        lines += [f"  {line}" for line in body]

        if suffix in (".json", ".mid", ".midi", ".csv", ".tsv", ".txt"):
            lines += [f"  {line}" for line in _try_loader(path)]

    lines.append(
        "\nWhat to look for: which file holds one row or entry per sung note, and\n"
        "whether it is one file per recording or one for the corpus. If nothing\n"
        "here does, the notes may be derivable from another annotation -- a Praat\n"
        "tier, or a score -- which is a conversion rather than a loader."
    )
    return "\n".join(lines)


def dry_run(pairing) -> int:
    """Report what was paired and how each annotation was read, without scoring.

    The two ways a benchmark silently reports nonsense are a mis-paired file and
    a misread annotation, and both are cheap to see and expensive to discover
    after a long run. The pitch and time ranges are the giveaway: a pitch range
    of 20-40 means a frequency column was read as MIDI, and a first onset far
    from the start of the audio means the reference is not aligned to it.
    """
    from .annotations import load_annotation

    print("\nAnnotation check:")
    header = (f"  {'item':<22} {'annotation file':<30} {'notes':>6} {'first':>8} "
              f"{'last':>9} {'pitches':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    problems = 0
    for item in pairing.items:
        try:
            annotation = load_annotation(item.annotation)
        except Exception as exc:
            print(f"  {item.item_id[:22]:<22} {item.annotation.name[:30]:<30} "
                  f"FAILED: {type(exc).__name__}: {exc}")
            problems += 1
            continue

        if not annotation.notes:
            print(f"  {item.item_id[:22]:<22} {item.annotation.name[:30]:<30} "
                  f"{'0':>6}  no notes read")
            problems += 1
            continue

        onsets = [n[0] for n in annotation.notes]
        pitches = [n[2] for n in annotation.notes]
        interpretation = ", ".join(f"{k}={v}" for k, v in annotation.inferences.items())
        print(
            f"  {item.item_id[:22]:<22} {item.annotation.name[:30]:<30} "
            f"{len(annotation.notes):>6} "
            f"{min(onsets):>7.2f}s {max(n[1] for n in annotation.notes):>8.2f}s "
            f"{round(min(pitches)):>4}-{round(max(pitches)):<5}"
        )
        print(f"  {'':<22} {interpretation}")

        # The footer asks the reader to eyeball this range. A check that runs
        # every time is what actually catches it: no voice sings below the
        # bottom of a bass or above the top of a soprano, so a median outside
        # that means a column that is not a pitch was read as one.
        median_pitch = sorted(pitches)[len(pitches) // 2]
        if not PLAUSIBLE_SUNG_MIDI[0] <= median_pitch <= PLAUSIBLE_SUNG_MIDI[1]:
            print(f"  {'':<22} WARN: median pitch is MIDI {median_pitch:.0f}, outside any "
                  "sung range -- a column that is not the pitch was read as one")
            problems += 1
        for warning in annotation.warnings:
            print(f"  {'':<22} WARN: {warning}")
            problems += 1

    # Unpaired audio counts as a problem here too. Reporting "all annotations
    # read cleanly" while a third of the dataset failed to pair is exactly the
    # kind of reassuring summary that gets skimmed past.
    problems += len(pairing.unpaired_audio)

    print()
    if problems:
        print(f"{problems} thing(s) worth checking above before running the benchmark.")
        if pairing.unpaired_audio:
            print(f"  ({len(pairing.unpaired_audio)} of them are audio files with no "
                  "annotation, listed at the top.)")
    else:
        print(f"All {len(pairing.items)} items paired and read cleanly.")
    print(
        "\nSanity checks: pitch ranges should look like sung notes (roughly MIDI 40-84).\n"
        "A range near 20-40 means a frequency column was read as MIDI. A first onset\n"
        "far from the start of its audio suggests the reference is not aligned to it --\n"
        "the benchmark will flag that too, but it is cheaper to know now."
    )
    return 0 if problems == 0 else 1



PLAUSIBLE_SUNG_MIDI = (36, 88)
