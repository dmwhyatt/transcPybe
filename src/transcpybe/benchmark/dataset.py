"""Pairing benchmark audio with its reference annotations.

Datasets rarely name the two identically. vocadito ships ``vocadito_1.wav``
beside ``vocadito_1_notes.csv``; others put annotations in a parallel directory,
or use a different extension for the same stem. Pairing is therefore done by
matching stems with a documented set of fallbacks, and every pairing decision is
recorded so an item silently matched to the wrong reference is visible.

Unpaired audio is reported rather than skipped quietly: a benchmark that scores
twelve of forty files and says nothing is worse than one that fails.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .annotations import ANNOTATION_SUFFIXES

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aif", ".aiff")

# Suffixes datasets append to an annotation stem that the audio stem lacks.
COMMON_ANNOTATION_TAGS = ("_notes", "_note", "_annotation", "_annotations",
                          "_gt", "_ref", "_melody", "_midi", "_pitch")

# When one recording has several annotation files -- note-level, f0 contour,
# lyrics -- these words in the remainder of the filename mark the one wanted.
# Datasets also ship per-annotator variants such as _notesA and _notesB, which
# an exact tag match misses, so the remainder is searched rather than compared.
NOTE_ANNOTATION_HINTS = ("note", "melody", "midi", "transcription")
NON_NOTE_HINTS = ("f0", "pitch_contour", "contour", "lyric", "phoneme",
                  "word", "syllable", "beat", "chord", "tempo")


@dataclass
class Item:
    """One benchmark recording and its reference."""

    item_id: str
    audio: Path
    annotation: Path
    match: str = "exact"      # how the pairing was made

    def to_dict(self) -> Dict[str, str]:
        return {
            "item_id": self.item_id,
            "audio": str(self.audio),
            "annotation": str(self.annotation),
            "match": self.match,
        }


@dataclass
class Pairing:
    items: List[Item] = field(default_factory=list)
    unpaired_audio: List[Path] = field(default_factory=list)
    unused_annotations: List[Path] = field(default_factory=list)
    audio_stems: List[str] = field(default_factory=list)
    annotation_stems: List[str] = field(default_factory=list)

    def diagnose(self, limit: int = 8) -> str:
        """Show both sides when pairing goes badly.

        Filename conventions vary enough between datasets that the useful thing
        to print is the names themselves, rather than another guess about them.
        """
        lines = ["", "What the two sides actually look like:"]
        lines.append(f"  audio stems ({len(self.audio_stems)}):")
        for stem in self.audio_stems[:limit]:
            lines.append(f"    {stem}")
        if len(self.audio_stems) > limit:
            lines.append(f"    ... and {len(self.audio_stems) - limit} more")
        lines.append(f"  annotation stems ({len(self.annotation_stems)}):")
        for stem in self.annotation_stems[:limit]:
            lines.append(f"    {stem}")
        if len(self.annotation_stems) > limit:
            lines.append(f"    ... and {len(self.annotation_stems) - limit} more")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [f"{len(self.items)} items paired"]
        by_match: Dict[str, int] = {}
        for item in self.items:
            by_match[item.match] = by_match.get(item.match, 0) + 1
        if set(by_match) - {"exact"}:
            lines.append("  matched by: " + ", ".join(
                f"{k} x{v}" for k, v in sorted(by_match.items())))
        if self.unpaired_audio:
            lines.append(f"  {len(self.unpaired_audio)} audio files had no annotation:")
            for path in self.unpaired_audio[:5]:
                lines.append(f"    {path.name}")
            if len(self.unpaired_audio) > 5:
                lines.append(f"    ... and {len(self.unpaired_audio) - 5} more")
        if self.unused_annotations:
            lines.append(f"  {len(self.unused_annotations)} annotations had no audio")

        # One annotation standing in for many recordings is the failure that
        # does not announce itself: every item pairs, every file reads cleanly,
        # and the score is computed against another recording's notes.
        distinct = len({item.annotation for item in self.items})
        if self.items and distinct < len(self.items):
            shared = len(self.items) - distinct
            lines += [
                f"  WARNING: {len(self.items)} recordings share {distinct} annotation "
                f"file(s); {shared} are reusing another recording's notes.",
                "    Pairing matches within a recording's own directory first, so this",
                "    means the annotation is genuinely elsewhere. Check --annotations,",
                "    and compare a few items in --dry-run: identical note counts and",
                "    time ranges across different recordings confirm it.",
            ]
        return "\n".join(lines)


def _normalise(stem: str) -> str:
    """Strip the tags datasets add to annotation filenames, and case."""
    lowered = stem.lower()
    for tag in COMMON_ANNOTATION_TAGS:
        if lowered.endswith(tag):
            lowered = lowered[: -len(tag)]
            break
    return lowered.strip("_-. ")


def find_files(root: Path, suffixes: Sequence[str]) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in suffixes else []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    )


def _prefix_candidates(audio_stem: str, annotation_files: Sequence[Path]) -> List[Path]:
    """Annotations whose stem begins with the audio stem.

    This is the general form of the tag rule: a dataset that names a clip
    ``vocadito_1`` and its annotations ``vocadito_1_notesA`` and
    ``vocadito_1_f0`` is matched here, where comparing against a fixed list of
    tags fails because the tag is not the end of the name.
    """
    lowered = audio_stem.lower()
    out = []
    for path in annotation_files:
        stem = path.stem.lower()
        if stem == lowered:
            out.append(path)
        elif stem.startswith(lowered) and stem[len(lowered):len(lowered) + 1] in ("_", "-", "."):
            out.append(path)
    return out


def _choose_note_annotation(candidates: Sequence[Path], audio_stem: str) -> Optional[Path]:
    """Pick the note-level annotation from several for the same recording."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def remainder(path: Path) -> str:
        stem = path.stem.lower()
        return stem[len(audio_stem):].strip("_-.") if stem.startswith(audio_stem.lower()) else stem

    preferred = [p for p in candidates
                 if any(hint in remainder(p) for hint in NOTE_ANNOTATION_HINTS)]
    if preferred:
        return sorted(preferred)[0]

    # Nothing looks like notes; at least rule out what clearly is not.
    remaining = [p for p in candidates
                 if not any(hint in remainder(p) for hint in NON_NOTE_HINTS)]
    return sorted(remaining or candidates)[0]


def pair_dataset(
    audio_dir: Path,
    annotation_dir: Optional[Path] = None,
    annotation_contains: Optional[str] = None,
) -> Pairing:
    """Pair every audio file with its annotation.

    Tried in order: identical stem, stem after stripping a known annotation tag,
    then annotation stem containing the audio stem. The last is a last resort and
    is reported as such, because it is the one that can go wrong.
    """
    audio_dir = Path(audio_dir)
    annotation_root = Path(annotation_dir) if annotation_dir else audio_dir

    audio_files = find_files(audio_dir, AUDIO_SUFFIXES)
    annotation_files = find_files(annotation_root, ANNOTATION_SUFFIXES)

    if annotation_contains:
        needle = annotation_contains.lower()
        annotation_files = [p for p in annotation_files if needle in p.stem.lower()]

    # A recording's own directory is tried before the corpus at large. Corpora
    # that number segments inside a folder per song reuse stems across folders --
    # M4Singer has 700 folders each starting at 0000 -- and a single index keyed
    # on the bare stem hands every song the first song's annotations, pairing the
    # whole corpus against a hundred-odd files without failing.
    in_dir: Dict[Tuple[str, str], Path] = {}
    in_dir_normalised: Dict[Tuple[str, str], Path] = {}
    by_stem: Dict[str, Path] = {}
    by_normalised: Dict[str, Path] = {}
    stem_counts: Dict[str, int] = {}
    normalised_counts: Dict[str, int] = {}
    for path in annotation_files:
        in_dir.setdefault((str(path.parent), path.stem), path)
        in_dir_normalised.setdefault((str(path.parent), _normalise(path.stem)), path)
        by_stem.setdefault(path.stem, path)
        by_normalised.setdefault(_normalise(path.stem), path)
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1
        key = _normalise(path.stem)
        normalised_counts[key] = normalised_counts.get(key, 0) + 1

    pairing = Pairing()
    used: set = set()

    for audio in audio_files:
        annotation, how = None, ""
        here = str(audio.parent)
        normalised = _normalise(audio.stem)
        if (here, audio.stem) in in_dir:
            annotation, how = in_dir[(here, audio.stem)], "exact"
        elif (here, normalised) in in_dir_normalised:
            annotation, how = in_dir_normalised[(here, normalised)], "tag-stripped"
        elif audio.stem in by_stem and stem_counts[audio.stem] == 1:
            annotation, how = by_stem[audio.stem], "exact"
        elif normalised in by_normalised and normalised_counts[normalised] == 1:
            annotation, how = by_normalised[normalised], "tag-stripped"
        elif stem_counts.get(audio.stem, 0) > 1:
            # Several annotations share this stem and none is beside the audio.
            # Choosing one would be a guess, and a guess here scores a recording
            # against another recording's notes without ever looking wrong.
            pairing.unpaired_audio.append(audio)
            continue
        else:
            candidates = _prefix_candidates(audio.stem, annotation_files)
            chosen = _choose_note_annotation(candidates, audio.stem)
            if chosen is not None:
                how = "prefix" if len(candidates) == 1 else f"prefix (1 of {len(candidates)})"
                annotation = chosen
            else:
                target = _normalise(audio.stem)
                substring = [
                    path for stem, path in by_normalised.items() if target and target in stem
                ]
                if len(substring) == 1:
                    annotation, how = substring[0], "substring"

        if annotation is None:
            pairing.unpaired_audio.append(audio)
            continue

        used.add(annotation)
        pairing.items.append(Item(item_id=audio.stem, audio=audio,
                                  annotation=annotation, match=how))

    # Corpora that number segments inside a directory per song -- M4Singer has
    # 20896 files across 700 folders, every one of them 0000.wav, 0001.wav --
    # give every recording the same stem. The stem alone is the cache key and
    # the reported item name, so identical ids would collapse the whole corpus
    # onto a handful of cache entries and score the same few recordings
    # repeatedly, at a plausible-looking number.
    _assign_unique_item_ids(pairing.items)

    pairing.unused_annotations = [p for p in annotation_files if p not in used]
    pairing.audio_stems = [p.stem for p in audio_files]
    pairing.annotation_stems = [p.stem for p in annotation_files]
    return pairing


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """Truncated SHA-256, enough to pin a file's identity in a report."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def dataset_digest(items: Sequence[Item]) -> str:
    """One digest standing for the whole dataset.

    Recorded with every result so a score can always be tied back to exactly the
    files that produced it, and so a changed or partially downloaded dataset
    cannot be mistaken for the one a previous number came from.
    """
    combined = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.item_id):
        combined.update(item.item_id.encode())
        combined.update(file_digest(item.audio).encode())
        combined.update(file_digest(item.annotation).encode())
    return combined.hexdigest()[:16]


def pair_annotators(
    annotation_dir: Path, tag_a: str, tag_b: str
) -> List[Tuple[str, Path, Path]]:
    """Pair two annotators' files for the same recording.

    Datasets that ship independent annotations of the same audio -- vocadito's
    ``_notesA`` and ``_notesB`` -- let the benchmark measure its own ceiling.
    Two people listening to the same singing disagree about where notes begin
    and end, and no transcriber can be scored above the rate at which they
    agree with each other. Reporting a model's F without that number invites
    reading a low score as a bad model when it is really a hard dataset.
    """
    annotation_files = find_files(annotation_dir, ANNOTATION_SUFFIXES)

    by_key_a: Dict[str, Path] = {}
    by_key_b: Dict[str, Path] = {}
    for path in annotation_files:
        if tag_a in path.stem:
            by_key_a[path.stem.replace(tag_a, "")] = path
        if tag_b in path.stem:
            by_key_b[path.stem.replace(tag_b, "")] = path

    return [(key, by_key_a[key], by_key_b[key])
            for key in sorted(set(by_key_a) & set(by_key_b))]


def annotation_tags(annotation_dir: Path, limit: int = 12) -> List[Tuple[str, int]]:
    """The trailing name parts that distinguish annotations of one recording.

    Datasets name these inconsistently -- vocadito's are notesA1 and notesA2,
    not the notesA and notesB its paper's prose suggests -- so a failed pairing
    should show what is on disk rather than leave the names to be guessed at
    twice.
    """
    counts: Dict[str, int] = {}
    for path in find_files(annotation_dir, ANNOTATION_SUFFIXES):
        if "_" in path.stem:
            tag = path.stem.rsplit("_", 1)[1]
            counts[tag] = counts.get(tag, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def pair_estimates(items: Sequence[Item], estimates_dir: Path) -> Dict[str, Path]:
    """Match each benchmark item to a transcription produced by another system.

    Scoring a competitor means running it over the same audio and reading its
    output back through the same metric. Comparing published figures instead
    compares datasets: a system reporting 0.77 on studio recordings of trained
    singers is not beating one reporting 0.41 on phone recordings of untrained
    ones, and on some datasets 0.77 is above what two annotators manage.
    """
    files = find_files(estimates_dir, ANNOTATION_SUFFIXES)
    found: Dict[str, Path] = {}
    for item in items:
        stem = item.audio.stem
        exact = [p for p in files if p.stem == stem]
        chosen = exact[0] if exact else None
        if chosen is None:
            candidates = _prefix_candidates(stem, files)
            chosen = _choose_note_annotation(candidates, stem)
        if chosen is not None:
            found[item.item_id] = chosen
    return found


def _assign_unique_item_ids(items: List[Item]) -> None:
    """Widen item ids with parent directories until every one is distinct.

    The same width is used for the whole dataset so names stay predictable: a
    flat directory keeps its filenames, and a tree keeps song and segment.
    """
    if not items:
        return

    for level in range(1, 5):
        names = [_id_at_level(item.audio, level) for item in items]
        if len(set(names)) == len(names):
            for item, name in zip(items, names):
                item.item_id = name
            return

    # Deeper nesting than the cap, or genuinely identical paths. Uniqueness
    # matters more than readability, because the cache key depends on it.
    seen: Dict[str, int] = {}
    for item in items:
        base = _id_at_level(item.audio, 4)
        seen[base] = seen.get(base, 0) + 1
        item.item_id = base if seen[base] == 1 else f"{base}__{seen[base]}"


def _id_at_level(path: Path, level: int) -> str:
    """The stem, prefixed by up to ``level - 1`` enclosing directory names."""
    parts = list(path.parent.parts[-(level - 1):]) if level > 1 else []
    return "/".join([*parts, path.stem])


def group_of(item: Item) -> str:
    """The performance an item belongs to, where items are segments of one."""
    return item.item_id.rsplit("/", 1)[0] if "/" in item.item_id else item.item_id


def stratified_sample(items: Sequence[Item], n: int, seed: int = 0) -> List[Item]:
    """Draw ``n`` items spread across performances rather than taken in order.

    Directory order is not a sample. Taking the first 40 items of a corpus laid
    out one folder per song gives 40 segments of the first song or two, by one
    singer -- a measurement of that song, reported as a measurement of the
    dataset. Drawing round-robin across performances covers as many as the
    budget allows, and seeding it keeps the choice reproducible so two runs are
    comparable.
    """
    items = list(items)
    if n <= 0 or n >= len(items):
        return items

    buckets: Dict[str, List[Item]] = {}
    for item in items:
        buckets.setdefault(group_of(item), []).append(item)

    rng = random.Random(seed)
    order = sorted(buckets)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(buckets[key])

    chosen: List[Item] = []
    depth = 0
    while len(chosen) < n:
        added = False
        for key in order:
            if depth < len(buckets[key]):
                chosen.append(buckets[key][depth])
                added = True
                if len(chosen) == n:
                    break
        if not added:
            break
        depth += 1
    return sorted(chosen, key=lambda i: i.item_id)


def load_item_metadata(spec: str, item_ids: Sequence[str]) -> Tuple[Dict[str, str], str]:
    """Join a metadata CSV to benchmark items, returning item_id -> group.

    ``spec`` is ``path.csv:column``. Which column holds the identifier is found
    by testing each against the item ids rather than assumed, because datasets
    name it track_id, item_name, id, or the filename, and guessing wrong joins
    nothing while looking like it worked.
    """
    path_text, _, column = spec.rpartition(":")
    if not path_text or not column:
        raise ValueError(f"expected FILE:COLUMN, got {spec!r}")
    path = Path(path_text)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} has no rows")
    if column not in rows[0]:
        raise ValueError(f"{path.name} has no column {column!r}; "
                         f"it has {sorted(rows[0])}")

    wanted = {str(i) for i in item_ids}
    normalised = {_normalise(i): i for i in wanted}

    best_column, best_map, best_hits = "", {}, 0
    for candidate in rows[0]:
        if candidate == column:
            continue
        mapping: Dict[str, str] = {}
        for row in rows:
            key = str(row.get(candidate, "")).strip()
            if not key:
                continue
            # Datasets identify a track by a bare number as often as by its
            # filename, so a stem containing the key counts as a match.
            target = (key if key in wanted
                      else normalised.get(_normalise(key))
                      or next((i for i in wanted if i.endswith(f"_{key}")), None))
            if target:
                mapping[target] = str(row.get(column, "")).strip()
        if len(mapping) > best_hits:
            best_column, best_map, best_hits = candidate, mapping, len(mapping)

    if not best_map:
        raise ValueError(f"no column in {path.name} matched any item id "
                         f"(tried {sorted(rows[0])})")
    return best_map, best_column


def pair_from_file(path: Path) -> Pairing:
    """Read an explicit audio-to-annotation pairing from a CSV.

    Name matching cannot reach every corpus. Separated stems are all called
    lead-vocal.wav and identified by their directory, while their references are
    named by year and rank, so the two share no string to match on. Where the
    join needs outside knowledge -- a title, an artist, a catalogue number -- it
    belongs in a file that can be checked and corrected by hand, not in a
    heuristic that fails quietly.

    Columns: ``audio`` and ``annotation`` are required, ``item_id`` optional and
    defaulting to the audio's directory and stem.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path.name} has no rows")

    missing = {"audio", "annotation"} - set(rows[0])
    if missing:
        raise ValueError(f"{path.name} needs columns {sorted(missing)}; "
                         f"it has {sorted(rows[0])}")

    pairing = Pairing()
    for number, row in enumerate(rows, start=2):
        audio = Path(str(row["audio"]).strip()).expanduser()
        annotation = Path(str(row["annotation"]).strip()).expanduser()
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            item_id = f"{audio.parent.name}/{audio.stem}" if audio.parent.name \
                else audio.stem
        # A row naming a file that is not there is a mistake in the mapping, and
        # dropping it silently would score a subset while reporting the whole.
        if not audio.exists():
            pairing.unpaired_audio.append(audio)
            continue
        if not annotation.exists():
            raise ValueError(f"{path.name} line {number}: no such annotation: "
                             f"{annotation}")
        pairing.items.append(Item(item_id=item_id, audio=audio,
                                  annotation=annotation, match="explicit"))

    seen: Dict[str, int] = {}
    for item in pairing.items:
        seen[item.item_id] = seen.get(item.item_id, 0) + 1
    duplicated = [k for k, n in seen.items() if n > 1]
    if duplicated:
        raise ValueError(f"{path.name} repeats item_id {duplicated[:5]}; "
                         "ids are cache keys and must be distinct")
    return pairing
