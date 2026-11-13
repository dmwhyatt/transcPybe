"""Transcribing a folder of audio files.

A batch is where progress reporting stops being a nicety: a few hundred songs
on a GPU box runs for hours, usually detached under tmux, and the only way to
know whether it is healthy is what it writes to the log.

Three properties matter for a run that long, and all three are here: it is
resumable, so an interrupted run does not start over; a single bad file does
not end the batch; and the estimated time remaining is weighted by audio
duration rather than file count, because songs are not all the same length.

A file is skipped only if its outputs exist and its ``.notes.json`` records the
settings this run would use, so a rerun after a change of settings or version
(2.0.0 changed the default preset and pitch tracker) redoes the files made
otherwise rather than leaving a folder of mixed transcriptions.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import TranscriptionConfig
from .core import TranscriptionResult, transcribe_file
from .midi_io import write_csv, write_json, write_midi
from .progress import ProgressReporter, format_duration, null_reporter

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aif", ".aiff", ".opus")

# Settings that change how a run goes but not its notes. A file made with other
# values of these is still current: "auto" resolves to a different device on
# another machine, and the batch size and memory logging change only speed and logs.
RUN_ONLY_SETTINGS = frozenset({"device", "crepe_batch_size", "gpu_memory_log_every"})
_NOT_RECORDED = object()


def settings_changed(json_path: Path, config: TranscriptionConfig) -> Optional[List[Tuple[str, object, object]]]:
    """How the settings recorded in ``json_path`` differ from those ``config`` runs with.

    Returns (setting, recorded, current) for each difference, an empty list if
    there is none, or None if there is no record to compare against. A setting
    the record lacks (an older version's output) counts as a difference.
    """
    from .landmarks import preset_config

    if not json_path.exists():
        return None
    try:
        recorded = json.loads(json_path.read_text())["config"]
        if not isinstance(recorded, dict):
            raise TypeError
    except (OSError, ValueError, KeyError, TypeError):
        return [("settings", "unreadable", "this run's")]
    current = preset_config(config).to_dict()       # what transcription records: the preset's decoder settings applied
    return [(k, recorded.get(k, _NOT_RECORDED), v) for k, v in current.items()
            if k not in RUN_ONLY_SETTINGS and recorded.get(k, _NOT_RECORDED) != v]


def _describe(changes) -> str:
    shown = [f"{k} {'(not recorded)' if old is _NOT_RECORDED else repr(old)} -> {new!r}" for k, old, new in changes[:3]]
    return ", ".join(shown) + (f" and {len(changes) - 3} more" if len(changes) > 3 else "")


@dataclass
class BatchItem:
    """One file's outcome."""

    path: Path
    status: str                     # 'done', 'skipped' or 'failed'
    n_notes: int = 0
    duration: float = 0.0
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class BatchResult:
    items: List[BatchItem] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def done(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == "done"]

    @property
    def failed(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == "failed"]

    @property
    def skipped(self) -> List[BatchItem]:
        return [i for i in self.items if i.status == "skipped"]

    def summary(self) -> str:
        lines = [
            f"Batch finished in {format_duration(self.elapsed)}: "
            f"{len(self.done)} transcribed, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed"
        ]
        if self.done:
            notes = sum(i.n_notes for i in self.done)
            audio = sum(i.duration for i in self.done)
            lines.append(f"  {notes} notes over {format_duration(audio)} of audio")
        for item in self.failed:
            lines.append(f"  FAILED {item.path.name}: {item.error}")
        return "\n".join(lines)


def find_audio_files(
    root: Path, extensions: Sequence[str] = AUDIO_EXTENSIONS, recursive: bool = True
) -> List[Path]:
    """Audio files under ``root``, sorted so runs are reproducible."""
    root = Path(root)
    if root.is_file():
        return [root]

    suffixes = {e.lower() for e in extensions}
    pattern = "**/*" if recursive else "*"
    found = [
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith(".")
    ]
    return sorted(found)


def probe_duration(path: Path) -> Optional[float]:
    """Read a file's duration from its header, without decoding it.

    Used only to weight the batch ETA. Returns None for formats soundfile
    cannot inspect cheaply, in which case the ETA falls back to file counts.
    """
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        return None
    return None


MAX_NAME_LEVEL = 6
"""How many trailing path components a name may use before giving up on
readability and falling back to a numeric suffix."""


def _name_at_level(path: Path, level: int) -> str:
    """A candidate output name built from ``level`` of surrounding context.

    0 is the bare stem, 1 is the containing directory alone -- which is what
    makes a tree of ``<song>/lead-vocal.wav`` stems read as song names -- and
    higher levels prepend more directories to the stem.
    """
    directories = list(path.parts[:-1])
    if level == 0 or not directories:
        return path.stem
    if level == 1:
        return directories[-1]
    tail = directories[-(level - 1):]
    return "__".join([*tail, path.stem])


def unique_output_names(paths: Sequence[Path]) -> Dict[Path, str]:
    """Give every input a distinct output name, using as little context as possible.

    Naming outputs after the file stem alone breaks badly on separated stems,
    where every file is called ``lead-vocal.wav``: they would all write to
    ``lead-vocal.mid``, and because finished work is skipped, a corpus of two
    hundred songs would quietly produce one transcription.

    The same level is used for the whole batch rather than resolving collisions
    file by file, so names stay consistent and predictable: a flat directory
    keeps its filenames untouched, and a tree of stems is named by song.
    """
    paths = list(paths)
    if not paths:
        return {}

    # Naming by the containing directory alone is right when every file shares
    # one stem, as separated stems all do, and misleading otherwise: it would
    # turn albums/A/track1.wav into "A" and throw the track name away.
    stems_all_identical = len({p.stem for p in paths}) == 1

    for level in range(MAX_NAME_LEVEL + 1):
        if level == 1 and not stems_all_identical:
            continue
        names = [_name_at_level(p, level) for p in paths]
        if len(set(names)) == len(names):
            return dict(zip(paths, names))

    # Pathological input (identical paths, or nesting deeper than the cap).
    # Uniqueness matters more than elegance here.
    assigned: Dict[Path, str] = {}
    seen: Dict[str, int] = {}
    for path in paths:
        base = _name_at_level(path, MAX_NAME_LEVEL)
        seen[base] = seen.get(base, 0) + 1
        assigned[path] = base if seen[base] == 1 else f"{base}__{seen[base]}"
    return assigned


def output_paths(path: Path, output_dir: Path, name: Optional[str] = None) -> dict:
    stem = name or path.stem
    return {
        "midi": output_dir / f"{stem}.mid",
        "json": output_dir / f"{stem}.notes.json",
        "csv": output_dir / f"{stem}.notes.csv",
    }


def transcribe_batch(
    paths: Iterable[Path],
    output_dir: Path,
    config: Optional[TranscriptionConfig] = None,
    *,
    progress: Optional[ProgressReporter] = None,
    skip_existing: bool = True,
    continue_on_error: bool = True,
    write_midi_files: bool = True,
    write_json_files: bool = True,
    write_csv_files: bool = True,
) -> BatchResult:
    """Transcribe many files, reporting per-file and overall progress."""
    config = config or TranscriptionConfig()
    config.validate()

    paths = list(paths)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = progress or null_reporter()

    names = unique_output_names(paths)
    durations = {p: probe_duration(p) for p in paths}
    total_audio = sum(d for d in durations.values() if d)
    known_all = all(d is not None for d in durations.values())

    progress.log(
        f"Batch: {len(paths)} files"
        + (f", {format_duration(total_audio)} of audio" if total_audio else "")
        + f" -> {output_dir}"
    )

    result = BatchResult()
    started = time.time()
    audio_done = 0.0
    unverified_noted = False

    for index, path in enumerate(paths, start=1):
        outputs = output_paths(path, output_dir, names.get(path))
        wanted = [
            outputs["midi"] if write_midi_files else None,
            outputs["json"] if write_json_files else None,
            outputs["csv"] if write_csv_files else None,
        ]
        wanted = [p for p in wanted if p is not None]

        if skip_existing and wanted and all(p.exists() for p in wanted):
            changes = settings_changed(outputs["json"], config)
            if not changes:
                progress.log(f"[{index}/{len(paths)}] {path.name}: already done, skipping")
                if changes is None and not unverified_noted:
                    progress.log("  (no .notes.json beside these outputs, so their settings cannot be "
                                 "checked; --overwrite redoes them)")
                    unverified_noted = True
                result.items.append(BatchItem(path=path, status="skipped",
                                              duration=durations.get(path) or 0.0))
                audio_done += durations.get(path) or 0.0
                continue
            progress.log(f"[{index}/{len(paths)}] {path.name}: made with other settings "
                         f"({_describe(changes)}); transcribing again")

        header = f"[{index}/{len(paths)}] {path.name}"
        if durations.get(path):
            header += f" ({format_duration(durations[path])})"
        progress.log(header)

        # Each file gets its own audio-duration reference so that the per-stage
        # realtime factors describe this file, not the batch.
        file_progress = ProgressReporter(
            enabled=progress.enabled, stream=progress.stream, logger=progress.logger,
            interval=progress.interval, audio_seconds=durations.get(path),
        )

        file_started = time.time()
        try:
            transcription = transcribe_file(path, config, file_progress)
        except KeyboardInterrupt:
            progress.log("Interrupted; stopping the batch.")
            raise
        except Exception as exc:
            logger.exception("Failed to transcribe %s", path)
            progress.log(f"  FAILED: {type(exc).__name__}: {exc}")
            result.items.append(BatchItem(path=path, status="failed",
                                          error=f"{type(exc).__name__}: {exc}",
                                          duration=durations.get(path) or 0.0))
            audio_done += durations.get(path) or 0.0
            if not continue_on_error:
                break
            continue

        _write_outputs(transcription, outputs, config,
                       write_midi_files, write_json_files, write_csv_files)

        elapsed = time.time() - file_started
        result.items.append(BatchItem(
            path=path, status="done", n_notes=transcription.n_notes,
            duration=transcription.duration, elapsed=elapsed,
        ))
        audio_done += durations.get(path) or transcription.duration
        progress.log(f"  {transcription.n_notes} notes in {format_duration(elapsed)}")

        progress.log("  " + _overall_line(
            index, len(paths), time.time() - started,
            audio_done, total_audio, known_all,
        ))

    result.elapsed = time.time() - started
    return result


def _overall_line(
    index: int, total_files: int, elapsed: float,
    audio_done: float, total_audio: float, weight_by_audio: bool,
) -> str:
    """Overall batch progress, weighted by audio duration where possible."""
    if weight_by_audio and total_audio > 0:
        fraction = min(audio_done / total_audio, 1.0)
    else:
        fraction = index / max(total_files, 1)

    text = f"overall {index}/{total_files} files ({fraction * 100:.0f}%), elapsed {format_duration(elapsed)}"
    if fraction > 0:
        remaining = elapsed * (1.0 - fraction) / fraction
        text += f", ETA {format_duration(remaining)}"
    return text


def _write_outputs(
    result: TranscriptionResult, outputs: dict, config: TranscriptionConfig,
    midi: bool, json_: bool, csv: bool,
) -> None:
    # The config the transcription ran with: a preset's decoder settings, the
    # device CREPE actually used.
    config = result.config
    if midi:
        write_midi(result.notes, outputs["midi"], config)
    if json_:
        write_json(result.notes, outputs["json"], config,
                   extra={"source": result.source, "duration": result.duration,
                          "device": result.device,
                          "tuning_offset_cents": 100.0 * result.tuning_offset})
    if csv:
        write_csv(result.notes, outputs["csv"])
