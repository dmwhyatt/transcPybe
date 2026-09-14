"""Command-line interface for vocal melody transcription.

    python -m transcpybe lead-vocal.wav -o out/
    python -m transcpybe stems/ -o midi/ --log-file run.log
    python -m transcpybe --self-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import TranscriptionConfig
from .progress import ProgressReporter, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcpybe",
        description="Transcribe monophonic singing into MIDI notes "
                    "(integer pitches, floating-point onsets). Give it a file, "
                    "or a directory to transcribe every audio file inside.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("audio", nargs="?",
                        help="Input audio file, or a directory of audio files")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory for outputs (default: alongside the input)")
    parser.add_argument("--name", default=None,
                        help="Base name for output files (single file only)")

    formats = parser.add_argument_group("output formats")
    formats.add_argument("--no-midi", action="store_true", help="Skip writing the .mid file")
    formats.add_argument("--no-json", action="store_true", help="Skip writing the .json file")
    formats.add_argument("--no-csv", action="store_true", help="Skip writing the .csv file")
    formats.add_argument("--print-notes", action="store_true",
                         help="Print the full note table to stdout")
    formats.add_argument("--max-rows", type=int, default=None,
                         help="Limit rows when printing (default: all)")

    monitoring = parser.add_argument_group("progress and logging")
    monitoring.add_argument("--no-progress", action="store_true",
                            help="Suppress progress reporting entirely")
    monitoring.add_argument("--progress-interval", type=float, default=5.0,
                            help="Seconds between progress lines when output is "
                                 "redirected to a file rather than a terminal")
    monitoring.add_argument("--log-file", default=None,
                            help="Append timestamped logs to this file as well as "
                                 "the console; survives a lost tmux session")
    monitoring.add_argument("--gpu-memory-every", type=int, default=0,
                            dest="gpu_memory_log_every",
                            help="Log GPU memory every N CREPE batches (0 disables)")
    monitoring.add_argument("-v", "--verbose", action="store_true",
                            help="Include debug-level detail in the log")
    monitoring.add_argument("-q", "--quiet", action="store_true",
                            help="Suppress the closing summary")

    model = parser.add_argument_group("pitch model")
    model.add_argument("--f0-backend", choices=("crepe", "pyin"), default="crepe",
                       help="crepe: neural, most accurate. pyin: no torch needed")
    model.add_argument("--crepe-model", choices=("full", "tiny"), default="full",
                       help="tiny is roughly 14x faster on CPU and less accurate")
    model.add_argument("--device", default="auto",
                       help="Torch device: auto (cuda, else mps, else cpu), or an "
                            "explicit cpu / cuda / cuda:1 / mps")
    model.add_argument("--crepe-batch-size", type=int, default=512,
                       help="CREPE frames per forward pass. Raise it on a large "
                            "GPU for speed, lower it if memory is tight")
    model.add_argument("--seed", type=int, default=0,
                       help="Seed for CREPE's +/-20 cent pitch dither, making a "
                            "run reproducible")
    model.add_argument("--no-seed", action="store_true",
                       help="Leave the dither unseeded, as torchcrepe does by default")
    model.add_argument("--fmin", type=float, default=55.0, help="Lowest f0 tracked, in Hz")
    model.add_argument("--fmax", type=float, default=1100.0, help="Highest f0 tracked, in Hz")

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--min-note", type=float, default=0.055, dest="min_note_duration_s",
                        help="Discard notes shorter than this, in seconds")
    tuning.add_argument("--pitch-sigma", type=float, default=0.55,
                        dest="pitch_sigma_semitones",
                        help="Emission width in semitones; larger absorbs more vibrato")
    tuning.add_argument("--p-self", type=float, default=0.985,
                        help="Note persistence; raise for fewer, longer notes")
    tuning.add_argument("--note-level-floor", type=float, default=30.0,
                        dest="note_level_floor_db",
                        help="Drop notes this many dB below the performance level. "
                             "Lower it to reject more separation residue, at the risk "
                             "of losing genuinely quiet singing")
    tuning.add_argument("--onset-n-fft", type=int, default=512,
                        dest="onset_n_fft",
                        help="FFT window for the onset envelope, in samples. 512 "
                             "is 32 ms at 16 kHz; raise it for a smoother envelope "
                             "at the cost of placing attacks less precisely")
    tuning.add_argument("--onset-aggregate", choices=("mean", "median"),
                        default="mean", dest="onset_aggregate",
                        help="How spectral flux is combined across frequency. "
                             "median only fires when most of the spectrum rises, "
                             "which misses vowel-initial onsets")
    tuning.add_argument("--onset-delta", type=float, default=1.0,
                        dest="onset_delta_ratio",
                        help="Peak threshold as a multiple of the onset envelope's "
                             "median. Raise it if noisy input over-segments")
    tuning.add_argument("--onset-max-size", type=int, default=9,
                        dest="onset_max_size",
                        help="Frequency max-filter width suppressing vibrato. "
                             "Raise it if sustained notes fragment")
    tuning.add_argument("--release-drop", type=float, default=1.0,
                        dest="release_drop_db",
                        help="End a note where it falls this many dB below its own "
                             "sustained level. Small by design: it locates the top "
                             "of the decay, not a point part-way down it")
    tuning.add_argument("--no-onsets", action="store_true",
                        help="Disable spectral-flux onset refinement")
    tuning.add_argument("--no-glides", action="store_true",
                        help="Disable portamento handling")
    tuning.add_argument("--no-release-trim", action="store_true",
                        help="Report note ends where the pitch decoder left them, "
                             "without trimming the release")

    batch = parser.add_argument_group("batch mode (directory input)")
    batch.add_argument("--no-recursive", action="store_true",
                       help="Do not descend into subdirectories")
    batch.add_argument("--overwrite", action="store_true",
                       help="Re-transcribe files whose outputs already exist "
                            "(default is to skip them, so a run is resumable)")
    batch.add_argument("--stop-on-error", action="store_true",
                       help="Abort the batch on the first failure "
                            "(default is to log it and carry on)")

    output = parser.add_argument_group("midi")
    output.add_argument("--program", type=int, default=0, dest="midi_program",
                        help="General MIDI program number for the output track")
    output.add_argument("--tempo", type=float, default=120.0, dest="midi_tempo",
                        help="Tempo written to the MIDI header; does not move note times")

    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic validation suite and exit")

    return parser


def config_from_args(args: argparse.Namespace) -> TranscriptionConfig:
    config = TranscriptionConfig(
        f0_backend=args.f0_backend,
        crepe_model=args.crepe_model,
        crepe_batch_size=args.crepe_batch_size,
        device=args.device,
        seed=None if args.no_seed else args.seed,
        gpu_memory_log_every=args.gpu_memory_log_every,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        min_note_duration_s=args.min_note_duration_s,
        pitch_sigma_semitones=args.pitch_sigma_semitones,
        p_self=args.p_self,
        note_level_floor_db=args.note_level_floor_db,
        onset_n_fft=args.onset_n_fft,
        onset_aggregate=args.onset_aggregate,
        onset_max_size=args.onset_max_size,
        onset_delta_ratio=args.onset_delta_ratio,
        release_drop_db=args.release_drop_db,
        use_onsets=not args.no_onsets,
        absorb_glides=not args.no_glides,
        trim_note_releases=not args.no_release_trim,
        midi_program=args.midi_program,
        midi_tempo=args.midi_tempo,
    )
    config.validate()
    return config


def run_self_test(progress: ProgressReporter) -> int:
    """Transcribe the synthetic suite and report accuracy."""
    from .core import transcribe_audio
    from .evaluate import format_scores, score
    from .synth import ALL_CASES

    config = TranscriptionConfig()
    rows = []
    with progress.stage("self-test", total=len(ALL_CASES)) as stage:
        for name, make_case in ALL_CASES.items():
            audio, ground_truth = make_case()
            result = transcribe_audio(audio, config)
            rows.append((name, score(result.notes, ground_truth)))
            stage.advance(1, detail=name)

    print()
    print(format_scores(rows))
    return 0


def _run_single(args, config, progress) -> int:
    from .core import transcribe_file
    from .midi_io import format_note_table, write_csv, write_json, write_midi

    audio_path = Path(args.audio)
    result = transcribe_file(audio_path, config, progress)

    output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent
    stem = args.name or audio_path.stem
    written = []

    if not args.no_midi:
        written.append(write_midi(result.notes, output_dir / f"{stem}.mid", config))
    if not args.no_json:
        written.append(write_json(
            result.notes, output_dir / f"{stem}.notes.json", config,
            extra={"source": str(audio_path), "duration": result.duration,
                   "device": result.device},
        ))
    if not args.no_csv:
        written.append(write_csv(result.notes, output_dir / f"{stem}.notes.csv"))

    if args.print_notes:
        print(format_note_table(result.notes, args.max_rows))

    if not args.quiet:
        print()
        print(result.summary())
        for path in written:
            print(f"  wrote {path}")
    return 0


def _run_batch(args, config, progress) -> int:
    from .batch import find_audio_files, transcribe_batch

    root = Path(args.audio)
    paths = find_audio_files(root, recursive=not args.no_recursive)
    if not paths:
        print(f"error: no audio files found under {root}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else root
    result = transcribe_batch(
        paths, output_dir, config,
        progress=progress,
        skip_existing=not args.overwrite,
        continue_on_error=not args.stop_on_error,
        write_midi_files=not args.no_midi,
        write_json_files=not args.no_json,
        write_csv_files=not args.no_csv,
    )

    if not args.quiet:
        print()
        print(result.summary())
    return 1 if result.failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logger = setup_logging(verbose=args.verbose, log_file=args.log_file)
    # Always route permanent lines through the logger, so every line carries a
    # timestamp and reaches the log file. The live in-place progress line still
    # goes straight to the terminal, where it belongs.
    progress = ProgressReporter(
        enabled=not args.no_progress,
        logger=logger,
        interval=args.progress_interval,
    )

    if args.self_test:
        return run_self_test(progress)

    if not args.audio:
        build_parser().print_help()
        return 2

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"error: no such file or directory: {audio_path}", file=sys.stderr)
        return 1

    config = config_from_args(args)

    try:
        if audio_path.is_dir():
            return _run_batch(args, config, progress)
        return _run_single(args, config, progress)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
