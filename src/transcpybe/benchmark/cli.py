"""Command line for benchmarking the transcriber against reference annotations.

    python -m transcpybe.benchmark AUDIO_DIR --annotations ANNOT_DIR
    python -m transcpybe.benchmark --synthetic
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..config import TranscriptionConfig
from ..progress import ProgressReporter, setup_logging
from .compare import annotator_agreement, compare_results
from .dataset import pair_dataset
from .inspect import dry_run, inspect_directory
from .runner import format_result, run_benchmark
from .tuning import run_tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcpybe-benchmark",
        description="Score the transcriber against reference annotations, with "
                    "confidence intervals and reproducibility metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", nargs="?", help="Directory of benchmark audio")
    parser.add_argument("-a", "--annotations", default=None,
                        help="Directory of reference annotations "
                             "(default: alongside the audio)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write the full result as JSON to this path")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache transcriptions here so re-scoring is instant. "
                             "The cache is keyed on the configuration, so changing "
                             "a parameter re-transcribes rather than scoring the old one")

    parser.add_argument("--annotation-contains", default=None,
                        help="Only consider annotation files whose name contains this. "
                             "Use it when a dataset ships several annotations per "
                             "recording, e.g. --annotation-contains notesB to pin one "
                             "annotator")
    parser.add_argument("--agreement", nargs=2, metavar=("TAG_A", "TAG_B"),
                        help="Score one annotator against another instead of running the "
                             "transcriber, e.g. --agreement notesA notesB. This measures "
                             "the ceiling: no transcriber can be scored meaningfully above "
                             "the rate at which two people agree with each other.")
    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--tune", action="store_true",
                        help="Search the segmentation parameters against this dataset. "
                             "Splits recordings into fit, select and report; searches on "
                             "fit, chooses on select, and scores the report set once.")
    tuning.add_argument("--tune-trials", type=int, default=60)
    tuning.add_argument("--tune-beta", type=float, default=1.5,
                        help="Recall weight. Above 1 favours recall, which guards against "
                             "a reference that lists only part of what was sung: plain F "
                             "would reward transcribing less.")
    tuning.add_argument("--tune-objective", choices=("onset", "sequence"),
                        default="onset",
                        help="What to optimise. 'onset' needs the reference to share a "
                             "clock with the audio. 'sequence' compares the notes in "
                             "order and ignores time, which is the only honest choice "
                             "against a reference notated at a nominal tempo.")
    parser.add_argument("--f0-cache", default=".f0-cache", metavar="DIR",
                        help="Where f0 and onset tracks are kept. Alignment and tuning "
                             "both read them, and computing them is most of the cost, "
                             "so they share one directory.")
    tuning.add_argument("--split-file", default="split.json",
                        help="Where the fit/select/report split is written, and read back "
                             "on later runs so the report set stays the report set.")
    tuning.add_argument("--stratify-by", default=None, metavar="FILE:COLUMN",
                        help="Stratify the split by a metadata column, e.g. a decade.")
    tuning.add_argument("--tuned-config", default="tuned_config.json",
                        help="Where the winning configuration is written.")

    parser.add_argument("--inspect", default=None, metavar="DIR",
                        help="Report what a dataset directory actually contains -- file "
                             "census, layout, and a preview of each kind of annotation "
                             "with what the loader makes of it. Run this before "
                             "believing any score from a new dataset.")
    parser.add_argument("--pairs", default=None, metavar="FILE",
                        help="An explicit audio-to-annotation mapping (CSV with columns "
                             "audio, annotation, optionally item_id). For corpora where "
                             "no name matching can work -- separated stems all called "
                             "lead-vocal.wav, against references named by year and rank.")
    parser.add_argument("--annotations-json", default=None, metavar="FILE",
                        help="One JSON holding annotations for the whole corpus, keyed "
                             "by recording id, as M4Singer and MIR-ST500 ship them. "
                             "Expanded to one file per recording and paired normally.")
    parser.add_argument("--items", default=None, metavar="FILE",
                        help="Restrict the run to the recording ids listed in this file, "
                             "one per line. Needed to reproduce a published test split: "
                             "scoring a model on recordings it was trained on is not a "
                             "result.")
    parser.add_argument("--estimates", default=None, metavar="DIR",
                        help="Score transcriptions already produced by another system "
                             "(a directory of .mid/.csv files, matched to the audio by "
                             "name) instead of running this transcriber. The only "
                             "honest way to compare against a competitor: same audio, "
                             "same references, same tolerances.")
    parser.add_argument("--by-metadata", default=None, metavar="FILE:COLUMN",
                        help="With --compare, break the comparison down by a column of "
                             "a metadata CSV, e.g. vocadito_metadata.csv:language. Turns "
                             "a claim about why one system trails into a measurement: a "
                             "deficit that tracks the column is domain, one that does not "
                             "is the model.")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved result files (-o) on the recordings "
                             "they share, with a paired bootstrap. Answers whether a "
                             "change helped, which two separate confidence intervals "
                             "cannot.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pair the files and read the annotations, then stop. "
                             "Shows what matched what and how each annotation was "
                             "interpreted, in seconds, before committing to a long run")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Score N recordings drawn across the whole corpus rather "
                             "than the first N in directory order. On a corpus laid out "
                             "one folder per song, --limit measures one song and reports "
                             "it as the dataset.")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for --sample, so two runs draw the same recordings.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N items, for a quick sanity check")

    parser.add_argument("--synthetic", action="store_true",
                        help="Benchmark against the built-in synthetic suite instead. "
                             "Checks the harness end to end; it is not a substitute "
                             "for real annotated audio")

    model = parser.add_argument_group("transcriber")
    model.add_argument("--f0-backend", choices=("crepe", "pyin"), default="crepe")
    model.add_argument("--crepe-model", choices=("full", "tiny"), default="full")
    model.add_argument("--device", default="auto")
    model.add_argument("--seed", type=int, default=0)

    stats = parser.add_argument_group("scoring")
    stats.add_argument("--time-offset", type=float, default=0.0, metavar="SECONDS",
                       help="Shift every transcribed onset by this much before scoring, "
                            "to correct a known difference in what the reference calls a "
                            "note's start. Stated in the report and the saved result, "
                            "because the number is then conditional on it.")
    stats.add_argument("--align", action="store_true",
                       help="Warp each reference onto its recording before scoring, for "
                            "references notated at a nominal tempo. Aligned to the f0 "
                            "contour, which is the same for every configuration, so the "
                            "warp cannot absorb the differences being measured. "
                            "References that cannot be placed are reported and skipped.")

    stats.add_argument("--onset-shift", type=float, default=0.0, metavar="SECONDS",
                       help="Shift transcribed note starts only, leaving ends alone. For "
                            "a reference that calls something else the start of a note -- "
                            "a syllable including its consonant, say. Use this rather than "
                            "--time-offset when the note ends already agree.")
    stats.add_argument("--bootstrap", type=int, default=2000,
                       help="Bootstrap resamples for the confidence interval")
    stats.add_argument("--no-alignment-check", action="store_true",
                       help="Skip the global time-offset search (faster, but a "
                            "misaligned reference will go unnoticed)")
    stats.add_argument("--worst", type=int, default=10,
                       help="How many worst-scoring items to list")

    parser.add_argument("--log-file", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def _synthetic_dataset(directory: Path) -> Path:
    """Write the synthetic suite out as audio plus CSV annotations."""
    import soundfile as sf

    from ..synth import ALL_CASES

    directory.mkdir(parents=True, exist_ok=True)
    for name, make_case in ALL_CASES.items():
        audio, ground_truth = make_case()
        sf.write(directory / f"{name}.wav", audio, 16000)
        rows = ["onset,offset,pitch"]
        rows += [f"{onset:.6f},{offset:.6f},{pitch}" for onset, offset, pitch in ground_truth]
        (directory / f"{name}.csv").write_text("\n".join(rows) + "\n")
    return directory


# Roughly a bass low D to a soprano high E, wide enough that no real singing
# annotation falls outside it.
PLAUSIBLE_SUNG_MIDI = (36, 88)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logger = setup_logging(log_file=args.log_file)
    progress = ProgressReporter(enabled=not args.quiet, logger=logger)

    if args.inspect:

        print(inspect_directory(Path(args.inspect)))
        return 0

    if args.compare:
        # A saved result holds scores, not notes, so nothing here can be applied
        # after the fact. Accepting these silently lets a correction look tested
        # when it was never applied -- the run comes back identical and reads as
        # evidence that the correction made no difference.
        inert = [name for name, value in (
            ("--onset-shift", args.onset_shift),
            ("--time-offset", args.time_offset),
            ("--estimates", args.estimates),
            ("--cache-dir", args.cache_dir),
            ("--limit", args.limit),
            ("--align", args.align),
            ("--sample", args.sample),
            ("--annotations", args.annotations),
        ) if value]
        if inert:
            print(f"error: {', '.join(inert)} cannot apply to --compare, which reads "
                  "scores that are already computed.", file=sys.stderr)
            print("  Score with the flag and save the result, then compare that:",
                  file=sys.stderr)
            print("    benchmark <audio> --estimates DIR --onset-shift 0.041 "
                  "-o corrected.json", file=sys.stderr)
            print("    benchmark --compare corrected.json ours.json", file=sys.stderr)
            return 2
        return compare_results(Path(args.compare[0]), Path(args.compare[1]),
                               by_metadata=args.by_metadata)

    if args.synthetic:
        import tempfile

        audio_dir = _synthetic_dataset(Path(tempfile.mkdtemp(prefix="synthetic_benchmark_")))
        annotation_dir = None
        progress.log(f"Synthetic dataset written to {audio_dir}")
    elif args.audio:
        audio_dir = Path(args.audio)
        annotation_dir = Path(args.annotations) if args.annotations else None
        if not audio_dir.exists():
            print(f"error: no such directory: {audio_dir}", file=sys.stderr)
            return 1
    elif args.pairs:
        audio_dir, annotation_dir = Path("."), None
    else:
        build_parser().print_help()
        return 2

    # Agreement needs only the annotations, so it runs before pairing with audio.
    if args.agreement:
        return annotator_agreement(annotation_dir or audio_dir, *args.agreement)

    if args.annotations_json:
        import tempfile

        from .annotations import load_keyed

        try:
            keyed = load_keyed(args.annotations_json)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: could not read {args.annotations_json}: {exc}", file=sys.stderr)
            return 1
        # Expanded to one file per recording so the ordinary pairing, its name
        # matching and its dry-run all apply unchanged.
        annotation_dir = Path(tempfile.mkdtemp(prefix="keyed_annotations_"))
        empty = 0
        for item_id, annotation in keyed.items():
            if not annotation.notes:
                empty += 1
                continue
            (annotation_dir / f"{item_id}.csv").write_text(
                "\n".join(f"{s},{e},{p}" for s, e, p in annotation.notes))
        print(f"Expanded {len(keyed) - empty} keyed annotations from "
              f"{Path(args.annotations_json).name} to {annotation_dir}")
        if empty:
            print(f"  {empty} had no readable notes and were skipped")

    if args.pairs:
        from .dataset import pair_from_file

        try:
            pairing = pair_from_file(Path(args.pairs))
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        pairing = pair_dataset(audio_dir, annotation_dir,
                               annotation_contains=args.annotation_contains)
    print(pairing.summary())
    if not pairing.items:
        print(pairing.diagnose())
        print(
            "\nerror: no audio could be paired with an annotation.\n"
            "  Annotations are matched to audio by filename: same stem, then the same\n"
            "  stem with a tag such as _notes, then any annotation whose name begins\n"
            "  with the audio stem. Compare the two lists above.\n"
            "  If they live in separate trees, pass --annotations.",
            file=sys.stderr,
        )
        return 1

    ambiguous = [item for item in pairing.items if "1 of" in item.match]
    if ambiguous:
        print(
            f"\nNote: {len(ambiguous)} recording(s) had several annotations and the "
            "note-level one was chosen. The file used is shown per item below; pass "
            "--annotation-contains to pin a different one."
        )

    if args.items:
        try:
            wanted = {line.strip() for line in Path(args.items).read_text().splitlines()
                      if line.strip() and not line.startswith("#")}
        except OSError as exc:
            print(f"error: could not read {args.items}: {exc}", file=sys.stderr)
            return 1
        # Split files name recordings in whatever separator the corpus uses --
        # M4Singer's own metadata writes Alto-1#newboy#0000 where the pairing
        # builds Alto-1#newboy/0000 -- so ids are compared with separators
        # flattened rather than requiring the two to agree.
        def normalise(name: str) -> str:
            return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

        wanted_normal = {normalise(w) for w in wanted}
        before = len(pairing.items)
        pairing.items = [i for i in pairing.items
                         if i.item_id in wanted or i.audio.stem in wanted
                         or normalise(i.item_id) in wanted_normal]
        print(f"\nRestricted to the {len(pairing.items)} of {before} paired items listed "
              f"in {Path(args.items).name} ({len(wanted)} ids given).")
        if not pairing.items:
            print("error: no paired item matched the id list", file=sys.stderr)
            return 1

    if args.sample:
        from .dataset import group_of, stratified_sample

        before = len(pairing.items)
        pairing.items = stratified_sample(pairing.items, args.sample, seed=args.sample_seed)
        groups = len({group_of(i) for i in pairing.items})
        print(f"\nSampled {len(pairing.items)} of {before} recordings across {groups} "
              f"performance(s), seed {args.sample_seed}.")

    if args.limit:
        pairing.items = pairing.items[: args.limit]
        print(f"\nLimited to the first {len(pairing.items)} items.")

    if args.dry_run:
        return dry_run(pairing)

    if args.tune:
        return run_tuning(pairing, args, progress)

    estimates = None
    if args.estimates:
        from .dataset import pair_estimates

        estimates_dir = Path(args.estimates)
        if not estimates_dir.exists():
            print(f"error: no such directory: {estimates_dir}", file=sys.stderr)
            return 1
        estimates = pair_estimates(pairing.items, estimates_dir)
        missing = [i.item_id for i in pairing.items if i.item_id not in estimates]
        if missing:
            print(f"\n{len(missing)} item(s) had no transcription in {estimates_dir} "
                  f"and were dropped: {', '.join(missing[:5])}"
                  f"{' ...' if len(missing) > 5 else ''}")
            pairing.items = [i for i in pairing.items if i.item_id in estimates]
        if not pairing.items:
            print("error: no item had a matching transcription", file=sys.stderr)
            return 1
        print(f"Scoring {len(pairing.items)} transcriptions from {estimates_dir} "
              f"(this transcriber will not run)")

    config = TranscriptionConfig(
        f0_backend=args.f0_backend,
        crepe_model=args.crepe_model,
        device=args.device,
        seed=args.seed,
    )
    config.validate()

    result = run_benchmark(
        pairing, config,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        progress=progress,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
        check_alignment=not args.no_alignment_check,
        estimates=estimates,
        time_offset=args.time_offset,
        onset_shift=args.onset_shift,
        align=args.align,
        align_cache=Path(args.f0_cache),
    )

    print()
    print(format_result(result, worst=args.worst))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull result: {path}")

    if args.synthetic:
        print("\nNote: these are synthetic signals with generated ground truth. "
              "They verify the harness, not real-world accuracy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
