"""The tuning command: fit parameters, then say what the fit is worth.

Separated from tune.py, which holds the search itself. The library can be used
without a command line, and the command can grow options without the search
growing arguments.

Tuning borrows its metrics and its dataset loading from
``transcription-benchmarking``, which is why it is an optional extra rather
than part of the base install. Scoring lives there and only there: this
package transcribes, and does not ship a benchmark of its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import TranscriptionConfig


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="transcpybe-tune",
        description="Fit transcription parameters on one part of a corpus and "
                    "report what the fit is worth on a part it never saw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="Directory of audio to tune against")
    parser.add_argument("-a", "--annotations", default=None,
                        help="Directory of reference annotations "
                             "(default: alongside the audio)")
    parser.add_argument("--annotation-contains", default=None,
                        help="Only consider annotation files whose name contains this")
    parser.add_argument("--pairs", default=None, metavar="FILE",
                        help="An explicit audio-to-annotation CSV, for corpora where "
                             "no name matching can work")

    parser.add_argument("--trials", dest="tune_trials", type=int, default=60)
    parser.add_argument("--beta", dest="tune_beta", type=float, default=1.5,
                        help="F-beta weighting. Above 1 favours recall, which is "
                             "the safer direction when a reference may be "
                             "incomplete: a note the reference lacks is scored as "
                             "a false positive whether or not it was sung.")
    parser.add_argument("--objective", dest="tune_objective",
                        choices=("onset", "sequence"), default="onset",
                        help="What to maximise. 'onset' needs references sharing a "
                             "clock with the audio; 'sequence' ignores time and "
                             "works on references that do not.")
    parser.add_argument("--align", action="store_true",
                        help="Warp each reference onto its recording first, for "
                             "references notated at a nominal tempo. Fitted once, "
                             "against an f0 track no searched parameter can change.")
    parser.add_argument("--cv", type=int, default=None, metavar="K",
                        help="Cross-validate over K folds instead of using one "
                             "fit/select/report split. Every recording is scored by "
                             "a search that never saw it, so the estimate rests on "
                             "the whole corpus rather than the fifth of it a single "
                             "split reports on. Worth it below a few hundred "
                             "recordings, where a held-out fifth is too small to "
                             "show anything. It spends the corpus: afterwards no "
                             "part of it is unseen.")
    parser.add_argument("--split-file", default="split.json",
                        help="Where the fit/select/report split is kept. An existing "
                             "one is reused, so the report set cannot quietly become "
                             "a set the parameters have already seen.")
    parser.add_argument("--stratify-by", default=None, metavar="FILE:COLUMN",
                        help="Balance the split across a column of a metadata CSV")
    parser.add_argument("--tuned-config", default="tuned_config.json",
                        help="Where to write the winning configuration")
    parser.add_argument("--f0-cache", default=".f0-cache", metavar="DIR",
                        help="Cache the f0 and onset tracks here. Each recording is "
                             "analysed once and every trial replays from it, which is "
                             "what makes a search of this size affordable.")

    model = parser.add_argument_group("transcriber")
    model.add_argument("--f0-backend", choices=("crepe", "pyin"), default="crepe")
    model.add_argument("--crepe-model", choices=("full", "tiny"), default="full")
    model.add_argument("--device", default="auto")
    model.add_argument("--seed", type=int, default=0)

    parser.add_argument("--log-file", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    try:
        from transcription_benchmarking.dataset import pair_dataset, pair_from_file
    except ImportError:
        print("error: tuning needs the benchmark package for its metrics and "
              "dataset loading.\n  pip install 'transcpybe[tune]'", file=sys.stderr)
        return 1

    from .progress import ProgressReporter, setup_logging

    args = build_parser().parse_args(argv)
    logger = setup_logging(log_file=args.log_file)
    progress = ProgressReporter(enabled=not args.quiet, logger=logger)

    if args.pairs:
        try:
            pairing = pair_from_file(Path(args.pairs))
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        audio_dir = Path(args.audio)
        if not audio_dir.exists():
            print(f"error: no such directory: {audio_dir}", file=sys.stderr)
            return 1
        pairing = pair_dataset(
            audio_dir, Path(args.annotations) if args.annotations else None,
            annotation_contains=args.annotation_contains)
    print(pairing.summary())
    if not pairing.items:
        print("error: no audio could be paired with an annotation. Run\n"
              "  transcription-benchmark <dir> --dry-run\n"
              "to see what matched what.", file=sys.stderr)
        return 1

    return run_tuning(pairing, args, progress)



def describe_refusals(aligned: int, refused) -> str:
    """What the alignment could not place, and whether that is a problem.

    A bare count reads as incidental whether it is 5 or 150, so this reports a
    share. Refusals are also not a random sample -- they are the recordings
    whose references fit worst -- so past a quarter of the corpus it says what
    tuning on the remainder actually means.
    """
    attempted = aligned + len(refused)
    share = len(refused) / attempted if attempted else 0.0
    lines = [f"\nAligned {aligned} references; {len(refused)} refused "
             f"({share:.0%}) and left out of the split."]

    # Grouped: one reason repeated 140 times and seven scattered ones are
    # different problems, and the first five lines of either look alike.
    kinds = {}
    for item_id, reason in refused:
        if "covers only" in reason:
            kind = "the reference covers only part of the recording"
        elif "not a tempo" in reason:
            kind = "the warp is not a single tempo"
        else:
            kind = reason
        kinds.setdefault(kind, []).append(item_id)
    for kind, ids in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {len(ids):>4}  {kind}")
        lines.append(f"        e.g. {', '.join(ids[:3])}")

    if share > 0.25:
        lines.append(
            "\n  That is a large share to discard, and the refusals are not a\n"
            "  random sample: they are the recordings whose references fit worst.\n"
            "  Tuning on what is left fits the agreeable part of the corpus. Read\n"
            "  a refusal or two before trusting the result -- a vocal that stops\n"
            "  for an instrumental section covers only part of its recording\n"
            "  legitimately, and would be refused for being right.")
    return "\n".join(lines)


def _run_cross_validation(pairing, args, progress, config, cache, references) -> int:
    """Tune and report across folds, and say what the resulting number is not."""
    import dataclasses

    from .tune import DEFAULT_SPACE, cross_validate

    items = [i for i in pairing.items if i.item_id in references]
    print(f"\nCross-validating over {args.cv} folds of {len(items)} recordings, "
          f"{args.tune_trials} configurations each.")
    print("Each fold fits and selects on the others and is scored by a search that\n"
          "never saw it, so every recording contributes to the estimate.")

    total = args.cv * args.tune_trials
    with progress.stage("cross-validation", total=total) as stage:
        try:
            result = cross_validate(
                config, items, references, cache, folds=args.cv,
                space=DEFAULT_SPACE, trials=args.tune_trials, beta=args.tune_beta,
                seed=args.seed, objective=args.tune_objective,
                on_fold=lambda k, f: stage.advance(
                    args.tune_trials,
                    detail=f"fold {k + 1}: {f['baseline']:.3f} -> {f['tuned']:.3f}"))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    delta = result["delta"]
    print(f"\nOut-of-fold over all {delta['n']} recordings, {args.tune_objective} "
          f"F-beta at beta={args.tune_beta}:")
    print(f"  {'baseline':<10}{result['baseline']:>9.3f}")
    print(f"  {'tuned':<10}{result['tuned']:>9.3f}")
    print(f"  {'change':<10}{delta['mean']:>+9.3f}   95% CI "
          f"[{delta['low']:+.3f}, {delta['high']:+.3f}]"
          f"   better on {delta['better']}, worse on {delta['worse']}")

    kept = result["kept_default"]
    folds = len(result["folds"])
    if kept:
        print(f"\n  {kept} of {folds} folds found nothing that beat the defaults, and "
              "kept them.\n"
              "  That happens during fitting, on data the fold chose freely -- so it is\n"
              "  a stronger statement than the interval below, which can only say that\n"
              "  what one fold did find failed to carry.")
        if kept == folds:
            print("  No fold moved anything. There is nothing to read in the table "
                  "below.")

    crosses_zero = delta["low"] <= 0.0 <= delta["high"]
    if crosses_zero:
        print("\n  The interval includes zero, so this is consistent with the "
              "parameters\n  making no difference at all on this corpus.")
    else:
        print("\n  The interval excludes zero, so the parameters do carry a real "
              "difference\n  on this corpus. Whether it transfers is a question for "
              "another one.")

    if not result["stability"]:
        print("\nNo fold found a configuration that beat the default on its own "
              "selection\n  set, so there is nothing to show here. The defaults won "
              "every fold.")
        _print_cv_caveat(args)
        return _write_cv_result(args, config, result, delta)

    print("\nWhat each fold chose:")
    print(f"  {'parameter':<30}{'default':>9}{'median':>9}{'range':>20}"
          f"{'moved':>8}{'spread':>8}{'agree':>7}")
    for name, stats in sorted(result["stability"].items()):
        window = (f"{stats['low']:.4g}" if stats["moved"] < 2
                  else f"{stats['low']:.4g} to {stats['high']:.4g}")
        spread = "-" if stats["spread"] != stats["spread"] else f"{stats['spread']:.0%}"
        moved = f"{stats['moved']}/{stats['folds']}"
        print(f"  {name:<30}{stats['baseline']:>9.4g}{stats['median']:>9.4g}"
              f"{window:>20}{moved:>8}{spread:>8}{stats['agreement']:>7.0%}")
    print(
        "\n  Moved is how many folds changed the parameter at all; the rest kept\n"
        "  the default, and are the strongest evidence in the table. Spread is how\n"
        "  much of the searched range the choices cover, so near 100% is what\n"
        "  drawing at random looks like -- it needs two folds and is shown as - \n"
        "  below that, since one value has no spread rather than none. Agree is how\n"
        "  many of the folds that moved it went the same way.\n"
        "\n  Read moved first. A row moved by one fold of five is one opinion, and\n"
        "  in a single run it would have printed as a chosen value."
    )

    _print_cv_caveat(args)
    return _write_cv_result(args, config, result, delta)


def _print_cv_caveat(args) -> None:
    print(
        "\n  This answers whether there is anything in these parameters, not what\n"
        "  the tuned model scores. Every recording has been fitted on in some\n"
        "  fold, so this corpus is spent: it can no longer report an unseen number,\n"
        "  and the folds are several answers rather than one configuration to keep."
    )


def _write_cv_result(args, config, result, delta) -> int:
    import dataclasses

    Path(args.tuned_config).write_text(json.dumps({
        "cross_validation": {
            "folds": result["folds"], "delta": delta,
            "stability": result["stability"],
            "baseline": result["baseline"], "tuned": result["tuned"],
        },
        "beta": args.tune_beta, "trials": args.tune_trials,
        "objective": args.tune_objective,
        "base_config": dataclasses.asdict(config),
    }, indent=2, default=str))
    print(f"\nFull result: {args.tuned_config}")
    return 0


def run_tuning(pairing, args, progress) -> int:
    """Fit the parameters, then report what the fit is worth on held-out data."""
    import dataclasses

    if args.tune_objective == "onset" and not args.align:
        print(
            "\nNote: tuning on onsets assumes the references share a clock with the\n"
            "  audio. If they were notated rather than aligned, pass --align, or use\n"
            "  --objective sequence. A benchmark run says which you have.",
        )
    elif args.tune_objective == "onset" and args.align:
        print(
            "\nNote: a warp fixes global drift, not local timing. An onset counts\n"
            "  only within 50 ms, so if what the warp leaves behind is wider than\n"
            "  that, this objective is mostly noise and no setting will move it --\n"
            "  the search will report a flat surface and a null result. Against a\n"
            "  notated reference, --objective sequence asks what notes rather than\n"
            "  when, which a warp does not have to get right.",
        )

    from transcription_benchmarking.annotations import load_annotation
    from transcription_benchmarking.dataset import load_item_metadata
    from .tune import (DEFAULT_SPACE, StageCache, Split, make_split, random_search,
                       score_config)

    config = TranscriptionConfig(
        f0_backend=args.f0_backend, crepe_model=args.crepe_model,
        device=args.device, seed=args.seed)
    cache = StageCache(Path(args.f0_cache), config)

    references = {}
    refused = []
    for item in pairing.items:
        annotation = load_annotation(item.annotation)
        if not annotation.notes:
            continue
        notes = annotation.notes
        if args.align:
            # Warped once, here, against an f0 track no searched parameter can
            # change. Tuning against unaligned references would optimise the
            # mismatch: on this corpus that scores 0.06 however good the
            # transcription is.
            from transcription_benchmarking.align import align_reference

            track, _ = cache.get(item, config)
            warp = align_reference(notes, track.f0_hz, track.voicing, track.times,
                                   voicing_threshold=config.voicing_threshold)
            if not warp.ok:
                refused.append((item.item_id, warp.reason))
                continue
            notes = warp.warp_notes(notes)
        references[item.item_id] = notes

    if args.align:
        print(describe_refusals(len(references), refused))

    if len(references) < 6:
        print(f"error: only {len(references)} recordings have readable references; "
              "a split needs more than that", file=sys.stderr)
        return 1

    if args.cv:
        return _run_cross_validation(pairing, args, progress, config, cache,
                                     references)

    item_ids = [i.item_id for i in pairing.items if i.item_id in references]
    groups, stratified_by = {}, ""
    if args.stratify_by:
        try:
            groups, id_column = load_item_metadata(args.stratify_by, item_ids)
            stratified_by = args.stratify_by
            matched = len(groups) / len(item_ids) if item_ids else 0.0
            print(f"Stratifying by {args.stratify_by} (joined on {id_column}, "
                  f"{len(groups)} of {len(item_ids)} matched, {matched:.0%})")
            if matched < 0.8:
                print(f"  Only {matched:.0%} joined, so the split is mostly unstratified "
                      "and the\n  balance this was meant to guarantee is not there. "
                      "Check how the\n  metadata names its rows against how the corpus "
                      "names its files.")
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Reuse an existing split rather than drawing a new one, so the report set
    # cannot quietly become a set the parameters have already seen.
    split_path = Path(args.split_file)
    if split_path.exists():
        split = Split.from_dict(json.loads(split_path.read_text()))
        print(f"Reusing the split in {split_path}")
    else:
        split = make_split(item_ids, groups, seed=args.seed, stratified_by=stratified_by)
        split_path.write_text(json.dumps(split.to_dict(), indent=2))
        print(f"Wrote a new split to {split_path}")

    by_id = {i.item_id: i for i in pairing.items}
    fit = [by_id[k] for k in split.fit if k in by_id]
    select = [by_id[k] for k in split.select if k in by_id]
    report = [by_id[k] for k in split.report if k in by_id]
    print(f"  {len(fit)} to fit, {len(select)} to select, {len(report)} to report")

    print(f"\nSearching {args.tune_trials} configurations over "
          f"{len(DEFAULT_SPACE)} parameters, {args.tune_objective} F-beta at "
          f"beta={args.tune_beta}.")
    print("The f0 of each recording is computed once; trials replay from it.")
    with progress.stage("tuning", total=args.tune_trials) as stage:
        result = random_search(
            config, fit, select, references, cache, trials=args.tune_trials,
            beta=args.tune_beta, seed=args.seed, objective=args.tune_objective,
            on_trial=lambda n, s: stage.advance(1, detail=f"fit {s['fbeta']:.3f}"))

    best = result["best"]
    print(f"\n{'':<12}{'fit':>9}{'select':>9}")
    print(f"{'baseline':<12}{result['baseline_fit']['fbeta']:>9.3f}"
          f"{result['baseline_select']['fbeta']:>9.3f}")
    print(f"{'tuned':<12}{best['fit']['fbeta']:>9.3f}{best['select']['fbeta']:>9.3f}"
          f"   (trial {best['trial']} of {args.tune_trials})")

    if best["changes"]:
        print("\nParameters chosen:")
        for name, value in sorted(best["changes"].items()):
            current = getattr(config, name)
            mark = " " if abs(float(value) - float(current)) < 1e-9 else "*"
            print(f"  {mark} {name:<32} {current!r:>10} -> {value:.4g}")

    # Scored once, at the end, on recordings no trial and no selection touched.
    before = score_config(config, report, references, cache, beta=args.tune_beta,
                          objective=args.tune_objective)
    after = score_config(best["config"], report, references, cache, beta=args.tune_beta,
                         objective=args.tune_objective)
    print(f"\nHeld-out report set ({before['n']} recordings), scored once:")
    print(f"  {'':<10}{'F-beta':>9}{'precision':>11}{'recall':>9}")
    print(f"  {'baseline':<10}{before['fbeta']:>9.3f}{before['precision']:>11.3f}"
          f"{before['recall']:>9.3f}")
    print(f"  {'tuned':<10}{after['fbeta']:>9.3f}{after['precision']:>11.3f}"
          f"{after['recall']:>9.3f}")
    print(f"  {'change':<10}{after['fbeta'] - before['fbeta']:>+9.3f}"
          f"{after['precision'] - before['precision']:>+11.3f}"
          f"{after['recall'] - before['recall']:>+9.3f}")
    print(
        "\n  This is the number to quote. The fit and select figures above are\n"
        "  what the search optimised and what it chose on, and both are biased\n"
        "  upward by the search itself. If the held-out change is small, the\n"
        "  tuning found the fit set rather than the problem."
    )

    Path(args.tuned_config).write_text(json.dumps({
        "config": dataclasses.asdict(best["config"]),
        "beta": args.tune_beta, "trials": args.tune_trials,
        "split_file": str(split_path),
        "held_out": {"baseline": before, "tuned": after},
    }, indent=2, default=str))
    print(f"\nWinning configuration: {args.tuned_config}")
    return 0




if __name__ == "__main__":
    sys.exit(main())
