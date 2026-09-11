"""The tuning command: fit parameters, then say what the fit is worth.

Separated from tune.py, which holds the search itself. The library can be used
without a command line, and the command can grow options without the search
growing arguments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import TranscriptionConfig

def run_tuning(pairing, args, progress) -> int:
    """Fit the parameters, then report what the fit is worth on held-out data."""
    import dataclasses

    from .annotations import load_annotation
    from .dataset import load_item_metadata
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
            from .align import align_reference

            track, _ = cache.get(item, config)
            warp = align_reference(notes, track.f0_hz, track.voicing, track.times,
                                   voicing_threshold=config.voicing_threshold)
            if not warp.ok:
                refused.append((item.item_id, warp.reason))
                continue
            notes = warp.warp_notes(notes)
        references[item.item_id] = notes

    if args.align:
        print(f"Aligned {len(references)} references; {len(refused)} refused "
              "and left out of the split.")
        for item_id, reason in refused[:5]:
            print(f"  {item_id}: {reason}")
    if len(references) < 6:
        print(f"error: only {len(references)} recordings have readable references; "
              "a split needs more than that", file=sys.stderr)
        return 1

    item_ids = [i.item_id for i in pairing.items if i.item_id in references]
    groups, stratified_by = {}, ""
    if args.stratify_by:
        try:
            groups, id_column = load_item_metadata(args.stratify_by, item_ids)
            stratified_by = args.stratify_by
            print(f"Stratifying by {args.stratify_by} (joined on {id_column}, "
                  f"{len(groups)} of {len(item_ids)} matched)")
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    # Reuse an existing split rather than drawing a new one, so the report set
    # cannot quietly become a set the parameters have already seen.
    if args.tune_objective == "onset" and not args.align:
        print(
            "\nNote: tuning on onsets assumes the references share a clock with the\n"
            "  audio. If they were notated rather than aligned, pass --align, or use\n"
            "  --tune-objective sequence. A benchmark run says which you have.",
        )

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


