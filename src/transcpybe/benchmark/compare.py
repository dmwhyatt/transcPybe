"""Putting two runs, or two annotators, side by side.

Both answer the same shape of question -- is this better than that, and by
enough to believe -- so both live here rather than in the argument parser.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

def compare_results(before: Path, after: Path, by_metadata: Optional[str] = None) -> int:
    """Report a paired comparison of two saved benchmark results."""
    from .metrics import PRF, ItemScore, compare_runs

    def load(path: Path) -> Dict[str, ItemScore]:
        payload = json.loads(path.read_text())
        scores = {}
        for item in payload["items"]:
            scores[item["item_id"]] = ItemScore(
                item_id=item["item_id"],
                n_reference=item["n_reference"], n_estimated=item["n_estimated"],
                con=PRF(**item["con"]), conp=PRF(**item["conp"]),
                conpoff=PRF(**item["conpoff"]),
            )
        return scores

    try:
        items_a, items_b = load(before), load(after)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: could not read both result files: {exc}", file=sys.stderr)
        return 2

    print(f"\nPaired comparison: {before.name} -> {after.name}")
    shared = set(items_a) & set(items_b)
    if not shared:
        print("error: the two runs share no recordings", file=sys.stderr)
        return 2
    print(f"  {len(shared)} recordings in common\n")

    any_significant = False
    for metric, label in (("con_f", "COn      (onset)"),
                          ("conp_f", "COnP     (onset+pitch)"),
                          ("conpoff_f", "COnPOff  (+offset)")):
        result = compare_runs(items_a, items_b, metric)
        if result is None:
            continue
        any_significant |= result.significant
        mark = "*" if result.significant else " "
        print(f"  {mark} {label:<24} {result.mean_a:.3f} -> {result.mean_b:.3f}  "
              f"{result.delta:+.3f}  95% CI [{result.low:+.3f}, {result.high:+.3f}]  "
              f"better on {result.n_better}, worse on {result.n_worse}")

    if by_metadata:
        breakdown_by_metadata(items_a, items_b, by_metadata, before.stem, after.stem)

    print(
        "\n  Resampled over the per-recording differences, so the difficulty of each\n"
        "  recording -- which both runs share -- is removed rather than counted twice."
    )
    if any_significant:
        print("  * marks an interval that excludes zero.")
    else:
        print("  No interval excludes zero: nothing here is distinguishable from noise.")
    return 0


def breakdown_by_metadata(items_a, items_b, spec: str, label_a: str, label_b: str) -> None:
    """Report both systems' scores per group of a metadata column."""
    import statistics

    from .dataset import load_item_metadata

    shared = sorted(set(items_a) & set(items_b))
    try:
        groups, id_column = load_item_metadata(spec, shared)
    except (OSError, ValueError) as exc:
        print(f"\n  could not break down by {spec}: {exc}", file=sys.stderr)
        return

    matched = [k for k in shared if k in groups]
    if not matched:
        print(f"\n  no item matched {spec}", file=sys.stderr)
        return

    by_group: Dict[str, List[str]] = {}
    for key in matched:
        by_group.setdefault(groups[key] or "(blank)", []).append(key)

    field = spec.rpartition(":")[2]
    print(f"\n  By {field} (joined on {id_column}, {len(matched)} of {len(shared)} "
          f"items matched):")
    print(f"    {field:<16} {'n':>3} {label_a[:12]:>12} {label_b[:12]:>12} {'delta':>8}")
    rows = []
    for group, keys in by_group.items():
        mean_a = statistics.fmean(items_a[k].conp.f for k in keys)
        mean_b = statistics.fmean(items_b[k].conp.f for k in keys)
        rows.append((group, len(keys), mean_a, mean_b, mean_b - mean_a))
    for group, n, mean_a, mean_b, delta in sorted(rows, key=lambda r: -r[4]):
        print(f"    {group[:16]:<16} {n:>3} {mean_a:>12.3f} {mean_b:>12.3f} {delta:>+8.3f}")
    print(
        "\n    COnP F per group. A gap that varies with this column is about the\n"
        "    material; one that is flat across it is about the models."
    )



def annotator_agreement(annotation_dir: Path, tag_a: str, tag_b: str) -> int:
    """Score annotator B against annotator A, using the benchmark's own metrics.

    Running the same machinery over two human annotations does two jobs at once:
    it establishes the ceiling for the dataset, and it checks the metrics
    themselves against a figure someone else has published.
    """
    from .annotations import load_annotation
    from .dataset import annotation_tags, pair_annotators
    from .metrics import aggregate_scores, score_item

    pairs = pair_annotators(annotation_dir, tag_a, tag_b)
    if not pairs:
        # Guessing tag names from a dataset's documentation is how this fails,
        # so show the ones actually on disk rather than only that it failed.
        print(f"error: no recording had both a '{tag_a}' and a '{tag_b}' annotation "
              f"in {annotation_dir}", file=sys.stderr)
        tags = annotation_tags(annotation_dir)
        if tags:
            print("  tags present in the annotation filenames:", file=sys.stderr)
            for tag, count in tags:
                print(f"    {tag:<20} {count} file(s)", file=sys.stderr)
        return 2

    scores = []
    for key, path_a, path_b in pairs:
        first, second = load_annotation(path_a), load_annotation(path_b)
        if first.notes and second.notes:
            scores.append(score_item(key.strip("_-. "), first.notes, second.notes))

    if not scores:
        print("error: no pair had notes on both sides", file=sys.stderr)
        return 2

    aggregate = aggregate_scores(scores)
    print(f"\nAnnotator agreement: '{tag_b}' scored against '{tag_a}' over "
          f"{len(scores)} recordings")
    print(f"  {'metric':<28}{'F':>7}   95% CI on F")
    for key, label in (("con", "COn      (onset)"),
                       ("conp", "COnP     (onset+pitch)"),
                       ("conpoff", "COnPOff  (+offset)")):
        interval = aggregate.macro[f"{key}_f"]
        print(f"  {label:<28}{interval.mean:>7.3f}   "
              f"[{interval.low:.3f}-{interval.high:.3f}]")
    print(
        "\nThis is the ceiling. A transcriber scoring near these numbers is\n"
        "disagreeing with the reference about as much as its two authors\n"
        "disagree with each other, and the remaining gap is mostly a question\n"
        "about what counts as a note rather than an error the model can fix."
    )
    return 0


