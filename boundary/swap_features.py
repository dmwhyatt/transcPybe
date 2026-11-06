"""Point a labelled manifest at shards of the same mixes made with another pitch tracker.

    python -m boundary.swap_features LABELLED.jsonl OUT.jsonl --shards DIR --expect-f0-backend pesto

Round 2 trained on ``m4gts-musdb/all-lm.jsonl``: round 1's shard rows, relabelled with each note's landmarks
(``relabel``). Rerunning ``prepare`` with round 1's vocals, accompaniment, seed and parts but another
``--f0-backend`` makes the same mixes, because mixing is seeded by each recording's place and a ``mix_id`` is a
digest of the vocal, its notes, the seed and the mix setup. Only the pitch features differ.

This writes LABELLED's rows, each with ``shard`` and ``index`` taken from the row of the same ``mix_id`` in DIR's
shard manifests (``part-*.jsonl``); everything else, the labels included, stays LABELLED's. It stops unless:
- every part in DIR was made with ``--expect-f0-backend`` and the same vocals, accompaniment, seed and mixing as
  LABELLED's own shards (their ``.done`` records, when those are found beside them);
- every LABELLED row has a twin in DIR, with the same padding, setup and variant.

Training with ``--label-voicing LABELLED.jsonl`` then checks each pair's targets and spectrograms as it loads them
(TRACKER-RETRAIN-PLAN.md). Runs in seconds on a login node: it reads manifests, not shards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import items as manifest

MIX_KEYS = ("vocals", "accomp", "accomp_depth", "var_db", "variants", "seed")   # prepare's .done fields that set a mix
SAME = ("pad_before", "setup", "variant")                                    # row fields that must match their twin's


def done_keys(shard_dir: Path) -> dict:
    """Each part's ``prepare`` arguments, from its ``.done`` record, by part name."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(shard_dir).glob("part-*.done"))}


def swap(labelled: list, new: list) -> list:
    by_id = {}
    for r in new:
        if r["mix_id"] in by_id:
            raise SystemExit(f"{r['mix_id']} appears twice in the new shards")
        by_id[r["mix_id"]] = r
    out, missing = [], []
    for r in labelled:
        twin = by_id.get(r["mix_id"])
        if twin is None:
            missing.append(r["mix_id"])
            continue
        differ = [k for k in SAME if r.get(k) != twin.get(k)]
        if differ:
            raise SystemExit(f"{r['mix_id']}: {', '.join(differ)} differ between the labelled row and the new shard's")
        out.append(dict(r, shard=twin["shard"], index=twin["index"], features_swapped_from=r.get("shard")))
    if missing:
        raise SystemExit(f"{len(missing)} labelled rows have no twin in the new shards, e.g. {missing[0]}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labelled", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--shards", type=Path, required=True, help="the new run's shards/ folder")
    ap.add_argument("--expect-f0-backend", required=True, help="the tracker the new shards must have been made with")
    args = ap.parse_args(argv)

    labelled = manifest.read(args.labelled)
    new_keys = done_keys(args.shards)
    parts = sorted(args.shards.glob("part-*.jsonl"))
    if not parts or len(parts) != len(new_keys):
        raise SystemExit(f"{args.shards}: {len(parts)} shard manifests and {len(new_keys)} .done records; "
                         "is every part finished?")
    wrong = [n for n, k in new_keys.items() if k.get("f0_backend") != args.expect_f0_backend]
    if wrong:
        raise SystemExit(f"{len(wrong)} parts were not made with {args.expect_f0_backend}, e.g. {wrong[0]}: "
                         f"{new_keys[wrong[0]].get('f0_backend')}")
    old_dirs = {str(Path(r["shard"]).parent) for r in labelled if "shard" in r}
    old_keys = {n: k for d in old_dirs for n, k in done_keys(Path(d)).items()}
    if old_keys:
        if set(old_keys) != set(new_keys):
            raise SystemExit(f"the labelled shards have parts {sorted(old_keys)[:2]}..., the new ones "
                             f"{sorted(new_keys)[:2]}...: not the same split into parts")
        for n, k in new_keys.items():
            differ = [f for f in MIX_KEYS if k.get(f) != old_keys[n].get(f)]
            if differ:
                raise SystemExit(f"{n}: {', '.join(differ)} differ from the labelled run's, so the mixes would too")
    else:
        print(f"note: no .done records beside the labelled shards ({', '.join(sorted(old_dirs)) or 'none'}); "
              "the mixes are checked by the twins' padding here and their spectrograms in training")

    rows = swap(labelled, [r for p in parts for r in manifest.read(p)])
    n = manifest.write(args.out, rows)
    print(f"{n} rows -> {args.out}: labels from {args.labelled}, {args.expect_f0_backend} features from "
          f"{len(parts)} parts in {args.shards}" + (f"; mixing arguments match the labelled run's {len(old_keys)} parts"
                                                     if old_keys else ""))


if __name__ == "__main__":
    main()
