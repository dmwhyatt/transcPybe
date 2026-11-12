"""One part of a dataset, from labelled vocals to a feature shard, in one job on a node's own disk.

    python -m boundary.prepare VOCALS.jsonl OUT_DIR --accomp ROOT [--accomp-depth 2] [--var-db 20 35]
        [--variants 1] [--seed 0] [--part I/N] --scratch DIR [--workers 8] [--device auto]

For part I of N, all inside ``--scratch``:
1. mix the vocals whose position modulo N is I (``mix --part``: each keeps its
   position, so its mix is the one a whole run would make);
2. take each mix as its stem, with no separator (``separate --backend mix``).
   On 21 October, mixing the accompaniment 20-35 dB under the voice recovered
   77-89% of what separation gained, so the pre-registered rule dropped
   separation from the bulk data (diagnostics-plan.md on musix, rule (a));
3. compute features with ``--workers`` processes sharing the GPU;
4. pack them into one shard (``boundary.pack``), then copy it to
   ``OUT_DIR/shards/part-IIII-of-NNNN.npz`` with its manifest ``.jsonl``.

Why it is laid out like this on CSD3:
- The per-recording files stay on the node's local disk, which CSD3 deletes
  when the job ends. Only shards reach hpc-work, the Lustre filesystem meant
  for large files (https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html).
- Mixing, features and packing run in one GPU job. On the SL3 service level a
  CPU-only job could wait hours behind thousands of others (one waited over
  3 h on 21 October), and a separate job per stage would wait once per stage.
- A part is the unit of an array job, which the batch guide prefers to a loop
  of submissions (https://docs.hpc.cam.ac.uk/hpc/user-guide/batch.html).
  SL3 jobs stop at 12 hours (https://docs.hpc.cam.ac.uk/hpc/user-guide/policies.html),
  so choose N to keep each part well inside that.

Resumable: a finished part leaves ``part-....done`` holding its arguments, and a
rerun with the same arguments skips it. An unfinished part starts again from
scratch. The ``mix`` and ``stem`` paths in a shard's rows name files that went
with the scratch disk; mixing is deterministic, so ``mix`` rebuilds any of them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from . import features, items as manifest, mix, pack, separate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocals", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--accomp", required=True, help="root of multitrack songs, or 'synthetic' for tests")
    ap.add_argument("--accomp-depth", type=int, default=2, help="2 for MUSDB18-HQ, 1 for MoisesDB")
    ap.add_argument("--var-db", type=float, nargs=2, default=[20.0, 35.0], metavar=("LO", "HI"),
                    help="vocal over accompaniment, dB; 20 35 is the no-separator stand-in")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", metavar="I/N")
    ap.add_argument("--scratch", type=Path, required=True, help="node-local working folder; removed afterwards")
    ap.add_argument("--workers", type=int, default=8, help="feature processes sharing the GPU")
    ap.add_argument("--mix-workers", type=int, default=0, help="mixing processes (default: every core)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--crepe-model", default="full")
    ap.add_argument("--f0-backend", default=None, help="override, e.g. pyin for tests")
    ap.add_argument("--keep-scratch", action="store_true", help="leave the working folder, for debugging")
    args = ap.parse_args(argv)

    part, n_parts = (int(x) for x in args.part.split("/"))
    name = f"part-{part:04d}-of-{n_parts:04d}"
    shards = args.out_dir.resolve() / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    done = shards / f"{name}.done"
    key = json.dumps(dict(vocals=str(args.vocals.resolve()), accomp=args.accomp, accomp_depth=args.accomp_depth,
                          var_db=args.var_db, variants=args.variants, seed=args.seed, crepe_model=args.crepe_model,
                          f0_backend=args.f0_backend), sort_keys=True)
    if done.exists() and done.read_text() == key:
        print(f"{name}: already done ({shards / name}.npz)")
        return

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    work = (args.scratch / name).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        mix.main([str(args.vocals), str(work / "mix"), "--accomp", args.accomp, "--accomp-depth", str(args.accomp_depth),
                  "--variants", str(args.variants), "--seed", str(args.seed), "--var-db", *map(str, args.var_db),
                  "--workers", str(args.mix_workers or cores), "--part", args.part])
        separate.main([str(work / "mix" / "mixes.jsonl"), str(work / "mix"), "--backend", "mix"])
        features.main([str(work / "mix" / "stems.jsonl"), str(work / "features"), "--device", args.device,
                       "--crepe-model", args.crepe_model, "--workers", str(args.workers)]
                      + (["--f0-backend", args.f0_backend] if args.f0_backend else []))
        local = work / f"{name}.npz"
        rows = pack.pack(manifest.read(work / "features" / "features.jsonl"), local)
        final = shards / f"{name}.npz"
        shutil.copyfile(local, shards / f"{name}.part.npz")       # one large write to the shared filesystem
        os.replace(shards / f"{name}.part.npz", final)
        manifest.write(shards / f"{name}.jsonl", [dict(r, shard=str(final)) for r in rows])
        done.write_text(key)
        print(f"{name}: {len(rows)} recordings -> {final}")
    finally:
        if not args.keep_scratch:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
