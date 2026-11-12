"""Pack many feature files into one shard, for a shared filesystem that is slow with small files.

    python -m boundary.pack FEATURES.jsonl SHARD.npz

Writes ``SHARD.npz``: every recording's ``mel``, ``aux`` and ``y`` end to end
(float16, compressed), with ``offsets`` marking where each starts. Also writes
``SHARD.jsonl``: the feature manifest's rows, each with ``shard`` (the .npz)
and ``index`` (its place in it) in place of ``features``. Training reads either
kind of row (train.load_split).

CSD3 keeps active job data on its Lustre filesystems, which suit large files,
and gives each node a local disk that is wiped when the job ends
(https://docs.hpc.cam.ac.uk/hpc/user-guide/io_management.html). So the
per-recording files are made on the node's disk, and only shards go to
hpc-work. That also keeps well inside the quota of about a million files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import items as manifest

FIELDS = ("mel", "aux", "y")


def pack(rows: list, shard: Path) -> list:
    """Write the shard and return its rows (``shard`` and ``index`` in place of ``features``)."""
    parts = {k: [] for k in FIELDS}
    offsets = [0]
    for r in rows:
        with np.load(r["features"]) as z:
            for k in FIELDS:
                parts[k].append(z[k].astype(np.float16))
        offsets.append(offsets[-1] + len(parts["y"][-1]))
    shard = Path(shard).resolve()
    tmp = shard.with_name(shard.stem + ".part.npz")
    np.savez_compressed(tmp, offsets=np.array(offsets, dtype=np.int64),
                        **{k: np.concatenate(v) if v else np.zeros((0, 1), np.float16) for k, v in parts.items()})
    tmp.replace(shard)
    out = []
    for i, r in enumerate(rows):
        row = {k: v for k, v in r.items() if k != "features"}
        out.append(dict(row, shard=str(shard), index=i))
    return out


def read(shard, index: int, data: dict) -> dict:
    """One recording's arrays from a shard; ``data`` caches each shard's arrays once loaded."""
    if shard not in data:
        with np.load(shard) as z:
            data[shard] = {k: z[k] for k in ("offsets",) + FIELDS}
    z = data[shard]
    a, b = z["offsets"][index], z["offsets"][index + 1]
    return {k: z[k][a:b] for k in FIELDS}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", type=Path)
    ap.add_argument("shard", type=Path)
    args = ap.parse_args(argv)
    rows = pack(manifest.read(args.features), args.shard)
    manifest.write(args.shard.with_suffix(".jsonl"), rows)
    print(f"{len(rows)} recordings -> {args.shard}")


if __name__ == "__main__":
    main()
