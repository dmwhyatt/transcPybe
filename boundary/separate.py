"""Run every mix through the lead-vocal separator the pipeline uses.

    python -m boundary.separate MIXES.jsonl OUT_DIR --backend docker [--image sep-bsroformer53:b24ab72]
    python -m boundary.separate MIXES.jsonl OUT_DIR --backend apptainer --sif sep-bsroformer53.sif
    python -m boundary.separate MIXES.jsonl OUT_DIR --backend dry       # no separator: the vocal as mixed

Writes ``OUT_DIR/stems/<mix_id>.wav`` and ``OUT_DIR/stems.jsonl`` (each mix's
line plus ``stem``). Resumable: a mix whose stem exists is not separated again,
and the work goes to the container in batches, so a job that runs out of time
loses one batch at most.

The separator is the one in the pipeline's image (BS-RoFormer 53-stem,
chunk 176400, its memory patches). The image is built on musix from
audio-symbolic-pipeline (see ~/sepbuild/Dockerfile there); on CSD3, which has
no Docker, the same image converted with ``apptainer build``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from . import items as manifest

HERE = Path(__file__).resolve().parent


def _stage(paths, stage: Path) -> None:
    """Hard-link (or copy) a batch into an empty folder the container can see whole."""
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for p in paths:
        dst = stage / Path(p).name
        try:
            os.link(p, dst)
        except OSError:
            shutil.copy2(p, dst)


def _container_cmd(backend: str, stage: Path, store: Path, args) -> list:
    inner = ["python", "/boundary/mss_bulk.py", "--input-folder", "/in", "--store-dir", "/out"]
    if backend == "docker":
        return ["docker", "run", "--rm", "--device", "nvidia.com/gpu=all", "-u", f"{os.getuid()}:{os.getgid()}",
                "-e", "HOME=/tmp", "-v", f"{stage}:/in:ro", "-v", f"{store}:/out", "-v", f"{HERE}:/boundary:ro",
                "--entrypoint", inner[0], args.image] + inner[1:]
    return ["apptainer", "exec", "--nv", "--cleanenv", "-B", f"{stage}:/in", "-B", f"{store}:/out",
            "-B", f"{HERE}:/boundary", str(args.sif)] + inner


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mixes", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--backend", choices=("docker", "apptainer", "dry"), required=True)
    ap.add_argument("--image", default="sep-bsroformer53:b24ab72")
    ap.add_argument("--sif", type=Path)
    ap.add_argument("--batch", type=int, default=400)
    args = ap.parse_args(argv)

    rows = manifest.read(args.mixes)
    stems = args.out_dir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    for r in rows:
        # The dry backend's stem is the dry vocal itself: nothing to separate, nothing to copy.
        r["stem"] = r["dry"] if args.backend == "dry" and "dry" in r else str(stems / f"{r['mix_id']}.wav")
    todo = [r for r in rows if not Path(r["stem"]).exists()]
    print(f"{len(rows)} mixes, {len(todo)} to separate ({args.backend})", flush=True)

    if args.backend == "dry":
        if any("dry" not in r for r in rows):
            raise SystemExit("the dry backend needs mixes rendered with --keep-dry or --accomp none")
    else:
        stage, store = (args.out_dir / "_stage").resolve(), (args.out_dir / "_store").resolve()   # mounts need absolute paths
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            _stage([r["mix"] for r in batch], stage)
            shutil.rmtree(store, ignore_errors=True)
            store.mkdir(parents=True)
            code = subprocess.call(_container_cmd(args.backend, stage, store, args))
            got = 0
            for r in batch:
                out = store / Path(r["mix"]).stem / "lead-vocal.wav"
                if out.exists() and out.stat().st_size > 0:
                    shutil.move(str(out), r["stem"])
                    got += 1
            print(f"batch {b // args.batch + 1}: exit {code}, {got}/{len(batch)} stems", flush=True)
            if got == 0:
                raise SystemExit("separator produced nothing for a whole batch; stopping")
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(store, ignore_errors=True)

    done = [r for r in rows if Path(r["stem"]).exists()]
    manifest.write(args.out_dir / "stems.jsonl", done)
    print(f"{len(done)}/{len(rows)} stems -> {stems}")


if __name__ == "__main__":
    main()
