"""Run every mix through the lead-vocal separator the pipeline uses.

    python -m boundary.separate MIXES.jsonl OUT_DIR --backend docker [--image sep-bsroformer53:b24ab72]
    python -m boundary.separate MIXES.jsonl OUT_DIR --backend apptainer --sif sep-bsroformer53.sif
    python -m boundary.separate MIXES.jsonl OUT_DIR --backend dry       # no separator: the vocal as mixed
    python -m boundary.separate MIXES.jsonl OUT_DIR --backend mix       # no separator: the whole mix (with
                                                                         # mix --var-db 20 35, a bleed stand-in)

Writes ``OUT_DIR/stems/<mix_id>.wav`` and ``OUT_DIR/stems.jsonl`` (each mix's
line plus ``stem``). Resumable: a mix whose stem exists is not separated again
(mix ids carry a digest of everything that made the mix, so an existing stem
is always this mix's), and the work goes to the container in batches, so a
job that runs out of time loses one batch at most.

A mix the separator fails on is listed in ``OUT_DIR/failed.jsonl`` and makes
the run exit non-zero once every batch has been tried, unless
``--allow-missing``: a dataset that silently loses its longest mixes, or
everything over one accompaniment, is worse than a failed job.

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
from .progress import Progress

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
    # --cleanenv keeps the host's Python and module variables out of the container, but it would also drop
    # the GPU Slurm allocated: pass that through, or the separator may land on another job's GPU.
    gpus = [arg for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES") if var in os.environ
            for arg in ("--env", f"{var}={os.environ[var]}")]
    return (["apptainer", "exec", "--nv", "--cleanenv"] + gpus
            + ["-B", f"{stage}:/in", "-B", f"{store}:/out", "-B", f"{HERE}:/boundary", str(args.sif)] + inner)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mixes", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--backend", choices=("docker", "apptainer", "dry", "mix"), required=True)
    ap.add_argument("--image", default="sep-bsroformer53:b24ab72")
    ap.add_argument("--sif", type=Path)
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--allow-missing", action="store_true", help="exit 0 even if some mixes failed to separate")
    args = ap.parse_args(argv)
    if args.backend == "apptainer" and (args.sif is None or not args.sif.exists()):
        raise SystemExit(f"--backend apptainer needs --sif pointing at the separator image (got {args.sif})")

    out_dir = args.out_dir.resolve()          # manifests carry absolute paths; mounts need them too
    rows = manifest.read(args.mixes)
    stems = out_dir / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    for r in rows:
        # The dry backend's stem is the dry vocal itself: nothing to separate, nothing to copy.
        r["stem"] = (r["dry"] if args.backend == "dry" and "dry" in r else r["mix"] if args.backend == "mix"
                     else str(stems / f"{r['mix_id']}.wav"))
    todo = [r for r in rows if not Path(r["stem"]).exists()]
    print(f"{len(rows)} mixes, {len(todo)} to separate ({args.backend})", flush=True)

    failed = []
    if args.backend in ("dry", "mix"):
        if args.backend == "dry" and any("dry" not in r for r in rows):
            raise SystemExit("the dry backend needs mixes rendered with --keep-dry or --accomp none")
        failed = todo
    else:
        stage, store = out_dir / "_stage", out_dir / "_store"
        prog = Progress(len(todo), "separate", every_s=0)
        for b in range(0, len(todo), args.batch):
            batch = todo[b:b + args.batch]
            _stage([r["mix"] for r in batch], stage)
            shutil.rmtree(store, ignore_errors=True)
            store.mkdir(parents=True)
            code = subprocess.call(_container_cmd(args.backend, stage, store, args))
            got = []
            for r in batch:
                out = store / Path(r["mix"]).stem / "lead-vocal.wav"
                if out.exists() and out.stat().st_size > 0:
                    shutil.move(str(out), r["stem"])
                    got.append(r["mix_id"])
            missing = [r for r in batch if r["mix_id"] not in set(got)] if len(got) < len(batch) else []
            failed += missing
            print(f"batch {b // args.batch + 1}: exit {code}, {len(got)}/{len(batch)} stems"
                  + (f"; failed: {' '.join(r['mix_id'] for r in missing[:10])}{' ...' if len(missing) > 10 else ''}"
                     if missing else ""), flush=True)
            prog.tick(len(batch))
            if not got:
                break                  # the separator itself is broken: no point in trying the rest
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(store, ignore_errors=True)

    done = [r for r in rows if Path(r["stem"]).exists()]
    manifest.write(out_dir / "stems.jsonl", done)
    manifest.write(out_dir / "failed.jsonl", [r for r in rows if not Path(r["stem"]).exists()])
    print(f"{len(done)}/{len(rows)} stems -> {stems}")
    if failed and not args.allow_missing:
        raise SystemExit(f"{len(rows) - len(done)} mixes have no stem (see {out_dir / 'failed.jsonl'}); "
                         "rerun to retry them, or pass --allow-missing to accept the loss")


if __name__ == "__main__":
    main()
