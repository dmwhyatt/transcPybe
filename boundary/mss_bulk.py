"""Run inside the separator's container: BS-RoFormer over a whole folder in one model load.

    python /boundary/mss_bulk.py --input-folder /in --store-dir /out

The pipeline's own wrapper (separate_lead_vocal.py) separates one file per
call, which reloads a 53-stem model every time. This runs the same MSS
inference with the same patched settings over every file in a folder, writing
``<store-dir>/<name>/lead-vocal.wav``. It imports the wrapper's own config
patch, so chunk size and checkpointing cannot drift from the pipeline's.

Python 3.9, the container's: no newer syntax here.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

WRAPPER_DIR = "/app/pipeline/docker/wrappers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--store-dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, WRAPPER_DIR)
    from separate_lead_vocal import _patch_config  # noqa: E402  (the pipeline's own)

    mss_root = Path(os.environ.get("MSS_ROOT", "/app/mss"))
    config_src = Path(os.environ["MSS_CONFIG"])
    checkpoint = Path(os.environ["MSS_CHECKPOINT"])
    chunk_size = int(os.environ.get("ROFORMER_CHUNK_SIZE", "176400"))
    use_checkpoint = os.environ.get("ROFORMER_USE_TORCH_CHECKPOINT", "1") == "1"
    with tempfile.TemporaryDirectory(prefix="bsroformer_bulk_") as tmp:
        config_dst = Path(tmp) / config_src.name
        _patch_config(config_src, config_dst, chunk_size, use_checkpoint, os.environ.get("ROFORMER_NUM_OVERLAP"))
        env = dict(os.environ, PYTHONPATH=str(mss_root), PYTORCH_ALLOC_CONF="expandable_segments:True")
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        cmd = [sys.executable, str(mss_root / "inference.py"), "--model_type", "bs_roformer",
               "--config_path", str(config_dst), "--start_check_point", str(checkpoint),
               "--input_folder", args.input_folder, "--store_dir", args.store_dir, "--disable_detailed_pbar"]
        print(" ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=str(mss_root), env=env))


if __name__ == "__main__":
    main()
