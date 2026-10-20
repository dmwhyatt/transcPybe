"""Train the boundary network on feature files.

    python -m boundary.train FEATURES.jsonl [MORE.jsonl ...] --out RUN_DIR [--seed 0] [--steps 30000]

Training draws random 4 s crops; validation runs whole recordings and picks
onsets as inference would. Two numbers decide the best checkpoint, and both are
logged at every evaluation:

- onset F at 50 ms, over every validation note;
- recall of re-sung notes: notes that follow a note on the same pitch with less
  than 50 ms between them. These are what transcPybe's decoder merges, and the
  reason this network exists, so they are scored on their own rather than left
  to be averaged away among the easy onsets.

The checkpoint kept is the one with the best harmonic mean of the two: recall
alone is won by a network that fires everywhere, and the harmonic mean will not
let a high recall make up for a collapsed F.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import HOP_S, items as manifest
from .infer import pick_peaks

TOL = 0.05


def load_split(rows, split):
    out = []
    for r in rows:
        if r.get("split") != split:
            continue
        z = np.load(r["features"])
        out.append(dict(id=r["mix_id"], notes=r["notes"], mel=z["mel"].astype(np.float32),
                        aux=z["aux"].astype(np.float32), y=z["y"].astype(np.float32)))
    return out


def batch(data, rng, size, crop):
    lens = np.array([len(d["y"]) for d in data], dtype=float)
    idx = rng.choice(len(data), size=size, p=lens / lens.sum())
    n_mel, n_aux = data[0]["mel"].shape[1], data[0]["aux"].shape[1]
    mel = np.zeros((size, crop, n_mel), np.float32)
    aux = np.zeros((size, crop, n_aux), np.float32)
    y = np.zeros((size, crop, 3), np.float32)
    mask = np.zeros((size, crop), np.float32)
    for b, i in enumerate(idx):
        d = data[i]
        t = len(d["y"])
        a = int(rng.integers(0, max(1, t - crop + 1)))
        n = min(crop, t - a)
        mel[b, :n], aux[b, :n], y[b, :n], mask[b, :n] = d["mel"][a:a + n], d["aux"][a:a + n], d["y"][a:a + n], 1
    return mel, aux, y, mask


def resung(notes):
    """Onsets of notes sung again on the previous note's pitch with under 50 ms between."""
    return [n[0] for p, n in zip(notes, notes[1:]) if abs(n[2] - p[2]) < 0.5 and n[0] - p[1] < 0.05]


def evaluate(model, data, device, threshold=0.5):
    import mir_eval
    from .infer import posteriors

    model.eval()
    fs, hit, tot = [], 0, 0
    for d in data:
        p = posteriors(model, d["mel"], d["aux"], device)
        est = pick_peaks(p[:, 0], threshold) * HOP_S
        ref = np.array([n[0] for n in d["notes"]])
        fs.append(mir_eval.onset.f_measure(ref, est, window=TOL)[0] if len(ref) else 1.0)
        for t in resung(d["notes"]):
            tot += 1
            hit += int(np.any(np.abs(est - t) <= TOL))
    model.train()
    return dict(onset_f=float(np.mean(fs)), resung_recall=hit / tot if tot else float("nan"), n_resung=tot)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop-s", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--pos-weight", type=float, default=8.0, help="for the onset and offset targets, which are rare")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--no-aux", action="store_true", help="ablation: log-mel only, no f0 features")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    import torch
    from .model import BoundaryNet, NetConfig, save

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rows = [r for f in args.features for r in manifest.read(f)]
    train, valid = load_split(rows, "train"), load_split(rows, "valid")
    if not train or not valid:
        raise SystemExit(f"need both splits: {len(train)} train, {len(valid)} valid")
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(args), features=[str(f) for f in args.features], device=device,
                   n_train=len(train), n_valid=len(valid)), open(args.out / "args.json", "w"), indent=1, default=str)

    model = BoundaryNet(NetConfig(n_mels=train[0]["mel"].shape[1], n_aux=train[0]["aux"].shape[1],
                                  use_aux=not args.no_aux)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    pw = torch.tensor([args.pos_weight, args.pos_weight, 1.0], device=device)
    head_w = torch.tensor([1.0, 0.5, 0.5], device=device)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crop = int(round(args.crop_s / HOP_S))
    log = open(args.out / "log.jsonl", "a")
    best, t0, run_loss = -1.0, time.time(), 0.0
    for step in range(1, args.steps + 1):
        mel, aux, y, mask = (torch.from_numpy(x).to(device) for x in batch(train, rng, args.batch, crop))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(mel, aux)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw, reduction="none")
        loss = ((loss * head_w).sum(-1) * mask).sum() / mask.sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, valid, device)
            f, r = m["onset_f"], m["resung_recall"]
            score = f if math.isnan(r) else (2 * f * r / (f + r) if f + r > 0 else 0.0)
            rec = dict(step=step, loss=run_loss / args.eval_every, lr=sched.get_last_lr()[0],
                       elapsed_s=round(time.time() - t0), score=score, **m)
            run_loss = 0.0
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            save(args.out / "last.pt", model, step=step, metrics=m)
            if score > best:
                best = score
                save(args.out / "best.pt", model, step=step, metrics=m)
    print(f"best score {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
