"""Train the boundary network on feature files.

    python -m boundary.train FEATURES.jsonl [MORE.jsonl ...] --out RUN_DIR [--seed 0] [--steps 30000]
    python -m boundary.train FEATURES.jsonl --eval-only CHECKPOINT [--extra-valid NAME=FEATURES.jsonl ...]

Training draws random 4 s crops. A recording shorter than that is joined to
further random recordings until the crop is filled, rather than padded with
zeros: padding would reach BatchNorm's statistics and start the backward GRU
in frames that never occur at inference. Validation runs whole recordings and
picks onsets as inference would. Two numbers decide the best checkpoint, and
both are logged at every evaluation:

- onset F at 50 ms, over every validation note;
- recall of re-sung notes: notes that follow a note on the same pitch with less
  than 50 ms between them. These are what transcPybe's decoder merges, and the
  reason this network exists, so they are scored on their own rather than left
  to be averaged away among the easy onsets.

The checkpoint kept is the one with the best harmonic mean of the two: recall
alone is won by a network that fires everywhere, and the harmonic mean will not
let a high recall make up for a collapsed F.

Also logged at every evaluation, never used to choose a checkpoint:

- precision and recall separately, and estimated onsets per reference note, so
  a network that over-fires is visible as such;
- onset timing: for each reference onset, the nearest estimate within 150 ms,
  as estimate minus reference. Logged as the median and quartiles in ms, and the
  share more than 25 ms early or late.

Timing against the validation labels only checks that the network learned
their timing. They follow the same convention as the training labels, so a
network that starts notes at the consonant looks perfect against them. A
convention mismatch shows up only against labels in another convention. Each
``--extra-valid NAME=FEATURES.jsonl`` adds such a set; the network is evaluated
on its ``valid`` split too, and the results are logged under ``extra.NAME``.
An example is M4Singer's validation singers with vowel onsets
(``sources --onsets vowel``) beside the consonant-onset training set: the
dry pilot started notes 40-70 ms before the annotators of Molina and vocadito.

Resumable: ``last.pt`` holds the optimiser, schedule and random states as well
as the weights, and a rerun with the same arguments carries on from it with
the best score so far. A rerun with different arguments into the same
``--out`` is refused rather than mixed into the old run.
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
TIMING_WINDOW = 0.15      # a reference onset counts for timing if an estimate is within 150 ms
TIMING_MARGIN = 0.025     # early or late: more than 25 ms out
EVAL_BATCH = 32
RESUME_KEYS = ("features", "seed", "steps", "batch", "crop_s", "lr", "weight_decay", "warmup", "pos_weight", "no_aux")


def load_split(rows, split):
    out = []
    for r in manifest.split(rows, split):
        with np.load(r["features"]) as z:
            out.append(dict(id=r["mix_id"], notes=r["notes"], mel=z["mel"].astype(np.float32),
                            aux=z["aux"].astype(np.float32), y=z["y"].astype(np.float32)))
    return out


class Crops:
    """Random fixed-length crops, a recording drawn in proportion to its length."""

    def __init__(self, data, crop: int):
        self.data, self.crop = data, crop
        lens = np.array([len(d["y"]) for d in data], dtype=float)
        self.p = lens / lens.sum()

    def _one(self, rng, i):
        parts, t = [i], len(self.data[i]["y"])
        while t < self.crop:                 # too short: carry on into other recordings, never into zeros
            j = int(rng.integers(len(self.data)))
            parts.append(j)
            t += len(self.data[j]["y"])
        a = int(rng.integers(0, t - self.crop + 1))
        if len(parts) == 1:
            d = self.data[i]
            return d["mel"][a:a + self.crop], d["aux"][a:a + self.crop], d["y"][a:a + self.crop]
        return tuple(np.concatenate([self.data[j][k] for j in parts])[a:a + self.crop] for k in ("mel", "aux", "y"))

    def batch(self, rng, size):
        mel, aux, y = zip(*(self._one(rng, int(i)) for i in rng.choice(len(self.data), size=size, p=self.p)))
        return np.stack(mel), np.stack(aux), np.stack(y)


def resung(notes):
    """Onsets of notes sung again on the previous note's pitch with under 50 ms between."""
    return [n[0] for p, n in zip(notes, notes[1:]) if abs(n[2] - p[2]) < 0.5 and n[0] - p[1] < 0.05]


def onset_timing(ref: np.ndarray, est: np.ndarray, window: float = TIMING_WINDOW) -> list:
    """Estimate minus reference, in seconds, for each reference onset whose nearest estimate is within ``window``."""
    if not len(est):
        return []
    out = []
    for r in ref:
        d = float(est[np.argmin(np.abs(est - r))] - r)
        if abs(d) <= window:
            out.append(d)
    return out


def timing_summary(offsets) -> dict:
    if not len(offsets):
        return dict(timing_median_ms=float("nan"), timing_p25_ms=float("nan"), timing_p75_ms=float("nan"),
                    early_share=float("nan"), late_share=float("nan"), n_timed=0)
    ms = np.asarray(offsets) * 1000.0
    p25, med, p75 = np.percentile(ms, [25, 50, 75])
    return dict(timing_median_ms=round(float(med), 1), timing_p25_ms=round(float(p25), 1),
                timing_p75_ms=round(float(p75), 1), early_share=round(float(np.mean(ms < -TIMING_MARGIN * 1000)), 4),
                late_share=round(float(np.mean(ms > TIMING_MARGIN * 1000)), 4), n_timed=int(len(ms)))


def evaluate(model, data, device, threshold=0.5):
    """Whole recordings, in length-sorted padded batches; the model is told each length, so
    every recording's posterior is what it would be on its own."""
    import mir_eval
    import torch

    model.eval()
    fs, ps, rs, hit, tot, n_est, n_ref, offsets = [], [], [], 0, 0, 0, 0, []
    order = sorted(range(len(data)), key=lambda k: len(data[k]["y"]))
    with torch.no_grad():
        for b in range(0, len(order), EVAL_BATCH):
            chunk = [data[k] for k in order[b:b + EVAL_BATCH]]
            lens = [len(d["mel"]) for d in chunk]
            t = max(lens)
            mel = np.zeros((len(chunk), t, chunk[0]["mel"].shape[1]), np.float32)
            aux = np.zeros((len(chunk), t, chunk[0]["aux"].shape[1]), np.float32)
            for i, d in enumerate(chunk):
                mel[i, :lens[i]], aux[i, :lens[i]] = d["mel"], d["aux"]
            p = torch.sigmoid(model(torch.from_numpy(mel).to(device), torch.from_numpy(aux).to(device),
                                    torch.tensor(lens)))[..., 0].float().cpu().numpy()
            for i, d in enumerate(chunk):
                est = pick_peaks(p[i, :lens[i]], threshold) * HOP_S
                ref = np.array([n[0] for n in d["notes"]])
                f, pr, rc = mir_eval.onset.f_measure(ref, est, window=TOL) if len(ref) else (1.0, 1.0, 1.0)
                fs.append(f)
                ps.append(pr)
                rs.append(rc)
                n_est, n_ref = n_est + len(est), n_ref + len(ref)
                offsets += onset_timing(ref, est)
                for onset in resung(d["notes"]):
                    tot += 1
                    hit += int(np.any(np.abs(est - onset) <= TOL))
    model.train()
    return dict(onset_f=float(np.mean(fs)), resung_recall=hit / tot if tot else float("nan"), n_resung=tot,
                onset_p=float(np.mean(ps)), onset_r=float(np.mean(rs)),
                onsets_per_note=round(n_est / n_ref, 3) if n_ref else float("nan"), **timing_summary(offsets))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--eval-only", type=Path, metavar="CHECKPOINT",
                    help="evaluate an existing checkpoint on the valid splits, print the results and stop")
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
    ap.add_argument("--extra-valid", action="append", default=[], metavar="NAME=FEATURES.jsonl",
                    help="another set to evaluate (its valid split) and log, never used to choose a checkpoint")
    args = ap.parse_args(argv)

    import torch
    from .model import BoundaryNet, NetConfig, save

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    features = [str(f.resolve()) for f in args.features]
    run_args = dict(vars(args), features=features, out=str(args.out.resolve()) if args.out else None)
    rows = [r for f in features for r in manifest.read(f)]
    if args.eval_only is None and args.out is None:
        raise SystemExit("--out is required unless --eval-only is given")
    train = [] if args.eval_only else load_split(rows, "train")
    valid = load_split(rows, "valid")
    if (not train and not args.eval_only) or not valid:
        raise SystemExit(f"need both splits: {len(train)} train, {len(valid)} valid")
    extra = {}
    for spec in args.extra_valid:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--extra-valid wants NAME=FEATURES.jsonl, not {spec!r}")
        extra[name] = load_split(manifest.read(Path(path).resolve()), "valid")
        if not extra[name]:
            raise SystemExit(f"--extra-valid {name}: no valid split in {path}")

    if args.eval_only:
        from .model import load
        model = load(args.eval_only, device)
        res = dict(checkpoint=str(args.eval_only.resolve()), **evaluate(model, valid, device))
        res["extra"] = {name: evaluate(model, data, device) for name, data in extra.items()}
        print(json.dumps(res, indent=1))
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = BoundaryNet(NetConfig(n_mels=train[0]["mel"].shape[1], n_aux=train[0]["aux"].shape[1],
                                  use_aux=not args.no_aux)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    amp = device == "cuda"                   # bf16 autocast: no loss scaling needed

    args.out.mkdir(parents=True, exist_ok=True)
    last, start, best = args.out / "last.pt", 1, -1.0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        old = ck.get("run_args", {})
        changed = [k for k in RESUME_KEYS if old.get(k) != run_args[k]]
        if changed:
            raise SystemExit(f"{args.out} holds a run with different {', '.join(changed)}; use a new --out")
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        if amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start, best = ck["step"] + 1, ck["best"]
        print(f"resuming {args.out} at step {start} (best score so far {best:.4f})", flush=True)
    with open(args.out / "args.json", "w") as f:
        json.dump(dict(run_args, device=device, n_train=len(train), n_valid=len(valid)), f, indent=1, default=str)

    crops = Crops(train, int(round(args.crop_s / HOP_S)))
    pw = torch.tensor([args.pos_weight, args.pos_weight, 1.0], device=device)
    head_w = torch.tensor([1.0, 0.5, 0.5], device=device)
    with open(args.out / "log.jsonl", "a" if start > 1 else "w") as log:
        t0, run_loss, since = time.time(), 0.0, 0
        for step in range(start, args.steps + 1):
            mel, aux, y = (torch.from_numpy(x).to(device) for x in crops.batch(rng, args.batch))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(mel, aux)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y, pos_weight=pw,
                                                                       reduction="none")
            loss = (loss * head_w).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss, since = run_loss + loss.item(), since + 1
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, valid, device)
                m_extra = {name: evaluate(model, data, device) for name, data in extra.items()}
                f, r = m["onset_f"], m["resung_recall"]
                score = f if math.isnan(r) else (2 * f * r / (f + r) if f + r > 0 else 0.0)
                rec = dict(step=step, loss=run_loss / since, lr=sched.get_last_lr()[0],
                           elapsed_s=round(time.time() - t0), score=score, **m)
                if m_extra:
                    rec["extra"] = m_extra
                run_loss, since = 0.0, 0
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                if score > best:
                    best = score
                    save(args.out / "best.pt", model, step=step, metrics=m)
                save(last, model, step=step, metrics=m, best=best, run_args=run_args, opt=opt.state_dict(),
                     sched=sched.state_dict(), np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if amp else None)
    print(f"best score {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
