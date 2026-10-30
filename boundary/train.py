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

A fourth output marks only re-sung note starts: the onset target, kept at
the notes ``resung`` picks out. Every note start where the pitch changes is
already a boundary for transcPybe's pitch decoder, so a split there can only
add a note; this output is meant to tell the decoder where a split is needed.
Its target is worked out from each recording's notes as it loads, so feature
files and shards hold three targets and need no rebuilding. It is trained at
half weight beside the offset and sounding outputs, and logged at every
evaluation (``resung_head_*``) without taking part in choosing the checkpoint.

``--onset-labels`` chooses what the onset output learns (round 2, ``ROUND2-PLAN.md``):

- ``stored`` (the default): the targets in the shard, made from the manifest's
  notes, as in every run before round 2;
- ``vowel-voicing``: the same, except that each note opening on a vowel starts
  at its voicing onset. That is the first frame within 150 ms after the label
  whose stored voicing (the f0 track the network is given) is at least 0.3,
  after at least 3 frames below it. A note with no such frame, or one the move
  would leave under 30 ms, keeps its start. M4Singer starts these notes a
  median 70 ms before voicing; human-labelled sets start them within about
  10 ms of it;
- ``landmarks``: two onset outputs instead of one. The first (column 0) learns
  where each note's syllable starts sounding, its consonant or labelled start.
  A fifth output learns where its vowel starts, or for a note opening on a
  vowel, its voicing onset by the rule above. The decoder places a note's start
  between the two (``infer.landmark_track``), so the convention is chosen when
  decoding, not by retraining.

Both need the manifest's per-note ``landmarks`` (``sources --onsets between``,
carried by ``relabel``). The offset, sounding and re-sung targets stay the
stored ones. The checkpoint rule uses the onset output against the onsets it
learned, so for ``landmarks`` that is the articulation onsets. The vowel
output's F is logged beside it. ``--report-labels`` prints how many notes the
vowel rule moves, how far, and the gap between the landmarks, then stops. A
training run refuses to start if the median move is outside 20-120 ms, which
would mean the rule is not doing what it is meant to.

``--onset-width`` widens the targets of the onset outputs (round 3,
``ROUND3-PLAN.md``). At the usual 2 frames an output marks a start with a spike.
At 10 frames it says how close each frame is to the nearest start: the target
falls steadily from 1 at the start to nearly 0 about 100 ms away, so one start
should give one hump rather than two spikes. Peaks are then picked at least that
far apart. It needs ``--onset-labels landmarks``. ``--onset-pos-weight`` sets the
weight of those outputs' positive frames alone, and at 1 their outputs estimate
the closeness itself. The offset and re-sung outputs keep ``--pos-weight``.

``--channels``, ``--hidden`` and ``--dropout`` set the network's size and
regularisation; the checkpoint records them, so inference needs no flags.
``--augment`` varies every training crop afresh (``augment``): the mixes are
fixed on disk, so without it the network sees the same audio on every pass.
``--stop-after`` ends a run early while the learning-rate schedule still runs
over ``--steps``, so a short run follows the first part of a long one.

Resumable: ``last.pt`` holds the optimiser, schedule and random states as well
as the weights, and a rerun with the same arguments carries on from it with
the best score so far. A rerun with different arguments into the same
``--out`` is refused rather than mixed into the old run.

With MLFLOW_TRACKING_URI set, each evaluation is also sent to that MLflow
server (``track``), so runs can be watched in a browser as they go.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

from . import HOP_S, items as manifest
from .features import LEVEL_RANGE, TARGET_WIDTH, targets
from .infer import MIN_GAP_FRAMES, pick_peaks
from .track import Tracker, context_tags

TOL = 0.05
TIMING_WINDOW = 0.15      # a reference onset counts for timing if an estimate is within 150 ms
TIMING_MARGIN = 0.025     # early or late: more than 25 ms out
EVAL_BATCH = 32
RESUME_KEYS = ("features", "seed", "steps", "batch", "crop_s", "lr", "weight_decay", "warmup", "pos_weight", "no_aux",
               "channels", "hidden", "dropout", "augment", "onset_labels", "onset_width", "onset_pos_weight")
ONSET_LABELS = ("stored", "vowel-voicing", "landmarks")
VOICING_COLUMN = 3        # aux: pitch, pitched, periodicity, voicing, level (features.frame_features)
VOICED_AT = 0.3           # the decoder's own voicing threshold
VOWEL_REACH_S = 0.15      # how far after its label a vowel-initial note's voicing onset is looked for
MIN_NOTE_S = 0.03
VOWEL_MOVE_MS = (20.0, 120.0)   # the median move the rule must make, or training stops (ROUND2-PLAN.md)
MEL_DB_PER_UNIT = 19.0    # one unit of the standardised log-mel: the median per-stem std in dB (dry4k, 30 stems)
LEVEL_DB_PER_UNIT = 40.0  # the level feature's scale (features.frame_features)
EQ_MAX_DB = 6.0
COMPRESS_MAX = 0.5        # loud frames lose up to half their level above the crop's median
MASK_MAX_BANDS = 8


def augment(mel, aux, rng):
    """One training crop varied in the feature domain, differently on every draw.

    A smooth random EQ of up to 6 dB across the mel bands; half the time a
    compressor that pulls frames louder than the crop's median down by up to
    half the difference; half the time a block of up to 8 mel bands masked. The
    f0 features were computed once and keep their values, except level, which
    follows the compressor. Targets are not touched.
    """
    mel, aux = mel.copy(), aux.copy()
    n = mel.shape[1]
    x = np.linspace(0.0, np.pi, n)
    curve = sum(rng.uniform(-1, 1) * np.cos(k * x) for k in (1, 2, 3))
    mel += (curve * rng.uniform(0, EQ_MAX_DB) / MEL_DB_PER_UNIT / (np.abs(curve).max() + 1e-9))[None, :]
    if rng.random() < 0.5:
        lv = mel.mean(1)
        cut = rng.uniform(0, COMPRESS_MAX) * np.maximum(0.0, lv - np.median(lv))
        mel -= cut[:, None]
        aux[:, 4] = np.clip(aux[:, 4] - cut * MEL_DB_PER_UNIT / LEVEL_DB_PER_UNIT, *LEVEL_RANGE)
    if rng.random() < 0.5:
        w = int(rng.integers(1, MASK_MAX_BANDS + 1))
        f0 = int(rng.integers(0, n - w + 1))
        mel[:, f0:f0 + w] = mel.mean()
    return mel, aux


def voicing_onset(voicing: np.ndarray, start: float, end: float, reach: float = VOWEL_REACH_S,
                  threshold: float = VOICED_AT):
    """The first frame after ``start``, within ``reach``, whose voicing is at least ``threshold`` after at least
    3 frames below it, in seconds; None if there is none, or if starting there would leave the note under 30 ms."""
    s0 = int(round(start / HOP_S))
    last = min(len(voicing) - 1, int(round((start + reach) / HOP_S)))
    for i in range(max(s0 + 1, 3), last + 1):
        if voicing[i] >= threshold and (voicing[i - 3:i] < threshold).all():
            t = round(i * HOP_S, 4)
            return t if t < end - MIN_NOTE_S else None
    return None


def onset_labels(r, voicing: np.ndarray, mode: str):
    """What the onset outputs learn for one manifest row: (notes for column 0, notes for the vowel output or None,
    the vowel rule's moves in seconds). See the module docstring for the modes."""
    notes = r["notes"]
    if mode == "stored":
        return notes, None, []
    lm = r.get("landmarks")
    if lm is None or len(lm) != len(notes):
        raise SystemExit(f"{r.get('mix_id', r.get('id'))}: --onset-labels {mode} needs the manifest's per-note "
                         "landmarks (sources --onsets between, carried by relabel)")
    moves, vowel_starts = [], []
    for (on, off, _), (a, v, opens_on_vowel) in zip(notes, lm):
        if opens_on_vowel:
            t = voicing_onset(voicing, a, off)
            if t is not None:
                moves.append(t - a)
            vowel_starts.append(t if t is not None else a)
        else:
            vowel_starts.append(v)
    if mode == "vowel-voicing":   # half-way notes, with vowel-initial ones moved to their voicing onset
        col0 = [[w if o else on, off, p] for (on, off, p), w, (_, _, o) in zip(notes, vowel_starts, lm)]
        return col0, None, moves
    articulation = [[a, off, p] for (_, off, p), (a, _, _) in zip(notes, lm)]
    vowel = [[w, off, p] for (_, off, p), w in zip(notes, vowel_starts)]
    return articulation, vowel, moves


def load_split(rows, split, mode: str = "stored", onset_width: int = TARGET_WIDTH):
    """One split's recordings, from per-recording feature files or from shards (boundary.pack).

    ``mode`` is ``--onset-labels``: with ``stored`` the onset target is the shard's; otherwise column 0 is rebuilt
    from ``onset_labels``, and ``landmarks`` adds the vowel target as a fifth column. Rebuilt onset targets spread
    ``onset_width`` frames either side (``--onset-width``)."""
    from .pack import read

    out, shards = [], {}
    for r in manifest.split(rows, split):
        if "shard" in r:
            z = read(r["shard"], r["index"], shards)
        else:
            with np.load(r["features"]) as f:
                z = {k: f[k] for k in ("mel", "aux", "y")}
        y = z["y"].astype(np.float32)
        # the fourth target comes from the notes, so they must be the notes the stored targets came from
        if not np.allclose(targets(r["notes"], len(y))[:, 0], y[:, 0], atol=0.01):
            raise SystemExit(f"{r['mix_id']}: the stored onset targets don't match its notes, so the re-sung target "
                             "worked out from them would be misplaced")
        aux = z["aux"].astype(np.float32)
        col0, vowel, moves = onset_labels(r, aux[:, VOICING_COLUMN], mode)
        if mode != "stored":
            y[:, 0] = targets(col0, len(y), onset_width)[:, 0]
        y = np.concatenate([y, resung_target(r["notes"], len(y))[:, None]], axis=1)
        if vowel is not None:
            y = np.concatenate([y, targets(vowel, len(y), onset_width)[:, :1]], axis=1)
        d = dict(id=r["mix_id"], notes=r["notes"], onset_notes=col0, mel=z["mel"].astype(np.float32), aux=aux, y=y,
                 moves=moves)
        if vowel is not None:
            d["vowel_onsets"] = [n[0] for n in vowel]
            d["landmark_gaps"] = [w[0] - a[0] for a, w in zip(col0, vowel) if w[0] > a[0]]
        out.append(d)
    return out


def label_report(data, mode: str) -> dict:
    """How the onset labels differ from the stored ones: notes opening on a vowel, how many the rule moves and how
    far, and (landmarks) the gap from articulation to vowel."""
    if mode == "stored":
        return {}
    ms = lambda xs, q: round(float(np.percentile(np.asarray(xs) * 1000, q)), 1) if len(xs) else float("nan")
    moves = [m for d in data for m in d["moves"]]
    n_notes = sum(len(d["notes"]) for d in data)
    rep = dict(n_notes=n_notes, n_moved=len(moves), moved_share=round(len(moves) / n_notes, 4) if n_notes else 0.0,
               move_median_ms=ms(moves, 50), move_p25_ms=ms(moves, 25), move_p75_ms=ms(moves, 75))
    if mode == "landmarks":
        gaps = [g for d in data for g in d["landmark_gaps"]]
        rep.update(n_gaps=len(gaps), gap_median_ms=ms(gaps, 50), gap_p25_ms=ms(gaps, 25), gap_p75_ms=ms(gaps, 75))
    return rep


def resung_target(notes, n_frames: int) -> np.ndarray:
    """The fourth output's target: the onset target, kept only at re-sung note starts (``resung``)."""
    keep = set(resung(notes))
    return targets([n for n in notes if n[0] in keep], n_frames)[:, 0]


class Crops:
    """Random fixed-length crops, a recording drawn in proportion to its length."""

    def __init__(self, data, crop: int, augment: bool = False):
        self.data, self.crop, self.augment = data, crop, augment
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
        crops = [self._one(rng, int(i)) for i in rng.choice(len(self.data), size=size, p=self.p)]
        if self.augment:
            crops = [(*augment(m, a, rng), y) for m, a, y in crops]
        mel, aux, y = zip(*crops)
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


def evaluate(model, data, device, threshold=0.5, min_gap=MIN_GAP_FRAMES):
    """Whole recordings, in length-sorted padded batches; the model is told each length, so
    every recording's posterior is what it would be on its own. ``min_gap`` spaces the onset outputs' peaks."""
    import mir_eval
    import torch

    model.eval()
    fs, ps, rs, hit, tot, n_est, n_ref, offsets = [], [], [], 0, 0, 0, 0, []
    head_hit, head_est, head_est_hit, head_tot = 0, 0, 0, 0   # the fourth output, where the network has one
    vowel_fs = []                                    # the fifth (landmarks), against the vowel onsets it learned
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
                                    torch.tensor(lens))).float().cpu().numpy()
            for i, d in enumerate(chunk):
                onset_notes = d.get("onset_notes", d["notes"])   # what the onset output learned (--onset-labels)
                est = pick_peaks(p[i, :lens[i], 0], threshold, min_gap) * HOP_S
                ref = np.array([n[0] for n in onset_notes])
                f, pr, rc = mir_eval.onset.f_measure(ref, est, window=TOL) if len(ref) else (1.0, 1.0, 1.0)
                fs.append(f)
                ps.append(pr)
                rs.append(rc)
                n_est, n_ref = n_est + len(est), n_ref + len(ref)
                offsets += onset_timing(ref, est)
                for onset in resung(onset_notes):
                    tot += 1
                    hit += int(np.any(np.abs(est - onset) <= TOL))
                if p.shape[-1] > 3:
                    re_sung = np.array(resung(d["notes"]))   # the fourth output's target is the stored notes'
                    head_tot += len(re_sung)
                    est_r = pick_peaks(p[i, :lens[i], 3], threshold) * HOP_S
                    head_est += len(est_r)
                    head_hit += sum(int(np.any(np.abs(est_r - o) <= TOL)) for o in re_sung)
                    head_est_hit += sum(int(np.any(np.abs(re_sung - e) <= TOL)) for e in est_r) if len(re_sung) else 0
                if p.shape[-1] > 4 and "vowel_onsets" in d:
                    est_v = pick_peaks(p[i, :lens[i], 4], threshold, min_gap) * HOP_S
                    ref_v = np.array(d["vowel_onsets"])
                    vowel_fs.append(mir_eval.onset.f_measure(ref_v, est_v, window=TOL)[0] if len(ref_v) else 1.0)
    model.train()
    head = {}
    if p.shape[-1] > 3:
        head = dict(resung_head_recall=head_hit / head_tot if head_tot else float("nan"),
                    resung_head_precision=head_est_hit / head_est if head_est else float("nan"),
                    resung_head_per_note=round(head_est / head_tot, 3) if head_tot else float("nan"))
    if vowel_fs:
        head["vowel_onset_f"] = float(np.mean(vowel_fs))
    return dict(onset_f=float(np.mean(fs)), resung_recall=hit / tot if tot else float("nan"), n_resung=tot,
                onset_p=float(np.mean(ps)), onset_r=float(np.mean(rs)),
                onsets_per_note=round(n_est / n_ref, 3) if n_ref else float("nan"), **timing_summary(offsets), **head)


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
    ap.add_argument("--stop-after", type=int, metavar="STEP",
                    help="stop at this step; the learning-rate schedule still runs over --steps")
    ap.add_argument("--no-aux", action="store_true", help="ablation: log-mel only, no f0 features")
    ap.add_argument("--channels", type=lambda s: tuple(int(c) for c in s.split(",")), default=(32, 64, 96),
                    help="convolution channels per block, comma-separated")
    ap.add_argument("--hidden", type=int, default=128, help="projection and GRU width")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--augment", action="store_true", help="vary every training crop afresh (see augment)")
    ap.add_argument("--onset-labels", choices=ONSET_LABELS, default="stored",
                    help="what the onset output learns (see the module docstring); round 2's two arms")
    ap.add_argument("--onset-width", type=int, default=TARGET_WIDTH,
                    help="frames an onset target spreads either side (see the module docstring)")
    ap.add_argument("--onset-pos-weight", type=float, default=None,
                    help="positive weight of the onset outputs alone (default: --pos-weight)")
    ap.add_argument("--report-labels", action="store_true",
                    help="print how --onset-labels changes the training labels, then stop")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--extra-valid", action="append", default=[], metavar="NAME=FEATURES.jsonl",
                    help="another set to evaluate (its valid split) and log, never used to choose a checkpoint")
    args = ap.parse_args(argv)

    import torch
    from .model import BoundaryNet, NetConfig, save

    if args.onset_width != TARGET_WIDTH and args.onset_labels != "landmarks":
        raise SystemExit("--onset-width needs --onset-labels landmarks: the other modes keep the stored targets' width")
    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    features = [str(f.resolve()) for f in args.features]
    run_args = dict(vars(args), features=features, out=str(args.out.resolve()) if args.out else None)
    rows = [r for f in features for r in manifest.read(f)]
    if args.eval_only is None and args.out is None and not args.report_labels:
        raise SystemExit("--out is required unless --eval-only or --report-labels is given")
    train = [] if args.eval_only else load_split(rows, "train", args.onset_labels, args.onset_width)
    valid = load_split(rows, "valid", args.onset_labels, args.onset_width)
    if (not train and not args.eval_only) or not valid:
        raise SystemExit(f"need both splits: {len(train)} train, {len(valid)} valid")
    extra = {}
    for spec in args.extra_valid:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--extra-valid wants NAME=FEATURES.jsonl, not {spec!r}")
        extra[name] = load_split(manifest.read(Path(path).resolve()), "valid", args.onset_labels, args.onset_width)
        if not extra[name]:
            raise SystemExit(f"--extra-valid {name}: no valid split in {path}")

    labels = label_report(train or valid, args.onset_labels)
    if args.report_labels:
        print(json.dumps(dict(onset_labels=args.onset_labels, split="train" if train else "valid", **labels), indent=1))
        return
    if labels and not args.eval_only and not (VOWEL_MOVE_MS[0] <= labels["move_median_ms"] <= VOWEL_MOVE_MS[1]):
        raise SystemExit(f"the vowel rule's median move is {labels['move_median_ms']} ms, outside {VOWEL_MOVE_MS}: "
                         "it is not doing what ROUND2-PLAN.md means it to, so this run stops here")

    min_gap = max(MIN_GAP_FRAMES, args.onset_width) if args.onset_width != TARGET_WIDTH else MIN_GAP_FRAMES
    if args.eval_only:
        from .model import load
        model = load(args.eval_only, device)
        res = dict(checkpoint=str(args.eval_only.resolve()), **evaluate(model, valid, device, min_gap=min_gap))
        res["extra"] = {name: evaluate(model, data, device, min_gap=min_gap) for name, data in extra.items()}
        print(json.dumps(res, indent=1))
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    n_out = train[0]["y"].shape[1]            # 4, or 5 with --onset-labels landmarks
    model = BoundaryNet(NetConfig(n_mels=train[0]["mel"].shape[1], n_aux=train[0]["aux"].shape[1],
                                  channels=args.channels, hidden=args.hidden, dropout=args.dropout,
                                  use_aux=not args.no_aux, n_out=n_out)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    amp = device == "cuda"                   # bf16 autocast: no loss scaling needed

    args.out.mkdir(parents=True, exist_ok=True)
    last, start, best = args.out / "last.pt", 1, -1.0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        old = ck.get("run_args", {})
        changed = [k for k in RESUME_KEYS if old.get(k, ap.get_default(k)) != run_args[k]]   # older runs lack newer keys
        if changed:
            raise SystemExit(f"{args.out} holds a run with different {', '.join(changed)}; use a new --out")
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        # map_location moved every tensor to the device, the saved generator states too; torch wants them on the CPU
        torch.set_rng_state(ck["torch_rng"].cpu())
        if amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start, best = ck["step"] + 1, ck["best"]
        print(f"resuming {args.out} at step {start} (best score so far {best:.4f})", flush=True)
    info = dict(run_args, device=device, n_train=len(train), n_valid=len(valid),
                n_params=sum(p.numel() for p in model.parameters()), labels=labels)
    with open(args.out / "args.json", "w") as f:
        json.dump(info, f, indent=1, default=str)
    out = args.out.resolve()
    tracker = Tracker.start(out, f"{out.parent.name}/{out.name}", tags=context_tags(out), params={
        k: " ".join(v) if k == "features" else v for k, v in info.items() if v is not None and v != []})
    # Slurm ends a job at its time limit with SIGTERM; as SystemExit it lets the tracker mark the run KILLED
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))

    crops = Crops(train, int(round(args.crop_s / HOP_S)), augment=args.augment)
    end = min(args.steps, args.stop_after or args.steps)
    # onset, offset, sounding, re-sung onset, and with landmarks the vowel onset, weighted as the onset
    onset_pw = args.pos_weight if args.onset_pos_weight is None else args.onset_pos_weight
    pw = torch.tensor([onset_pw, args.pos_weight, 1.0, args.pos_weight, onset_pw][:n_out], device=device)
    head_w = torch.tensor([1.0, 0.5, 0.5, 0.5, 1.0][:n_out], device=device)
    with tracker, open(args.out / "log.jsonl", "a" if start > 1 else "w") as log:
        t0, run_loss, since = time.time(), 0.0, 0
        for step in range(start, end + 1):
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
            if step % args.eval_every == 0 or step == end:
                m = evaluate(model, valid, device, min_gap=min_gap)
                m_extra = {name: evaluate(model, data, device, min_gap=min_gap) for name, data in extra.items()}
                f, r = m["onset_f"], m["resung_recall"]
                score = f if math.isnan(r) else (2 * f * r / (f + r) if f + r > 0 else 0.0)
                el = time.time() - t0
                rec = dict(step=step, loss=run_loss / since, lr=sched.get_last_lr()[0], elapsed_s=round(el),
                           eta_s=round(el / (step - start + 1) * (end - step)), best=max(best, score), score=score, **m)
                if m_extra:
                    rec["extra"] = m_extra
                run_loss, since = 0.0, 0
                log.write(json.dumps(rec) + "\n")
                log.flush()
                tracker.log(rec, step)
                print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [train] step {step}/{end}  "
                      f"score {score:.3f} (best {rec['best']:.3f})  onset F {f:.3f}  re-sung {r:.3f}  "
                      f"timing {m['timing_median_ms']:+.0f} ms  per note {m['onsets_per_note']:.2f}  "
                      f"loss {rec['loss']:.3f}  eta {rec['eta_s'] // 60} min"
                      + (f"  re-sung output R {m['resung_head_recall']:.2f} P {m['resung_head_precision']:.2f}"
                         if "resung_head_recall" in m else "")
                      + (f"  vowel onset F {m['vowel_onset_f']:.3f}" if "vowel_onset_f" in m else ""), flush=True)
                if score > best:
                    best = score
                    save(args.out / "best.pt", model, step=step, metrics=m)
                save(last, model, step=step, metrics=m, best=best, run_args=run_args, opt=opt.state_dict(),
                     sched=sched.state_dict(), np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if amp else None)
    print(f"best score {best:.4f} -> {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
