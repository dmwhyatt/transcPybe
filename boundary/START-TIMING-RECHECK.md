# Start-timing recheck: what is left after the snap change

Written 27 October 2026, before any measurement with it. This repeats the start-timing diagnostic (`start-timing-plan.md`, sha 96874d25…) on frozen candidate 3. It selects nothing and spends nothing. It informs one decision: whether the remaining start errors are the decoder's, which a cheaper decoder change could fix, or the network's, which only retraining could.

## Why

Frozen candidate 3 (`frozen-candidate-3.json`, sha f7f70082…) snaps note starts to the network's onsets from up to 150 ms away, in both directions. On the test half (test check 2):
- COnP@50 is still 0.526 against GAME's 0.572.
- Roughly 0.027 of the 0.039 COnPOff@50 gap to GAME is starts, and 0.012 is ends.

Before the snap change, the diagnostic attributed the near misses this way:
- 51% were onsets the network had right but the decoder did not use (+0.049 if fixed perfectly);
- 43% were onsets the network itself put elsewhere (+0.036);
- 7% were onsets it missed (+0.006).

The snap change was aimed at the first group.

## What is measured

The same data, pairs, contexts and measurements as the diagnostic, with these changes:
- **The candidate** is frozen candidate 3's development runs (`m4gts-aug-s{s}-B-plain-t0.5-sm99-w15b`), 3 seeds.
- **M1** also lists frozen candidate 2, beside preset B, GAME and MIR-ST500.
- **The check:** the unchanged files must reproduce candidate 3's cached COnPOff@50 (0.3240).
- **M3's network outputs** are the same cached per-seed outputs (`arms/post-m4gts-aug-s{s}/`). They do not depend on decoder settings, and M0 has already reproduced them.
- **M3's sub-classes of A follow candidate 3's snap**, which reaches 150 ms both ways, so there is no direction class:
  - *A-window*: every right peak is more than 150 ms from the transcribed start;
  - *A-other*: a right peak is within reach, but the start did not end up there.
- **M2's shift test** is reported only.

## How it is read, fixed now

These are the diagnostic's rule 2, applied to candidate 3. From M4, let S_A be the stakes of class A, and S_BC those of B and C together.

- **S_A ≥ S_BC:** the decoder is still the bottleneck. A decoder change comes before retraining. The largest A sub-class says which:
  - *A-window*: a wider reach, though the vocadito check showed a wider window costs a cappella;
  - *A-other*: a decoder diagnostic of why reachable peaks are not taken.
- **S_BC > S_A:** the network is now the bottleneck, and retraining is the route.
  - If B's stakes exceed C's, the retrain targets timing.
  - Otherwise it targets recall.

Either way, any retrained model needs a fresh held-out set for confirmation, because RWC-P's test half is spent.

## Steps

1. This note is hashed on musix (`start-timing-recheck.md` and `.sha256` in `~/boundary-work-20261020`).
2. `arms/start_timing3.py` runs it, writing `arms/start-timing3.txt`. No GPU, a few minutes.
