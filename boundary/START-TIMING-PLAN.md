# Start-timing diagnostic: why found notes start 50–150 ms off

Written 27 October 2026, before any measurement with it. This is a diagnostic. It selects no candidate and spends no test check. It decides only what to plan next: a decoder change, or retraining.

## Why

On the RWC-P test half (test check 1, `arms/tc1.txt`), frozen candidate 2 is level with GAME on finding notes, but far behind once starts must be within 50 ms:

| System | COnP@150 | COnP@50 | COnPOff@50 |
|---|---|---|---|
| Frozen candidate 2 | 0.665 | 0.487 | 0.267 |
| GAME | 0.679 | 0.572 | 0.338 |
| MIR-ST500 | 0.671 | 0.567 | 0.347 |

COnPOff@50 is roughly COnP@50 times the share of those notes whose end is also right. That share is 54.8% for the candidate and 59.1% for GAME. So the gap of 0.071 to GAME splits roughly as follows:
- **Ends, about 0.02.** At GAME's share, the candidate would score 0.487 × 0.591 = 0.288.
- **Starts, about 0.05.** The remaining 0.085 on COnP@50, at 59.1%.

The ends side has been tried from the decoder (the ends plan failed). The starts side has not been looked at directly. The onset ceiling (20 October) says it is worth looking at: with the reference starts, preset B's COnP@50 on the development half reached 0.563, GAME's level.

The question: for notes the candidate finds within 150 ms but not within 50 ms ("near misses"):
- Is the error a consistent shift?
- Did the network mark the right time, but the decoder not use it?
- Or did the network miss the time, or put it elsewhere?

## What is measured

**Data.** The RWC-P development half: the 48 songs with note references. The existing MIDI files of:
- frozen candidate 2, 3 seeds (`m4gts-aug-s{s}-B-plain-t0.5-sm99`);
- preset B (`sp-f-delta0.6`);
- GAME (`game`) and MIR-ST500 (`mirst500`).

Nothing is decoded except the reproduction check (M0).

**Pairs.** mir_eval's note matching at an onset tolerance of 150 ms, pitch within 50 cents, no offset (as COnP@150). For each pair, the start error is e = transcribed start − reference start. A pair is:
- **near** if |e| ≤ 50 ms;
- a **near miss** if 50 ms < |e| ≤ 150 ms.

Each reference note has a **context**:
- *after a rest*: the previous reference note ends more than 50 ms before it (or there is none);
- *legato, new pitch*: otherwise, and the pitch differs from the previous note;
- *legato, same pitch*: otherwise, and the pitch is the same.

### M0. Reproduction check

Each seed's network outputs are recomputed with the frozen code (boundary 7432d9a, CPU, as the sounding plan decoded). Each song is decoded again with transcPybe 3361803 and frozen candidate 2's configuration. The MIDI must match the existing run byte for byte, for all 144 song-seed decodes. If any differs, the attribution (M3) is not run, and the reason is reported. The outputs are cached as `arms/post-m4gts-aug-s{s}/`.

### M1. Shape of the error

For each system, report:
- pairs at 150 ms, and the near-miss share;
- the near misses' median and quartiles of e;
- the share of near misses that are early (e < 0) and late;
- the same broken down by context.

### M2. Shift test

Every transcribed start moves by a constant δ. Ends do not move. A start never comes within 10 ms of its own end, and never goes below 0. COnP@50 and COnPOff@50 are rescored exactly (mir_eval), each song averaged over the 3 seeds. δ runs from −100 to +100 ms in steps of 10 ms. At δ = 0 the score must reproduce frozen candidate 2's cached COnPOff@50 (0.2930).

The shift is cross-validated. The songs, in sorted order, are split into two folds by alternate position. δ is chosen on one fold by the highest COnPOff@50, and applied to the other. The held-out gain over δ = 0 is taken per song, over both folds (48 songs), with a 95% bootstrap interval (10,000 resamples, seed 0).

A second version uses two shifts. One applies to transcribed starts that touch the previous transcribed note (within 30 ms of its end), the other to the rest. Each runs from −100 to +100 ms in steps of 20 ms, cross-validated the same way. The decoder could apply either version, since it uses only the transcription.

### M3. Attribution (candidate only, each seed with its own outputs)

The decoder was given the onset output's peaks at 0.5 (`pick_peaks`, as in the hand-over). With decoder options as frozen, it snaps a start to the nearest peak within 50 ms, and only earlier. Each near miss falls in exactly one class:
- **A, the network had it:** a peak lies within 50 ms of the reference start. Sub-classes:
  - *A-direction*: that peak is also the nearest peak to the transcribed start, within 50 ms of it, and later than it. Snapping only earlier kept the start.
  - *A-window*: that peak is more than 50 ms from the transcribed start, outside the snap window.
  - *A-other*: anything else, such as a nearer, wrong peak won the snap.
- **B, the network misplaced it:** no peak lies within 50 ms of the reference start, but one lies within 50 ms of the transcribed start.
- **C, the network missed it:** no peak lies near either. Reported split by whether the onset output's maximum within 50 ms of the reference start reaches 0.3 (*C-weak*) or not (*C-none*).

### M4. Stakes

For each class, and for each A sub-class, each near miss of that class has its start moved onto its reference start, with ends unchanged. The gain in COnPOff@50 over the unchanged files is taken from the 3-seed mean. This is what a perfect fix of that class would be worth. M3's counts say how often; M4 says how much.

### M5. Reachability (reported)

Of the reference notes behind the candidate's near misses, report the share that GAME, and the share that MIR-ST500, match within 50 ms.

## Decision rules, fixed now

These decide only which plan to write next. Each next plan is pre-registered and judged by protocol amendment 3 as usual.

1. **Shift.** Either version of the cross-validated shift (M2) raises COnPOff@50 by at least +0.01, with the 95% interval excluding zero. If so, part of the gap is a timing convention. The next step includes a decoder plan for a start offset, without retraining for it.
2. **Decoder or network.** From M4, let S_A be the stakes of class A, and S_BC those of B and C together.
   - **S_A ≥ S_BC:** the decoder is the bottleneck, and no retraining comes next. The next step is a decoder plan with options that already exist:
     - `onset_snap_direction` "both", if A-direction carries the larger share of S_A;
     - a wider `onset_snap_window_s`, if A-window does;
     - a decoder diagnostic of A-other first, if A-other does.
   - **S_BC > S_A:** the network is the bottleneck, and retraining is the route.
     - If B's stakes exceed C's, the retrain targets timing: the onset label's position, or a sharper target.
     - Otherwise it targets recall. If C-weak carries most of C's stakes, a check of the peak threshold comes before retraining.
3. **Both:** if rule 1 passes and rule 2 points to the network, both follow. The shift plan comes first, as it is cheap.

## Steps

1. **Musix, before anything is measured:** this plan is hashed (`start-timing-plan.md` and `.sha256` in `~/boundary-work-20261020`).
2. **M0:** `arms/start_post.py`, 12 processes on the CPU (2 threads each, as the sounding plan used). It writes `arms/post-m4gts-aug-s{s}/` and `arms/start-post.log`.
3. **M1–M5 and the rules:** `arms/start_timing.py`, writing `arms/start-timing.txt`.
4. **Cost:** no GPU and no CSD3 time, about an hour on musix.
