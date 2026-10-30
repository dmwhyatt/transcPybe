# A cappella preset and seed-averaging plan

Written 30 October 2026, after the stage split (`~/phoneme-gap-20261030/timing.txt` on musix) and before any decode with these settings. Reviewed by the user the same day, who kept SSVD train&valid among the tuning sets and the +0.01 bar in Rule B. Hashed on musix before any decode.

## Why

On Molina, v2 loses to phoneme-informed almost entirely at note starts, and mostly on timing rather than on finding notes:

| System | COn@50 | COn@150 | Median start error | Start spread (IQR) | COnPOff@50 |
|---|---|---|---|---|---|
| Phoneme-informed | 0.931 | 0.954 | +2 ms | 24 ms | 0.712 |
| GAME | 0.941 | 0.968 | −3 ms | 18 ms | 0.506 |
| v2, *f* = 0.75 | 0.806 | 0.935 | −14 ms | 30 ms | 0.602 |
| v2, *f* = 0.5 | 0.745 | 0.934 | −23 ms | 33 ms | 0.563 |
| v2, *f* = 0.25 | 0.656 | 0.931 | −32 ms | 37 ms | 0.498 |

About 90% of the gap is at the onset stage. Pitch given a right start is level (0.884 against 0.886), and so are ends given a right start and pitch (0.845 at 20 dB against 0.864).

Three things follow:
- **Starts get later and tighter as *f* rises,** and *f* above 0.75 has never been decoded.
- **The snap window was chosen on separated pop.** v2 snaps starts up to 0.15 s both ways. On vocadito, measured on frozen candidate 2 (`orchestrator/bench-work-vocadito-snap-20261027/snapmean.txt`), a 0.05 s window both ways gave COnPOff +0.015* and +0.017*, and a 0.15 s window earlier only gave −0.070* and −0.067*. No narrower window has been tried on the landmark networks.
- **We don't know what kind of timing error this is.** The start-timing recheck attributed 81% of the RWC-P near-miss stake to the network's own mistimed onsets. If each seed errs differently, averaging helps. If they share the error, the cause is more likely in the labels or conventions, and the next network has to address that instead.

So this plan asks two questions:
- **Part A (seed averaging):** does averaging the three seeds' outputs help, and how much of the start error is seed noise?
- **Part B (a cappella preset):** is there a better *f* and snap window for a cappella, and does it hold on a set it was not tuned on?

Part A comes first because Part B is decoded from whichever outputs it leaves us.

## What has been seen

- **vocadito and Molina:** seen many times, and *f* = 0.75 was chosen after seeing them. They are tuning sets here. Nothing measured on them counts as confirmation.
- **RWC-P development half:** tuned on. **RWC-P test half:** spent by test checks 1 and 2, and not used here.
- **MIR-ST500 test songs:** nothing scored; two checks unspent. Not used here.
- **SSVD train&valid (67 recordings, 3,348 notes):** the development set of the 13 October transfer test. It is also phoneme-informed's training data. No BoundaryNet network has been decoded on it.
- **SSVD test (127 recordings, 5,203 notes, 74.4 minutes):** not pristine, and this is declared rather than ignored.
  - It was scored once, on 13 October (`bench-work-molina-20261012/xfer2`, REPORT section 14.4), to judge an onset filter with an older transcPybe tree and a madmom detector.
  - No v2 network, *f*, threshold or decoder setting was chosen on it, and neither the filter nor the detector is in v2. No BoundaryNet network, phoneme-informed, GAME, MIR-ST500 or Omnizart has been decoded on it.
  - Those older systems' aggregate scores were seen. That run's secondary condition also showed two-way snapping raising COn (0.708 to 0.746 for one detector). This grid holds the direction at both ways, so that result chooses nothing here.

## Scoring

Estimates are whole-semitone MIDI, as for every system in our tables. References are as published. COnPOff@50 is mir_eval's note F-measure (onset within 50 ms, pitch within 50 cents, offset within 20% of the reference note or 50 ms, whichever is larger), computed per recording and averaged.

Each comparison pairs recordings, using a 95% bootstrap interval (10,000 resamples, seed 0). "3-seed mean" means each recording's score averaged over the seeds.

**Notes in silence,** on the a cappella sets, uses the MIR-ST500 protocol's definition, because most of these sets have no pitch-track reference in our runs: a transcribed note counts if at least 80% of its span lies outside every reference note. RWC-P uses `dev/score.py`'s frame-level measure.

## Part A: averaging the seeds

**Averaged outputs:** the mean of the three networks' per-frame probabilities, for all five outputs, before peak picking. Nothing else changes. Threshold 0.5, the 250 ms pairing window and the decoder settings stay as they are.

**Decoded:** round 2's landmark networks (`runs/r2-lm`, the chosen checkpoints), averaged, on:
- the RWC-P development half at *f* = 0.5, 20 dB, snap 0.15 s both ways: the About page's separated-pop row;
- vocadito, Molina and SSVD train&valid at *f* = 0.75, 20 dB, snap 0.15 s both ways: the About page's a cappella row, called **the control** below.

### Rule A, fixed now

Averaging is **adopted** if all of these hold against the 3-seed mean at the same setting:
1. RWC-P development half: COnPOff@50 is higher, with the interval excluding zero.
2. It beats at least 2 of the 3 seeds individually on that score.
3. On that half, COnP@150 falls by at most 0.005, RPA falls by at most 0.005, and notes in silence rise by at most 0.5 points.
4. On each of vocadito A1, vocadito A2 and Molina, COnPOff is not below the control's with the interval excluding zero.

Otherwise each seed stays separate, and Part B uses the 3-seed mean.

### Diagnostic, selecting nothing

This informs the next network. On the RWC-P development half and on Molina, take each reference note paired within 150 ms by all three seeds, ignoring pitch, and record each seed's start error. Report:
- the IQR of one seed's errors, and of the errors of the three seeds' mean;
- the **seed share:** the variance between seeds within a note, as a share of the total variance of single-seed errors.

The reading is fixed now. If errors are roughly normal, removing the seed share would shrink the IQR by a factor of √(1 − share):

| Seed share | IQR could fall by | Reading |
|---|---|---|
| Under 20% | under about 10% | Seeds make much the same errors. The next network must target labels or conventions, not noise |
| 20–35% | about 10–20% | Both matter |
| Over 35% | over about 20% | Seed noise matters. Averaging and a timing output are both worth pursuing |

## Part B: the a cappella preset

**Grid:** 9 settings at 15 dB, snap both ways.

| | Window 0.05 s | Window 0.10 s | Window 0.15 s |
|---|---|---|---|
| *f* = 0.75 | ✓ | ✓ | ✓ |
| *f* = 0.875 | ✓ | ✓ | ✓ |
| *f* = 1.0 | ✓ | ✓ | ✓ |

15 dB is fixed rather than searched, because it is already known to suit these sets better than 20 dB (Molina 0.612 against 0.602, A1 0.478 against 0.474, A2 0.566 against 0.560). At *f* = 1.0 a paired start lands on the vowel peak. Lone peaks behave as they do now.

**Tuning sets:** vocadito A1, vocadito A2, Molina and SSVD train&valid. **Selection score:** the plain mean of the four sets' COnPOff@50.

SSVD train&valid is included because it is separate recordings with the same onset convention as the test set: the start of pitched sound, marked where the second harmonic appears. Phoneme-informed's thresholds were tuned on the same recordings.

Every setting and the control use the outputs Part A leaves us: averaged if adopted, otherwise the 3-seed mean.

### Rule B, fixed now

Each setting is judged against the control, and the highest-scoring setting that passes becomes the **preset candidate**. A setting passes if all of these hold:
1. The selection score rises by at least +0.01, with the interval excluding zero. The bootstrap resamples recordings within each set and keeps vocadito's two annotators together.
2. On each of the four sets, COnPOff is not below the control's with the interval excluding zero.
3. COnP@150 falls by at most 0.005, and notes in silence rise by at most 0.5 points, both on the mean of the four sets.

If two passing settings are within 0.002, the one nearer the control wins: the wider window, then the smaller *f*.

**If no setting passes, the control stays, no check is spent, and this plan ends.** No further variant is tried without a new written plan.

## The check

The preset candidate is frozen and hashed first: its settings, its code and whether it uses averaged outputs. Claude asks before spending the check. It runs once, on SSVD test. A candidate that fails is not re-tuned and re-tested on this set.

**Decoded at the check, and not before:**
- the preset candidate;
- the control, as a 3-seed mean: the row on the About page today;
- transcPybe v1 at its defaults;
- phoneme-informed, GAME, MIR-ST500 and Omnizart, through `bench.py` with their usual adapters.

Pitch tracks are computed once per recording, on the GPU, and shared by every transcPybe and boundary decode. If a recording fails for any system, it is dropped for every system, and this is reported.

**Validity check, before anything else is read:** phoneme-informed's COn must be within 0.01 of the paper's 0.9569 on this set (Table 1). COn does not depend on pitch rounding, so it should reproduce, as it did on Molina (0.9308 against 0.9305). If it does not, stop and find out why.

### Adoption, fixed now

The candidate is **confirmed as the a cappella preset** if all of these hold against the control:
1. COnPOff@50 is higher, with the interval excluding zero.
2. COnP@150 falls by at most 0.005.
3. Notes in silence rise by at most 0.5 points.

Otherwise the control stays.

### Against phoneme-informed, fixed now

This is the reason for the plan, but it gates nothing. The result is stated in these words only, from the paired interval on COnPOff@50:
- **Ahead:** the interval is entirely above zero.
- **Level:** the interval includes zero.
- **Behind:** the interval is entirely below zero.

**What that verdict can and cannot show.** SSVD is phoneme-informed's own corpus, with the same annotators and conventions it trained on, so it has a home advantage on this set, as the MIR-ST500 system does on MIR-ST500. Level would be a good result. Behind would not reverse a lead elsewhere, and ahead would be strong evidence. Molina and vocadito are reported next to it and labelled as tuned on.

### Reported, deciding nothing

- each seed, and every setting in the grid on each tuning set;
- COn@50, COn@100, COn@150, COnP@50, COnP@150 and COnPOff@150 for each system;
- the stage split, and each system's median start error and IQR;
- on Part A, the 0.4 threshold for the averaged outputs, because averaging lowers a peak when seeds disagree;
- on Part A, the same averaging for frozen candidate 4's networks on the RWC-P development half.

## Not in this plan

- No retraining, no new data and no CSD3 time.
- No use of the RWC-P test half, and no MIR-ST500 check. If averaging is adopted, using a MIR-ST500 check for it is a separate decision.
- No change to the separated-pop setting, except averaging if Rule A adopts it.
- No change to whole-semitone output.
- No change to thresholds, the pairing window, `sustain_merge_above` or checkpoints.
- SSVD stays on musix and is used for scoring only. Our copy has no licence file, so it is not trained on and nothing from it is committed.
- Changing the demo and the About page is a separate step, after the check.

## Steps

1. **Code:** boundary 1424b78, musix copy `code-1424b78`. It is 714e6f9's code plus `infer.averaged_posteriors`, the `--average-with` option in `rwc_dev.py` and a test that averaging one network gives that network's outputs. `arms/acap_preset.py`, derived from `arms/r2g_acap.py`, decodes the a cappella sets.
2. **Musix, before any decode:** this plan is hashed (`acap-preset-plan.md` and `.sha256` in `~/boundary-work-20261020`).
3. **Reproduction, before anything new:** the new code must give identical notes to `new-r2lmg0.75-r20-s0` on Molina and to `r2lmg-s0-B-plain-t0.5-lm0.5-sm99-w15b-r20` on the RWC-P development half. If it does not, stop.
4. **Part A:** pitch tracks for SSVD train&valid, then the averaged decodes and scoring, in new work folders `orchestrator/bench-work-{vocadito,molina,ssvd-dev}-preset-20261030`. RWC-P uses transcPybe d96e83b and the a cappella sets use 4f35a6e, as the existing runs do. The reported averaging of frozen candidate 4's networks uses its own frozen tree, 3361803. Rule A and the diagnostic go in `arms/seeds.txt`.
5. **Part B:** the grid and scoring. Rule B goes in `arms/acap-preset.txt`.
6. **Only if there is a preset candidate:** freeze and hash it, ask the user, then run the check in a new work folder. The results go in `arms/acap-check.txt`.
7. **Cost:**
   - Parts A and B: no CSD3 time, about an hour on musix. The landmark decodes took about 4 s per recording for 30 decodes (`arms/r2g-acap.log`), and this needs about 40 for 145 recordings. A few minutes of GPU cover the pitch tracks.
   - The check: about 1–2 hours on musix, mostly the rival systems. This is an estimate, and GAME is the unknown. It is not run alongside other heavy jobs.
