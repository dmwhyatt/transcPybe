# Hand-over plan: fewer false starts from the landmarks

Written 30 October 2026, after Parts A and B of `acap-preset-plan.md` (sha 6bf6aca2…) and the start diagnostics (`arms/leak.txt`, `arms/cands.txt`, `arms/false-starts.txt`), and before any decode with these settings. The user chose this stage on the same day. No network is trained. Hashed on musix before any decode.

## Why

The landmarks already hold most right note starts, but there are too many of them. Starts handed to the decoder, averaged outputs, threshold 0.5:

| Set | Recall | Precision | Best F, by threshold | Finished notes, COn@50 |
|---|---|---|---|---|
| Molina | 0.960 | 0.779 | 0.900 at 0.7 | 0.835 |
| vocadito A1 | 0.761 | 0.817 | 0.784 at 0.5 | 0.728 |
| vocadito A2 | 0.900 | 0.814 | 0.861 at 0.6 | 0.799 |
| SSVD train&valid | 0.918 | 0.764 | 0.845 at 0.6 | 0.802 |
| RWC-P development half | 0.818 | 0.623 | 0.708 at 0.6 | 0.689 |

Two findings point at the hand-over:
- **Lone peaks are the weak ones.** A start paired from both outputs lands on a reference start 86–92% of the time a cappella and 74% on RWC-P. Lone vowels manage 45–65% (35%), and lone articulations 30–61% (32%). Lone peaks are 20–28% of all starts.
- **False starts crowd real ones.** A cappella, 70–77% of false starts lie 50–150 ms from a reference start.

The mechanism is in transcPybe's `snap_segment_starts`. With snapping both ways, a note start goes to the nearest onset, whatever its strength or kind. So a stray peak near a right one can take the start: on Molina, 8.4% of reference notes have a right landmark and a note whose start is off. Every onset may also cut a note. The decoder's cut bar is half the recording's median peak strength, which any peak above 0.5 clears, so it filters nothing here.

## What has been seen

All four sets are tuning sets, and the diagnostics that suggested these changes used them. Scores here are flattered, and nothing here counts as confirmation. This plan touches no test data: not SSVD test, the RWC-P test half or the MIR-ST500 songs. It spends no check.

## Scoring

As `acap-preset-plan.md`: whole-semitone MIDI against references as published, COnPOff@50 per recording and averaged, paired 95% bootstrap intervals (10,000 resamples, seed 0).

## What is decoded

The averaged outputs of round 2's three landmark networks, on a 3 × 3 grid.

**Peak threshold:** 0.5, 0.6 or 0.7, on both onset outputs.

**Lone peaks:**

| Name | What happens |
|---|---|
| `all` | As now: a lone peak is treated like a paired start |
| `drop` | A lone peak within 150 ms of a paired start is removed. Other lone peaks are unchanged |
| `snap` | As `drop`, and the other lone peaks may time a note but never cut one (NaN strength) |

`t0.5 all` is **the control** on each set. Each preset keeps everything else:
- **A cappella:** `frozen-acap-1.json`'s decoder (*f* = 0.875, snap 0.15 s both ways, 15 dB) on vocadito, Molina and SSVD train&valid. The control is that candidate.
- **Separated pop:** *f* = 0.5, snap 0.15 s both ways, 20 dB on the RWC-P development half. The control is the averaged row from Part A.

The pairing window and every other decoder setting stay as they are.

## Decision rules, fixed now

### Rule C: a cappella

Each of the 8 other settings is judged against the control on the selection score, the plain mean of the four tuning sets' COnPOff@50. A setting passes if all of these hold:
1. The selection score rises by at least +0.01, with the interval excluding zero. The bootstrap resamples recordings within each set and keeps vocadito's two annotators together.
2. On each of the four sets, COnPOff is not below the control's with the interval excluding zero.
3. COnP@150 falls by at most 0.005, and notes in silence rise by at most 0.5 points, both on the mean of the four sets.

The highest-scoring passing setting becomes **the a cappella candidate**. If two passing settings are within 0.002, the one nearer the control wins: the lower threshold, then `all`, `drop`, `snap` in that order.

### Rule D: separated pop

Each of the 8 other settings is judged against the control on the RWC-P development half. A setting passes if all of these hold:
1. COnPOff@50 rises by at least +0.01, with the interval excluding zero.
2. COnP@150 falls by at most 0.005.
3. RPA falls by at most 0.005.
4. Notes in silence rise by at most 0.5 points.

The highest-scoring passing setting becomes **the separated-pop candidate**, with the same tie rule.

The two rules are judged separately, because each preset is a separate setting.

### If nothing passes

That preset stays as it is. No further variant is tried without a new written plan.

### Reported, deciding nothing

- every setting on every set: COnPOff@50, COn@50, COnP@50 and the number of notes;
- the averaged candidate 4 networks on RWC-P (0.3388), beside the separated-pop result;
- **doubled peaks:** of the false starts 50–150 ms from a reference start, the share whose reference start also has a right start within 50 ms. The diagnostics suggest these are second peaks for one note, but have not shown it.

## After the rules

- **A cappella candidate:** frozen and hashed as `frozen-acap-2.json`. It replaces `frozen-acap-1.json` as the candidate for the SSVD test check, whose terms in `acap-preset-plan.md` are otherwise unchanged. Claude asks before spending it.
- **Separated-pop candidate:** frozen and hashed as `frozen-sep-1.json`. Whether to use a MIR-ST500 check on it is a separate decision. Claude asks first.
- Changing the demo and the About page is a separate step.

## Steps

1. **Code:** boundary 1fd3d6b, musix copy `code-1fd3d6b`: 1424b78 plus the `lone` option in `infer.landmark_onsets` and `--lone` in `rwc_dev.py`, with tests, the default unchanged. `arms/acap_handover.py`, derived from `arms/acap_preset.py`, decodes the a cappella sets.
2. **Musix, before any decode:** this plan is hashed (`handover-plan.md` and `.sha256` in `~/boundary-work-20261020`).
3. **Decode:** the grid, in new work folders `orchestrator/bench-work-{vocadito,molina,ssvd-dev}-handover-20261030`, and as `ho-lm-…` runs on RWC-P. RWC-P uses transcPybe d96e83b and the a cappella sets 4f35a6e, as before.
4. **Reproduction, before anything is scored:** `t0.5 all` must give identical notes to `f0.875-w0.15-avg` on all three a cappella folders and to `avg-lm-B-plain-t0.5-lm0.5-sm99-w15b-r20` on RWC-P. If it does not, stop.
5. **Rules and reports:** `arms/handover.txt`.
6. **Cost:** no GPU and no CSD3 time, under an hour on musix.
