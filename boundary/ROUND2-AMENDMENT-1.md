# Round 2, amendment 1: decoding the landmark networks on the frame grid

Written 29 October 2026, **after** round 2's results had been seen (`arms/round2.txt` on musix). It is hashed on musix before any landmark network is decoded again.

## What went wrong

Round 2's landmark arm (lm) scored far below the control on the RWC-P development half. At its best fraction, *f* = 0.5, COnPOff@50 was −0.037 (significant) and COnP@150 −0.030. The decodes, not the networks, look to blame:

- **The networks trained normally.** Articulation-onset F was 0.73, vowel-onset F 0.74–0.75, and the median onset timing within 1 ms of the labels (`csd3-runs/r2-lm/seed*/log.jsonl`).
- **Same-pitch notes were merged about twice as often** as with the control: 1,622–2,346 against 965. The decodes also found fewer notes: 14,900–16,100 against 16,900.
- **The scores do not change smoothly with *f*.** 0.5 was best, 0.25 and 0.75 worse, and 0.35 and 0.65 worst. That is the order in which an interpolated onset misses the 10 ms frame grid.

The cause is `infer.landmark_onsets` in boundary 81f34b4. It placed each onset at articulation + *f* × (vowel − articulation) without rounding it to a frame. transcPybe's decoder works on the f0 grid, and an onset between frames stops its same-pitch merge seeing the split the onset made. `infer.py`'s own documentation already warned of this for the single-onset hand-over.

## The fix

Each paired onset is rounded to the nearest frame (half a frame rounds up) and given as that frame's time, exactly as `onset_track` gives a peak's. Lone vowel and articulation peaks were already on frames. Nothing else in the hand-over changes. The fix is in the boundary commit that follows 81f34b4 (named in the run logs). A test checks every fraction against the frame-grid times.

Unchanged:
- the networks (`runs/r2-lm`, the chosen checkpoints);
- the fractions (0.25, 0.35, 0.5, 0.65, 0.75), the 0.5 peak threshold and the 250 ms pairing window;
- the decoder settings: candidate 3 at 20 dB, and 15 dB reported;
- f50-vi's and the control's runs, which do not use this code;
- round 2's rules and reports.

## What is decoded

lm's 3 seeds at the five fractions, at 20 dB and 15 dB, on the RWC-P development half (runs named `r2lmg-s<s>-…`). vocadito and Molina are decoded too, reported only, in new work folders.

## Decision rules

Round 2's rules 2, 3 and 4 are applied again with these decodes in place of the first lm decodes. f50-vi's result stands as recorded (+0.0022, not significant). Frozen candidate 4 is not frozen until this is judged: if lm at its best *f* is eligible under rule 3, it becomes frozen candidate 4; otherwise rule 4 freezes candidate 3 at 20 dB.

**This is the only re-decode.** If lm fails again, no further decoder variant is tried without a new written plan.

## Steps

1. This amendment is hashed on musix (`round2-amendment-1.md` and `.sha256` in `~/boundary-work-20261020`).
2. The fixed code is copied to musix. lm is decoded and scored on the development half (`arms/decode-r2g.sh`), then vocadito and Molina (`arms/acap-r2g.sh`). The rules go in `arms/round2g.txt`.
3. **Cost:** no GPU and no CSD3 time, about an hour on musix.
