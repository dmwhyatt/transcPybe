# Sounding plan, amendment 1

Written 26 October 2026, before any decode with the sounding merge.

**What prompted it.**
- Protocol amendment 3 made COnPOff@50 the selection metric and replaced this plan's rules with its own.
- Re-judging the development half under it (`arms/rejudge.txt`) passed 34 of 44 candidates. It froze the highest, round 1's both-datasets-with-compression network at B-plain 0.5, as `frozen-candidate-1.json` (COnPOff@50 0.2841, +0.0252 against B).
- The user chose to run this plan and an ends plan before spending a test check.

**1. A third hand-over.** The "network" and "own" hand-overs stay as written. A third is added: frozen candidate 1's networks and decoder.

| Hand-over | Networks | Decoder | Onsets |
|---|---|---|---|
| frozen | m4gts-aug, chosen checkpoints, 3 seeds | B-plain, onset threshold 0.5, every onset may cut | the network's |

It runs at the same four settings (`s0`, `s95`, `s98`, `s99`), and each seed uses its own outputs. Its `s0` must reproduce frozen candidate 1 exactly (COnPOff@50 0.2841) before anything is judged.

This makes 36 decodes and nine candidates: the three hand-overs at `s95`, `s98` and `s99`.

**2. Decision rules.**
1. **Against B:** each candidate is judged by protocol amendment 3's four conditions.
2. **Against frozen candidate 1:** the highest passing candidate replaces frozen candidate 1 as the one to test only if it also beats it on COnPOff@50 (seed means, paired over songs) with the 95% interval excluding zero. Otherwise frozen candidate 1 stays. This stops a small development-half difference, among many candidates, from choosing what is tested.
3. **No test check yet:** none is spent after this plan. That waits for the ends plan and the user's go-ahead.

**3. Reported, deciding nothing:**
- each hand-over's settings against its own `s0`;
- COnPOff@150, COnP@150, COnP@50 and COn@150;
- notes transcribed, and the note-error breakdown;
- the boundary diagnostic on every setting's notes.
