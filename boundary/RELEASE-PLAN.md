# Release plan: ending notes later before a rest

Written 27 October 2026, after the ends recheck (`ends-recheck.md`, sha 7c2a0d84…; `arms/ends3.txt` and `~/singing-data-20261027/ends-data.txt` on musix), and before any decode with these settings. Judged by protocol amendment 4 (COnPOff@50), with the cross-annotator guard of the round-2 draft.

## Why

On the RWC-P development half, frozen candidate 3's notes with the right start and pitch have:

| System | Right end | Early end | Late end |
|---|---|---|---|
| Frozen candidate 3 | 59% | 22% | 19% |
| GAME | 62% | 9% | 29% |
| MIR-ST500 | 65% | 18% | 17% |

Candidate 3's early ends are a median 120 ms early, and three-quarters of them are notes before a rest, not legato.

The labels are not the cause. Measured against the audio's voicing offset, M4Singer's training labels end notes 19–24 ms after voicing stops, and RWC-P's references 16 ms after. So the decoder ends notes early.

The likeliest step is transcPybe's release trim. It ends a note at the first frame where its level has fallen `release_drop_db` (15 dB) below the note's own sustained level, taking at most `release_min_fraction` (half) of the note. 15 dB was chosen on the synthetic corpus, and preset B's other settings were chosen on COnP@150, which does not score ends. The trim has never been tuned on separated pop.

## What is decoded

Frozen candidate 3, exactly as frozen:
- code: boundary 7432d9a and transcPybe 3361803;
- the m4gts-aug chosen checkpoints, each seed with its own outputs;
- B-plain, threshold 0.5, every onset may cut, `sustain_merge_above` 0.99, `onset_snap_window_s` 0.15, `onset_snap_direction` both.

It is decoded on the RWC-P development half at four settings:

| Setting | Release trim |
|---|---|
| `r15` | on, 15 dB: frozen candidate 3's existing runs, not decoded again |
| `r20` | on, 20 dB |
| `r25` | on, 25 dB |
| `roff` | off (`trim_note_releases` false) |

This makes 9 new decodes (3 settings × 3 seeds), run 3 at a time while GAME's MIR-ST500 run shares musix.

**Cross-annotator guard.** vocadito (annotators A1 and A2, scored separately) and Molina are decoded at all four settings. This uses the a cappella runs' code (transcPybe 4f35a6e, boundary 956a9ce; `net3_bench.py`'s configuration, with `r15` being candidate 3 there). The guard compares each setting with `r15` decoded by that same code.

## Decision rules, fixed now

Each comparison takes each song's (or recording's) score averaged over the 3 seeds, pairs it, and uses a 95% bootstrap interval (10,000 resamples, seed 0).

1. **Against B:** each of `r20`, `r25` and `roff` is judged by amendment 4's conditions:
   - COnPOff@50 at least +0.01 over B, with the interval excluding zero;
   - at least 2 of 3 seeds above B;
   - RPA down by at most 0.005;
   - notes in silence up by at most 0.5 points;
   - COnP@150 down by at most 0.005.
2. **Against frozen candidate 3:** the highest passing setting becomes **frozen candidate 4** only if both hold:
   - it beats `r15` on COnPOff@50 with the interval excluding zero;
   - its cross-annotator guard holds: on each of vocadito A1, vocadito A2 and Molina, its COnPOff is not below `r15`'s with the interval excluding zero.

   Frozen candidate 4 is frozen and hashed before anything else. It is eligible for one of the MIR-ST500 protocol's two checks. Claude asks before spending it. It may instead be combined with round 2's outcome; that is the user's call.
3. **Otherwise** frozen candidate 3 stays.

**Reported, deciding nothing:**
- each setting against `r15`, on COnPOff@50, COnP@150, COnPOff@150, RPA, OA, voicing recall and false alarm, and notes in silence;
- the ends breakdown of `arms/ends3.py` for each setting: right, early and late ends, and how many early ends come before a rest;
- the a cappella COn, COnP and COnPOff for each setting and annotation set.

## Steps

1. **Musix, before any decode:** this plan is hashed (`release-plan.md` and `.sha256` in `~/boundary-work-20261020`).
2. `arms/decode-release.sh`: the 9 development decodes, scoring with `dev/score.py`, then `arms/release.py` (the rules and the development-half reports, written to `arms/release.txt`).
3. The a cappella decodes and scoring, in new work folders `orchestrator/bench-work-{vocadito,molina}-release-20261027`, then the guard (`arms/release-acap.txt`).
4. **Cost:** no GPU and no CSD3 time, about an hour on musix.
