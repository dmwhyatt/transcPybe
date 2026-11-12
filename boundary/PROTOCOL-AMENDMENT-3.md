# RWC-P protocol, amendment 3: COnPOff as the selection metric

Written 2026-10-26, before COnPOff had been computed for any candidate but the one named below, and before any learned-model candidate had been decoded on the test half.

**Why.** The user's priority is COnPOff, which needs a note's end to be right as well as its onset and pitch. `dev/score.py` has never scored note ends. On the development half, transcPybe is close to GAME on COnP@150 (−0.008, not significant) but far behind on COnPOff@50 (−0.068, significant). Note ends are therefore the main gap.

**What had been seen when this was written.** COnPOff on the development half, from `arms/conpoff.py`, for:
- the published systems;
- transcPybe as shipped and as preset B;
- one candidate: round 1's M4Singer-only network, B-plain at 0.7, which scored +0.0207 [+0.0128, +0.0279] against B.

No other candidate's COnPOff had been computed. The test half has not been decoded by any learned-model candidate.

**The metric.** COnPOff@50 is mir_eval's standard note F-measure with offsets, computed per song and averaged over the 48 note-level songs. A transcribed note matches a reference note when:
- its onset is within 50 ms;
- its pitch is within 50 cents;
- its offset is within 20% of the reference note's length, or 50 ms, whichever is larger.

It is computed by a separate script from the same MIDI files as `dev/score.py`, which stays as it is. That script must reproduce `dev/score.py`'s COnP@150 and COnP@50 for B exactly, as `arms/conpoff.py` does.

**Development half.** Amendment 2's rules stand, with COnPOff@50 in place of COnP@150 and one guard added. A candidate may go to the test half only if all four hold:
1. it beats B by at least 0.01 COnPOff@50 (on the mean over its 3 seeds, paired over songs), with the 95% bootstrap interval excluding zero;
2. at least 2 of its 3 seeds beat B on COnPOff@50 on their own;
3. amendment 2's two guards hold: frame overall accuracy falls by no more than 0.005, and notes in silence rise by no more than 0.5 points;
4. **new guard:** COnP@150 falls by no more than 0.005 against B, so that better ends are not bought with worse onsets or pitches.

**Applies to:**
- every candidate scored from now on, including those of the sounding plan (`sounding-plan.md`), whose own rules used COnP@150 and are replaced by these before any of its decodes;
- every candidate already scored on the development half: round 1's 18, round 1's final checkpoints, the decoder plan's and the pitch plan's. They are re-judged under these rules from their existing MIDI files, and all are listed, including those that fail.

**Test half.** Amendment 2's limit stands: at most two candidates over the whole learned-model line, none used so far. Each is frozen and hashed before its check, and checked once with all 3 seeds against B's test-half replay. It is confirmed if:
- the 3-seed mean beats B on COnPOff@50 with the 95% interval excluding zero;
- the three guards hold on the test half.

Claude asks before spending a check. A candidate that fails is not re-tuned and re-tested.

**Also reported, not gating:**
- COnP@150, COnP@50, COn@150 and COnPOff@150;
- seg_errors' error types;
- GAME, MIR-ST500 and the other published systems on COnPOff, for context.
