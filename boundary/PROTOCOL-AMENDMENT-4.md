# RWC-P protocol, amendment 4: guard pitch accuracy instead of overall accuracy

Written 2026-10-27 at the user's decision, **after** the snap plan's results (`snap-plan.md`, `arms/snap.txt`) and the sounding plan's had been seen. It is not blind. It changes a guard after candidates have been judged by it, so the development-half numbers of whatever it selects are flattered. The test half is the safeguard: no candidate selected under this amendment has been decoded there.

## Why

Amendment 2's overall-accuracy (OA) guard was meant to stop note-level gains that damage frame-level pitch output. Two plans in a row have been blocked by it alone:
- **Sounding plan:** 8 of 9 candidates.
- **Snap plan:** all 4 candidates. COnPOff@50 rose by up to +0.031 over frozen candidate 2, while raw pitch accuracy (RPA) rose against B (+0.005 to +0.010).

The snap plan's breakdown (`arms/snap-where.txt`) shows the mechanism. RWC-P's note reference and its F0 reference disagree at note starts:
- Moving a start earlier after a gap voices frames that the F0 reference marks unvoiced: about half the moved frames, consonants and breaths.
- Moving a legato start later, onto the attack the note reference marks, unvoices frames that the F0 reference marks sung.

So any change that brings starts closer to the note reference costs OA through voicing, whatever it does to pitch. OA was guarding pitch output, and pitch itself is measured directly by RPA.

## The change

In the development-half rules of amendment 3, condition 3's first guard changes:
- **Old guard (removed):** frame OA falls by no more than 0.005 against B.
- **New guard:** frame RPA falls by no more than 0.005 against B. RPA is mir_eval's raw pitch accuracy against the F0 reference, from `dev/score.py`'s caches.

Unchanged:
- the bar (+0.01 COnPOff@50, with the interval excluding zero);
- at least 2 of 3 seeds beating B;
- the silence guard (notes in silence rise by no more than 0.5 points);
- the COnP@150 guard (falls by no more than 0.005).

OA is still reported, with voicing recall and false alarms.

## Applies to

**Every development-half candidate judged under amendment 3, re-judged from its existing MIDI files.** All are listed, including those that fail:
- round 1's chosen checkpoints (18) and final checkpoints (18);
- the decoder plan's (3);
- the pitch plan's (p1, p2f–p5f);
- the sounding plan's (9);
- the ends plan's (3);
- the snap plan's (4).

**Selection.** The highest passing candidate on COnPOff@50 becomes **frozen candidate 3**, but only if it beats frozen candidate 2 on COnPOff@50 with the 95% interval excluding zero (paired over songs, 3-seed means). Otherwise frozen candidate 2 stays. Frozen candidate 3 is frozen and hashed before anything else is done with it.

## Test half

Amendment 2's limit stands. Two checks over the whole learned-model line; one is used (test check 1, frozen candidate 2, confirmed), so one remains.

A check is confirmed if:
- the 3-seed mean beats B on COnPOff@50 with the 95% interval excluding zero;
- the silence, COnP@150 and **RPA** guards hold on the test half.

OA is reported. Claude asks before spending the check. A candidate that fails is not re-tuned and re-tested. Test check 1's result stands as recorded.
