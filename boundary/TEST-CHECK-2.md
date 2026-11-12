# Test check 2: frozen candidate 3 on the RWC-P test half

Written 27 October 2026, before frozen candidate 3's settings have been decoded on the test half. This is the second and last test check that protocol amendment 2 allows. Claude asks the user before spending it.

**The candidate.** `frozen-candidate-3.json` (sha256 f7f70082…), exactly as frozen. It is frozen candidate 2 with note starts snapped to the network's onsets from up to 150 ms away, in both directions:
- **Networks:** the m4gts-aug chosen checkpoints, seeds 0–2. These are the same files as frozen candidate 2's; the hashes are in the record.
- **Decoder:** B-plain, onset threshold 0.5, every onset may cut, `sustain_merge_above` 0.99, `onset_snap_window_s` 0.15, `onset_snap_direction` "both".
- **Code:** boundary 7432d9a and transcPybe 3361803, from the musix copies the development runs used.

It was selected under protocol amendment 4, after the snap plan's results had been seen. Its development-half numbers are flattered for that reason, as well as by ordinary selection, so this check is the real test.

**How it is decoded, with no code change.** As in test check 1: a work folder whose song list holds the 50 test songs, with its pitch-track cache linked to the test half's (`dev/f0cache/sp-test`). The same command as the development runs then decodes the three seeds.

**A check before any test decode.** The same route on the development half must reproduce the development runs (`m4gts-aug-s<s>-B-plain-t0.5-sm99-w15b`) byte for byte. If any file differs, nothing is decoded on the test half until the cause is found.

**The rule (protocol amendment 4), fixed before this was written.** Preset B's test replay (`dev/runs/sp-test/B-full`) is the baseline, on all 50 test songs. The candidate is confirmed if:
1. the 3-seed mean beats B on COnPOff@50, with the 95% bootstrap interval over songs excluding zero (10,000 resamples, seed 0);
2. frame raw pitch accuracy (RPA) falls by no more than 0.005;
3. notes in silence rise by no more than 0.5 points;
4. COnP@150 falls by no more than 0.005.

There is no +0.01 bar on the test half.

**Reported alongside, deciding nothing:**
- each seed;
- COnP@150, COnP@50, COnPOff@150, and frame overall accuracy (OA);
- frozen candidate 2's test runs from test check 1, and the candidate's difference from them;
- GAME and MIR-ST500 on the same songs, from their existing test outputs.

**Whatever the result,** the candidate is not re-tuned and re-tested. No test checks remain after this one. If it is not confirmed, frozen candidate 2, confirmed by test check 1, stays the result to ship.
