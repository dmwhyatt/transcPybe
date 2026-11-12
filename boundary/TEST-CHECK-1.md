# Test check 1: frozen candidate 2 on the RWC-P test half

Written 26 October 2026, before any learned-model candidate has been decoded on the test half. This is the first of the two test checks protocol amendment 2 allows; the user approved spending it on frozen candidate 2.

**The candidate.** `frozen-candidate-2.json` (sha256 4bc9d2be…), exactly as frozen:
- networks: the m4gts-aug chosen checkpoints, seeds 0–2 (hashes in the record);
- decoder: B-plain, onset threshold 0.5, every onset may cut, `sustain_merge_above` 0.99;
- code: boundary 7432d9a and transcPybe 3361803, from the musix copies the development runs used.

**How it is decoded, with no code change.** `rwc_dev.py` reads its song list, pitch tracks and output folder from a work folder. A work folder is built for the check:
- its song list holds the 50 test songs;
- its pitch-track cache links to the test half's cache (`dev/f0cache/sp-test`), which preset B's test replay used;
- audio, references and `dev/frozen.json` link to the RWC work folder.

The same command as the development runs then decodes the three seeds.

**A check before any test decode.** The same kind of work folder is built for the development half, and the three seeds are decoded through it. Their MIDI files must be byte-identical to frozen candidate 2's development runs (`m4gts-aug-s<s>-B-plain-t0.5-sm99`). If any differs, nothing is decoded on the test half until the cause is found.

**The rule (protocol amendment 3), fixed before this was written.** Preset B's test replay (`dev/runs/sp-test/B-full`) is the baseline, on all 50 test songs. The candidate is confirmed if:
1. the 3-seed mean beats B on COnPOff@50, with the 95% bootstrap interval over songs excluding zero (10,000 resamples, seed 0);
2. frame overall accuracy falls by no more than 0.005;
3. notes in silence rise by no more than 0.5 points;
4. COnP@150 falls by no more than 0.005.

There is no +0.01 bar on the test half. The scores come from `dev/score.py` (frame accuracy, silence, COnP) and from the same COnPOff computation as the development half (mir_eval: onset 50 ms, pitch 50 cents, offset within 20% of the reference note or 50 ms).

**Reported alongside, deciding nothing:**
- each seed;
- COnP@150, COnP@50 and COnPOff@150;
- GAME and MIR-ST500 on the same songs, from their existing test outputs.

**Whatever the result,** the candidate is not re-tuned and re-tested. One test check remains.
