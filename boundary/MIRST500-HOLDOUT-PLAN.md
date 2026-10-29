# MIR-ST500 held-out protocol

Written 27 October 2026 and reviewed by the user. It was hashed on musix before any system was scored on these songs and before any transcPybe or boundary decode of them.

## Why

RWC-P's test half is spent: test check 1 confirmed frozen candidate 2, and test check 2 confirmed frozen candidate 3. The start-timing recheck (`start-timing-recheck.md`, sha 069f9ac3…) found that 81% of candidate 3's remaining start errors are the network's own mistimed onsets. So the next gain needs a retrained network. That network can still be selected on the RWC-P development half under protocol amendment 4, but it needs a fresh held-out set to confirm it. This protocol makes MIR-ST500's test split that set.

## What has been seen so far

- Whether each test song's YouTube link still works: 82 of 100 do.
- The audio's durations, file sizes and hashes.
- How many test songs and notes differ between the 2020-10-14 and current annotation releases: 70 songs, about 500 of 31,311 notes. This is a count only; no note was inspected.
- The first separated stem's format.

No system has been scored on these songs, and no transcPybe or boundary decode of them has run. GAME's and the MIR-ST500 system's outputs are being generated now (`orchestrator/bench-work-mirst500-20261027/others.sh`), but they are not scored until the first check.

## The set

- **Songs:** MIR-ST500's test split, songs 401–500. These 82 are still on YouTube:

  401, 403, 404, 405, 408–416, 418–423, 426–432, 434–436, 438–442, 444, 447, 448, 450–477, 480, 481, 483, 485–490, 492–494, 496–500.

  The other 18 are unavailable or private (402, 406, 407, 417, 424, 425, 433, 437, 443, 445, 446, 449, 478, 479, 482, 484, 491, 495). If a song fails for any system at any check, it is dropped for every system in that check, and this is reported.
- **Audio:** downloaded on 27 October with yt-dlp 2026.10.21 from the dataset's own links, as YouTube served it (opus). Each file's hash is in `~/mirst500-test-20261027/manifest.tsv`.
- **Annotations:** `MIR-ST500_corrected.json` from the dataset repository (github.com/york135/singing_transcription_ICASSP2021), commit 5460c98, git blob dbea88df. Each note is [onset, offset, MIDI pitch].
- **Lead-vocal stems:** BS-RoFormer 53-stem (`sep-bsroformer53:b24ab72`), with the same command as RWC-P. Each song is first decoded to 44.1 kHz stereo float WAV.
- **Pitch tracks for transcPybe decodes:** computed once per song, as RWC-P's were. This uses `dev/replay.py`'s f0 step, with transcPybe tree spurious-c7d3ab2, default settings and the GPU. The tracks are cached and used by every transcPybe and boundary decode.

## Never used for

Training, tuning, choosing checkpoints, thresholds or settings, designing labels, or any decision other than the checks below. No song is listened to, plotted, or inspected by hand. Nothing is decoded on these songs except at a check.

## Metric

COnPOff@50 is mir_eval's note F-measure:
- onset within 50 ms;
- pitch within 50 cents;
- offset within 20% of the reference note's length or 50 ms, whichever is larger.

It is computed per song and averaged over the songs, against the labels **as published, with no time shift**.

Also reported, not deciding:
- the same scores with the labels shifted 30 ms later (a follow-up paper by the dataset's authors reports that its onsets sit about 30 ms early against the audio);
- COnP@50, COnP@150, COn@50 and COnPOff@150.

## A check

1. **The candidate:** one candidate, frozen and hashed before its check, typically a retrained network with its decoder settings, all 3 seeds. It is decoded with its frozen code, through a work folder that points `rwc_dev.py` at these songs (as in test checks 1 and 2). Before that, the same route on the RWC-P development half must reproduce the candidate's development runs byte for byte.
2. **The incumbent:** frozen candidate 3 (confirmed by test check 2), with its frozen code and 3 seeds. It is decoded at the first check, not before, along with preset B through `dev/replay.py`'s frozen B-full.
3. **Confirmed if all hold:**
   - The candidate's 3-seed mean beats frozen candidate 3's 3-seed mean on COnPOff@50. Each song's score is averaged over seeds and paired over songs, and the 95% bootstrap interval (10,000 resamples, seed 0) must exclude zero.
   - COnP@150 falls by no more than 0.005 against frozen candidate 3.
   - Notes in silence rise by no more than 0.5 points against frozen candidate 3. MIR-ST500 has no pitch-track reference, so a transcribed note counts as in silence if at least 80% of its span lies outside every reference note. The share is taken per song and averaged.
4. **There is no frame-level guard.** MIR-ST500 has no F0 reference, so frame pitch accuracy and overall accuracy cannot be computed. The RWC-P development half's pitch-accuracy guard (amendment 4) still applies at selection.
5. **Reported, deciding nothing:**
   - each seed;
   - preset B, and the candidate against B;
   - GAME;
   - the MIR-ST500 system, marked as having a home advantage: it trained on songs 1–400 of this dataset, with the same annotators and conventions. GAME's report says it trained on private data only.

## Limits

At most two checks over the retraining line. A candidate that fails is not re-tuned and re-tested on this set. Claude asks the user before spending each check. If no candidate is confirmed, frozen candidate 3 stays the result to ship.
