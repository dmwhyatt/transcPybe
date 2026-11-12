# Ends recheck: note ends on frozen candidate 3, and where each set puts its ends

Written 27 October 2026, before any measurement with it. Two diagnostics. They select nothing and spend no check. They inform one decision: whether the round-2 plan (`ROUND2-PLAN.md`, a draft, not yet hashed) should change how note ends are labelled or decoded.

## Why

On the RWC-P test half (test check 2), of frozen candidate 3's notes with the right start and pitch (COnP@50 matches):

| System | Share that also have the right end |
|---|---|
| Frozen candidate 3 | 56.8% |
| GAME | 59.1% |
| MIR-ST500 | 61.1% |

That is roughly 0.012 of COnPOff@50 against GAME and 0.022 against MIR-ST500. The ends diagnostic (26 October) found:
- wrong ends were about half early and half late;
- late ends mostly ran a legato note into the next one;
- snapping ends to the offset output's peaks gained little.

Since then, snapping starts both ways has moved shared legato boundaries, and round 2 would move starts again. The labels for note ends have never been checked against the audio.

## A. Ends on frozen candidate 3 (RWC-P development half)

Run as `arms/diag_ends.py` was, on `m4gts-aug-s{s}-B-plain-t0.5-sm99-w15b`, 3 seeds pooled. The network outputs are each seed's own, cached in `arms/post-m4gts-aug-s{s}/`; start_post.py's M0 check confirmed them. For the notes whose start (50 ms) and pitch match, as COnP@50 counts them, it reports:
- right, early and late ends, by COnPOff's rule (within 20% of the reference note's length or 50 ms, whichever is larger);
- late ends: how many run into the next transcribed note, and how many have a reference gap of more than 50 ms after them;
- early ends: how many run into the next note;
- for wrong ends, whether an offset peak (0.5) lies within the tolerance of the true end, and whether one is reachable: after the note's start and before the next note.

Also reported:
- the same shares for frozen candidate 2, preset B, GAME and MIR-ST500 (without the peak measures for the last three);
- for candidate 3 and candidate 2, how many matched notes changed from right to wrong end, or wrong to right, between them.

**End snapping on top of candidate 3, reported only.** Candidate 3's settings are decoded with boundary 956a9ce and transcPybe d96e83b, the ends plan's code, at `end_snap_window_s` 0 (e0), 0.10, 0.20 and 0.30, 3 seeds each.
- **Check:** e0 must reproduce candidate 3's development runs byte for byte.
- **Reported against e0:** COnPOff@50, COnP@150, RPA and notes in silence, each with its 95% interval; also amendment 4's conditions against B.

Nothing is selected. A decoder change for ends would need its own written plan.

## B. Where each set puts its note ends

The yardstick is the start-convention check's (`~/singing-data-20261027/conventions_data.py`), turned around. For every labelled note end followed by a rest (the next labelled note starts at least 100 ms later, or there is none), it measures the voicing offset near the end: the first unvoiced frame of transcPybe's pYIN track (16 kHz, 10 ms, voicing at least 0.3), after at least 5 voiced frames and followed by at least 3 unvoiced ones. The search runs from 350 ms before the label to 100 ms after it, and the label must not lie in a later voiced stretch.

The measure is label end minus voicing offset; negative means the label ends earlier. Ends are grouped by how the syllable closes:
- a vowel;
- a nasal coda (Mandarin finals ending -n or -ng, Japanese N, English or Korean final m, n or ng in CSD);
- another consonant (CSD only).

Measured on:
- **M4Singer** (400 clips, the same seed-0 sample): MIDI ends;
- **PJS:** the end of the phone before a pause;
- **CSD** (50 recordings): note ends;
- **Annotated-VocalSet** (300 files): the end of a 'Sound' segment before a 'Rest';
- **RWC-P's development half**, its note references, against two yardsticks: pYIN on the separated stems, and the hand-annotated F0 reference's own voicing.

vocadito, Molina and MIR-ST500 are not measured. They serve as guards or held-out sets, and their labels are not used to design training labels.

## How it is read

There are no fixed thresholds, because this informs the round-2 draft rather than deciding anything. It will be summarised for the user with four questions:
- is candidate 3's end problem now mostly early, late, legato or before rests;
- do the offset peaks mark the right ends;
- does end snapping help on top of candidate 3;
- do the training sets' end conventions differ from RWC-P's by enough, in the direction of the errors, to justify an end-label change in round 2?

## Steps

1. This note is hashed on musix (`ends-recheck.md` and `.sha256` in `~/boundary-work-20261020`).
2. A: `arms/decode-ends3.sh` (12 decodes, then scoring) and `arms/ends3.py`, writing `arms/ends3.txt`.
3. B: `~/singing-data-20261027/ends_data.py`, writing `ends-data.txt`.
4. No GPU, about an hour on musix alongside GAME's run.
