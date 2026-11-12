# Decision record: a slide or ornament within one syllable is one note

Recorded 30 October 2026. **The decision is the user's, made in conversation that day after seeing the results below. It is not blind.** This record was hashed on musix before `frozen-sep-1.json` was frozen and before anything was decoded under it.

## The decision

When a singer slides or ornaments within one syllable, the transcription shows **one note, not several**. The output is a melody line, and these movements are part of how a note is sung.

The question had to be decided because the two RWC-P references disagree. The note reference holds one note across such a movement, and the F0 reference records the movement. No measure on RWC-P can settle which is right.

## What led to it

From `arms/false-marks.txt` and `arms/handover.txt`, on the RWC-P development half:
- 27.4% of the false landmarks (2,028 of 7,404) lie inside a reference note where the pitch moves by at least half a semitone. The F0 reference marks singing at 97% of them, so these are real movements, and 75% are lone peaks.
- The `snap` setting stops lone peaks cutting notes, so notes stop following those movements. Under `handover-plan.md` (sha 4043b11c…), `t0.5 snap` had the highest COnPOff@50 of any setting, but it failed Rule D's third condition, the only one it failed:

| Measure, against the control | `t0.5 snap` |
|---|---|
| COnPOff@50 | 0.3469 against 0.3362: +0.0108 [+0.0054, +0.0164] |
| COnP@150 | +0.0141 |
| Notes in silence | −0.16 points |
| RPA | 0.6413 against 0.6517: −0.0104 [−0.0119, −0.0088] |

- It makes 328.4 notes per song, against 344.4 for the control and 323.8 in the note reference.

## What changes

1. **Rule D's third condition is set aside for this one result.** That condition, RPA falling by at most 0.005 against the control, was this plan's own and stricter than the protocol's. Rule D's other conditions stand, and `t0.5 snap` meets them.
2. **`t0.5 snap` becomes the separated-pop candidate,** frozen and hashed as `frozen-sep-1.json`, as `handover-plan.md` provides. Its settings are the averaged outputs of round 2's landmark networks, threshold 0.5, `lone="snap"`, *f* = 0.5, snap 0.15 s both ways, 20 dB.
3. **Later plans on separated pop take this decision as given.** They judge RPA against preset B, as amendment 4 does, rather than against the candidate they are trying to beat. Otherwise any change that follows fewer ornaments would fail by construction.

## What does not change

- **Protocol amendment 4's guard, which still holds.** It limits RPA against preset B, and `t0.5 snap` is +0.0004 [−0.0027, +0.0035] (0.6413 against 0.6409). So the MIR-ST500 protocol's selection condition is met as written, and that protocol is not amended.
- **Note-level pitch guards.** COnP@150 and notes in silence still guard against real damage to pitch and to where notes fall.
- **Whole-semitone output.**
- **The a cappella candidate.** `frozen-acap-2.json` stays as Rule C chose it. Rule C has no RPA condition, and its winner, `t0.6 drop`, still lets lone peaks far from a pair cut notes. So the a cappella preset does not yet fully follow this decision. Changing that needs a new written plan.
- **The demo and the About page,** which are a separate step.

## What it costs, and what would undo it

- **Frame-level pitch follows the voice less closely:** RPA falls by 0.0104 and overall accuracy from 0.7639 to 0.7582. RPA remains level with GAME's 0.6422 on this half and above phoneme-informed's 0.6019 and MIR-ST500's 0.5956. Anyone needing the pitch detail should use the F0 track rather than the MIDI.
- **The result is unconfirmed.** The candidate was chosen on the development half, after a rule was set aside with the result in view, so its score there is flattered. It has not been run on the RWC-P test half, which is spent, or on the MIR-ST500 songs. Claude asks before spending a MIR-ST500 check.
- **If a MIR-ST500 check fails,** the candidate is not re-tuned and re-tested on that set. The decision about what a note is stands unless the user changes it, but this setting would not ship as confirmed.
