# Round 2 training plan: onset timing

Written 27 October 2026 and hashed on musix before any round-2 code was written or any network trained. The values settled before hashing are listed at the end.

How a candidate is selected on RWC-P is fixed by protocol amendments 3 and 4 (RWC work directory on musix). How it is confirmed is fixed by the MIR-ST500 held-out protocol (`mirst500-holdout-plan.md`, sha c7d7b324…). This plan adds what is trained, how the arms are compared, and the steps. It changes neither of those.

## The question

The start-timing recheck (`start-timing-recheck.md`, sha 069f9ac3…) looked at frozen candidate 3's note starts on the development half that are 50–150 ms off.
- **81% are the network's own mistimed onsets.** Fixing them perfectly would be worth +0.054 COnPOff@50.
- **No constant shift helps.** The errors go both ways.

The decoder can no longer fix them. The network has to place its onsets better.

Two findings say the onset *labels* are a large part of that.

1. **Label position matters.** The 21 October pilots, re-scored at 50 ms (`arms/conventions.txt`), showed:

   | Label position (consonant start = 0, vowel start = 1) | Effect on COnPOff@50 | Near misses |
   |---|---|---|
   | Moving from 0 to 0.5 | +0.020 | 34% → 26% |
   | Moving from 0 to 1 | −0.027 | starts go late |

   The 0.5 used since has never been tuned.
2. **Annotators disagree about where a sung note starts.** The 27 October convention check (`~/singing-data-20261027/conventions-data.txt` on musix) measured label positions against the audio's voicing onset (pYIN):
   - CSD's human labels sit 40–70 ms later than our training labels.
   - M4Singer marks notes that open on a vowel a median 70 ms before voicing. Human-made sets put them within about 10 ms of it.

Tuning a single label convention on RWC-P risks fitting RWC-P's score-aligned references rather than anything that generalises. So round 2 trains the network to find acoustic landmarks, and leaves the convention to the decoder.

## Stage 1

### What is held fixed

Round 1's winning recipe, as frozen candidate 3 uses it:
- **Data:** M4Singer + GTSinger, each vocal mixed once over MUSDB18-HQ backing 20–35 dB down, with no separator. The shards are `m4gts-musdb/shards/` on CSD3, reused unchanged. Training is 14,362 M4Singer clips (20.4 h) plus 25,494 GTSinger clips (71.7 h): 92.1 hours in all. Validation is 4.0 h and 7.6 h respectively.
- **Training:** `--augment` (the m4gts-aug arm).
- **Network:** base size, 3 seeds (0, 1, 2), 30,000 steps, evaluation every 500 steps.
- **Checkpoint:** `train.py`'s rule, as in round 1.
- **Offset, sounding and re-sung outputs:** trained from the same half-way notes as round 1, so they are unchanged.

### The decoder: release trim at 20 dB (decided after seeing results)

The release plan (`release-plan.md`, sha 980ccf47…; `arms/release.txt`) decoded frozen candidate 3 with transcPybe's release trim loosened.
- **On the RWC-P development half:** COnPOff@50 rose at every setting: 20 dB +0.0096, 25 dB +0.0108, off +0.0107, each with the interval excluding zero. Right ends went from 59% to 61%.
- **On a cappella:** every setting lost. At 20 dB only Molina's loss is significant (−0.006). At 25 dB and off, the losses are larger and wider.
- **By that plan's rules,** frozen candidate 3 stayed.

The trim ends notes that run on through a room's reverberation. Separated stems carry little of it, so there the trim cuts real singing short. On a cappella room recordings it is still needed.

On 27 October, after seeing those results, the user decided to fold the release trim into round 2. The setting is **20 dB**: the mildest one tried, which keeps most of the gain. This is not blind. Two safeguards remain:
- the MIR-ST500 check, which is separated pop and untouched;
- the a cappella results, reported for every round-2 decode.

transcPybe's own default (15 dB) does not change. The 20 dB setting belongs to the separated-pop preset only.

**"Candidate 3 at 20 dB"** below means frozen candidate 3's networks and decoder with `release_drop_db` 20. Its development runs are the release plan's (`…-w15b-r20`).

### The arms

| Arm | Onset labels | Trained? |
|---|---|---|
| **f50** | half-way (0.5), as round 1 | **round 1's m4gts-aug networks, not retrained**: the control, and frozen candidate 3's networks. At the primary setting it is candidate 3 at 20 dB |
| **f50-vi** | half-way, and vowel-initial notes moved to their voicing onset | new |
| **lm** | two landmark outputs instead of one onset output; the convention is chosen at decode time | new |

#### f50-vi: the vowel-initial correction

f50 with one change: each note that opens on a vowel has its onset moved to the first voicing onset after it. That is the first frame whose stored voicing is at least 0.3, after at least 3 frames below 0.3, and within 150 ms of the label. A note stays where it is if:
- there is no such onset, as for a note continuing a vowel (a melisma) or following a voiced note without a gap;
- the move would leave it shorter than 30 ms.

The voicing is the stored features' own voicing column: the f0 track the network sees, computed when the shards were made.

"Opens on a vowel" follows each source's phoneme tier:
- **M4Singer:** the note's first phone is not a pinyin initial, and not silence or breath;
- **GTSinger:** `gtsinger_is_vowel` is true of its first phone.

This corrects a quirk of one dataset's labelling, measured against the audio, so it is expected to carry over to other annotators.

#### lm: two landmarks, convention at decode time

The onset output is replaced by two outputs:
- **articulation onset**: where the note's syllable starts sounding. That is the consonant's start, or for a note opening on a vowel, its labelled start;
- **vowel onset**: where the vowel starts. That is the phoneme tier's vowel start, or for a note opening on a vowel, its voicing onset by f50-vi's rule, or its labelled start if there is none.

Both targets have the same shape and weight as round 1's onset target. For a note with no consonant and no voicing onset, the two coincide. The network gains one output (five in all). Offset, sounding and re-sung targets are unchanged.

**At decode time,** the two outputs' peaks (0.5, `pick_peaks`) are paired into note starts at a fraction *f*:
- Each vowel peak takes the nearest articulation peak before it, within 250 ms (the sources' long-consonant limit), unless another vowel peak lies between them. Its onset is articulation + *f* × (vowel − articulation), with strength the larger of the two peaks.
- A vowel peak with no articulation peak gives an onset at the vowel peak.
- An articulation peak with no vowel peak within 250 ms after it gives an onset at the articulation peak.

The resulting onsets go to the decoder exactly as round 1's onset peaks do, with the same split policy and sounding output. *f* = 0.5 corresponds to round 1's convention.

Choosing *f* needs no retraining, so the convention can be checked across annotators, and later re-tuned for a different target, from one set of networks.

### The code changes

1. **Rebuilding targets** (both new arms). `train.py` gains an option to rebuild the stored targets from the manifest's notes when it loads, instead of reading them from the shard. Relabelled manifests then reuse the shards' audio features, and the expensive preparation (about 6 GPU-hours) doesn't need repeating.
2. **Phoneme facts per note** (both new arms).
   - `sources.py` records, for each note, whether it opens on a vowel and where its vowel starts.
   - `relabel.py` carries both.
3. **The vowel-initial move** (both new arms): the rule above, applied when the targets are rebuilt.
4. **Landmark targets** (lm): a fifth target and output. `train.py`'s checkpoint rule uses the articulation output's onset F in place of round 1's onset F.
5. **The landmark hand-over** (lm): pairing, as above, in `infer.py`, with an `rwc_dev.py` option for *f*.
   - With the option off, it must decode a round-1 network byte for byte as boundary 7432d9a does. This is checked on the development half before any lm decode.

**Checks before training:**
- at *f* = 0.5 with no vowel move, the rebuilt targets must equal the stored ones within `train.py`'s existing tolerance, for every recording;
- report the share of vowel-initial notes that move and the median and quartiles of the move, on the training set. If the median is not between +20 and +120 ms, the rule is not doing what it is meant to: stop and report;
- report, for lm, the median and quartiles of vowel onset minus articulation onset.

**Tests:**
- rebuilt targets match stored ones for unchanged notes, and move with moved notes;
- a vowel-initial note moves to a voicing onset within reach, and a note in continuous voicing stays put;
- a consonant-initial note is never moved by the vowel rule;
- the pairing gives the interpolated onset, a lone vowel peak, and a lone articulation peak.

### How each arm is scored

On musix, by amendment 4, on the RWC-P development half (50 songs, 48 at note level).
- **Primary setting:** candidate 3 at 20 dB's decoder:
  - B-plain, onset threshold 0.5, every onset may cut;
  - `sustain_merge_above` 0.99;
  - `onset_snap_window_s` 0.15, `onset_snap_direction` both;
  - `release_drop_db` 20.

  Each arm is decoded exactly as the incumbent, and only the onsets differ.
- **f50-vi:** decoded with the frozen code (boundary 7432d9a, transcPybe 3361803) at the primary setting. Thresholds 0.3 and 0.7 are reported.
- **lm:** decoded with the landmark hand-over at *f* = 0.25, 0.35, 0.5, 0.65 and 0.75, otherwise at the primary setting.
- **f50:** its scores are the release plan's existing development runs of candidate 3 at 20 dB. Frozen candidate 3's own runs (15 dB) are reported alongside.

Each comparison takes each song's COnPOff@50 averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval (10,000 resamples, seed 0).

**Cross-annotator results, reported only.** vocadito (annotators A1 and A2, separately) and Molina are decoded with the same code and the same 20 dB trim for:
- candidate 3 at 20 dB (the release plan's a cappella runs, `new-net3-r20`);
- f50-vi;
- lm at its chosen *f*.

For each set, each arm's COnPOff is reported against candidate 3 at 20 dB's, with its 95% interval. That isolates the label changes from the trim, whose a cappella cost is already known. These results decide nothing. The user's priority is separated pop, and on 27 October they chose to report them rather than let them block a change. The check that a label convention generalises beyond RWC-P's annotators is the MIR-ST500 held-out check, on separated pop with different annotators.

### Decision rules, fixed when this plan is hashed

1. **The vowel-initial move** joins the recipe if f50-vi beats f50 at the primary setting, with the interval excluding zero.
2. **The landmark design.**
   - *f* is the value with the highest development COnPOff@50 among the five.
   - lm at that *f* replaces the single onset output in the recipe if it beats f50 with the interval excluding zero.
3. **A candidate.** An arm, at its setting, is eligible for the MIR-ST500 check only if all hold:
   - it passes amendment 4's conditions against B (+0.01 COnPOff@50 with the interval excluding zero, at least 2 of 3 seeds, and the RPA, silence and COnP@150 guards);
   - it beats candidate 3 at 20 dB on COnPOff@50 with the interval excluding zero, so the label change must add to the trim's gain.

   The highest eligible arm is frozen and hashed, with its *f* if it is lm, and its 20 dB trim. With the user's go-ahead, it uses one of the MIR-ST500 protocol's two checks. That protocol compares against frozen candidate 3 (15 dB), so the check confirms the whole package: the label change and the trim together.
4. **If none is eligible,** candidate 3 at 20 dB is frozen and hashed as **frozen candidate 4**: the trim alone, which beat frozen candidate 3 on the development half by +0.0096. With the user's go-ahead, it uses one of the MIR-ST500 checks against frozen candidate 3. Stage 2 starts from whatever rules 1 and 2 adopted.

Rules 1 and 2 are judged separately. If both adopt their change, the combination is stage 2's starting recipe and is judged there.

**Reported, deciding nothing:**
- COnP@50, COnP@150, COnPOff@150, and the median start error;
- the start-timing recheck's measures for each arm: the near-miss share, the share that is the network's own timing, and its stakes. Also, especially for f50-vi, the near-miss share and median start error for notes after a rest;
- for lm, COnPOff@50 against *f* on each set (RWC-P, vocadito A1, vocadito A2, Molina): how much the best convention differs between annotators;
- every arm at 15 dB as well as 20 dB, on the development half and a cappella: the trim's cost and gain, kept separate from the labels';
- for each arm, the validation peak step, and onset timing on M4Singer's held-out singers.

### Steps

1. **Code (local, then musix and CSD3):** the changes above, their checks and tests. Commit, without pushing until asked.
2. **CSD3, login node (nice):**
   - build the vocal manifests with the per-note facts with `sources.py`, then the shard manifests with `relabel.py`;
   - check that, with the new fields ignored, they reproduce the existing manifest's notes exactly;
   - the vowel-move and landmark-gap statistics are worked out inside a short job, not on the login node, since they read the shards.

   The phoneme timings are on CSD3 for every clip round 1 used: M4Singer's 20,896 TextGrids and GTSinger's 28,628 singing `.json` files, checked on 27 October.
3. **CSD3:** two training jobs (f50-vi, lm), one A100 each, 3 seeds side by side, `--augment`, 3 h each as in round 1. At most 6 GPU-hours.
4. **Copy** each run's `best.pt`, `last.pt`, `args.json` and `log.jsonl` to musix.
5. **musix,** one job at a time beside anything else heavy (GAME's run was killed by memory pressure on 27 October):
   - decode f50-vi (3 seeds × 3 thresholds) and lm (3 seeds × 5 fractions) on the development half at 20 dB;
   - decode each arm's primary setting again at 15 dB, reported only;
   - decode the cross-annotator sets at 20 dB (and 15 dB, reported);
   - score, and apply the rules (`arms/round2.txt`).

## Stage 2: more data (outline; planned in writing after stage 1)

- **Candidates:** Annotated-VocalSet (6.4 h, with the VocalSet audio), CSD (4.9 h) and PJS (0.45 h), downloaded to musix on 27 October. Their licences are clear to train on and to ship.
- **Before any of them enters training:** each set's onsets are put on stage 1's footing.
  - **PJS** has phoneme timings, so both landmarks are exact.
  - **CSD** has none, so its consonant and vowel starts must be estimated from the audio.
  - **Annotated-VocalSet** has only vowels: its legato starts move from the 'Sound' segment to the transition start, a median 93 ms earlier.
- **In-domain data,** pending decisions:
  - DALI's audio, which needs Cambridge's view on copyright;
  - a small hand-labelled set of separated pop.

## Settled before hashing (27 October)

- **The phoneme timings** `sources.py` needs are on CSD3.
- **The fraction arms** (0.35, 0.65) are replaced by lm, whose fraction is chosen at decode time from 0.25, 0.35, 0.5, 0.65 and 0.75. The grid brackets the useful range, since the pilots showed 0 and 1 are both worse.
- **No capped-lead arm.** lm knows each note's consonant length, so a capped lead could be tried at decode time if wanted.
- **The release trim** is part of round 2's separated-pop decoder at **20 dB** (the user's decision, after the release plan's results). 25 dB and off gained about 0.001 more on RWC-P, which is within noise. 20 dB keeps more protection for wet mixes whose separated vocal still carries reverb.
- **The vowel rule:**
  - a reach of **150 ms**: M4Singer's vowel-initial notes start a median 70 ms before voicing, three-quarters within about 110 ms;
  - a voicing threshold of **0.3**: the decoder's own, so labels and decoder agree on where singing starts.
- **The pairing window is 250 ms:** the sources' long-consonant limit, so the decoder pairs the same events the labels do.
- **The cross-annotator results are reported only** (the user's choice). They were a guard in an earlier draft.

**Still open:** the exact option names, which are recorded in the code commits.
