# Retraining the landmark networks on PESTO's and pYIN's pitch tracks

Written 6 November 2026, before any training; revised the same day, before any job ran, to make the training data on CPU nodes. Exploratory: no held-out data is left (MIR-ST500's two checks, SSVD's test recordings and RWC-P's test half are spent), so every score here comes from sets the presets were tuned on. This plan fixes in advance what is compared and what decides, so those sets are used to compare, never to choose a setting. Nothing here can be called a confirmed result.

## Why

The demo now offers PESTO and pYIN beside CREPE full. The networks learned only from CREPE full's tracks, and on 4 November (`arms/frontends.txt`, tuning sets, the same presets) swapping in another tracker cost:

| Tracker | vocadito and Molina | RWC-P dev | SSVD train and valid |
| --- | --- | --- | --- |
| PESTO | −0.09 to −0.14 | −0.008 | level |
| pYIN | −0.10 to −0.14 | −0.15 | level |

There are two separate causes:

1. **The network's inputs.** Three of its five pitch features are the tracker's own output. Periodicity in particular means something different for each tracker. Retraining addresses this.
2. **The note decoder.** It makes notes only where the track's voicing is high, whatever the network says. PESTO passed 84% of vocadito's sung frames at 0.3, against CREPE full's 99%. Retraining cannot change this. A voicing map can: a monotone map from the tracker's confidence to CREPE full's, fitted on training data.

## What is trained (CSD3)

- **Data.** Round 1's 45 parts of M4Singer and GTSinger over MUSDB18-HQ. The vocals, accompaniment and number of parts are read from round 1's `.done` records, and mixing is seeded per recording, so the mixes are the same. Only the pitch features are made again, with transcPybe `4e23d5b` (`claude/pitch-front-ends`):
  - PESTO: `pesto-pitch` 2.0.1, checkpoint `mir-1k_g7`, in 30 s chunks, on the CPU, as on 4 November;
  - pYIN: librosa's, unchanged since 4 November.
- **Labels.** Round 2's (`m4gts-musdb/all-lm.jsonl` and `m4-lm.jsonl`), unchanged. `swap_features` points their rows at the new shards. The vowel targets of vowel-initial notes are placed by CREPE full's voicing (`--label-voicing`), so the targets are exactly round 2's and only the inputs differ. Each row must match its CREPE twin's notes, landmarks, stored targets and spectrogram, or training stops.
- **Recipe.** Round 2's `r2-lm`, exactly: 30,000 steps, batch 32, 4 s crops, learning rate 1e-3, weight decay 0.01, warm-up 500, positive weight 8, evaluation every 500 steps, channels 32/64/96, hidden 128, dropout 0.2, `--augment`, `--onset-labels landmarks`, and M4Singer's validation singers as the extra set. Seeds 0–2 train side by side on one A100, and each keeps its `best.pt` by round 2's validation rule.
- **Voicing map.** A quantile map from the tracker's periodicity to CREPE full's, over every frame of the validation singers' mixes (`voicing_map`, inside the training job). It is never fitted on a set we score on.
- **Where, and the budget.**
  - The 45 data parts run on Ice Lake CPU nodes (`prepare-cpu.sbatch`), 32 cores each, since neither tracker needs a GPU. Round 1's parts took 5–8.5 minutes on A100 nodes. On CPU nodes this is not measured yet, so each part is allowed 30 minutes, and the trial part measures it. At the limits that is 720 core-hours per tracker, from the CPU account.
  - Training runs on one A100, allowed 3 hours; round 2's took 1.92. That is about 2 GPU-hours per tracker, at most 3.

## What is compared (musix, after the runs are copied there)

**Tracks:**
- PESTO and pYIN: the tuning sets' tracks from 4 November (`frontends/f0`, on the CPU). On 6 November the demo's GPU PESTO, without TF32, scored within 0.001 of them on every set.
- CREPE full: the cached tracks the presets ran on.

**Arms, for each tracker T:**

| Arm | Networks | Track to the networks | Track to the note decoder |
| --- | --- | --- | --- |
| A | round 2 (CREPE full) | CREPE full | CREPE full |
| B | round 2 (CREPE full) | T | T, as it is |
| C | retrained on T | T | T, as it is |
| D | retrained on T | T | T, voicing mapped |

A is what the demo ships with CREPE full. B is what it does now with T.

**Decoding.** The two confirmed presets, exactly, with the three seeds averaged as the presets do:
- the a cappella preset on vocadito (both annotators), Molina, and SSVD's training and validation recordings;
- the separated-pop preset on RWC-P's development half (48 songs with notes).

Nothing else is varied: no grid over *f*, thresholds or trims, and no other checkpoint.

**Scoring.** COnPOff@50 at whole semitones, as on 4 November. Differences are paired by recording: a bootstrap of 10,000 resamples, seed 0, 95% intervals. Reported beside it, deciding nothing:
- COnP@50 and COn@50;
- COnPOff@50 with measured pitch;
- notes per recording;
- the share of frames voiced at 0.3;
- each arm against A, which measures how far T's best pipeline is from CREPE full's.

## Decisions, fixed now

For each tracker, over the five columns (vocadito A1, vocadito A2, Molina, SSVD train and valid, RWC-P dev):

1. **Retraining helps** if C beats B with the interval clear of zero on at least three columns, and is below B with the interval clear of zero on none.
2. **The voicing map helps** if D beats C by the same rule.
3. **The demo.**
   - If rule 1 holds, the demo pairs T with its own networks. If rule 2 holds too, it also applies the map.
   - The page says those networks were trained on T, and that the results are unconfirmed.
   - Otherwise every tracker keeps round 2's networks.
   - Deploying still needs the user's go.
4. **A failed job** is rerun unchanged. A seed that fails twice is reported, and the remaining seeds are averaged and the gap is said.

## Running it

On CSD3, from a login node (`nice -n 19`), with code `claude/boundary-model` and `claude/pitch-front-ends` from the 6 November bundle:

```bash
nice -n 19 ~/venvs/boundary/bin/pip install --no-deps pesto-pitch==2.0.1 torchaudio==2.11.0 omegaconf==2.4.0 PyYAML==6.0.3 tqdm==4.70.1
nice -n 19 bash boundary/slurm/retrain-tracker.sh pesto
nice -n 19 bash boundary/slurm/retrain-tracker.sh pyin
```

`retrain-tracker.sh` checks the setup, then submits three jobs, each running only after the one before succeeds:
1. part 0 as a trial, on an Ice Lake node;
2. the other 44 parts, on Ice Lake nodes;
3. the training (`train-tracker.sbatch`), on an A100.

`PREP=prepare` sends the parts to A100 nodes instead, as round 1 ran; PESTO then runs on the GPU, where its tracks match the CPU's within 0.004 cents.

The outputs are:
- shards: `$DATA/m4gts-musdb-{pesto,pyin}/shards`;
- runs: `$DATA/runs/r2-lm-{pesto,pyin}`, holding the seeds, the swapped manifests and `voicing-map.json`.

Copy each run's `seed*/{best.pt,last.pt,args.json,log.jsonl}` and `voicing-map.json` to musix's `csd3-runs/`.
