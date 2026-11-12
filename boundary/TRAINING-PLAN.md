# Training plan: does GTSinger help, and does compression?

Written 23 October 2026, before any of these networks has been trained. The data was still being prepared: 18 of 45 parts were done, and no network had seen any of it.

How a network is scored on RWC-P is already fixed in `protocol-amendment-2.md` (RWC work directory on musix, 20 October). This plan adds what is trained, the comparisons between the arms, and the steps. Nothing here changes amendment 2.

## The question

The 4k-clip pilots peaked after about 10 passes over their data, then got worse. A smaller network, dropout and augmentation all peaked at the same height. So the limit looked like data volume. This round adds GTSinger, which gives nearly three times as many training clips (39,856 against 14,362) and singers in languages other than Chinese. It asks whether that helps on real separated pop.

## What is held fixed

These carry over from the 21 October diagnostics:
- **Labels:** half-way onsets, `between-0.5`.
- **Audio:** each vocal mixed once over MUSDB18-HQ backing, 20–35 dB down, with no separator.
- **Data:** the shards in `m4gts-musdb/shards/` on CSD3 (`prepare`, 22–23 October).
- **Network:** base size, 3 seeds (0, 1, 2), the default 30,000 steps.
- **Checkpoint:** each seed keeps the checkpoint with the best score on its own validation split, by `train.py`'s rule (the harmonic mean of onset F and re-sung recall).

Evaluation runs every 500 steps instead of 1,000, because the pilots peaked early.

## The three arms

| Arm | Training data | Clips (train / valid) | Extra validation, for information |
|---|---|---|---|
| **A: m4** | M4Singer | 14,362 / 2,755 | GTSinger valid |
| **B: m4gts** | M4Singer + GTSinger | 39,856 / 5,241 | M4Singer valid |
| **C: m4gts-aug** | as B, with `--augment` | 39,856 / 5,241 | M4Singer valid |

GTSinger's validation singers are EN-Alto-2 and KO-Soprano-2. Other singers' versions of their songs are left out of training (option 1, 22 October).

**Why arm C.** Compression was the only augmentation that narrowed the gap between our training audio and real pop stems: the domain classifier's dynamics AUC fell by 0.07. The report listed it as likely in the recipe, but it has never been tested on RWC-P. `--augment` applies a compressor in the feature domain, on half of the crops, together with a mild random EQ and band masking. That only approximates compression of the audio. Arm C needs no new data and costs about 3 GPU-hours.

The arms select checkpoints on different validation sets, A on M4Singer alone and B and C on both. Each arm is a recipe as it would be used, and RWC-P decides between them.

## How each arm is scored

On musix, by amendment 2. For each of the 9 networks (3 arms × 3 seeds):
- run `boundary/rwc_dev.py` on the RWC-P development half (50 songs, 48 at note level);
- decode with variants `B` and `B-plain` at thresholds 0.3, 0.5 and 0.7;
- score with `dev/score.py` against B's development replay (`sp-f-delta0.6`).

That makes 54 scored runs. All of them go in the record, including the ones that fail.

**Primary setting for comparing arms:** `B-plain` at threshold 0.7, with every peak splitting. This was the best setting for half-way labels on 21 October (0.6633 against B's 0.6589). The other five settings are reported but decide nothing between arms.

Each comparison takes each song's COnP@150 averaged over the 3 seeds, pairs it over songs, and uses a 95% bootstrap interval.

## Decision rules, fixed now

1. **GTSinger joins the recipe** if B beats A at the primary setting, with the interval excluding zero.
   - If the interval includes zero, the question stays open. The next recipe keeps GTSinger unless B's point estimate is below A's.
2. **`--augment` joins the recipe** if C beats B at the primary setting, with the interval excluding zero. Otherwise it stays off.
3. **Against B, the shipped preset (amendment 2).** Every arm × setting is a candidate. A candidate may go to the RWC-P test half only if all of these hold:
   - it beats B by at least 0.01 COnP@150, with the interval excluding zero;
   - at least 2 of its 3 seeds beat B on their own;
   - frame overall accuracy falls by no more than 0.005;
   - notes in silence rise by no more than 0.5 points.
4. **If several candidates pass,** the one with the highest development score is frozen: its checkpoints, variant and threshold are hashed. It is then checked once on the test half, as one of the two test checks amendment 2 allows over the whole learned-model line. None has been used yet. Claude asks before spending one.
5. **If none passes,** there is no test check. The record says so, and the next step returns to the decoder hand-over. The 20 October pilots inflated note counts when decoding with network onsets (Molina: 3,397 notes against 2,269).

Rules 1 and 2 decide only the recipe, not whether anything ships. Rule 3 decides that.

**Reported, deciding nothing:**
- COnP@50, COn@150 and the error types;
- for each arm, the step where validation peaks and how far it falls by the end. If data volume is the limit, B should peak later in steps and fall less after its peak;
- the extra validation scores: does the M4Singer-only network do worse on GTSinger singers, and does adding GTSinger cost anything on M4Singer singers?
- Molina and vocadito for the best candidate, as the a cappella generality check.

## Steps

**0. Wait for the data.** When `prepare` finishes, all 45 parts should be in `m4gts-musdb/shards/`.

**1. Make the three manifests** on a CSD3 login node. This reads 45 small files and takes seconds:

```bash
cd /rds/user/$USER/hpc-work/boundary/m4gts-musdb && python3 -c "
import glob, json
rows = [l for f in sorted(glob.glob('shards/*.jsonl')) for l in open(f)]
for name, keep in (('all', None), ('m4', 'm4singer'), ('gts', 'gtsinger')):
    sel = [l for l in rows if keep is None or json.loads(l)['source'] == keep]
    open(name + '.jsonl', 'w').writelines(sel); print(name, len(sel))"
```

Expected counts: all 45,097, m4 17,117, gts 27,980.

**2. Submit the three training jobs.** Each takes one A100, with the three seeds side by side:

```bash
cd ~/transcPybe; D=/rds/user/$USER/hpc-work/boundary
TRAIN_ARGS="--eval-every 500 --extra-valid gts=$D/m4gts-musdb/gts.jsonl" bash boundary/slurm/submit.sh train FEATURES=$D/m4gts-musdb/m4.jsonl RUN=$D/runs/m4 SEEDS="0 1 2"
TRAIN_ARGS="--eval-every 500 --extra-valid m4=$D/m4gts-musdb/m4.jsonl" bash boundary/slurm/submit.sh train FEATURES=$D/m4gts-musdb/all.jsonl RUN=$D/runs/m4gts SEEDS="0 1 2"
TRAIN_ARGS="--eval-every 500 --augment --extra-valid m4=$D/m4gts-musdb/m4.jsonl" bash boundary/slurm/submit.sh train FEATURES=$D/m4gts-musdb/all.jsonl RUN=$D/runs/m4gts-aug SEEDS="0 1 2"
```

**Time.** Each job keeps `train.sbatch`'s 3 h. The basis is 2,000 steps in 105 s for one seed on an A100 (21 October), which puts 30,000 steps at about 26 min of training per seed. Three seeds share the node's 32 cores, and the larger validation sets make each evaluation longer, but neither has been measured. The dashboard shows each run's time remaining after its first evaluation. Runs resume, so a job that times out is resubmitted with the same command and carries on.

**Memory.** Each seed holds its features as float32. That hasn't been measured at this size. `train.sbatch` estimates 33 GB per seed for the earlier plan, which mixed both datasets twice, so this plan should need about half of that: roughly 50 GB per job against the 250 GB that comes with the GPU.

**Cost.** At most 9 GPU-hours, plus about 6 for `prepare`.

**3. Watch** on the dashboard (MLflow on musix, experiment `boundary`). Things to look at:
- score against step;
- the step where it peaks;
- onsets per note, where over-firing shows;
- timing median;
- the `extra.*` scores.

Nothing seen there changes the rules above.

**4. Copy the checkpoints to musix.** Run from your Mac. It moves a few MB:

```bash
ssh hpc 'cd /rds/user/dmw56/hpc-work/boundary/runs && tar cf - m4*/seed*/best.pt m4*/seed*/args.json m4*/seed*/log.jsonl' | ssh musix 'mkdir -p ~/boundary-work-20261020/csd3-runs && tar xf - -C ~/boundary-work-20261020/csd3-runs'
```

**5. Score on musix:** `rwc_dev.py` for the 9 checkpoints, then `dev/score.py`, then the seed-mean paired bootstrap. Claude runs this part. It reads RWC-P only on musix, and only the development half.

**6. Apply the rules.** Write the results into the report, with every arm and setting listed.

## Amendment 1: a fourth output (23 October, before any training job of this round started)

**What changes.** Every network in this round has a fourth output, for re-sung note starts: a note on the previous note's pitch (within half a semitone) that starts less than 50 ms after it ends, as `train.resung` defines them. Its target is the onset target kept only at those notes. It trains at half weight, with positive frames weighted 8 like onset and offset. The target is worked out from each recording's notes when it loads, so the shards are unchanged. Code: commit 76d7192.

**What stays the same:**
- the three arms and their data;
- how checkpoints are chosen: the onset output's F-measure and re-sung recall, as before;
- the decode: `rwc_dev.py` uses the onset output only;
- the primary setting and every decision rule above.

**Why now.** A later experiment will let only this output split notes in the decoder, while every onset is still used for timing. Training it now means that experiment needs no retraining. It will be planned separately, in writing, before any decode uses this output.

**Checked before adopting.** On the 21 October dry4k pilot's data and settings (seed 0, evaluation every 250 steps), 1,000 steps with the fourth output scored 0.789, 0.807, 0.827 and 0.820, against 0.791, 0.810, 0.820 and 0.822 without it. The onset output is unaffected, within seed-to-seed noise.

**Reported, deciding nothing:** `resung_head_recall`, `resung_head_precision` and `resung_head_per_note` at each evaluation, for each arm.

## Amendment 2: also score the final checkpoints (26 October, during training, before any network of this round has been decoded on RWC-P)

**What prompted it.** Only the training curves on the dashboard had been seen when this was written. Nothing had been decoded or scored on RWC-P.
- The combined arms' own validation score stayed flat and noisy, between 0.65 and 0.67, from step 2,000 to 9,000.
- Over the same steps, their score on M4Singer's validation singers kept rising, from 0.757 at step 1,000 to 0.805 at step 8,000.
- Their validation set mixes M4Singer's and GTSinger's held-out singers, and the GTSinger part did not improve.
- So the checkpoint the rule picks for those arms may come from early in training, for reasons unrelated to how well they transcribe pop.
- The M4Singer-only arm peaked at step 6,000 and then declined slowly.

**What is added.** Each of the 9 runs' final checkpoint (`last.pt`, step 30,000) is also decoded and scored on the RWC-P development half, at the same six settings. They are named `<arm>-s<seed>-last-<setting>`. The checkpoints copied to musix in step 4 therefore include `last.pt`.

**Reported, deciding nothing:**
- the tables of section "How each arm is scored", for the final checkpoints;
- for each arm, the final checkpoint minus the chosen one on COnP@150 at the primary setting: the seed mean, paired over songs, with its 95% interval.

**What stays the same.** Rules 1 to 5 use the checkpoints chosen by `train.py`'s rule, exactly as written. A final checkpoint that would pass rule 3 is not eligible for a test check in this round; using it would need a new plan, written before it is tested.
