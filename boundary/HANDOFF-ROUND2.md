# Handoff: BoundaryNet for transcPybe (29 Oct 2026)

## Goal

Make transcPybe (sung melody → MIDI) transcribe **separated pop vocals** well. The lever is **BoundaryNet**, a small learned network for note onsets, offsets and sounding, handed to transcPybe's decoder. The headline metric is **COnPOff@50** (start within 50 ms, pitch within 50 cents, end right). The comparison target is GAME.

## Where things are

| What | Where |
|---|---|
| transcPybe code | `~/transcPybe`. PR #3 `claude/round1-onsets` (open). PR #5 `claude/spurious-notes` (on top of #3; decoder options such as the sounding merge and end snapping) |
| BoundaryNet code + plans | `~/transcPybe-boundary`, branch `claude/boundary-model` = PR #4, pushed 29 Oct. It includes `81f34b4` (round-2 labels and landmarks), `714e6f9` (landmark onsets on the frame grid), and the plan records in `boundary/` (`ROUND2-PLAN.md`, `ROUND2-AMENDMENT-1.md`, `RELEASE-PLAN.md`, `ENDS-RECHECK.md`, `MIRST500-HOLDOUT-PLAN.md`, each matching its musix hash) plus this handoff. `boundary/REPORT.md` is still uncommitted (the user hasn't decided) |
| Experiments (musix) | `~/boundary-work-20261020/`: plans and their `.sha256`, `frozen-candidate-*.json`, `arms/` (scripts and results), `csd3-runs/` (checkpoints), `code-<sha>/` (code snapshots) |
| RWC-P eval (musix, private) | `~/transcription-benchmarking/orchestrator/bench-work-rwc-20261019` (dev/test split, `dev/score.py`, `dev/scorecache/`) |
| MIR-ST500 held-out set (musix) | `~/mirst500-test-20261027` (82 songs, stems, annotations; GAME and MIR-ST500 outputs in `orchestrator/bench-work-mirst500-20261027`). **Nothing scored yet** |
| New training data (musix) | `~/singing-data-20261027`: CSD, PJS, Annotated-VocalSet + VocalSet; convention measurements in `conventions-data.txt` and `ends-data.txt` |
| Training (CSD3) | `/rds/user/dmw56/hpc-work/boundary` (shards in `m4gts-musdb/`, `runs/`). Round-2 manifests: `m4gts-musdb/{all,m4}-lm.jsonl` |
| Dashboard | MLflow on musix: `~/mlflow/mlflow.db` (read the sqlite directly), experiment `boundary` |
| Progress figure | https://claude.ai/artifact/TgbWy2XdZ7sDC1xM2N6rnd |
| Memory notes | `~/.claude/projects/-Users-davidwhyatt-transcPybe/memory/` (`transcpybe-onset-findings.md`, `model-training-decisions.md`) |

## How we work (the protocol)

- **Pre-register.** Every experiment gets a plan, hashed on musix before its runs, with fixed decision rules. Any change after seeing results is disclosed in a written amendment.
- **Selection** is on the **RWC-P dev half**, by protocol amendments 3 and 4:
  - COnPOff@50 at least +0.01 over preset B, with the 95% interval excluding zero;
  - at least 2 of 3 seeds above B;
  - guards: RPA, notes in silence, COnP@150.
- **RWC-P's test half is spent** (2 of 2 checks used).
- **Confirmation** is now the **MIR-ST500 held-out check** (`mirst500-holdout-plan.md`, sha c7d7b324…): 2 checks, against frozen candidate 3, labels scored as published. **Ask the user before spending a check.**

## Results so far

RWC-P test half, COnPOff@50:

| Version | COnPOff@50 |
|---|---|
| Preset B (no network) | 0.249 |
| Frozen candidate 2 (network + sounding merge) | 0.267 |
| Frozen candidate 3 (+ start snapping 0.15 s, both ways) | **0.299** |
| GAME | 0.338 |

Candidate 3 is level with GAME on COnP@150. The gap that remains is about 0.027 in note starts and 0.012 in note ends.

- **Release trim at 20 dB** gives +0.010 over candidate 3 on the dev half and slightly hurts a cappella. This is **frozen candidate 4** (`frozen-candidate-4.json`, sha 81207c6f…). It is eligible for a MIR-ST500 check, not spent.
- **Round 2** (`round2.txt`, `round2g.txt`): neither arm beat the control (candidate 3 at 20 dB, dev 0.334).
  - **Vowel-initial label fix** (f50-vi): +0.002 (not significant), but better starts and a cappella +0.025–0.032.
  - **Two-landmark network** (lm; the convention *f* is chosen at decode time): after a grid bug was fixed, −0.006 (not significant) at *f* = 0.5. On a cappella, *f* = 0.75 is much better: vocadito +0.09, Molina +0.045. That *f* was chosen after seeing those sets, so it is not confirmed.
- **A cappella** in general: the separated-pop configs are mixed there, so the network ships as a **separated-pop preset, not the default**.

## Open decisions (the user's)

1. Spend a MIR-ST500 check on frozen candidate 4 now, or wait for a stronger candidate.
2. Pursue the landmark network as one model with two presets (separated pop at *f* = 0.5, a cappella at *f* ≈ 0.75). That needs a new plan and a fresh a cappella held-out check.
3. Stage 2 data: in-domain separated pop would help most:
   - DALI's audio needs Cambridge's view on copyright;
   - or a small hand-labelled set.

   The studio sets (CSD, PJS, Annotated-VocalSet) need their onsets aligned before use.
4. Merge order: PR #3, then PR #5, plus PR #4. Building the actual preset is deferred until the user wants to ship.

## The user's standing rules

- **Commits and pushes:** commit or push only when asked. PR comments posted for the user say they came from a Claude agent.
- **Private data:** RWC-P lives and is scored on musix only; never commit Molina, RWC or other datasets. MIR-ST500's test songs are for scoring only: no training, tuning or listening.
- **Benchmark folders:** never reuse or delete `bench-work*` folders; make new ones.
- **CSD3 connection:**
  - run `ssh -O check hpc` first; if it fails, don't connect, ask the user to run `ssh hpc`;
  - then use `ssh -o ProxyCommand=false -o BatchMode=yes -o ConnectTimeout=10 hpc …`;
  - if the VPN drops, make no further login attempts;
  - never type or store CSD3 passwords or phone codes; HF tokens only via `hf auth login`.
- **CSD3 commands:** prefix every login-node command with `nice -n 19`. Size Slurm time, cores and memory from measurements, never the maximum (`sources` jobs needed 1:10 and 1:25). The user often prefers to be **given the commands** to run themselves.
- **musix:** run heavy jobs one at a time. GAME's container was killed by memory pressure when other jobs ran beside it.
