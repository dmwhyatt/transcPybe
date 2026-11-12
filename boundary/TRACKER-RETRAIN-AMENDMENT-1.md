# TRACKER-RETRAIN-PLAN amendment 1: the voicing map with round 2's networks

Written 7 November 2026. This is after pYIN's comparison (`tracker-retrain/report-pyin.txt` on musix). It is before any result from PESTO's retrained networks: PESTO's seed 1 is still training, and no PESTO arm C or D has been decoded. The plan (sha256 0b0bb9cf…84f7) is otherwise unchanged. Like the plan, this is exploratory.

## Why

On pYIN:
- Retraining alone changed nothing. C against B was level on four columns and better only on Molina (+0.013), so rule 1 does not hold.
- The voicing map helped on all five columns: D against C ranged from +0.026 (SSVD dev) to +0.148 (vocadito A2).

Rule 3 applies the map only together with the retrained networks. As written, pYIN would therefore keep neither.

The map changes only the track that the note decoder gets. The networks' inputs are untouched. So the arm the plan left out is round 2's networks with the map, and it may be the whole of D's gain.

## The added arm

**E:**
- networks: round 2's (`csd3-runs/r2-lm`, `best.pt` of seeds 0–2);
- track to the networks: T's;
- track to the note decoder: T's, voicing-mapped with `csd3-runs/r2-lm-T/voicing-map.json`.

The map is the one the training job fitted on the validation singers of the training data. It is never fitted on a set we score on.

Everything else is as for C and D:
- tracks, presets, sets, seeds averaged, networks on the CPU;
- scoring, bootstrap and intervals.

## The added rule, and rule 3 replaced

4. **The map helps round 2's networks** if E beats B with the interval clear of zero on at least three of the five columns, and is below B with the interval clear of zero on none.

3. **The demo**, for each tracker:
   - If rule 1 holds, it is as before: the demo pairs T with its own networks, and also applies the map if rule 2 holds.
   - If rule 1 does not hold and rule 4 does, the demo keeps round 2's networks and applies T's map to the note decoder's track.
   - Otherwise the demo keeps round 2's networks and no map.
   - The page says the map was fitted on training data and that the results are unconfirmed.
   - Deploying needs the user's go.

Reported beside these, deciding nothing: E against A, and E against D.

## What this amendment cannot claim

For pYIN it is post hoc: its B, C and D numbers were seen first. For PESTO it is written before any result from the retrained networks. PESTO's B, from 4 November, was known when the plan was written.
