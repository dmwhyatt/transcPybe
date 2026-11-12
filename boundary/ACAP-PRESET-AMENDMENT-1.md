# A cappella preset plan, amendment 1: the code tree for one reported arm

Written 30 October 2026, after `acap-preset-plan.md` (sha 6bf6aca2...) was hashed and before any decode under it.

**What changes.** The plan says the reported averaging of frozen candidate 4's networks uses that candidate's frozen transcPybe tree, 3361803. It cannot: `rwc_dev.py` passes the offset output's `end_times` to every hand-over, and `OnsetTrack` in 3361803 does not accept them. This arm uses transcPybe d96e83b instead, as the RWC-P landmark decodes do.

**A check is added.** Seed 0 of candidate 4's networks is decoded alone with this code and tree, and its notes are compared with the frozen run `m4gts-aug-s0-B-plain-t0.5-sm99-w15b-r20`. The result is reported. If the notes differ, the arm is reported with that caveat.

**What does not change.** This arm is reported only and decides nothing. Rules A and B, the check, the tuning sets, the grid and every other decode stand as hashed.
