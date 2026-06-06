# Final Compliance Report

## CMP4501 Submission Audit

Audit date: 2026-06-06

### Authors and Institution

- Kerim Elmali - 2282509
- Mohammad Siyam - 2267953
- Abdalla Hamuda - 2105020
- Department of Software Engineering
- Bahçeşehir University (BAU)

### Selected Track

Option A - Autonomous Driving with Highway-Env

### Requirement Status

| Requirement | Status | Evidence |
|---|---|---|
| Project title | Pass | README header |
| Student names and IDs | Pass | README header and footer |
| CMP4501 course code | Pass | README header |
| Selected track | Pass | Prominent below title |
| Evolution GIF near top | Pass | `assets/evolution.gif` |
| Reward function and LaTeX | Pass | README Section 4 |
| Reward justification | Pass | README Section 4 |
| PPO explanation | Pass | README Section 5 |
| Hyperparameters | Pass | README Section 5 and `src/config.py` |
| States and actions | Pass | README Section 3 |
| Training graphs | Pass | README Section 7 |
| Training analysis | Pass | README Section 7 |
| Evaluation analysis | Pass | README Section 8 |
| Challenges and solutions | Pass | README Section 10 |
| Repository structure | Pass | README Section 12 |
| Unsupported claims removed | Pass | Evidence-only wording used |

### Verified Results

- Training CSV records: 392.
- Final CSV timestep: 200,704.
- Peak rollout mean reward: 8.9845 at step 40,448.
- Final rollout mean reward: 7.9673.
- Logged run duration: 14,081.8 seconds.
- Final CSV FPS: 14.
- Evaluation episodes: five per checkpoint.
- Final mean return: 11.5362.
- Final mean episode length: 56.0.
- Final crash rate: 100%.
- Exact policy parameter count: 146,438.

### Repository Artifact Status

The final checkpoints, comparison JSON, MP4 files, and `logs/train.log` exist
in the working directory. They were previously excluded by `.gitignore`.
Explicit exceptions have been added for the named final artifacts. They must
be included in the final commit or submission package.

### Remaining Limitation

The saved deterministic evaluation contains only five episodes per checkpoint,
and all evaluated episodes crashed. The README states this directly and makes
no unsupported safety or convergence claim.
