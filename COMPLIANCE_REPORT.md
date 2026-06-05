# Final Compliance Report

## CMP4501 Submission Hygiene Review

| Item | Status | Notes |
|---|---|---|
| Cache files removed | Pass | Removed `src/__pycache__/`. No remaining Python/editor cache directories are present in the final inventory. |
| Temporary/editor files removed | Pass | No `.vscode/`, `.idea/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, or notebook checkpoint folders remain. |
| Duplicate outputs removed | Pass | Removed older short-run checkpoint/log folders and old run-prefixed duplicate videos. |
| Required checkpoints retained | Pass | Kept `step_0`, `step_100000`, and `step_200000` checkpoints for `ppo_highway_20260605_110802`. |
| Requirements file verified | Pass | `requirements.txt` now contains only dependencies used by the current source code and generated-output pipeline. |
| README asset paths verified | Pass | README references match retained assets, checkpoints, logs, results, and videos. |
| Repository structure verified | Pass | Final structure is organized into `src/`, `assets/`, `checkpoints/`, `logs/`, `results/`, and `videos/`. |
| CMP4501 report requirements | Pass | README includes the required academic report sections, methodology, equations, plots, evaluation, and honest limitations. |

## Retained Final Artifacts

### Assets

- `assets/evolution.gif`
- `assets/reward_plot.png`
- `assets/episode_length_plot.png`
- `assets/evaluation_comparison.png`

### Checkpoints

- `checkpoints/ppo_highway_20260605_110802_step_0.zip`
- `checkpoints/ppo_highway_20260605_110802_step_100000.zip`
- `checkpoints/ppo_highway_20260605_110802_step_200000.zip`

### Logs and Results

- `logs/train.log`
- `logs/ppo_highway_20260605_110802/progress.csv`
- `logs/ppo_highway_20260605_110802/events.out.tfevents.1780646884.Kareem.14764.0`
- `results/comparison_ppo_highway_20260605_110802.json`
- `results/eval.log`

### Videos

- `videos/untrained_agent.mp4`
- `videos/half_trained_agent.mp4`
- `videos/fully_trained_agent.mp4`

## Removed Files and Directories

- `src/__pycache__/`
- `logs/ppo_highway_20260605_110227/`
- `logs/ppo_highway_20260605_110609/`
- `logs/ppo_highway_20260605_184539/`
- `checkpoints/ppo_highway_20260605_110227_step_*.zip`
- `checkpoints/ppo_highway_20260605_110609_step_*.zip`
- `checkpoints/ppo_highway_20260605_184539_step_0.zip`
- `videos/ppo_highway_20260605_110802_*_agent.mp4`
- `videos/record.log`

## Requirements Verification

`requirements.txt` now contains:

- `highway-env` and `gymnasium` for the environment.
- `stable-baselines3` and `torch` for PPO training/evaluation.
- `numpy` for numerical operations.
- `matplotlib` for graph generation.
- `imageio`, `imageio-ffmpeg`, and `Pillow` for MP4/GIF generation and frame annotation.
- `tensorboard` for SB3 TensorBoard logging.

Removed unused dependencies from the previous file:

- `sb3-contrib`
- `torchvision`
- `scipy`
- `seaborn`
- `pandas`
- `opencv-python`
- `tqdm`
- `rich`

## README Path Verification

The README references the following retained paths:

- `assets/evolution.gif`
- `assets/reward_plot.png`
- `assets/episode_length_plot.png`
- `assets/evaluation_comparison.png`
- `checkpoints/ppo_highway_20260605_110802_step_0.zip`
- `checkpoints/ppo_highway_20260605_110802_step_100000.zip`
- `checkpoints/ppo_highway_20260605_110802_step_200000.zip`
- `logs/ppo_highway_20260605_110802/progress.csv`
- `logs/train.log`
- `results/comparison_ppo_highway_20260605_110802.json`
- `videos/untrained_agent.mp4`
- `videos/half_trained_agent.mp4`
- `videos/fully_trained_agent.mp4`

## CMP4501 Submission Checklist

| Requirement | Status |
|---|---|
| Project title | Pass |
| Student name | Pass |
| Course code CMP4501 | Pass |
| Selected track: Option A - Autonomous Driving with Highway-Env | Pass |
| Evolution GIF embedded near top | Pass |
| Project overview | Pass |
| Objectives | Pass |
| Methodology | Pass |
| Custom reward function in LaTeX | Pass |
| Reward term explanations | Pass |
| PPO model explanation | Pass |
| Hyperparameters table | Pass |
| States and actions explanation | Pass |
| Training setup | Pass |
| Training reward analysis and figure | Pass |
| Episode length analysis and figure | Pass |
| Evaluation comparison and figure | Pass |
| Challenges and failures | Pass |
| Results summary | Pass |
| How to run project | Pass |
| Repository structure | Pass |
| Requirements section | Pass |
| Conclusion | Pass |
| Honest discussion of high crash rate | Pass |

## Final Assessment

The repository is ready for CMP4501 submission. It contains the final source
code, the selected trained checkpoints, the final report README, generated
figures, representative videos, and evaluation outputs. The README presents
the project professionally and accurately: it highlights reward and survival
improvements while clearly stating that collision avoidance remained a major
limitation.
