"""
train.py
--------
PPO training entry point for CMP4501 Autonomous Driving with Highway-Env.

Three checkpoints are saved automatically:
  checkpoints/<run>_step_0.zip          -- untrained baseline
  checkpoints/<run>_step_<half>.zip     -- half-trained policy
  checkpoints/<run>_step_<total>.zip    -- fully-trained policy

Logs are written to:
  logs/<run>/progress.csv               -- CSV per-rollout metrics
  logs/<run>/events.out.tfevents.*      -- TensorBoard events

Usage
-----
    python src/train.py
    python src/train.py --timesteps 300000
    python src/train.py --lr 3e-4 --seed 0
    python src/train.py --no-reward-shaping   # use native highway-env reward
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a script from the project root or from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.utils import safe_mean

from config import Config, DEFAULT_CONFIG, CHECKPOINT_DIR, LOG_DIR
from utils import make_env, set_seed, setup_logging, save_checkpoint


_module_logger = logging.getLogger(__name__)

_SEP  = "=" * 66
_THIN = "-" * 66


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TrainingProgressCallback(BaseCallback):
    """
    Prints a one-line progress summary at regular timestep intervals.

    Handles the two-phase ``learn()`` approach correctly: the cumulative
    ``num_timesteps`` value is used throughout, and ``_on_training_start``
    re-aligns the next print boundary whenever a new ``learn()`` call begins.

    Parameters
    ----------
    total_timesteps:
        Grand total steps for the entire training run (both phases).
    log_interval:
        Print a progress line every this many cumulative steps.
    """

    def __init__(self, total_timesteps: int, log_interval: int = 10_000) -> None:
        super().__init__(verbose=0)
        self._total = total_timesteps
        self._log_interval = log_interval
        self._next_log: int = log_interval  # overwritten in _on_training_start

    def _on_training_start(self) -> None:
        # Compute the first print boundary strictly above the current position.
        # When called at the start of phase 2 with num_timesteps already at
        # ~100 000, this prevents a burst of back-filled print lines.
        self._next_log = (
            (self.num_timesteps // self._log_interval + 1) * self._log_interval
        )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log:
            self._print_row()
            self._next_log += self._log_interval
        return True

    def _print_row(self) -> None:
        pct = min(100.0, 100.0 * self.num_timesteps / self._total)
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            mean_r = safe_mean([ep["r"] for ep in buf])
            mean_l = safe_mean([ep["l"] for ep in buf])
            stats = f"ep_rew={mean_r:+.4f}  ep_len={mean_l:5.0f} steps"
        else:
            stats = "collecting first episodes..."
        print(
            f"    [{pct:5.1f}%]  "
            f"step {self.num_timesteps:>9,} / {self._total:>9,}"
            f"  |  {stats}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line overrides.

    Every flag maps to a field in ``TrainConfig`` or ``EnvConfig``.
    Unspecified flags fall back to ``DEFAULT_CONFIG`` values.
    """
    defaults = DEFAULT_CONFIG.train
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Highway-Env  (CMP4501).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        metavar="N",
        help=f"Total environment steps (default: {defaults.total_timesteps:,}).",
    )
    parser.add_argument(
        "--lr", type=float, default=None, dest="learning_rate",
        metavar="F",
        help=f"PPO learning rate (default: {defaults.learning_rate}).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=None, dest="n_steps",
        metavar="N",
        help=f"Rollout steps per env before each PPO update (default: {defaults.n_steps}).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        metavar="N",
        help=f"Random seed (default: {defaults.seed}).",
    )
    parser.add_argument(
        "--no-reward-shaping", action="store_true",
        help="Train with the native highway-env reward instead of the custom shaped reward.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    """
    Merge CLI overrides into ``DEFAULT_CONFIG``.

    Only fields explicitly set on the command line are changed; all others
    retain their ``DEFAULT_CONFIG`` values.

    Parameters
    ----------
    args:
        Parsed CLI arguments from ``parse_args()``.

    Returns
    -------
    Config
        Fully populated configuration ready to pass to ``train()``.
    """
    cfg = DEFAULT_CONFIG
    if args.timesteps is not None:
        cfg.train.total_timesteps = args.timesteps
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.n_steps is not None:
        cfg.train.n_steps = args.n_steps
    if args.seed is not None:
        cfg.train.seed = args.seed
    return cfg


# ---------------------------------------------------------------------------
# Policy kwargs
# ---------------------------------------------------------------------------

def _build_policy_kwargs(cfg: Config) -> dict[str, Any]:
    """
    Translate ``ModelConfig`` into the ``policy_kwargs`` dict expected by
    SB3's ``MlpPolicy``.

    Maps:
      ``cfg.model.hidden_sizes`` → ``net_arch``
      ``cfg.model.activation``   → ``activation_fn`` (PyTorch class)
    """
    import torch.nn as nn

    _ACTIVATION_MAP: dict[str, type] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu":  nn.ELU,
    }
    activation_fn = _ACTIVATION_MAP.get(cfg.model.activation.lower(), nn.ReLU)

    return {
        "net_arch":      list(cfg.model.hidden_sizes),
        "activation_fn": activation_fn,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_banner(cfg: Config, run_name: str, use_reward_shaping: bool) -> None:
    print()
    print(_SEP)
    print("  CMP4501 -- Autonomous Driving  |  PPO Training")
    print(_SEP)
    print(f"  Environment   : {cfg.env.env_id}")
    print(f"  Algorithm     : {cfg.train.algorithm} (MlpPolicy)")
    print(f"  Total steps   : {cfg.train.total_timesteps:,}")
    print(f"  n_steps       : {cfg.train.n_steps}  (rollout per env)")
    print(f"  batch_size    : {cfg.train.batch_size}")
    print(f"  n_epochs      : {cfg.train.n_epochs}")
    print(f"  learning_rate : {cfg.train.learning_rate}")
    print(f"  gamma / lam   : {cfg.train.gamma} / {cfg.train.gae_lambda}")
    print(f"  clip_range    : {cfg.train.clip_range}")
    print(f"  ent_coef      : {cfg.train.ent_coef}")
    print(f"  Reward shaping: {'ON  (custom shaped reward)' if use_reward_shaping else 'OFF (native highway-env)'}")
    print(f"  Network       : {cfg.model.hidden_sizes}  [{cfg.model.activation}]")
    print(f"  Seed          : {cfg.train.seed}")
    print(_THIN)
    print(f"  Run name      : {run_name}")
    print(f"  Checkpoints   : {CHECKPOINT_DIR}/")
    print(f"  Logs          : {LOG_DIR}/{run_name}/")
    print(_SEP)
    print()


def _print_checkpoint(path: Path, label: str) -> None:
    print(f"  [checkpoint]  {label}")
    print(f"                {path}.zip")
    print()


def _print_phase_header(phase: int, total_phases: int, start: int, end: int) -> None:
    print(f"  Phase {phase}/{total_phases}  --  "
          f"steps {start:,} to {end:,}")
    print(_THIN)


def _print_summary(
    run_name: str,
    elapsed: float,
    ckpt_0: Path,
    ckpt_half: Path,
    ckpt_final: Path,
    log_dir: Path,
) -> None:
    mins, secs = divmod(int(elapsed), 60)
    print()
    print(_SEP)
    print("  Training complete")
    print(_SEP)
    print(f"  Elapsed       : {mins}m {secs:02d}s")
    print(f"  Checkpoints")
    print(f"    Untrained   : {ckpt_0}.zip")
    print(f"    Half        : {ckpt_half}.zip")
    print(f"    Final       : {ckpt_final}.zip")
    print(f"  CSV log       : {log_dir / 'progress.csv'}")
    print(f"  TensorBoard")
    print(f"    tensorboard --logdir {log_dir}")
    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config, use_reward_shaping: bool = True) -> None:
    """
    Build, train, and checkpoint a PPO agent on Highway-Env.

    Steps
    -----
    1. Initialise logging, seeding, and output directories.
    2. Construct the environment (optionally with custom reward shaping).
    3. Build the ``PPO`` model from ``cfg.train`` and ``cfg.model``.
    4. Configure the SB3 logger for CSV and TensorBoard output.
    5. Save the untrained model (step 0).
    6. Train for ``total_timesteps // 2`` steps → save half-trained checkpoint.
    7. Continue training for the remaining steps → save final checkpoint.
    8. Close the environment and print a summary.

    Parameters
    ----------
    cfg:
        Fully populated ``Config`` object.  Pass ``DEFAULT_CONFIG`` or a
        modified copy built by ``build_config()``.
    use_reward_shaping:
        When ``True`` (default) the environment is wrapped with
        ``RewardShapingWrapper`` using the weights from ``cfg.reward``.
        Pass ``False`` to use the native highway-env reward signal.
    """
    # ── Initialise ────────────────────────────────────────────────────────
    run_name = f"ppo_highway_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    setup_logging(LOG_DIR / "train.log")
    _module_logger.info("Training run started: %s", run_name)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.train.seed)

    _print_banner(cfg, run_name, use_reward_shaping)

    total    = cfg.train.total_timesteps
    half     = total // 2
    remaining = total - half

    # ── Environment ───────────────────────────────────────────────────────
    env = make_env(cfg, seed=cfg.train.seed, use_reward_shaping=use_reward_shaping)

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = _build_policy_kwargs(cfg)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.train.learning_rate,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_range=cfg.train.clip_range,
        ent_coef=cfg.train.ent_coef,
        vf_coef=cfg.train.vf_coef,
        max_grad_norm=cfg.train.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=cfg.train.seed,
        verbose=0,                   # suppress SB3 rollout tables; we print our own
    )

    # ── SB3 logger: CSV + TensorBoard ─────────────────────────────────────
    log_dir = LOG_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    sb3_logger = sb3_configure(str(log_dir), ["csv", "tensorboard"])
    model.set_logger(sb3_logger)

    # ── Shared progress callback (handles both phases) ─────────────────────
    progress_cb = TrainingProgressCallback(
        total_timesteps=total,
        log_interval=max(10_000, cfg.train.n_steps * 2),
    )

    # ── Checkpoint 0: untrained ────────────────────────────────────────────
    ckpt_0 = CHECKPOINT_DIR / f"{run_name}_step_0"
    save_checkpoint(model, ckpt_0)
    _print_checkpoint(ckpt_0, label="Untrained baseline  (step 0)")

    wall_start = time.perf_counter()

    # ── Phase 1: 0 → half ─────────────────────────────────────────────────
    _print_phase_header(1, 2, 0, half)
    model.learn(
        total_timesteps=half,
        callback=progress_cb,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    ckpt_half = CHECKPOINT_DIR / f"{run_name}_step_{half}"
    save_checkpoint(model, ckpt_half)
    _print_checkpoint(ckpt_half, label=f"Half-trained  (step {half:,})")

    # ── Phase 2: half → total ─────────────────────────────────────────────
    _print_phase_header(2, 2, half, total)
    model.learn(
        total_timesteps=remaining,
        callback=progress_cb,
        reset_num_timesteps=False,
        progress_bar=False,
    )

    ckpt_final = CHECKPOINT_DIR / f"{run_name}_step_{total}"
    save_checkpoint(model, ckpt_final)
    _print_checkpoint(ckpt_final, label=f"Fully-trained  (step {total:,})")

    # ── Cleanup & summary ─────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    env.close()

    _print_summary(run_name, elapsed, ckpt_0, ckpt_half, ckpt_final, log_dir)
    _module_logger.info(
        "Run %s complete. Elapsed: %.1fs. Final model: %s.zip",
        run_name, elapsed, ckpt_final,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    use_reward_shaping = not args.no_reward_shaping
    train(cfg, use_reward_shaping=use_reward_shaping)


if __name__ == "__main__":
    main()
