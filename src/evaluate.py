"""
evaluate.py
-----------
Policy evaluation and checkpoint-comparison pipeline for CMP4501.

Stand-alone usage
-----------------
Evaluate one checkpoint:
    python src/evaluate.py --checkpoint checkpoints/<run>_step_200000.zip

Compare all three checkpoints from the latest training run automatically:
    python src/evaluate.py --compare

Compare a specific named run:
    python src/evaluate.py --compare --run ppo_highway_20260605_110802

Override the number of evaluation episodes:
    python src/evaluate.py --compare --episodes 20

All results are saved as JSON files under results/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, DEFAULT_CONFIG, CHECKPOINT_DIR, RESULTS_DIR
from utils import make_env, setup_logging

logger = logging.getLogger(__name__)

_SEP  = "=" * 72
_THIN = "-" * 72


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Aggregated metrics from a batch of evaluation episodes.

    Attributes
    ----------
    mean_return:
        Average undiscounted episode return across all episodes.
    std_return:
        Standard deviation of episode returns.
    mean_length:
        Average number of steps per episode.
    crash_rate:
        Fraction of episodes that ended in a collision (0–1).
    mean_speed:
        Average ego-vehicle speed (m/s) across all steps of all episodes.
    n_episodes:
        Number of episodes actually evaluated.
    checkpoint_tag:
        Human-readable label, e.g. ``"step_200000"`` or ``"untrained"``.
    raw_returns:
        Per-episode returns for downstream plotting.
    """

    mean_return: float = 0.0
    std_return: float = 0.0
    mean_length: float = 0.0
    crash_rate: float = 0.0
    mean_speed: float = 0.0
    n_episodes: int = 0
    checkpoint_tag: str = ""
    raw_returns: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"EvalResult [{self.checkpoint_tag}]  "
            f"n={self.n_episodes}  "
            f"return={self.mean_return:+.4f}+/-{self.std_return:.4f}  "
            f"crash={self.crash_rate:.1%}  "
            f"speed={self.mean_speed:.1f} m/s  "
            f"len={self.mean_length:.0f}"
        )


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_policy(
    agent: Any,
    env: Any,
    n_episodes: int = 10,
    deterministic: bool = True,
    checkpoint_tag: str = "",
) -> EvalResult:
    """
    Roll out the agent for ``n_episodes`` full episodes and aggregate metrics.

    Per-episode statistics collected:
      - Total (undiscounted) return — sum of shaped rewards.
      - Episode length — number of environment steps.
      - Crash flag — ``True`` if ``info["crashed"]`` was ever ``True``.
      - Per-step speed — from ``info["speed"]`` (m/s).

    Parameters
    ----------
    agent:
        SB3 model with a ``predict(obs, deterministic)`` method.
    env:
        Configured Gymnasium environment (``FlattenObservation`` wrapped).
        Must **not** be a ``VecEnv``; pass a plain ``Env`` instance.
    n_episodes:
        Number of complete episodes to run.
    deterministic:
        Pass ``True`` (default) for greedy evaluation, ``False`` for
        stochastic rollouts sampled from the policy.
    checkpoint_tag:
        Label stored in the returned ``EvalResult`` for display/saving.

    Returns
    -------
    EvalResult
        Aggregated evaluation metrics.
    """
    import numpy as np

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_crashed: list[bool] = []
    all_speeds: list[float] = []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        ep_any_crash = False

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            ep_return += float(reward)
            ep_steps  += 1

            if info.get("crashed", False):
                ep_any_crash = True
            if "speed" in info:
                all_speeds.append(float(info["speed"]))

        ep_returns.append(ep_return)
        ep_lengths.append(ep_steps)
        ep_crashed.append(ep_any_crash)

        logger.debug(
            "Episode %d/%d  return=%.4f  len=%d  crashed=%s",
            ep_idx + 1, n_episodes, ep_return, ep_steps, ep_any_crash,
        )

    arr = np.array(ep_returns, dtype=np.float64)
    return EvalResult(
        mean_return=float(arr.mean()),
        std_return=float(arr.std()),
        mean_length=float(np.mean(ep_lengths)),
        crash_rate=float(np.mean(ep_crashed)),
        mean_speed=float(np.mean(all_speeds)) if all_speeds else 0.0,
        n_episodes=n_episodes,
        checkpoint_tag=checkpoint_tag,
        raw_returns=ep_returns,
    )


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def save_eval_result(result: EvalResult, tag: str | None = None) -> Path:
    """
    Serialise an ``EvalResult`` to a timestamped JSON file under ``results/``.

    Parameters
    ----------
    result:
        Evaluation result to save.
    tag:
        Override for the filename stem.  Defaults to
        ``eval_<result.checkpoint_tag>``.

    Returns
    -------
    Path
        Absolute path of the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = tag or f"eval_{result.checkpoint_tag}"
    # Sanitise for use as a filename
    stem = re.sub(r"[^\w\-]", "_", stem)
    out_path = RESULTS_DIR / f"{stem}.json"

    payload = asdict(result)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Eval result saved: %s", out_path)
    return out_path


def save_comparison(results: list[EvalResult], run_tag: str = "comparison") -> Path:
    """
    Save a list of ``EvalResult`` objects as a single comparison JSON.

    Parameters
    ----------
    results:
        Ordered list of results (e.g. untrained, half, fully-trained).
    run_tag:
        Used in the output filename.

    Returns
    -------
    Path
        Absolute path of the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\-]", "_", run_tag)
    out_path = RESULTS_DIR / f"{stem}.json"
    payload = [asdict(r) for r in results]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Comparison saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_latest_run_checkpoints(
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict[str, Path]:
    """
    Discover the three canonical checkpoints from the most recent training run.

    Looks for ``*.zip`` files matching the pattern
    ``ppo_highway_<timestamp>_step_<N>.zip`` and groups them by run prefix.
    The most-recently-modified run is selected and its step-0, middle, and
    final checkpoints are returned.

    Returns
    -------
    dict[str, Path]
        Keys are ``"step_0"``, ``"step_half"``, ``"step_final"``; values
        are the corresponding ``.zip`` paths (without the ``.zip`` suffix,
        matching how SB3 expects them).

    Raises
    ------
    FileNotFoundError
        If no matching checkpoints are found.
    """
    zips = sorted(checkpoint_dir.glob("ppo_highway_*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError(
            f"No PPO checkpoints found in {checkpoint_dir}. "
            "Run python src/train.py first."
        )

    # Group by run prefix (everything before _step_)
    runs: dict[str, list[Path]] = {}
    for z in zips:
        match = re.match(r"^(ppo_highway_\d{8}_\d{6})_step_\d+\.zip$", z.name)
        if match:
            prefix = match.group(1)
            runs.setdefault(prefix, []).append(z)

    if not runs:
        raise FileNotFoundError("No valid checkpoint files found.")

    # Pick the run with the most-recent modification time
    latest_prefix = max(runs, key=lambda p: max(f.stat().st_mtime for f in runs[p]))
    run_files = sorted(runs[latest_prefix], key=lambda p: _step_from_path(p))

    if len(run_files) < 2:
        raise FileNotFoundError(
            f"Run '{latest_prefix}' has only {len(run_files)} checkpoint(s); "
            "expected at least 2 (step_0 + final)."
        )

    step_0   = run_files[0]
    step_final = run_files[-1]
    # Middle checkpoint — prefer the explicit midpoint, else use the middle index
    if len(run_files) >= 3:
        step_half = run_files[len(run_files) // 2]
    else:
        step_half = run_files[-1]  # fallback: use final as both half and final

    return {
        "step_0":     step_0.with_suffix(""),
        "step_half":  step_half.with_suffix(""),
        "step_final": step_final.with_suffix(""),
    }


def find_run_checkpoints(
    run_name: str,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> dict[str, Path]:
    """
    Discover checkpoints for a specific named run.

    Parameters
    ----------
    run_name:
        Run prefix, e.g. ``"ppo_highway_20260605_110802"``.

    Returns
    -------
    dict[str, Path]
        Same structure as ``find_latest_run_checkpoints``.
    """
    zips = sorted(
        checkpoint_dir.glob(f"{run_name}_step_*.zip"),
        key=lambda p: _step_from_path(p),
    )
    if not zips:
        raise FileNotFoundError(
            f"No checkpoints found for run '{run_name}' in {checkpoint_dir}."
        )

    if len(zips) < 2:
        raise FileNotFoundError(
            f"Run '{run_name}' has only {len(zips)} checkpoint(s)."
        )

    step_0    = zips[0]
    step_final = zips[-1]
    step_half  = zips[len(zips) // 2] if len(zips) >= 3 else zips[-1]

    return {
        "step_0":     step_0.with_suffix(""),
        "step_half":  step_half.with_suffix(""),
        "step_final": step_final.with_suffix(""),
    }


def _step_from_path(path: Path) -> int:
    """Extract the integer step count from a checkpoint filename."""
    match = re.search(r"_step_(\d+)", path.stem)
    return int(match.group(1)) if match else 0


# ---------------------------------------------------------------------------
# Comparison pipeline
# ---------------------------------------------------------------------------

def compare_checkpoints(
    checkpoints: dict[str, Path],
    cfg: Config,
    n_episodes: int = 10,
    use_reward_shaping: bool = True,
) -> list[EvalResult]:
    """
    Load and evaluate multiple checkpoints in sequence.

    Parameters
    ----------
    checkpoints:
        Ordered mapping of ``label → checkpoint_path`` (no ``.zip`` suffix).
        Keys become the ``checkpoint_tag`` in each ``EvalResult``.
    cfg:
        Project configuration.
    n_episodes:
        Episodes per checkpoint.
    use_reward_shaping:
        Whether to use the custom shaped reward during evaluation.

    Returns
    -------
    list[EvalResult]
        One result per checkpoint, in iteration order.
    """
    from stable_baselines3 import PPO

    results: list[EvalResult] = []

    for label, ckpt_path in checkpoints.items():
        print(f"\n  Evaluating  [{label}]  {ckpt_path}.zip")
        print(_THIN)

        env = make_env(cfg, seed=cfg.train.seed, use_reward_shaping=use_reward_shaping)
        agent = PPO.load(str(ckpt_path), env=env)

        t0 = time.perf_counter()
        result = evaluate_policy(
            agent, env,
            n_episodes=n_episodes,
            deterministic=cfg.eval.deterministic,
            checkpoint_tag=label,
        )
        elapsed = time.perf_counter() - t0
        env.close()

        _print_single_result(result, elapsed)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_single_result(result: EvalResult, elapsed: float | None = None) -> None:
    """Print a concise one-block summary for a single EvalResult."""
    elapsed_str = f"  ({elapsed:.1f}s)" if elapsed is not None else ""
    print(f"  Mean return  : {result.mean_return:+.4f} +/- {result.std_return:.4f}{elapsed_str}")
    print(f"  Crash rate   : {result.crash_rate:.1%}  "
          f"({int(result.crash_rate * result.n_episodes)}/{result.n_episodes} episodes)")
    print(f"  Mean speed   : {result.mean_speed:.2f} m/s")
    print(f"  Mean ep. len : {result.mean_length:.0f} steps")


def print_comparison_table(results: list[EvalResult]) -> None:
    """
    Print a side-by-side comparison table for a list of EvalResult objects.

    Columns: checkpoint label, mean return ± std, crash rate, speed, length.
    The best value in each numeric column is highlighted with an asterisk (*).
    """
    if not results:
        print("  (no results to compare)")
        return

    print()
    print(_SEP)
    print("  Checkpoint Comparison")
    print(_SEP)

    # Column widths
    lw = max(len(r.checkpoint_tag) for r in results) + 2
    lw = max(lw, 16)

    header = (
        f"  {'Label':<{lw}}"
        f"{'Mean Return':>16}"
        f"{'Std':>10}"
        f"{'Crash Rate':>12}"
        f"{'Speed (m/s)':>13}"
        f"{'Ep. Length':>12}"
    )
    print(header)
    print(_THIN)

    # Identify best values (for annotation)
    best_return = max(r.mean_return for r in results)
    best_crash  = min(r.crash_rate  for r in results)
    best_speed  = max(r.mean_speed  for r in results)
    best_length = max(r.mean_length for r in results)

    def _star(val: float, best: float, tol: float = 1e-6) -> str:
        return "*" if abs(val - best) < tol else " "

    for r in results:
        row = (
            f"  {r.checkpoint_tag:<{lw}}"
            f"{r.mean_return:>+15.4f}{_star(r.mean_return, best_return)}"
            f"{r.std_return:>10.4f}"
            f"{r.crash_rate:>11.1%}{_star(r.crash_rate,  best_crash)}"
            f"{r.mean_speed:>12.2f}{_star(r.mean_speed,  best_speed)}"
            f"{r.mean_length:>11.0f}{_star(r.mean_length, best_length)}"
        )
        print(row)

    print(_THIN)
    print("  * = best value in column")
    print()

    # Improvement summary (only if we have step_0 and a final entry)
    _print_improvement_summary(results)


def _print_improvement_summary(results: list[EvalResult]) -> None:
    """Print a delta block comparing the first and last checkpoint."""
    if len(results) < 2:
        return

    first = results[0]
    final = results[-1]

    d_return = final.mean_return - first.mean_return
    d_crash  = final.crash_rate  - first.crash_rate    # negative = improvement
    d_speed  = final.mean_speed  - first.mean_speed

    print("  Training improvement  (final vs untrained)")
    print(_THIN)
    print(f"  Return    : {first.mean_return:+.4f}  ->  {final.mean_return:+.4f}"
          f"  ({d_return:+.4f})")
    print(f"  Crash rate: {first.crash_rate:.1%}     ->  {final.crash_rate:.1%}"
          f"  ({d_crash:+.1%})")
    print(f"  Speed     : {first.mean_speed:.2f} m/s ->  {final.mean_speed:.2f} m/s"
          f"  ({d_speed:+.2f})")
    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Two modes
    ---------
    Single checkpoint:
        --checkpoint path/to/model.zip [--episodes N] [--stochastic]

    Three-way comparison (latest run, auto-discovered):
        --compare [--run <run_name>] [--episodes N]
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on Highway-Env  (CMP4501).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--checkpoint", type=Path, metavar="PATH",
        help="Path to a single checkpoint (.zip extension optional).",
    )
    mode.add_argument(
        "--compare", action="store_true",
        help="Evaluate and compare all three checkpoints from a training run.",
    )
    parser.add_argument(
        "--run", type=str, default=None, metavar="RUN_NAME",
        help=(
            "Named run prefix for --compare, e.g. 'ppo_highway_20260605_110802'. "
            "Defaults to the most-recently-modified run."
        ),
    )
    parser.add_argument(
        "--episodes", type=int, default=None, metavar="N",
        help="Number of evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic actions instead of greedy (deterministic) policy.",
    )
    parser.add_argument(
        "--no-reward-shaping", action="store_true",
        help="Evaluate with the native highway-env reward instead of the custom reward.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(RESULTS_DIR / "eval.log")

    cfg = DEFAULT_CONFIG
    if args.seed is not None:
        cfg.train.seed = args.seed
    cfg.eval.deterministic = not args.stochastic
    n_episodes = args.episodes if args.episodes is not None else cfg.eval.eval_episodes
    use_reward_shaping = not args.no_reward_shaping

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("  CMP4501 -- Autonomous Driving  |  Policy Evaluation")
    print(_SEP)
    print(f"  Episodes        : {n_episodes}")
    print(f"  Deterministic   : {cfg.eval.deterministic}")
    print(f"  Reward shaping  : {'ON' if use_reward_shaping else 'OFF'}")
    print(f"  Seed            : {cfg.train.seed}")
    print(_THIN)

    # ── Single-checkpoint mode ────────────────────────────────────────────
    if args.checkpoint is not None:
        from stable_baselines3 import PPO

        ckpt = args.checkpoint.with_suffix("") if args.checkpoint.suffix == ".zip" else args.checkpoint
        print(f"  Checkpoint      : {ckpt}.zip")
        print(_SEP)

        env   = make_env(cfg, seed=cfg.train.seed, use_reward_shaping=use_reward_shaping)
        agent = PPO.load(str(ckpt), env=env)

        t0 = time.perf_counter()
        result = evaluate_policy(
            agent, env,
            n_episodes=n_episodes,
            deterministic=cfg.eval.deterministic,
            checkpoint_tag=ckpt.stem,
        )
        elapsed = time.perf_counter() - t0
        env.close()

        print()
        _print_single_result(result, elapsed)

        out = save_eval_result(result)
        print(f"\n  Saved: {out}")

    # ── Comparison mode ────────────────────────────────────────────────────
    else:
        if args.run:
            checkpoints = find_run_checkpoints(args.run)
        else:
            checkpoints = find_latest_run_checkpoints()

        run_prefix = list(checkpoints.values())[0].parent
        print(f"  Run             : {checkpoints['step_0'].name.split('_step_')[0]}")
        print(f"  Checkpoints     : {CHECKPOINT_DIR.name}/")

        # Rename keys to descriptive labels for the table
        step_0_count   = _step_from_path(checkpoints["step_0"])
        step_h_count   = _step_from_path(checkpoints["step_half"])
        step_f_count   = _step_from_path(checkpoints["step_final"])

        labeled: dict[str, Path] = {
            f"Untrained (step {step_0_count:,})":          checkpoints["step_0"],
            f"Half-trained (step {step_h_count:,})":       checkpoints["step_half"],
            f"Fully-trained (step {step_f_count:,})":      checkpoints["step_final"],
        }

        results = compare_checkpoints(
            labeled, cfg,
            n_episodes=n_episodes,
            use_reward_shaping=use_reward_shaping,
        )

        print_comparison_table(results)

        run_id = checkpoints["step_final"].stem.split("_step_")[0]
        out = save_comparison(results, run_tag=f"comparison_{run_id}")
        print(f"  Results saved: {out}")

    print()


if __name__ == "__main__":
    main()
