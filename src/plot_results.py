"""
plot_results.py
---------------
Generate README-ready training and evaluation figures from existing logs.

Usage
-----
    python src/plot_results.py
    python src/plot_results.py --run ppo_highway_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ASSETS_DIR, LOG_DIR, RESULTS_DIR
from utils import setup_logging

logger = logging.getLogger(__name__)

RUN_RE = re.compile(r"^ppo_highway_\d{8}_\d{6}$")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_latest_run(log_root: Path = LOG_DIR, results_dir: Path = RESULTS_DIR) -> Path:
    """Return the latest run with progress.csv, preferring evaluated runs."""
    candidates = [
        path for path in log_root.iterdir()
        if path.is_dir() and (path / "progress.csv").exists() and RUN_RE.match(path.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No SB3 progress.csv files found under {log_root}.")

    evaluated = [
        path for path in candidates
        if (results_dir / f"comparison_{path.name}.json").exists()
    ]
    if evaluated:
        return max(evaluated, key=lambda path: (path.stat().st_mtime, path.name))

    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_run_dir(run: str | None = None, log_root: Path = LOG_DIR) -> Path:
    """Resolve an explicit run name or discover the latest run."""
    if run is None:
        return find_latest_run(log_root)

    run_dir = log_root / run
    progress_path = run_dir / "progress.csv"
    if not progress_path.exists():
        raise FileNotFoundError(f"Run '{run}' has no progress.csv at {progress_path}.")
    return run_dir


def load_training_log(log_path: Path) -> dict[str, list[float]]:
    """
    Parse SB3's progress.csv into numeric metric arrays.

    Non-numeric and blank cells are skipped for each metric independently.
    """
    if log_path.is_dir():
        log_path = log_path / "progress.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    metrics: dict[str, list[float]] = {}
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Training log has no header row: {log_path}")

        for field in reader.fieldnames:
            metrics[field] = []

        for row in reader:
            for field, raw_value in row.items():
                if raw_value is None or raw_value.strip() == "":
                    continue
                try:
                    metrics.setdefault(field, []).append(float(raw_value))
                except ValueError:
                    logger.debug("Skipping non-numeric CSV cell %s=%r", field, raw_value)

    return metrics


def load_eval_results(results_dir: Path, run: str | None = None) -> list[dict[str, Any]]:
    """
    Load a checkpoint-comparison JSON file from results/.

    If ``run`` is provided, ``comparison_<run>.json`` is preferred. Otherwise
    the newest comparison JSON is used.
    """
    if run is not None:
        candidate = results_dir / f"comparison_{run}.json"
        if candidate.exists():
            return _load_eval_json(candidate)
        logger.warning("No evaluation comparison found for %s; using latest comparison JSON.", run)

    comparison_files = sorted(
        results_dir.glob("comparison_ppo_highway_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not comparison_files:
        raise FileNotFoundError(f"No comparison_ppo_highway_*.json files found in {results_dir}.")

    return _load_eval_json(comparison_files[0])


def _load_eval_json(path: Path) -> list[dict[str, Any]]:
    """Load and validate a comparison JSON file."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError(f"Evaluation JSON must contain a list or object: {path}")

    logger.info("Loaded evaluation results: %s", path)
    return sorted(payload, key=lambda item: _step_from_label(str(item.get("checkpoint_tag", ""))))


def _step_from_label(label: str) -> int:
    """Extract an integer training step from labels such as 'step 200,000'."""
    match = re.search(r"step[\s_]+([\d,]+)", label)
    return int(match.group(1).replace(",", "")) if match else 0


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_common_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", output_path)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def _metric(metrics: dict[str, list[float]], key: str) -> list[float]:
    values = metrics.get(key)
    if not values:
        raise ValueError(f"Required metric '{key}' is missing or empty in progress.csv.")
    return values


def plot_return_curve(
    steps: list[int],
    returns: list[float],
    output_path: Path,
    window: int = 10,
    label: str = "Mean episode reward",
) -> None:
    """Plot SB3 rollout reward over training."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(steps, returns, color="#9A3412", alpha=0.35, linewidth=1.2, label="Raw SB3 mean")
    if len(returns) >= 3:
        smooth_window = min(window, max(2, len(returns) // 8))
        ax.plot(
            steps,
            _rolling_mean(returns, smooth_window),
            color="#1D4ED8",
            linewidth=2.3,
            label=f"{smooth_window}-point rolling mean",
        )
    _apply_common_style(
        ax,
        "Training Reward Over Time",
        "Training timesteps",
        "Mean episode reward",
    )
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_episode_length_curve(
    steps: list[int],
    lengths: list[float],
    output_path: Path,
    window: int = 10,
) -> None:
    """Plot SB3 rollout episode length over training."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(steps, lengths, color="#047857", alpha=0.45, linewidth=1.3, label="Raw SB3 mean")
    if len(lengths) >= 3:
        smooth_window = min(window, max(2, len(lengths) // 8))
        ax.plot(
            steps,
            _rolling_mean(lengths, smooth_window),
            color="#7C3AED",
            linewidth=2.3,
            label=f"{smooth_window}-point rolling mean",
        )
    _apply_common_style(
        ax,
        "Episode Length Over Time",
        "Training timesteps",
        "Mean episode length (steps)",
    )
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_crash_rate(
    steps: list[int],
    crash_rates: list[float],
    output_path: Path,
) -> None:
    """Plot fraction of evaluation episodes ending in a crash vs. training step."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(steps, crash_rates, marker="o", linewidth=2.2, color="#B91C1C", label="Crash rate")
    _apply_common_style(ax, "Evaluation Crash Rate", "Training timesteps", "Crash rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_comparison(
    runs: list[dict[str, Any]],
    metric: str,
    output_path: Path,
) -> None:
    """Overlay multiple metric series on a single axes for comparison."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for run in runs:
        ax.plot(
            run["steps"],
            run["values"],
            marker="o",
            linewidth=2.2,
            label=run["label"],
        )
    _apply_common_style(ax, f"{metric} Comparison", "Training timesteps", metric)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def plot_evaluation_comparison(eval_results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot evaluation return, speed, length, and crash rate by checkpoint."""
    labels = [str(result.get("checkpoint_tag", f"Checkpoint {idx + 1}")) for idx, result in enumerate(eval_results)]
    x_positions = list(range(len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Checkpoint Evaluation Comparison", fontsize=16, fontweight="bold")

    panels = [
        ("mean_return", "Mean return", "#1D4ED8"),
        ("mean_length", "Mean episode length", "#047857"),
        ("mean_speed", "Mean speed (m/s)", "#9A3412"),
        ("crash_rate", "Crash rate", "#B91C1C"),
    ]

    for ax, (key, ylabel, color) in zip(axes.flat, panels):
        values = [float(result.get(key, 0.0)) for result in eval_results]
        ax.bar(x_positions, values, color=color, alpha=0.86)
        _apply_common_style(ax, ylabel, "Checkpoint", ylabel)
        ax.set_xticks(x_positions, labels, rotation=18, ha="right")
        if key == "crash_rate":
            ax.set_ylim(0, 1.05)

    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------

def generate_all_plots(
    log_dirs: list[Path] | None = None,
    output_dir: Path = ASSETS_DIR,
    labels: list[str] | None = None,
    run: str | None = None,
) -> None:
    """Load logs and produce the standard README figures."""
    del labels

    if log_dirs:
        run_dir = log_dirs[0]
        if run_dir == LOG_DIR:
            run_dir = resolve_run_dir(run)
        elif run_dir.name != "progress.csv" and not (run_dir / "progress.csv").exists():
            run_dir = LOG_DIR / run_dir.name
    else:
        run_dir = resolve_run_dir(run)

    if run is None and RUN_RE.match(run_dir.name):
        run = run_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)

    training = load_training_log(run_dir / "progress.csv")
    steps = [int(value) for value in _metric(training, "time/total_timesteps")]
    rewards = _metric(training, "rollout/ep_rew_mean")
    lengths = _metric(training, "rollout/ep_len_mean")

    plot_return_curve(steps, rewards, output_dir / "reward_plot.png")
    plot_episode_length_curve(steps, lengths, output_dir / "episode_length_plot.png")

    eval_results = load_eval_results(RESULTS_DIR, run=run)
    plot_evaluation_comparison(eval_results, output_dir / "evaluation_comparison.png")

    logger.info("Generated README figures from run: %s", run_dir.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labeled training and evaluation graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run name, e.g. ppo_highway_YYYYMMDD_HHMMSS. Defaults to the latest logs run.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        nargs="+",
        default=None,
        help="Optional log directory override. The default is the latest run under logs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory to write plots into.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Reserved for compatibility with earlier plot CLI examples.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    setup_logging()
    generate_all_plots(args.log_dir, args.output_dir, args.labels, run=args.run)


if __name__ == "__main__":
    main()
