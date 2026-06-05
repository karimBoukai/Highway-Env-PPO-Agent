"""
record_video.py
---------------
Phase 6: Evolution Video Generation for CMP4501.

For each of the three training checkpoints (untrained, half-trained, fully-
trained), multiple deterministic evaluation episodes are tested first.  A
representative episode is selected according to checkpoint-specific criteria,
then that exact episode is recorded as an MP4.  The three selected episodes
are also assembled into one evolution GIF.

Output files
------------
  videos/untrained_agent.mp4              -- selected episode (step 0)
  videos/half_trained_agent.mp4           -- selected episode (step 100k)
  videos/fully_trained_agent.mp4          -- selected episode (step 200k)
  assets/evolution.gif                    -- combined evolution animation

Usage
-----
    python src/record_video.py                    # latest run, auto-discovered
    python src/record_video.py --run ppo_highway_20260605_110802
    python src/record_video.py --run ppo_highway_20260605_110802 --attempts 20
    python src/record_video.py --gif-fps 8        # lower fps = smaller file
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, DEFAULT_CONFIG, CHECKPOINT_DIR, VIDEO_DIR, ASSETS_DIR
from utils import make_env, setup_logging
from evaluate import find_latest_run_checkpoints, find_run_checkpoints, _step_from_path

logger = logging.getLogger(__name__)

_SEP  = "=" * 66
_THIN = "-" * 66

# ── Visual constants ──────────────────────────────────────────────────────────
_HEADER_H   = 26   # extra pixels prepended above the rendered frame
_FOOTER_H   = 22   # extra pixels appended below the rendered frame
_BG_DARK    = (12, 12, 30)      # header / footer background (dark navy)
_BG_TITLE   = (10, 10, 35)      # title card background
_COL_YELLOW = (255, 220, 60)    # label text colour
_COL_WHITE  = (240, 240, 240)   # step / return text colour
_COL_GREY   = (160, 160, 160)   # secondary info text


# ---------------------------------------------------------------------------
# Episode selection data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeMetrics:
    """Metrics collected from one deterministic candidate episode."""

    attempt: int
    seed: int
    episode_return: float
    episode_length: int
    crashed: bool
    average_speed: float
    lane_changes: int
    action_changes: int
    speed_std: float

    @property
    def smoothness_score(self) -> float:
        """Higher is smoother: fewer lane/action changes and lower speed variance."""
        return -(
            2.0 * float(self.lane_changes)
            + 0.5 * float(self.action_changes)
            + 0.1 * self.speed_std
        )


@dataclass(frozen=True)
class SelectedEpisode:
    """Selected candidate episode plus the human-readable selection reason."""

    metrics: EpisodeMetrics
    reason: str


# ---------------------------------------------------------------------------
# PIL helpers
# ---------------------------------------------------------------------------

def _load_font(size: int = 13) -> Any:
    """Return a PIL bitmap font; works regardless of installed TTF files."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()            # older Pillow


def _text_width(draw: Any, text: str, font: Any) -> int:
    """Return the rendered pixel width of *text*; compatible with all Pillow versions."""
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0]
    except AttributeError:
        w, _ = draw.textsize(text, font=font)   # Pillow < 9
        return w


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _add_label(
    frame: np.ndarray,
    title: str,
    step: int,
    ep_return: float,
) -> np.ndarray:
    """
    Add a header strip (agent label) and footer strip (step / return) to a
    raw rendered frame by prepending and appending coloured bands.

    The original frame content is never modified; the bands extend the canvas
    vertically so the rendering itself is fully visible.

    Parameters
    ----------
    frame:
        Raw RGB uint8 array of shape ``(H, W, 3)`` from ``env.render()``.
    title:
        Agent label, e.g. ``"Fully-Trained Agent (Step 200,000)"``.
    step:
        Current step index within the episode.
    ep_return:
        Cumulative shaped reward accumulated so far.

    Returns
    -------
    np.ndarray
        Annotated frame; shape ``(H + _HEADER_H + _FOOTER_H, W, 3)``.
    """
    from PIL import Image, ImageDraw

    H, W = frame.shape[:2]
    font    = _load_font(13)
    font_sm = _load_font(11)

    header = Image.new("RGB", (W, _HEADER_H), _BG_DARK)
    dh = ImageDraw.Draw(header)
    dh.text((6, (_HEADER_H - 13) // 2), title, fill=_COL_YELLOW, font=font)

    content = Image.fromarray(frame.astype(np.uint8))

    footer = Image.new("RGB", (W, _FOOTER_H), _BG_DARK)
    df = ImageDraw.Draw(footer)
    ret_text  = f"return: {ep_return:+.2f}"
    step_text = f"step {step}"
    df.text((6, (_FOOTER_H - 11) // 2), ret_text, fill=_COL_WHITE, font=font_sm)
    sw = _text_width(df, step_text, font_sm)
    df.text((W - sw - 6, (_FOOTER_H - 11) // 2), step_text, fill=_COL_GREY, font=font_sm)

    canvas = Image.new("RGB", (W, H + _HEADER_H + _FOOTER_H), _BG_DARK)
    canvas.paste(header,  (0, 0))
    canvas.paste(content, (0, _HEADER_H))
    canvas.paste(footer,  (0, _HEADER_H + H))

    return np.array(canvas)


def _make_title_card(
    agent_name: str,
    step_label: str,
    size: tuple[int, int],
    n_frames: int,
) -> list[np.ndarray]:
    """
    Return a list of identical title-card frames to act as a section divider
    in the evolution GIF.

    Parameters
    ----------
    agent_name:
        Prominent line of text, e.g. ``"Fully-Trained Agent"``.
    step_label:
        Subtitle, e.g. ``"Step 200,000"``.
    size:
        ``(width, height)`` in pixels — must match the annotated episode frames.
    n_frames:
        Number of frames to hold the card (controls how long it displays).

    Returns
    -------
    list[np.ndarray]
        All identical frames; each of shape ``(height, width, 3)``.
    """
    from PIL import Image, ImageDraw

    W, H = size
    img  = Image.new("RGB", (W, H), _BG_TITLE)
    draw = ImageDraw.Draw(img)

    font_big = _load_font(16)
    font_sm  = _load_font(13)

    tw_big = _text_width(draw, agent_name, font_big)
    tw_sm  = _text_width(draw, step_label, font_sm) if step_label else 0

    mid_y = H // 2
    title_y = mid_y - 8 if not step_label else mid_y - 20
    draw.text(((W - tw_big) // 2, title_y), agent_name, fill=_COL_YELLOW, font=font_big)
    if step_label:
        draw.text(((W - tw_sm)  // 2, mid_y + 6), step_label, fill=_COL_WHITE, font=font_sm)

    line_col = (50, 50, 100)
    draw.line([(W // 6, mid_y - 26), (5 * W // 6, mid_y - 26)], fill=line_col, width=1)
    draw.line([(W // 6, mid_y + 24), (5 * W // 6, mid_y + 24)], fill=line_col, width=1)

    card = np.array(img)
    return [card] * n_frames


# ---------------------------------------------------------------------------
# Episode recording
# ---------------------------------------------------------------------------

def evaluate_candidate_episodes(
    cfg: Config,
    ckpt_path: Path,
    attempts: int = 20,
    max_steps: int = 200,
    seed_offset: int = 0,
) -> list[EpisodeMetrics]:
    """
    Run multiple deterministic episodes and return per-episode metrics.

    The policy is deterministic; diversity comes from resetting the traffic
    environment with a different seed for each attempt.
    """
    from stable_baselines3 import PPO

    env = make_env(cfg, seed=cfg.train.seed, use_reward_shaping=True)
    agent = PPO.load(str(ckpt_path), env=env)
    candidates: list[EpisodeMetrics] = []

    previous_action: int | None
    for attempt in range(1, attempts + 1):
        seed = cfg.train.seed + seed_offset + attempt - 1
        obs, _ = env.reset(seed=seed)

        ep_return = 0.0
        ep_length = 0
        crashed = False
        speeds: list[float] = []
        lane_changes = 0
        action_changes = 0
        previous_action = None
        done = False

        while not done and ep_length < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            action_int = int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            ep_return += float(reward)
            ep_length += 1

            if info.get("crashed", False):
                crashed = True
            if "speed" in info:
                speeds.append(float(info["speed"]))
            if action_int in {0, 2}:
                lane_changes += 1
            if previous_action is not None and action_int != previous_action:
                action_changes += 1
            previous_action = action_int

        speed_arr = np.array(speeds, dtype=np.float64) if speeds else np.array([], dtype=np.float64)
        candidates.append(
            EpisodeMetrics(
                attempt=attempt,
                seed=seed,
                episode_return=ep_return,
                episode_length=ep_length,
                crashed=crashed,
                average_speed=float(speed_arr.mean()) if speed_arr.size else 0.0,
                lane_changes=lane_changes,
                action_changes=action_changes,
                speed_std=float(speed_arr.std()) if speed_arr.size else 0.0,
            )
        )

    env.close()
    return candidates


def select_untrained_episode(candidates: list[EpisodeMetrics]) -> SelectedEpisode:
    """Choose a poor untrained episode to make random/weak behavior visible."""
    crashed = [m for m in candidates if m.crashed]
    pool = crashed or candidates
    selected = min(pool, key=lambda m: (m.episode_length, m.episode_return, m.smoothness_score))
    reason = "Poor representative: shortest crashed/low-return episode."
    if not selected.crashed:
        reason = "Poor representative: shortest available low-return episode."
    return SelectedEpisode(selected, reason)


def select_half_trained_episode(
    candidates: list[EpisodeMetrics],
    untrained_reference: EpisodeMetrics,
) -> SelectedEpisode:
    """Choose a medium-quality half-trained episode that still shows mistakes."""
    returns = sorted(m.episode_return for m in candidates)
    median_return = returns[len(returns) // 2]

    better_than_untrained = [
        m for m in candidates
        if m.episode_return > untrained_reference.episode_return
    ]
    mistake_pool = [m for m in better_than_untrained if m.crashed]
    pool = mistake_pool or better_than_untrained or candidates

    selected = min(
        pool,
        key=lambda m: (
            abs(m.episode_return - median_return),
            abs(m.episode_length - untrained_reference.episode_length),
            -m.episode_return,
        ),
    )
    if selected.episode_return > untrained_reference.episode_return and selected.crashed:
        reason = "Medium-quality episode: better return than untrained while still showing mistakes."
    elif selected.episode_return > untrained_reference.episode_return:
        reason = "Medium-quality episode: closest to median return among improved attempts."
    elif selected.crashed:
        reason = "Fallback medium episode: closest to median half-trained return, still showing mistakes."
    else:
        reason = "Fallback medium episode: closest to median half-trained return."
    return SelectedEpisode(selected, reason)


def select_fully_trained_episode(candidates: list[EpisodeMetrics]) -> SelectedEpisode:
    """Choose the best fully-trained episode by safety, return, length, smoothness."""
    selected = max(
        candidates,
        key=lambda m: (
            not m.crashed,
            m.episode_return,
            m.episode_length,
            m.smoothness_score,
        ),
    )
    if selected.crashed:
        reason = "Best available episode: highest return/length despite crash."
    else:
        reason = "Best episode: no crash, then highest return, length, and smoothness."
    return SelectedEpisode(selected, reason)


def record_episode(
    cfg: Config,
    ckpt_path: Path,
    label: str,
    seed: int,
    max_steps: int = 200,
) -> tuple[list[np.ndarray], EpisodeMetrics]:
    """
    Record one deterministic episode, returning annotated RGB frames.

    An environment with ``render_mode="rgb_array"`` is created internally;
    each step's rendered frame is annotated via ``_add_label`` and appended
    to the list.

    Parameters
    ----------
    cfg:
        Project configuration (``EnvConfig`` and seed used).
    ckpt_path:
        Path to the SB3 checkpoint file (without ``.zip`` suffix).
    label:
        Display text for the header strip on each frame.
    max_steps:
        Hard cap on episode length (normal highway-env episodes are ≤80).

    Returns
    -------
    tuple[list[np.ndarray], float, bool]
        ``(annotated_frames, metrics)``
    """
    from stable_baselines3 import PPO

    env   = make_env(cfg, seed=cfg.train.seed, render_mode="rgb_array",
                     use_reward_shaping=True)
    agent = PPO.load(str(ckpt_path), env=env)

    obs, _ = env.reset(seed=seed)
    frame0 = env.render()

    frames: list[np.ndarray] = []
    ep_return = 0.0
    crashed   = False
    step      = 0
    done      = False
    speeds: list[float] = []
    lane_changes = 0
    action_changes = 0
    previous_action: int | None = None

    if frame0 is not None:
        frames.append(_add_label(frame0, label, step, ep_return))

    while not done and step < max_steps:
        action, _ = agent.predict(obs, deterministic=True)
        action_int = int(action)
        obs, reward, terminated, truncated, info = env.step(action_int)
        done       = terminated or truncated
        ep_return += float(reward)
        step      += 1

        if info.get("crashed", False):
            crashed = True
        if "speed" in info:
            speeds.append(float(info["speed"]))
        if action_int in {0, 2}:
            lane_changes += 1
        if previous_action is not None and action_int != previous_action:
            action_changes += 1
        previous_action = action_int

        raw = env.render()
        if raw is not None:
            frames.append(_add_label(raw, label, step, ep_return))

    env.close()
    speed_arr = np.array(speeds, dtype=np.float64) if speeds else np.array([], dtype=np.float64)
    metrics = EpisodeMetrics(
        attempt=0,
        seed=seed,
        episode_return=ep_return,
        episode_length=step,
        crashed=crashed,
        average_speed=float(speed_arr.mean()) if speed_arr.size else 0.0,
        lane_changes=lane_changes,
        action_changes=action_changes,
        speed_std=float(speed_arr.std()) if speed_arr.size else 0.0,
    )
    return frames, metrics


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_as_mp4(frames: list[np.ndarray], path: Path, fps: int = 15) -> bool:
    """
    Write *frames* to an MP4 via imageio's bundled ffmpeg.

    Returns ``True`` on success.  On failure logs a warning and returns
    ``False`` so the caller can fall back to GIF.
    """
    try:
        import imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            macro_block_size=None,
            format="ffmpeg",
            codec="libx264",
            quality=7,      # CRF-equivalent quality (1=best, 10=worst)
        )
        for f in frames:
            writer.append_data(f.astype(np.uint8))
        writer.close()
        return True
    except Exception as exc:
        logger.warning("MP4 write failed (%s) — falling back to GIF.", exc)
        return False


def save_as_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> Path:
    """
    Write *frames* to an animated GIF using PIL (always available).

    Parameters
    ----------
    fps:
        Frame rate; converted to milliseconds-per-frame for PIL.

    Returns
    -------
    Path
        The ``.gif`` file that was written.
    """
    from PIL import Image

    gif_path = path.with_suffix(".gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / fps))
    pil_frames  = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return gif_path


def save_video(frames: list[np.ndarray], stem: Path, fps: int = 15) -> Path:
    """
    Save *frames* as MP4 if imageio/ffmpeg is available, otherwise as GIF.

    Parameters
    ----------
    stem:
        Output path **without** an extension.  The correct extension is
        appended automatically.

    Returns
    -------
    Path
        The file actually written.
    """
    mp4_path = stem.with_suffix(".mp4")
    if save_as_mp4(frames, mp4_path, fps=fps):
        logger.info("Saved MP4: %s", mp4_path)
        return mp4_path

    gif_path = save_as_gif(frames, stem.with_suffix(".gif"), fps=fps)
    logger.info("Saved GIF (fallback): %s", gif_path)
    return gif_path


# ---------------------------------------------------------------------------
# Evolution GIF
# ---------------------------------------------------------------------------

def create_evolution_gif(
    segments: dict[str, tuple[str, str, list[np.ndarray]]],
    output_path: Path,
    fps: int = 10,
    max_frames_per_segment: int = 60,
    title_duration_s: float = 2.0,
) -> Path:
    """
    Assemble an evolution GIF from three labelled episode segments.

    Structure::

        [title card: "Untrained Agent" / "Step 0"]  -- 2 s
        [episode frames from untrained]              -- downsampled
        [title card: "Half-Trained Agent" / ...]
        [episode frames from half-trained]
        [title card: "Fully-Trained Agent" / ...]
        [episode frames from fully-trained]

    Parameters
    ----------
    segments:
        Ordered dict: ``display_label`` → ``(agent_name, step_label, frames)``.
        Iteration order determines the sequence in the GIF.
    output_path:
        Desired output path; the suffix is replaced with ``.gif``.
    fps:
        GIF frame rate.
    max_frames_per_segment:
        Maximum episode frames per segment after downsampling (every other
        frame is selected, so up to ``max_frames_per_segment * 2`` raw frames
        are consumed).
    title_duration_s:
        Duration of each title card in seconds.

    Returns
    -------
    Path
        Path of the ``.gif`` file written.
    """
    from PIL import Image

    gif_path = output_path.with_suffix(".gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    n_title  = max(1, int(fps * title_duration_s))
    dur_ms   = max(1, int(1000 / fps))
    combined: list[Image.Image] = []
    frame_size: tuple[int, int] | None = None

    for display_label, (agent_name, step_label, frames) in segments.items():
        if not frames:
            logger.warning("Segment '%s' has no frames — skipped.", display_label)
            continue

        H, W = frames[0].shape[:2]
        if frame_size is None:
            frame_size = (W, H)

        for tf in _make_title_card(agent_name, step_label, frame_size, n_title):
            combined.append(Image.fromarray(tf.astype(np.uint8)))

        episode_subset = frames[::2][:max_frames_per_segment]
        for f in episode_subset:
            img = Image.fromarray(f.astype(np.uint8))
            if img.size != frame_size:
                img = img.resize(frame_size, Image.LANCZOS)
            combined.append(img)

    if not combined:
        raise RuntimeError("No frames were assembled — all segments were empty.")

    combined[0].save(
        str(gif_path),
        save_all=True,
        append_images=combined[1:],
        duration=dur_ms,
        loop=0,
        optimize=False,
    )
    size_kb = gif_path.stat().st_size // 1024
    logger.info(
        "Evolution GIF: %s  (%d frames, %d KB)", gif_path, len(combined), size_kb,
    )
    return gif_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_selection_table(rows: list[tuple[str, SelectedEpisode]]) -> None:
    """Print the selected-episode summary table requested by Phase 8."""
    print(_SEP)
    print("  Representative Episode Selection")
    print(_SEP)
    header = (
        f"  {'Checkpoint':<16}"
        f"{'Return':>11}"
        f"{'Length':>10}"
        f"{'Crash':>10}"
        f"  Selection Reason"
    )
    print(header)
    print(_THIN)
    for checkpoint, selected in rows:
        metrics = selected.metrics
        crash_status = "Yes" if metrics.crashed else "No"
        print(
            f"  {checkpoint:<16}"
            f"{metrics.episode_return:>+11.3f}"
            f"{metrics.episode_length:>10}"
            f"{crash_status:>10}"
            f"  {selected.reason}"
        )
    print(_SEP)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the video recording pipeline."""
    parser = argparse.ArgumentParser(
        description="Record evolution videos for CMP4501 Highway-Env agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run", type=str, default=None, metavar="RUN_NAME",
        help=(
            "Run prefix, e.g. 'ppo_highway_20260605_110802'.  "
            "Defaults to the most-recently-modified run in checkpoints/."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the RNG seed used when resetting the environment.",
    )
    parser.add_argument(
        "--attempts", type=int, default=20,
        help="Deterministic candidate episodes to evaluate per checkpoint.",
    )
    parser.add_argument(
        "--mp4-fps", type=int, default=15, dest="mp4_fps",
        help="Frame rate for individual MP4 videos.",
    )
    parser.add_argument(
        "--gif-fps", type=int, default=10, dest="gif_fps",
        help="Frame rate for the evolution GIF.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, dest="max_steps",
        help="Hard cap on recorded episode length (normal episodes are <= 80).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1.")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(VIDEO_DIR / "record.log")

    cfg = DEFAULT_CONFIG
    if args.seed is not None:
        cfg.train.seed = args.seed

    # ── Discover checkpoints ──────────────────────────────────────────────
    if args.run:
        checkpoints = find_run_checkpoints(args.run)
    else:
        checkpoints = find_latest_run_checkpoints()

    run_name = checkpoints["step_final"].stem.split("_step_")[0]
    step_0   = _step_from_path(checkpoints["step_0"])
    step_h   = _step_from_path(checkpoints["step_half"])
    step_f   = _step_from_path(checkpoints["step_final"])

    print()
    print(_SEP)
    print("  CMP4501 -- Autonomous Driving  |  Evolution Video")
    print(_SEP)
    print(f"  Run              : {run_name}")
    print(f"  Checkpoints      : step {step_0:,} / {step_h:,} / {step_f:,}")
    print(f"  Attempts/checkpt : {args.attempts}")
    print(f"  Individual videos: {VIDEO_DIR}/")
    print(f"  Evolution GIF    : {ASSETS_DIR}/evolution.gif")
    print(_SEP)
    print()

    # ── Candidate evaluation and recording plan ───────────────────────────
    plan = [
        (
            f"Untrained Agent (Step {step_0:,})",
            f"Untrained Agent - Step {step_0:,}",
            "",
            checkpoints["step_0"],
            "untrained_agent",
            "untrained",
            0,
        ),
        (
            f"Half-Trained Agent (Step {step_h:,})",
            f"Half-Trained Agent - Step {step_h:,}",
            "",
            checkpoints["step_half"],
            "half_trained_agent",
            "half",
            10_000,
        ),
        (
            f"Fully-Trained Agent (Step {step_f:,})",
            f"Fully-Trained Agent - Step {step_f:,}",
            "",
            checkpoints["step_final"],
            "fully_trained_agent",
            "full",
            20_000,
        ),
    ]

    segments: dict[str, tuple[str, str, list[np.ndarray]]] = {}
    video_paths: list[Path] = []
    candidates_by_kind: dict[str, list[EpisodeMetrics]] = {}
    selected: dict[str, SelectedEpisode] = {}

    print("  Evaluating candidate episodes ...")
    print(_THIN)
    for display_label, _agent_name, _step_label, ckpt_path, _safe_name, kind, seed_offset in plan:
        t0 = time.perf_counter()
        candidates = evaluate_candidate_episodes(
            cfg,
            ckpt_path,
            attempts=args.attempts,
            max_steps=args.max_steps,
            seed_offset=seed_offset,
        )
        elapsed = time.perf_counter() - t0
        candidates_by_kind[kind] = candidates
        best_return = max(m.episode_return for m in candidates)
        no_crash = sum(1 for m in candidates if not m.crashed)
        print(
            f"  {display_label:<38}"
            f" attempts={len(candidates):>2}"
            f" no_crash={no_crash:>2}"
            f" best_return={best_return:+7.2f}"
            f" ({elapsed:.1f}s)"
        )
    print()

    selected["untrained"] = select_untrained_episode(candidates_by_kind["untrained"])
    selected["half"] = select_half_trained_episode(
        candidates_by_kind["half"],
        selected["untrained"].metrics,
    )
    selected["full"] = select_fully_trained_episode(candidates_by_kind["full"])

    print_selection_table(
        [
            ("Untrained", selected["untrained"]),
            ("Half-Trained", selected["half"]),
            ("Fully-Trained", selected["full"]),
        ]
    )

    for display_label, agent_name, step_label, ckpt_path, safe_name, kind, _seed_offset in plan:
        choice = selected[kind]
        print(f"  Recording selected episode  [{display_label}]")
        print(_THIN)
        print(f"  Selected attempt: {choice.metrics.attempt}  |  seed: {choice.metrics.seed}")

        t0 = time.perf_counter()
        frames, recorded_metrics = record_episode(
            cfg,
            ckpt_path,
            display_label,
            seed=choice.metrics.seed,
            max_steps=args.max_steps,
        )
        elapsed = time.perf_counter() - t0

        video_stem = VIDEO_DIR / safe_name
        video_path = save_video(frames, video_stem, fps=args.mp4_fps)
        video_paths.append(video_path)

        status = "CRASHED" if recorded_metrics.crashed else "survived"
        print(f"  Frames     : {len(frames)}"
              f"  |  return: {recorded_metrics.episode_return:+.4f}"
              f"  |  length: {recorded_metrics.episode_length}"
              f"  |  avg_speed: {recorded_metrics.average_speed:.2f} m/s"
              f"  |  {status}"
              f"  ({elapsed:.1f}s)")
        print(f"  Saved      : {video_path}")
        print()

        segments[display_label] = (agent_name, step_label, frames)

    # ── Evolution GIF ─────────────────────────────────────────────────────
    print(_THIN)
    print("  Assembling evolution GIF ...")
    gif_path = create_evolution_gif(
        segments,
        ASSETS_DIR / "evolution.gif",
        fps=args.gif_fps,
        max_frames_per_segment=60,
        title_duration_s=2.0,
    )
    size_kb = gif_path.stat().st_size // 1024
    print(f"  Evolution GIF : {gif_path}")
    print(f"  File size     : {size_kb:,} KB")

    # ── Final summary ─────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("  All recordings complete")
    print(_SEP)
    print("  Individual videos:")
    for vp in video_paths:
        print(f"    {vp}")
    print(f"  Evolution GIF : {gif_path}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
