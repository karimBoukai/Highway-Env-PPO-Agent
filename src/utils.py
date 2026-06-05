"""
utils.py
--------
Shared helper functions used across the project.

Keeps every other module free of boilerplate by centralising:
  - Random-seed management for reproducibility.
  - Logging configuration.
  - Environment factory (``make_env`` / ``make_env_fn``) that applies the
    project config and wraps the environment for PPO compatibility.
  - Environment validation (``validate_env``).
  - Checkpoint save/load helpers (stubs – implemented in a later phase).
  - Timing utility.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import gymnasium

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set the global random seed for Python and NumPy.

    Parameters
    ----------
    seed:
        Non-negative integer seed value.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    logger.info("Global seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure the root logger with a console handler and an optional file handler.

    Parameters
    ----------
    log_file:
        If provided, a ``FileHandler`` writing to this path is added.  The
        parent directory is created if it does not exist.
    level:
        Minimum log level (default: ``INFO``).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    cfg: Any,
    seed: int | None = None,
    render_mode: str | None = None,
    use_reward_shaping: bool = False,
) -> gymnasium.Env:
    """
    Build and configure a Highway-Env Gymnasium environment from a ``Config``.

    The environment is configured and wrapped with
    ``gymnasium.wrappers.FlattenObservation`` so the observation is a 1-D
    ``Box`` array suitable for use with Stable-Baselines3's ``MlpPolicy``.

    Observation shape
    -----------------
    ``obs_vehicles_count * len(obs_features)``
    e.g. 5 vehicles × 5 features = **25-dimensional** flat vector.

    Action space
    ------------
    ``Discrete(5)`` – LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER.

    Parameters
    ----------
    cfg:
        Project ``Config`` instance; ``cfg.env`` and ``cfg.reward`` fields
        are used.
    seed:
        If provided the environment is reset with this seed immediately after
        construction so the internal RNG state is deterministic.
    render_mode:
        Passed directly to ``gymnasium.make()``.  Use ``None`` for headless
        training, ``"rgb_array"`` for video recording.
    use_reward_shaping:
        When ``True`` the environment is additionally wrapped with
        ``RewardShapingWrapper`` using the weights from ``cfg.reward``.
        Defaults to ``False`` so environment validation and testing
        scripts see the native highway-env reward.

    Returns
    -------
    gymnasium.Env
        Configured and wrapped environment ready for SB3.
    """
    import gymnasium as gym
    import highway_env  # noqa: F401 – side-effect: registers highway-env IDs
    from gymnasium.wrappers import FlattenObservation

    env = gym.make(cfg.env.env_id, render_mode=render_mode)
    env.unwrapped.configure(cfg.env.to_highway_config())
    env = FlattenObservation(env)

    if use_reward_shaping:
        from reward import RewardShapingWrapper
        env = RewardShapingWrapper(
            env,
            weights=cfg.reward,
            speed_range=cfg.env.reward_speed_range,
        )

    if seed is not None:
        env.reset(seed=seed)

    logger.debug(
        "Created env '%s' | obs=%s | act=%s | reward_shaping=%s | seed=%s",
        cfg.env.env_id,
        env.observation_space.shape,
        env.action_space,
        use_reward_shaping,
        seed,
    )
    return env


def make_env_fn(
    cfg: Any,
    seed: int = 0,
    use_reward_shaping: bool = False,
) -> Callable[[], gymnasium.Env]:
    """
    Return a no-argument callable that constructs a seeded environment.

    This factory pattern is required by
    ``stable_baselines3.common.env_util.make_vec_env`` and similar utilities
    that create multiple parallel environments.

    Parameters
    ----------
    cfg:
        Project ``Config`` instance.
    seed:
        Seed for this particular environment instance.
    use_reward_shaping:
        Passed through to ``make_env``.

    Returns
    -------
    Callable[[], gymnasium.Env]
        Zero-argument function that returns a fresh, seeded environment.

    Example
    -------
    >>> from stable_baselines3.common.env_util import make_vec_env
    >>> vec_env = make_vec_env(make_env_fn(cfg, seed=0, use_reward_shaping=True), n_envs=4)
    """
    def _init() -> gymnasium.Env:
        return make_env(cfg, seed=seed, use_reward_shaping=use_reward_shaping)

    return _init


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def validate_env(env: gymnasium.Env) -> None:
    """
    Assert that ``env`` is correctly configured for PPO training.

    Checks performed
    ----------------
    1. Observation space is a 1-D ``Box`` (required by ``MlpPolicy``).
    2. Action space is ``Discrete`` (required by PPO with discrete actions).
    3. ``env.reset()`` returns an observation whose shape matches the space.
    4. ``env.step(action)`` returns a tuple of the correct types and shapes.

    Parameters
    ----------
    env:
        Environment to validate (should already be ``FlattenObservation``-
        wrapped before calling this function).

    Raises
    ------
    ValueError
        If the observation or action space is incompatible with PPO.
    RuntimeError
        If ``reset()`` or ``step()`` raise an unexpected exception.
    """
    import gymnasium
    from gymnasium import spaces

    # ── 1. Observation space ──────────────────────────────────────────────
    if not isinstance(env.observation_space, spaces.Box):
        raise ValueError(
            f"Observation space must be Box, got "
            f"{type(env.observation_space).__name__}. "
            "PPO MlpPolicy requires a continuous Box observation."
        )
    if len(env.observation_space.shape) != 1:
        raise ValueError(
            f"Observation space must be 1-D, got shape "
            f"{env.observation_space.shape}. "
            "Wrap the environment with FlattenObservation before validating."
        )

    # ── 2. Action space ───────────────────────────────────────────────────
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError(
            f"Action space must be Discrete, got "
            f"{type(env.action_space).__name__}. "
            "Set action_type='DiscreteMetaAction' in EnvConfig."
        )

    # ── 3. reset() ────────────────────────────────────────────────────────
    try:
        obs, info = env.reset()
    except Exception as exc:
        raise RuntimeError(f"env.reset() raised an unexpected error: {exc}") from exc

    if not isinstance(obs, (list, __import__("numpy").ndarray)):
        raise ValueError(f"reset() must return an array-like observation, got {type(obs)}.")
    if obs.shape != env.observation_space.shape:
        raise ValueError(
            f"reset() returned obs shape {obs.shape}, "
            f"expected {env.observation_space.shape}."
        )
    if not isinstance(info, dict):
        raise ValueError(f"reset() info must be a dict, got {type(info)}.")

    # ── 4. step() ─────────────────────────────────────────────────────────
    action = env.action_space.sample()
    try:
        obs2, reward, terminated, truncated, info2 = env.step(action)
    except Exception as exc:
        raise RuntimeError(f"env.step() raised an unexpected error: {exc}") from exc

    if obs2.shape != env.observation_space.shape:
        raise ValueError(
            f"step() returned obs shape {obs2.shape}, "
            f"expected {env.observation_space.shape}."
        )
    if not isinstance(reward, (int, float)):
        raise ValueError(f"step() reward must be a scalar, got {type(reward)}.")
    if not isinstance(terminated, bool):
        raise ValueError(f"step() terminated must be bool, got {type(terminated)}.")
    if not isinstance(truncated, bool):
        raise ValueError(f"step() truncated must be bool, got {type(truncated)}.")

    logger.info(
        "validate_env passed | obs=%s | act=%s | n_actions=%d",
        env.observation_space.shape,
        env.action_space,
        env.action_space.n,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers  (stubs – implemented in the training phase)
# ---------------------------------------------------------------------------

def save_checkpoint(
    agent: Any,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Persist an SB3 agent to disk using its built-in ``save()`` method.

    SB3 automatically appends ``.zip`` to the path, so do **not** include
    the extension in ``path``.

    Parameters
    ----------
    agent:
        SB3 model instance (``PPO``, ``DQN``, etc.) with a ``save()`` method.
    path:
        Destination path **without** extension, e.g.
        ``checkpoints/run_step_0``.  The file written will be
        ``checkpoints/run_step_0.zip``.
    metadata:
        Currently unused; reserved for future extension.

    Raises
    ------
    TypeError
        If ``agent`` does not have a ``save()`` method.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(agent, "save"):
        raise TypeError(
            f"{type(agent).__name__} has no save() method. "
            "Pass an SB3 model instance."
        )
    save_stem = path.with_suffix("")  # strip any accidental extension
    agent.save(str(save_stem))
    logger.info("Checkpoint saved: %s.zip", save_stem)


def load_checkpoint(path: Path, env: Any) -> Any:
    """
    Load a saved SB3 checkpoint and return the reconstructed model.

    Unlike the original stub, this uses SB3's class-method ``load()`` which
    returns a **new** model instance rather than modifying one in-place.

    Parameters
    ----------
    path:
        Path to the saved ``.zip`` file (extension optional).
    env:
        Gymnasium environment to attach to the loaded model.  Required so
        the model's observation and action spaces are validated.

    Returns
    -------
    Any
        Loaded SB3 model (typically ``PPO``).
    """
    from stable_baselines3 import PPO

    load_path = path.with_suffix("") if path.suffix == ".zip" else path
    model = PPO.load(str(load_path), env=env)
    logger.info("Checkpoint loaded: %s.zip", load_path)
    return model


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class Timer:
    """
    Simple wall-clock timer for measuring elapsed time in training loops.

    Usage
    -----
    >>> t = Timer()
    >>> t.start()
    >>> # ... do work ...
    >>> print(f"Elapsed: {t.elapsed():.2f}s")
    """

    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        """Record the current time as the start instant."""
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Return seconds elapsed since the last ``start()`` call."""
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        return time.perf_counter() - self._start

    def reset(self) -> None:
        """Reset the timer (equivalent to calling ``start()`` again)."""
        self.start()
