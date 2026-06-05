"""
reward.py
---------
Custom reward shaping for CMP4501 Autonomous Driving with Highway-Env.

Reward formula
--------------
The shaped reward is a weighted linear combination of five normalised
components.  Every raw component r_i(t) is bounded in [0, 1]; the sign of
each contribution is carried entirely by its scalar weight w_i:

    R(t) = w_spd  * r_spd(t)
         + w_lane * r_lane(t)
         + w_col  * r_col(t)
         + w_lc   * r_lc(t)
         + w_slow * r_slow(t)

Component definitions
---------------------
r_spd(t)   Speed reward – normalised ego speed inside target range [v_min, v_max]:
               r_spd = clip( (v(t) - v_min) / (v_max - v_min),  0, 1 )

r_lane(t)  Lane-discipline reward – normalised lane index (0 = leftmost, 1 = rightmost):
               r_lane = lane_id(t) / (N_lanes - 1)
           Rewards keeping to the rightmost (slow/keep-right) lane.

r_col(t)   Collision penalty indicator:
               r_col = 1  if crashed(t)  else  0

r_lc(t)    Unsafe lane-change indicator – penalises every lateral manoeuvre:
               r_lc = 1  if a(t) in {LANE_LEFT, LANE_RIGHT}  else  0

r_slow(t)  Low-speed penalty – normalised deficit below v_min:
               r_slow = clip( (v_min - v(t)) / v_min,  0, 1 )
           Only positive when v(t) < v_min.

Default weights
---------------
    w_spd  = +0.4   (encourage target-range speed)
    w_lane = +0.1   (encourage rightmost lane)
    w_col  = -1.0   (heavily discourage crashes)
    w_lc   = -0.1   (mildly discourage lateral manoeuvres)
    w_slow = -0.2   (additionally discourage crawling)

Theoretical bounds:
    R_max = w_spd + w_lane = +0.5   (full speed, rightmost lane, no crash)
    R_min = w_col + w_lc + w_slow  = -1.3   (crash + lane change + standing still)

LaTeX (for the CMP4501 methodology section)
-------------------------------------------
    R(t) = w_{\\text{spd}}  \\cdot r_{\\text{spd}}(t)
         + w_{\\text{lane}} \\cdot r_{\\text{lane}}(t)
         + w_{\\text{col}}  \\cdot r_{\\text{col}}(t)
         + w_{\\text{lc}}   \\cdot r_{\\text{lc}}(t)
         + w_{\\text{slow}} \\cdot r_{\\text{slow}}(t)

    r_{\\text{spd}}(t)  = \\text{clip}\\!
                           \\left(\\frac{v(t)-v_{\\min}}{v_{\\max}-v_{\\min}},0,1\\right)

    r_{\\text{lane}}(t) = \\frac{\\ell(t)}{L-1}

    r_{\\text{col}}(t)  = \\mathbf{1}[\\text{crashed}(t)]

    r_{\\text{lc}}(t)   = \\mathbf{1}\\bigl[a_t \\in
                           \\{\\text{LANE\\_LEFT},\\,\\text{LANE\\_RIGHT}\\}\\bigr]

    r_{\\text{slow}}(t) = \\text{clip}\\!
                           \\left(\\frac{v_{\\min}-v(t)}{v_{\\min}},0,1\\right)

Public API
----------
RewardWeights          – per-component weight configuration dataclass
RewardComponents       – named container for a single-step reward breakdown
compute_reward_components – returns RewardComponents (for logging / analysis)
compute_shaped_reward  – returns the scalar total reward (for wrappers)
RewardShapingWrapper   – Gymnasium Wrapper that replaces the native reward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import gymnasium


# ---------------------------------------------------------------------------
# Action constants (DiscreteMetaAction)
# ---------------------------------------------------------------------------

_LANE_LEFT: int = 0
_IDLE: int = 1
_LANE_RIGHT: int = 2
_FASTER: int = 3
_SLOWER: int = 4

_LANE_CHANGE_ACTIONS: frozenset[int] = frozenset({_LANE_LEFT, _LANE_RIGHT})


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """
    Per-component scalar weights for the shaped reward function.

    Positive weights encourage the corresponding behaviour; negative weights
    penalise it.  All raw components r_i are normalised to [0, 1], so the
    magnitude of each weight directly controls its contribution to R(t).

    Attributes
    ----------
    speed : float
        Weight for the speed reward component (default +0.4).
        Encourages travelling near the target speed range.
    lane : float
        Weight for the lane-discipline component (default +0.1).
        Encourages keeping to the rightmost lane.
    collision : float
        Weight for the collision-penalty component (default -1.0).
        Heavily discourages crashing into other vehicles.
    unsafe_lc : float
        Weight for the unsafe-lane-change component (default -0.1).
        Mildly penalises every lateral manoeuvre to reduce unnecessary
        weaving.
    low_speed : float
        Weight for the low-speed penalty component (default -0.2).
        Provides an additional penalty when the agent crawls below v_min,
        complementing the zero speed reward with an active punishment.
    """

    speed: float = 0.4
    lane: float = 0.1
    collision: float = -1.0
    unsafe_lc: float = -0.1
    low_speed: float = -0.2


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class RewardComponents:
    """
    Breakdown of a single-step shaped reward into its five components.

    All raw component values (``speed``, ``lane``, ``collision``,
    ``unsafe_lc``, ``low_speed``) lie in [0, 1].  The ``total`` field is
    their weighted sum, which may be negative.

    This container is returned by ``compute_reward_components`` and is
    intended for logging, debugging, and reward-breakdown plots.

    Attributes
    ----------
    speed : float
        r_spd(t) – normalised ego speed in [0, 1].
    lane : float
        r_lane(t) – normalised lane index in [0, 1].
    collision : float
        r_col(t) – 1.0 if crashed, else 0.0.
    unsafe_lc : float
        r_lc(t) – 1.0 if a lane-change action was taken, else 0.0.
    low_speed : float
        r_slow(t) – normalised speed deficit below v_min, in [0, 1].
    total : float
        R(t) – the weighted sum of all components.
    """

    speed: float
    lane: float
    collision: float
    unsafe_lc: float
    low_speed: float
    total: float

    def as_dict(self) -> dict[str, float]:
        """Return all fields as an ordered dict (convenient for logging)."""
        return {
            "speed": self.speed,
            "lane": self.lane,
            "collision": self.collision,
            "unsafe_lc": self.unsafe_lc,
            "low_speed": self.low_speed,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Private helpers – individual component calculations
# ---------------------------------------------------------------------------

def _r_speed(speed: float, v_min: float, v_max: float) -> float:
    """Normalised speed in target range; 0 below v_min, 1 at or above v_max."""
    if v_max <= v_min:
        return 0.0
    return float(np.clip((speed - v_min) / (v_max - v_min), 0.0, 1.0))


def _r_low_speed(speed: float, v_min: float) -> float:
    """Normalised speed deficit below v_min; 0 at or above v_min, 1 at rest."""
    if v_min <= 0.0:
        return 0.0
    return float(np.clip((v_min - speed) / v_min, 0.0, 1.0))


def _r_lane(info: dict[str, Any]) -> float:
    """
    Read the already-normalised right-lane reward from the env's info dict.

    Highway-Env computes this as ``lane_id / (N_lanes - 1)``, which maps the
    leftmost lane to 0.0 and the rightmost lane to 1.0.
    """
    return float(info.get("rewards", {}).get("right_lane_reward", 0.0))


def _r_collision(info: dict[str, Any]) -> float:
    """Return 1.0 if the ego vehicle crashed this step, else 0.0."""
    return 1.0 if info.get("crashed", False) else 0.0


def _r_unsafe_lc(info: dict[str, Any]) -> float:
    """Return 1.0 if a lateral manoeuvre (LANE_LEFT or LANE_RIGHT) was executed."""
    return 1.0 if int(info.get("action", _IDLE)) in _LANE_CHANGE_ACTIONS else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reward_components(
    info: dict[str, Any],
    weights: RewardWeights,
    speed_range: tuple[float, float] = (20.0, 30.0),
) -> RewardComponents:
    """
    Decompose a single environment step into its five reward components and
    compute the weighted total.

    Parameters
    ----------
    info :
        The ``info`` dict returned by ``env.step(action)``.  Expected keys:
        ``"speed"`` (float), ``"crashed"`` (bool), ``"action"`` (int),
        ``"rewards"`` (dict with ``"right_lane_reward"`` float).
        Missing keys are substituted with safe defaults (no reward, no crash,
        IDLE action), so the function is safe to call on synthetic dicts.
    weights :
        Per-component weights.  Obtain from ``DEFAULT_CONFIG.reward`` or
        construct a custom ``RewardWeights`` instance.
    speed_range :
        ``(v_min, v_max)`` in m/s defining the target speed band.  Should
        match ``EnvConfig.reward_speed_range``.

    Returns
    -------
    RewardComponents
        Named container with all five raw components and the scalar total.

    Example
    -------
    >>> from reward import RewardWeights, compute_reward_components
    >>> info = {'speed': 27.0, 'crashed': False, 'action': 1,
    ...         'rewards': {'right_lane_reward': 1.0}}
    >>> rc = compute_reward_components(info, RewardWeights(), (20.0, 30.0))
    >>> rc.total
    0.6
    """
    v_min, v_max = speed_range
    speed = float(info.get("speed", 0.0))

    r_spd = _r_speed(speed, v_min, v_max)
    r_lane = _r_lane(info)
    r_col = _r_collision(info)
    r_lc = _r_unsafe_lc(info)
    r_slow = _r_low_speed(speed, v_min)

    total = (
        weights.speed    * r_spd
        + weights.lane     * r_lane
        + weights.collision * r_col
        + weights.unsafe_lc * r_lc
        + weights.low_speed * r_slow
    )

    return RewardComponents(
        speed=r_spd,
        lane=r_lane,
        collision=r_col,
        unsafe_lc=r_lc,
        low_speed=r_slow,
        total=float(total),
    )


def compute_shaped_reward(
    info: dict[str, Any],
    weights: RewardWeights | None = None,
    speed_range: tuple[float, float] = (20.0, 30.0),
) -> float:
    """
    Compute the scalar shaped reward for one environment step.

    This is a convenience wrapper around ``compute_reward_components`` that
    discards the per-component breakdown and returns only the total.  Use
    ``compute_reward_components`` when you need the individual values for
    logging or analysis.

    Parameters
    ----------
    info :
        The ``info`` dict from ``env.step()``.
    weights :
        Reward weights.  Defaults to ``RewardWeights()`` if not supplied.
    speed_range :
        ``(v_min, v_max)`` target speed band in m/s.

    Returns
    -------
    float
        The weighted total reward R(t).
    """
    w = weights if weights is not None else RewardWeights()
    return compute_reward_components(info, w, speed_range).total


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

class RewardShapingWrapper(gymnasium.Wrapper):
    """
    Gymnasium ``Wrapper`` that replaces the environment's native reward with
    the custom shaped reward defined in this module.

    The wrapper intercepts ``step()`` and substitutes the returned reward with
    the output of ``compute_shaped_reward``.  All other environment attributes
    and methods (observation space, action space, ``reset()``, etc.) are
    delegated unchanged to the wrapped environment.

    Parameters
    ----------
    env :
        The environment to wrap.  Can be a raw ``HighwayEnv`` instance or one
        already wrapped with ``FlattenObservation``.
    weights :
        Reward weights to apply.  Defaults to ``RewardWeights()`` if not
        supplied.  Typically obtained from ``cfg.reward`` so that the weights
        remain configurable through ``config.py``.
    speed_range :
        ``(v_min, v_max)`` target speed band in m/s.  If ``None``, the value
        is read from ``env.unwrapped.config['reward_speed_range']``.

    Usage
    -----
    Standalone wrapping::

        env = gymnasium.make("highway-v0")
        env = FlattenObservation(env)
        env = RewardShapingWrapper(env, weights=cfg.reward)

    Via ``make_env`` (Phase 4)::

        env = make_env(cfg, use_reward_shaping=True)

    The ``info`` dict returned by ``step()`` is passed through unmodified, so
    downstream code (evaluation, logging) can still read the native reward
    components via ``info["rewards"]``.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        weights: RewardWeights | None = None,
        speed_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(env)
        self.weights: RewardWeights = weights if weights is not None else RewardWeights()

        if speed_range is not None:
            self._speed_range: tuple[float, float] = speed_range
        else:
            raw = env.unwrapped.config.get("reward_speed_range", [20.0, 30.0])
            self._speed_range = (float(raw[0]), float(raw[1]))

    def step(
        self, action: int
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Step the environment and replace the native reward.

        Parameters
        ----------
        action :
            Integer action from ``Discrete(5)`` action space.

        Returns
        -------
        tuple
            ``(observation, shaped_reward, terminated, truncated, info)``
            where ``shaped_reward`` is the output of
            ``compute_shaped_reward`` and all other fields are unchanged.
        """
        obs, _native_reward, terminated, truncated, info = self.env.step(action)
        shaped = compute_shaped_reward(info, self.weights, self._speed_range)
        return obs, shaped, terminated, truncated, info

    @property
    def reward_weights(self) -> RewardWeights:
        """Read-only access to the current reward weights."""
        return self.weights
