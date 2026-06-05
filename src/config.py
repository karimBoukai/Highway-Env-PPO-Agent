"""
config.py
---------
Central configuration for the Autonomous Driving project.

All hyperparameters, environment settings, paths, and training options are
defined here as a single dataclass so every other module imports one object
instead of scattered magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from reward import RewardWeights  # noqa: E402 – reward has no config dependency


# ---------------------------------------------------------------------------
# Root paths (relative to the project root, one level above src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_DIR = PROJECT_ROOT / "videos"
ASSETS_DIR = PROJECT_ROOT / "assets"


@dataclass
class EnvConfig:
    """
    Highway-Env environment settings.

    Observation design
    ------------------
    ``observation_type`` is set to ``"Kinematics"`` which returns a
    ``(obs_vehicles_count × len(obs_features))`` array.  After passing the
    environment through ``gymnasium.wrappers.FlattenObservation`` this becomes
    a 1-D ``Box`` of size ``obs_vehicles_count * len(obs_features)``, which is
    directly compatible with Stable-Baselines3's ``MlpPolicy``.

    Action design
    -------------
    ``"DiscreteMetaAction"`` produces a ``Discrete(5)`` space:
      0 – LANE_LEFT  |  1 – IDLE  |  2 – LANE_RIGHT
      3 – FASTER     |  4 – SLOWER
    """

    # ── Environment identity ───────────────────────────────────────────────
    env_id: str = "highway-v0"

    # ── Road / traffic ────────────────────────────────────────────────────
    lanes_count: int = 4
    vehicles_count: int = 20           # total vehicles on the road
    duration: int = 40                 # episode length in seconds
    policy_frequency: int = 2          # agent decisions per second
    simulation_frequency: int = 15    # physics steps per second

    # ── Observation ───────────────────────────────────────────────────────
    observation_type: str = "Kinematics"
    obs_vehicles_count: int = 5        # number of nearby vehicles to observe
    obs_features: list[str] = field(
        default_factory=lambda: ["presence", "x", "y", "vx", "vy"]
    )
    obs_features_range: dict[str, list[float]] = field(
        default_factory=lambda: {
            "x": [-100.0, 100.0],
            "y": [-100.0, 100.0],
            "vx": [-20.0, 20.0],
            "vy": [-20.0, 20.0],
        }
    )
    obs_normalize: bool = True         # normalise each feature to [-1, 1]
    obs_absolute: bool = False         # use relative coords w.r.t. ego

    # ── Action ────────────────────────────────────────────────────────────
    action_type: str = "DiscreteMetaAction"

    # ── Reward ────────────────────────────────────────────────────────────
    reward_speed_range: tuple[float, float] = field(
        default_factory=lambda: (20.0, 30.0)
    )
    normalize_reward: bool = True

    def to_highway_config(self) -> dict[str, Any]:
        """
        Serialise this dataclass into the dict format expected by
        ``env.unwrapped.configure()``.

        Returns
        -------
        dict
            A nested dict compatible with Highway-Env's internal config system.
        """
        return {
            "lanes_count": self.lanes_count,
            "vehicles_count": self.vehicles_count,
            "duration": self.duration,
            "policy_frequency": self.policy_frequency,
            "simulation_frequency": self.simulation_frequency,
            "reward_speed_range": list(self.reward_speed_range),
            "normalize_reward": self.normalize_reward,
            "observation": {
                "type": self.observation_type,
                "vehicles_count": self.obs_vehicles_count,
                "features": list(self.obs_features),
                "features_range": self.obs_features_range,
                "absolute": self.obs_absolute,
                "normalize": self.obs_normalize,
            },
            "action": {
                "type": self.action_type,
            },
        }


@dataclass
class ModelConfig:
    """Neural network / policy architecture settings."""

    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dueling: bool = True               # use dueling network heads
    noisy_nets: bool = False           # NoisyNet exploration


@dataclass
class TrainConfig:
    """
    PPO training loop and optimisation hyperparameters.

    All fields map directly to SB3's ``PPO.__init__`` or ``learn()`` arguments
    so there is one place to change any training knob.

    Checkpoint schedule
    -------------------
    Three checkpoints are saved automatically by ``train.py``:
      - step 0                     (untrained baseline)
      - step ``total_timesteps//2``  (half-trained)
      - step ``total_timesteps``     (fully-trained)
    """

    algorithm: str = "PPO"

    # ── Training duration ─────────────────────────────────────────────────
    total_timesteps: int = 200_000
    seed: int = 42
    n_envs: int = 1                  # parallel environments (DummyVecEnv)

    # ── PPO core hyperparameters ──────────────────────────────────────────
    learning_rate: float = 5e-4
    n_steps: int = 512               # rollout steps collected per env before update
    batch_size: int = 64             # PPO minibatch size  (must divide n_steps)
    n_epochs: int = 10               # gradient epochs per rollout buffer
    gamma: float = 0.99              # discount factor
    gae_lambda: float = 0.95         # GAE-Lambda smoothing parameter
    clip_range: float = 0.2          # PPO clipping epsilon
    ent_coef: float = 0.01           # entropy bonus coefficient (exploration)
    vf_coef: float = 0.5             # value-function loss weight
    max_grad_norm: float = 0.5       # gradient norm clipping threshold


@dataclass
class EvalConfig:
    """Evaluation and checkpointing settings."""

    eval_episodes: int = 10
    eval_freq: int = 5_000            # evaluate every N training steps
    save_freq: int = 10_000           # checkpoint every N training steps
    deterministic: bool = True        # greedy policy during evaluation
    render: bool = False


@dataclass
class Config:
    """
    Top-level configuration object – import this in every module.

    Reward weights are exposed as ``cfg.reward`` (a ``RewardWeights``
    instance) so they can be tuned centrally without touching ``reward.py``.
    For example::

        cfg = DEFAULT_CONFIG
        cfg.reward.collision = -2.0   # double the collision penalty
        cfg.reward.speed     = 0.6    # increase speed incentive
    """

    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)


# ---------------------------------------------------------------------------
# Module-level default instance (override fields as needed)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = Config()
