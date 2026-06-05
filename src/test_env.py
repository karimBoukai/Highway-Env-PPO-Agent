"""
test_env.py
-----------
Phase 2 environment validation script for CMP4501.

Verifies that the Highway-Env environment launches correctly, is configured
as specified, and is compatible with PPO (MlpPolicy + Discrete actions).

Usage
-----
    python src/test_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from utils import make_env, validate_env, set_seed, setup_logging


# ── Action labels for DiscreteMetaAction ─────────────────────────────────────
ACTION_LABELS: dict[int, str] = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

SEPARATOR = "=" * 62


def _section(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "-" * (len(title)))


def main() -> None:
    setup_logging()
    cfg = DEFAULT_CONFIG
    set_seed(cfg.train.seed)

    print()
    print(SEPARATOR)
    print("  CMP4501 – Applied Reinforcement Learning")
    print("  Phase 2: Environment Setup Validation")
    print(SEPARATOR)

    # ── Build environment ─────────────────────────────────────────────────
    print(f"\n  Building environment  '{cfg.env.env_id}'  (seed={cfg.train.seed}) ...")
    env = make_env(cfg, seed=cfg.train.seed)
    hw = env.unwrapped          # base HighwayEnv instance (unwraps all wrappers)
    hw_cfg = hw.config          # live highway-env config dict

    # ── Observation space ─────────────────────────────────────────────────
    _section("Observation Space")
    obs_space = env.observation_space
    obs_dim = obs_space.shape[0]
    n_vehicles = hw_cfg["observation"]["vehicles_count"]
    n_features = len(hw_cfg["observation"]["features"])
    print(f"    Space type   : {type(obs_space).__name__}")
    print(f"    Shape        : {obs_space.shape}  ({n_vehicles} vehicles x {n_features} features)")
    print(f"    Flat dim     : {obs_dim}")
    print(f"    dtype        : {obs_space.dtype}")
    print(f"    Bounds       : low={obs_space.low[0]:.1f}, high={obs_space.high[0]:.1f}")
    print(f"    Features     : {hw_cfg['observation']['features']}")
    print(f"    Normalised   : {hw_cfg['observation']['normalize']}")
    print(f"    Absolute     : {hw_cfg['observation']['absolute']}")
    print(f"    MlpPolicy OK : {len(obs_space.shape) == 1}")

    # ── Action space ──────────────────────────────────────────────────────
    _section("Action Space")
    act_space = env.action_space
    print(f"    Space type   : {type(act_space).__name__}({act_space.n})")
    print(f"    Num actions  : {act_space.n}")
    for idx, label in ACTION_LABELS.items():
        print(f"      {idx} -> {label}")

    # ── Road configuration ────────────────────────────────────────────────
    _section("Road Configuration")
    print(f"    Lanes        : {hw_cfg['lanes_count']}")
    print(f"    Vehicles     : {hw_cfg['vehicles_count']}")
    print(f"    Duration     : {hw_cfg['duration']} s")
    print(f"    Policy freq  : {hw_cfg['policy_frequency']} Hz")
    print(f"    Sim freq     : {hw_cfg['simulation_frequency']} Hz")

    # ── Reward configuration ──────────────────────────────────────────────
    _section("Reward Configuration")
    print(f"    Speed range  : {hw_cfg['reward_speed_range']} m/s")
    print(f"    Normalised   : {hw_cfg['normalize_reward']}")
    print(f"    Collision    : {hw_cfg.get('collision_reward', -1)}")
    print(f"    High speed   : {hw_cfg.get('high_speed_reward', 0.4)}")
    print(f"    Right lane   : {hw_cfg.get('right_lane_reward', 0.1)}")

    # ── Functional validation ─────────────────────────────────────────────
    _section("Functional Validation")
    print("    Running validate_env() ...")
    validate_env(env)
    print("    reset() + step() checks  : PASSED")
    print("    Observation shape check  : PASSED")
    print("    Action space check       : PASSED")
    print("    PPO MlpPolicy compatible : PASSED")

    # ── Sample episode (5 random steps) ──────────────────────────────────
    _section("Sample Episode (5 random steps)")
    obs, info = env.reset(seed=cfg.train.seed)
    print(f"    reset() obs shape : {obs.shape}")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(
            f"    step {step + 1}"
            f"  action={ACTION_LABELS[action]:<12}"
            f"  reward={reward:+.4f}"
            f"  crashed={info.get('crashed', '?')}"
            f"  speed={info.get('speed', 0.0):.1f} m/s"
            f"  done={done}"
        )
        if done:
            obs, info = env.reset()

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print(SEPARATOR)
    print("  All checks passed.")
    print(f"  Environment '{cfg.env.env_id}' is ready for PPO training.")
    print(f"  Observation : Box{obs_space.shape}  (flat, normalised Kinematics)")
    print(f"  Actions     : Discrete({act_space.n})  (DiscreteMetaAction)")
    print(SEPARATOR)
    print()


if __name__ == "__main__":
    main()
