"""
test_reward.py
--------------
Phase 3 reward-function demonstration for CMP4501.

Constructs six representative driving scenarios as synthetic ``info`` dicts
(matching the structure returned by highway-env's ``env.step()``) and prints
a full breakdown of every reward component together with the weighted total.

Run with:
    python src/test_reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG
from reward import RewardWeights, RewardComponents, compute_reward_components

# ── Formatting constants ──────────────────────────────────────────────────────
SEP = "=" * 76
THIN = "-" * 76
COL_W = 12   # width for each component column


# ---------------------------------------------------------------------------
# Synthetic scenario definitions
# ---------------------------------------------------------------------------
# Each entry is (label, info_dict).
# info keys must match what highway-env puts in info after env.step():
#   speed           – ego vehicle speed in m/s
#   crashed         – bool
#   action          – integer DiscreteMetaAction index
#   rewards         – dict with 'right_lane_reward' in [0, 1]
# ---------------------------------------------------------------------------

def _make_info(
    speed: float,
    crashed: bool,
    action: int,
    right_lane: float,
) -> dict:
    return {
        "speed": speed,
        "crashed": crashed,
        "action": action,
        "rewards": {"right_lane_reward": right_lane},
    }


ACTION_LABELS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

SCENARIOS: list[tuple[str, dict]] = [
    (
        "Optimal cruise",
        _make_info(speed=27.0, crashed=False, action=1, right_lane=1.0),
        # speed=27 (in [20,30]), rightmost lane, no crash, no lane change
    ),
    (
        "Fast but wrong lane",
        _make_info(speed=30.0, crashed=False, action=1, right_lane=0.0),
        # speed=30 (at target max), leftmost lane, no crash
    ),
    (
        "Slow crawl, middle lane",
        _make_info(speed=10.0, crashed=False, action=1, right_lane=0.33),
        # speed=10 (well below v_min=20), middle lane
    ),
    (
        "Aggressive lane change",
        _make_info(speed=25.0, crashed=False, action=0, right_lane=0.67),
        # speed=25 (in range), LANE_LEFT action, no crash
    ),
    (
        "Collision",
        _make_info(speed=16.0, crashed=True,  action=1, right_lane=0.0),
        # below v_min, crashed, IDLE
    ),
    (
        "Crash during lane change",
        _make_info(speed=14.0, crashed=True,  action=2, right_lane=0.33),
        # below v_min, crashed, LANE_RIGHT (worst case)
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    """Format a component value with consistent width and sign."""
    return f"{value:+{COL_W}.4f}"


def _bar(value: float, w_max: float = 0.5, width: int = 20) -> str:
    """
    Render a mini ASCII bar showing the contribution relative to the
    theoretical maximum positive reward (R_max = +0.5 by default).
    """
    clamped = max(-1.3, min(value, w_max))
    filled = int((clamped - (-1.3)) / (w_max - (-1.3)) * width)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo(weights: RewardWeights, speed_range: tuple[float, float]) -> None:
    v_min, v_max = speed_range

    print()
    print(SEP)
    print("  CMP4501 -- Phase 3: Custom Reward Function Demo")
    print(SEP)

    # ── Configuration summary ──────────────────────────────────────────────
    print()
    print("  Reward Weights (from DEFAULT_CONFIG.reward)")
    print(THIN)
    print(f"    w_speed     = {weights.speed:+.2f}   "
          "encourages target-range speed")
    print(f"    w_lane      = {weights.lane:+.2f}   "
          "encourages rightmost lane")
    print(f"    w_collision = {weights.collision:+.2f}   "
          "heavily penalises crashes")
    print(f"    w_unsafe_lc = {weights.unsafe_lc:+.2f}   "
          "mildly penalises lane changes")
    print(f"    w_low_speed = {weights.low_speed:+.2f}   "
          "additionally penalises crawling")
    print()
    print(f"  Speed range  : [{v_min:.0f}, {v_max:.0f}] m/s")
    print(f"  R_max (theory): {weights.speed + weights.lane:+.2f}  "
          "(full speed + rightmost lane)")
    print(f"  R_min (theory): {weights.collision + weights.unsafe_lc + weights.low_speed:+.2f}  "
          "(crash + lane change + standing still)")

    # ── Formula ───────────────────────────────────────────────────────────
    print()
    print("  Formula")
    print(THIN)
    print("  R(t) = w_speed * r_spd(t)  +  w_lane * r_lane(t)")
    print("       + w_collision * r_col(t)  +  w_unsafe_lc * r_lc(t)")
    print("       + w_low_speed * r_slow(t)")
    print()
    print("  where:")
    print(f"    r_spd  = clip((v - {v_min:.0f}) / ({v_max:.0f} - {v_min:.0f}), 0, 1)")
    print(f"    r_lane = lane_id / (N_lanes - 1)           in [0, 1]")
    print(f"    r_col  = 1 if crashed else 0               in {{0, 1}}")
    print(f"    r_lc   = 1 if action in {{LANE_LEFT,LANE_RIGHT}} else 0  in {{0,1}}")
    print(f"    r_slow = clip(({v_min:.0f} - v) / {v_min:.0f}, 0, 1)")

    # ── Component table ────────────────────────────────────────────────────
    header_components = (
        f"  {'Scenario':<26}"
        f"{'r_spd':>{COL_W}}"
        f"{'r_lane':>{COL_W}}"
        f"{'r_col':>{COL_W}}"
        f"{'r_lc':>{COL_W}}"
        f"{'r_slow':>{COL_W}}"
        f"{'TOTAL':>{COL_W}}"
        f"  Bar (-1.3 .. +0.5)"
    )
    print()
    print("  Scenario Breakdown")
    print(THIN)
    print(header_components)
    print(THIN)

    for label, info in SCENARIOS:
        rc: RewardComponents = compute_reward_components(info, weights, speed_range)
        action_label = ACTION_LABELS.get(int(info["action"]), "?")
        speed_val = info["speed"]

        row = (
            f"  {label:<26}"
            f"{rc.speed:>{COL_W}.4f}"
            f"{rc.lane:>{COL_W}.4f}"
            f"{rc.collision:>{COL_W}.4f}"
            f"{rc.unsafe_lc:>{COL_W}.4f}"
            f"{rc.low_speed:>{COL_W}.4f}"
            f"{rc.total:>{COL_W}.4f}"
            f"  {_bar(rc.total)}"
        )
        print(row)
        print(
            f"  {'':26}"
            f"  v={speed_val:.1f} m/s | "
            f"action={action_label} | "
            f"crashed={info['crashed']}"
        )

    print(THIN)

    # ── Weighted-contribution table ────────────────────────────────────────
    print()
    print("  Weighted Contributions  (weight x raw_component)")
    print(THIN)
    header_contrib = (
        f"  {'Scenario':<26}"
        f"{'spd*w':>{COL_W}}"
        f"{'lane*w':>{COL_W}}"
        f"{'col*w':>{COL_W}}"
        f"{'lc*w':>{COL_W}}"
        f"{'slow*w':>{COL_W}}"
        f"{'TOTAL':>{COL_W}}"
    )
    print(header_contrib)
    print(THIN)

    for label, info in SCENARIOS:
        rc = compute_reward_components(info, weights, speed_range)
        spd_c  = weights.speed     * rc.speed
        lane_c = weights.lane      * rc.lane
        col_c  = weights.collision * rc.collision
        lc_c   = weights.unsafe_lc * rc.unsafe_lc
        slow_c = weights.low_speed * rc.low_speed

        row = (
            f"  {label:<26}"
            f"{_fmt(spd_c)}"
            f"{_fmt(lane_c)}"
            f"{_fmt(col_c)}"
            f"{_fmt(lc_c)}"
            f"{_fmt(slow_c)}"
            f"{_fmt(rc.total)}"
        )
        print(row)

    print(THIN)

    # ── Sensitivity analysis ───────────────────────────────────────────────
    print()
    print("  Sensitivity: Collision weight sweep (other weights fixed)")
    print(THIN)
    crash_info = _make_info(speed=14.0, crashed=True, action=2, right_lane=0.0)
    print(f"  Scenario: crash during lane change  (v=14, LANE_RIGHT, crashed=True)")
    print(f"  {'w_collision':>14}  {'TOTAL':>10}")
    print(THIN)
    for w_col in [-0.5, -1.0, -1.5, -2.0, -3.0]:
        w_test = RewardWeights(
            speed=weights.speed,
            lane=weights.lane,
            collision=w_col,
            unsafe_lc=weights.unsafe_lc,
            low_speed=weights.low_speed,
        )
        rc = compute_reward_components(crash_info, w_test, speed_range)
        print(f"  {w_col:>14.2f}  {rc.total:>10.4f}")
    print(THIN)

    print()
    print(SEP)
    print("  Reward function verified. Ready for integration with PPO training.")
    print(SEP)
    print()


def main() -> None:
    cfg = DEFAULT_CONFIG
    run_demo(
        weights=cfg.reward,
        speed_range=cfg.env.reward_speed_range,
    )


if __name__ == "__main__":
    main()
