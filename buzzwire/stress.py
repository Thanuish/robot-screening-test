"""Stress check: every game driven like an impatient person (fast swings, the gripper shoved into
the table and the objects, grabbing at random), every control tick timed. A tick must never take
long: the simulation runs in real time, and a slow tick freezes the robot on every screen.

    python -m buzzwire.stress                 # every game, a few seeds; fails on a slow tick
    python -m buzzwire.stress --game tutorial --seeds 0 1 --seconds 120

Fails (exit code 1) if a tick other than a level change takes longer than --limit-ms, or an IK
solve didn't finish.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import config as C
from .cubes import CubesGame
from .game import BuzzWireGame
from .tower import TowerGame
from .tutorial import TutorialGame
from .rings import RingsGame

GAMES = {"tutorial": TutorialGame, "cubes": CubesGame, "tower": TowerGame, "rings": RingsGame, "wire": BuzzWireGame}   # (wire: no longer in the menu or the test)


def run(kind: str, seed: int, seconds: float):
    g = GAMES[kind](C.load())
    rng = np.random.default_rng(seed)
    hand, vel, yaw = np.array([0.0, 0.0, 1.3]), np.zeros(3), 0.0
    grip, closed = 0, False
    worst, failed, k = 0.0, 0, 0
    while k * g.dt < seconds and not g.finished:
        if k % 3 == 0:                                   # a phone's ~33 messages a second
            if rng.random() < 0.02:
                vel = rng.normal(0, 0.8, 3)              # a sudden swing
            vel += rng.normal(0, 0.05, 3) - 0.05 * vel
            vel[2] -= 0.02 if rng.random() < 0.5 else -0.015    # drawn down toward the table
            hand = np.clip(hand + vel * 0.03, [-0.4, -0.5, 0.6], [0.4, 0.5, 1.8])
            yaw += rng.normal(0, 0.08)
            if rng.random() < 0.01:
                closed = not closed
            grip = 0 if rng.random() < 0.005 else (1 if rng.random() < 0.05 else grip)
            q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            g.set_input({"right": {"pose": list(map(float, hand + rng.normal(0, 0.0015, 3))) + q, "grip": grip,
                                   "stick": [0, 1 if closed else -1], "epoch": 0}})
        level, ik = g.li, g.ik
        t = time.perf_counter()
        g.step()
        dt = time.perf_counter() - t
        if g.li == level:                                # (a level change builds a new scene: not counted)
            worst = max(worst, dt)
        failed += ik.failed
        ik.failed = 0
        k += 1
    return k, worst, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=list(GAMES), nargs="*", default=[k for k in GAMES if k != "wire"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--limit-ms", type=float, default=50.0)
    a = ap.parse_args()
    ok = True
    for kind in a.game:
        for seed in a.seeds:
            ticks, worst, failed = run(kind, seed, a.seconds)
            bad = worst * 1000 > a.limit_ms or failed
            ok &= not bad
            print(f"  {kind:8s} seed {seed}: {ticks} ticks, slowest {worst * 1000:5.1f} ms, unfinished IK solves {failed}"
                  f"{'   <-- TOO SLOW' if bad else ''}", flush=True)
    print("  OK" if ok else "  FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
