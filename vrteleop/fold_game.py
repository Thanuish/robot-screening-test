"""The jeans folding game: three folds in order, against the clock.

The steps are checked on the fold metrics (vrteleop/cloth.py fold_metrics): the footprint of
the jeans on the table relative to spread flat. A step counts once the fabric has settled
(nothing held, hardly anything moving), so throwing the jeans around doesn't score.

    1. lay one leg over the other     footprint <= 62 %  (flat jeans -> one leg wide)
    2. fold them in half              footprint <= 35 %  (hems up to the waist)
    3. fold once more                 footprint <= 22 %  (a compact stack)

The clock starts when an arm is first engaged after a reset and stops at the last step.
"""
from __future__ import annotations

import time

import numpy as np

STEPS = [
    ("Lay one leg over the other", 0.62),
    ("Fold in half: bring the hems up to the waist", 0.35),
    ("Fold once more into a compact stack", 0.22),
]
SETTLE_S = 1.0          # the fold must hold this long, untouched
MOVING = 0.004          # [m] a point moving more than this between two checks is moving
MOVING_MAX = 0.02       # ... and at most this fraction of the points may move


class FoldGame:
    def __init__(self):
        self.best: float | None = None
        self.reset()

    def reset(self):
        self.step = 0                     # steps done
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.step_times: list[float] = []
        self._prev = None
        self._ok_since: float | None = None

    @property
    def done(self) -> bool:
        return self.step >= len(STEPS)

    def elapsed(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end or time.perf_counter()) - self.t_start

    def start(self):
        """An arm was engaged: the clock starts (once per game)."""
        if self.t_start is None and not self.done:
            self.t_start = time.perf_counter()

    def update(self, verts: np.ndarray, metrics: dict, holding: bool) -> str | None:
        """Call every second or so; returns a message when a step was completed."""
        now = time.perf_counter()
        moving = 1.0
        if self._prev is not None and self._prev.shape == verts.shape:
            moving = float((np.abs(verts - self._prev).max(axis=1) > MOVING).mean())
        self._prev = verts.copy()
        if self.done or self.t_start is None:
            return None
        settled = not holding and moving <= MOVING_MAX
        if settled and metrics.get("coverage", 1.0) <= STEPS[self.step][1]:
            self._ok_since = self._ok_since or now
            if now - self._ok_since >= SETTLE_S:
                self._ok_since = None
                self.step += 1
                self.step_times.append(self.elapsed())
                if self.done:
                    self.t_end = now
                    t = self.elapsed()
                    record = self.best is None or t < self.best
                    self.best = t if record else self.best
                    return f"Folded in {t:.0f} s!" + ("  New best!" if record else f"  (best {self.best:.0f} s)")
                return f"Step {self.step} done - next: {STEPS[self.step][0].lower()}"
        else:
            self._ok_since = None
        return None

    def status(self) -> dict:
        return {
            "step": self.step,
            "steps": len(STEPS),
            "goal": STEPS[self.step][0] if not self.done else "Folded!",
            "target": STEPS[self.step][1] if not self.done else None,
            "time": round(self.elapsed(), 1),
            "running": self.t_start is not None and not self.done,
            "done": self.done,
            "best": round(self.best, 1) if self.best is not None else None,
        }
