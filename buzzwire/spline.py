"""The wire's shape: a smooth curve through control points (centripetal Catmull-Rom: it passes
through every point and never loops or overshoots between them), sampled evenly along its
length. The physics builds the wire from these samples (a chain of thin capsules); the game
measures progress along it.
"""
from __future__ import annotations

import numpy as np


def _catmull_rom(p0, p1, p2, p3, n: int, alpha=0.5):
    """n points on the segment p1 -> p2 (without p2), centripetal parametrisation."""
    def tj(ti, a, b):
        return ti + max(np.linalg.norm(b - a), 1e-9) ** alpha
    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False)[:, None]
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


class Wire:
    """A curve through `points` (N x 3, world), resampled every `step` metres along its length."""

    def __init__(self, points, step: float = 0.005):
        P = np.asarray(points, float)
        if len(P) < 2:
            raise ValueError("a wire needs at least 2 points")
        # end tangents: mirror the second / second-to-last point
        ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        dense = [_catmull_rom(*ext[i:i + 4], n=64) for i in range(len(P) - 1)]
        dense = np.vstack(dense + [P[-1:]])
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        self.length = float(s[-1])
        n = max(2, int(np.ceil(self.length / step)) + 1)
        self.s = np.linspace(0, self.length, n)                 # arc length of each sample
        self.pts = np.stack([np.interp(self.s, s, dense[:, k]) for k in range(3)], axis=1)
        t = np.gradient(self.pts, axis=0)
        self.tangents = t / np.linalg.norm(t, axis=1, keepdims=True)
        self.control = P

    def closest(self, p, near: float | None = None, window: float = 0.03):
        """(arc length, distance) of the point on the wire closest to p; with near, only within
        +/- window of that arc length (a continuous track: no jump to another part of a curve)."""
        if near is not None:
            i0 = max(0, int(np.searchsorted(self.s, near - window)) - 1)
            i1 = min(len(self.s), int(np.searchsorted(self.s, near + window)) + 1)
            if i1 - i0 >= 2:
                return self._closest(p, i0, i1)
        return self._closest(p, 0, len(self.s))

    def _closest(self, p, i0, i1):
        a, b = self.pts[i0:i1 - 1], self.pts[i0 + 1:i1]
        ab = b - a
        u = np.clip(np.einsum("ij,ij->i", np.asarray(p) - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
        q = a + ab * u[:, None]
        d = np.linalg.norm(q - p, axis=1)
        i = int(np.argmin(d)) + i0
        return float(self.s[i] + u[i - i0] * (self.s[i + 1] - self.s[i])), float(d[i - i0])

    def at(self, s: float):
        """(point, tangent) at arc length s."""
        s = float(np.clip(s, 0, self.length))
        p = np.stack([np.interp(s, self.s, self.pts[:, k]) for k in range(3)])
        t = np.stack([np.interp(s, self.s, self.tangents[:, k]) for k in range(3)])
        return p, t / np.linalg.norm(t)

    def min_bend_radius(self) -> float:
        """The tightest bend [m] (a ring can't follow a bend much tighter than its own size)."""
        t = self.tangents
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1, 1))
        ds = np.diff(self.s)
        k = ang / np.maximum(ds, 1e-9)
        return float(1 / k.max()) if k.max() > 1e-9 else float("inf")
