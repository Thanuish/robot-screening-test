"""A jitter buffer for the phone's controller messages.

A phone sends its pose ~60 times a second, but through the internet they arrive unevenly: a few
at once, then a gap. Fed to the robot as they arrive, the motion stutters. Here every message is
kept with the phone's own timestamp and played back evenly a little later: each control tick
gets the pose interpolated at (now - delay) on the phone's clock. The delay adapts to the
connection: it covers how late 95 % of the messages come (over the last few seconds), between
[controls] jitter_buffer_s and jitter_buffer_max_s, and changes slowly (never felt as a jump).
A longer gap holds the last pose.

The phone's clock is mapped to this one by the smallest (arrival - phone time) seen in the last
few seconds: the message that came through fastest.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

WINDOW_S = 3.0                 # the clock offset: the fastest message in this long
KEEP_S = 1.0                   # messages kept behind the playback time
COVER = 0.95                   # the delay covers this share of the messages
MARGIN = 0.01                  # [s] on top
GROW, SHRINK = 0.2, 0.03       # [s per s] how fast the delay may grow / shrink (playback 20 % slower / 3 % faster)


def _nlerp(q0, q1, a):
    q1 = q1 if np.dot(q0, q1) >= 0 else -q1
    q = q0 * (1 - a) + q1 * a
    return q / np.linalg.norm(q)


class Jitter:
    def __init__(self, delay: float, delay_max: float | None = None):
        self.lo = delay
        self.hi = max(delay, delay_max if delay_max is not None else delay)
        self.delay = delay                  # now (adapts between lo and hi)
        self.want = delay                   # what the connection needs
        self.msgs: deque = deque()          # (phone time, controller dict)
        self.offsets: deque = deque()       # (arrival, arrival - phone time)
        self.last = None                    # when sample() was last called

    def reset(self):
        self.msgs.clear()
        self.offsets.clear()
        self.delay = self.want = self.lo
        self.last = None

    def push(self, t_phone: float, ctrl: dict | None, now: float | None = None):
        now = time.perf_counter() if now is None else now
        if self.msgs and t_phone < self.msgs[-1][0] - 0.5:
            self.reset()                    # (the phone page restarted: a new clock)
        if self.msgs and t_phone <= self.msgs[-1][0]:
            return                          # (out of order)
        self.msgs.append((t_phone, ctrl))
        self.offsets.append((now, now - t_phone))
        while self.offsets and self.offsets[0][0] < now - WINDOW_S:
            self.offsets.popleft()
        if len(self.offsets) >= 10:
            o = np.fromiter((x for _, x in self.offsets), float)
            self.want = float(np.clip(np.quantile(o, COVER) - o.min() + MARGIN, self.lo, self.hi))

    def sample(self, now: float | None = None) -> dict | None:
        """The controller message for this moment (None: nothing yet)."""
        if not self.msgs:
            return None
        now = time.perf_counter() if now is None else now
        dt = 0.0 if self.last is None else max(0.0, now - self.last)
        self.last = now
        if self.want > self.delay:
            self.delay = min(self.want, self.delay + GROW * dt)
        else:
            self.delay = max(self.want, self.delay - SHRINK * dt)
        t = now - min(o for _, o in self.offsets) - self.delay
        while len(self.msgs) > 2 and self.msgs[1][0] < t - KEEP_S:
            self.msgs.popleft()
        prev = None
        for m in self.msgs:
            if m[0] > t:
                break
            prev = m
        if prev is None:
            return self.msgs[0][1]
        i = self.msgs.index(prev)
        if i + 1 >= len(self.msgs):
            return prev[1]                  # (a gap: hold the last pose)
        nxt = self.msgs[i + 1]
        a, b = prev[1], nxt[1]
        same = (a and b and a.get("pose") and b.get("pose") and a.get("epoch") == b.get("epoch")
                and a.get("grip") == b.get("grip"))
        if not same:
            return a
        f = (t - prev[0]) / (nxt[0] - prev[0])
        p0, p1 = np.asarray(a["pose"], float), np.asarray(b["pose"], float)
        pose = np.r_[p0[:3] + (p1[:3] - p0[:3]) * f, _nlerp(p0[3:7], p1[3:7], f)]
        return dict(a, pose=[round(float(v), 6) for v in pose])
