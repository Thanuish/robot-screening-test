"""A bot plays Robot Buzz Wire through the same controller messages as the phone (a pose ~30 times
a second, MOVE held, GRAB / RELEASE), in simulation time. Per level it picks the wand off the
start rest, guides the loop along the wire (centred on it, square to it), hangs the wand on the
end hook and lets go.

    python -m buzzwire.bot                    # clean run: every level, no touches
    python -m buzzwire.bot --clumsy           # wobbles more than the loop's clearance: touches
    python -m buzzwire.bot --levels straight  # some levels only
    python -m buzzwire.bot --replay           # also replay the input log: the same scores?
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time

import mujoco
import numpy as np

from . import config as C
from . import scene as SC
from .game import Game, replay

F = np.diag([1.0, -1.0, -1.0])        # the gripper's orientation relative to the wand when holding it
SEND_EVERY = 3                        # ticks between poses (~33 Hz, like a phone)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def wand_frame(level, s, full: bool):
    """The wand's ideal orientation at arc length s: its loop square to the wire, the handle
    level (pointing back toward the robot)."""
    p, t = level.wire.at(s)
    th = np.array([t[0], t[1], 0.0])
    th /= np.linalg.norm(th)
    x = np.array([th[1], -th[0], 0.0])
    if x[0] < 0:
        x = -x
    y = t if full else th
    x = x - np.dot(x, y) * y
    x /= np.linalg.norm(x)
    return p, np.stack([x, y, np.cross(x, y)], axis=1)


class Bot:
    def __init__(self, game: Game, clumsy=False, speed=0.035):
        self.g, self.clumsy, self.speed = game, clumsy, speed
        self.hand = np.array([0.0, 0.0, 1.0])
        self.grip, self.closed = 0.0, False
        self.c0 = None                                   # the hand when MOVE was pressed
        self.e0 = None

    # ---------------------------------------------------------------- sending
    def _send(self, pos=None, R=None):
        q = [1, 0, 0, 0] if R is None else list(_quat(R))
        p = self.hand if pos is None else pos
        self.g.set_input({"right": {"pose": list(map(float, p)) + list(map(float, q)), "grip": self.grip,
                                    "stick": [0.0, 1.0 if self.closed else -1.0], "epoch": 0}})

    def run_ticks(self, n):
        for _ in range(n):
            if self.g.finished:
                return
            self.g.step()

    def engage(self):
        """Press MOVE with the hand at rest; remember where the arm's target was."""
        self.grip = 1.0
        self.hand_R = np.eye(3)
        self._send(self.hand, self.hand_R)
        self.run_ticks(2)
        tel = self.g.teleop
        self.e0 = (tel._e0_pos.copy(), _mat(tel._e0_quat))
        self.c0 = self.hand.copy()

    def command(self, pos, R):
        """Send the hand pose that makes the arm's target (pos, R) (inverting the clutch mapping)."""
        scale = self.g.cfg.controls["motion_scale"]
        hp = self.c0 + (np.asarray(pos) - self.e0[0]) / scale
        hR = np.asarray(R) @ self.e0[1].T
        self._send(hp, hR)

    def move(self, pos, R, dur):
        """Glide the target to (pos, R) in dur seconds (smooth start and stop)."""
        tel = self.g.teleop
        p0, R0 = tel.target_pos.copy(), _mat(tel.target_quat)
        q0, q1 = _quat(R0), _quat(R)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        n = max(1, int(dur / self.g.dt))
        for k in range(1, n + 1):
            if k % SEND_EVERY == 0 or k == n:
                a = 0.5 - 0.5 * np.cos(np.pi * k / n)
                q = q0 * (1 - a) + q1 * a
                self.command(p0 + (np.asarray(pos) - p0) * a, _mat(q / np.linalg.norm(q)))
            self.run_ticks(1)
        self.run_ticks(int(0.3 / self.g.dt))

    def gripper(self, close, dur=1.0):
        self.closed = close
        tel = self.g.teleop
        self.command(tel.target_pos, _mat(tel.target_quat))
        self.run_ticks(int(dur / self.g.dt))

    # ---------------------------------------------------------------- one level
    def play_level(self):
        g = self.g
        lv, cfg, key = g.level, g.cfg, g.level.key
        full = lv.rotation == "full"
        while g.state == "countdown" and not g.finished:
            self.run_ticks(1)
        self.engage()
        # the handle on the start rest: approach from above, close, lift until the loop is centred
        c, q = g.sc.rest["start"]
        Rw = _mat(q)
        h = c + Rw @ SC.handle_centre_local(cfg, lv)
        Rg = Rw @ F
        self.move(h + [0, 0, 0.08], Rg, 2.0)
        self.move(h, Rg, 1.2)
        self.gripper(True)
        lift = lv.loop_radius - cfg.wand["loop_thickness"] / 2 - cfg.wire["sleeve_radius"]
        self.move(h + [0, 0, lift], Rg, 0.8)
        # the hold: the gripper target relative to the wand (as it is now held)
        wp, wq = g.wand_pose()
        tel = g.teleop
        T_w = (wp, _mat(wq))
        grip_in_wand = (T_w[1].T @ (tel.target_pos - wp), T_w[1].T @ _mat(tel.target_quat))

        def target_for(centre, Rwd):
            return centre + Rwd @ grip_in_wand[0], Rwd @ grip_in_wand[1]

        # along the wire
        s0 = cfg.wire["sleeve_length"] * 0.5
        s1 = lv.wire.length - cfg.wire["sleeve_length"] * 0.5
        p, R = wand_frame(lv, s0, full)
        self.move(*target_for(p, R), 0.8)
        clear = lv.loop_radius - cfg.wand["loop_thickness"] / 2 - cfg.wire["radius"]
        n = int((s1 - s0) / self.speed / g.dt)
        for k in range(1, n + 1):
            if g.level.key != key:
                return
            if k % SEND_EVERY == 0 or k == n:
                s = s0 + (s1 - s0) * k / n
                p, R = wand_frame(lv, s, full)
                if self.clumsy:                          # wobble up and down past the clearance
                    p = p + R[:, 2] * 1.4 * clear * np.sin(2 * np.pi * (s - s0) / 0.08)
                self.command(*target_for(p, R))
            self.run_ticks(1)
        # onto the end hook: above the rest, down, let go, away
        c, q = g.sc.rest["end"]
        Rw = _mat(q)
        tp, tR = target_for(c, Rw)
        self.move(tp + [0, 0, lift], tR, 1.0)
        self.move(tp + [0, 0, 0.002], tR, 0.8)
        self.gripper(False, 0.8)
        self.move(tp + [0, 0, 0.08], tR, 1.0)
        self.grip = 0.0
        self._send()
        while g.level.key == key and not g.finished:
            self.run_ticks(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clumsy", action="store_true")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--replay", action="store_true", help="replay the input log and compare the scores")
    a = ap.parse_args()
    cfg = C.load()
    out = tempfile.mkdtemp(prefix="buzzwire-bot-")
    g = Game(cfg, levels=a.levels, out_dir=out, player="bot")
    bot = Bot(g, clumsy=a.clumsy)
    t0 = time.perf_counter()
    sim0 = 0.0
    while not g.finished:
        bot.play_level()
    wall = time.perf_counter() - t0
    for k, r in g.results.items():
        print(f"  {k:9s} {'HUNG' if r['hung'] else 'not hung'}  time {r['time_s']:6.1f}s  touches {r['touches']:2d} "
              f"({r['touch_time_s']:.2f}s)  progress {r['progress']:.2f}  drops {r['drops']}  robot hits {r['robot_hits']}  "
              f"jerk {r['rms_jerk']}  score {r['score']}")
    print("  scores:", g.scores())
    sim = sum(r["time_s"] for r in g.results.values())
    print(f"  wall time {wall:.1f} s for about {sim + 3 * len(g.results):.0f} s of play")
    if a.replay:
        log = json.loads(open(f"{out}/inputs.json").read())
        g2 = replay(cfg, log, levels=a.levels)
        same = g2.scores() == g.scores() and all(g2.results[k] == g.results[k] for k in g.results)
        print(f"  replay: {g2.scores()}  -> {'IDENTICAL' if same else 'DIFFERENT'}")
    return g


if __name__ == "__main__":
    main()
