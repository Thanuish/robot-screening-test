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

F = SC.GRIP_IN_WAND
SEND_EVERY = 3                        # ticks between poses (~33 Hz, like a phone)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


wand_frame = SC.wand_frame


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

    step_fn = None                    # (a candidate session steps through Session.tick instead)

    def run_ticks(self, n):
        for _ in range(n):
            if self.g.finished:
                return
            (self.step_fn or self.g.step)()

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
        self._send()                                     # (the first level waits for a controller)
        while g.state in ("waiting", "countdown") and not g.finished:
            self.run_ticks(1)
        if lv.start_held:                                # the wand starts in the hand
            self.closed = True
            self.engage()
            lift = 0.0
        else:                                            # pick it up from the start rest
            self.engage()
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
        if lv.finish == "reach_end":                     # done at the far sleeve (the level ends by itself)
            while g.level.key == key and not g.finished:
                self.run_ticks(1)
            self.grip = 0.0
            return
        # onto the end hook: above the rest, down, let go, away
        lift = lv.loop_radius - cfg.wand["loop_thickness"] / 2 - cfg.wire["sleeve_radius"]
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


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


DOWN = np.diag([1.0, -1.0, -1.0])     # the gripper pointing down, fingers closing along world y (yaw 0)


class PickBot(Bot):
    """Pick-and-place games (cube tasks, tower): the gripper pointing down, turned about the vertical."""

    def pick_place(self, n, pos_of, dest_xy, dest_yaw=None, period=None, place_z=None, lift=0.12, half=None, rest_z=None):
        """Grab object n from above (fingers across its narrow side), carry it to dest_xy (turned to
        dest_yaw modulo period), set it down at place_z (its centre), let go, lift."""
        g = self.g
        p, q = pos_of(n)
        oyaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        hx, hy = (half[0], half[1]) if half is not None else (1, 1)
        if hx < hy - 1e-6:
            g_yaw = (oyaw % np.pi) - np.pi / 2
        elif hy < hx - 1e-6:
            g_yaw = ((oyaw + np.pi / 2) % np.pi) - np.pi / 2
        else:
            g_yaw = ((oyaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        Z = SC.TABLE_Z
        self.move(np.r_[p[:2], Z + lift], _rz(g_yaw) @ DOWN, 1.4)
        self.move(np.r_[p[:2], max(p[2], Z + 0.022)], _rz(g_yaw) @ DOWN, 1.0)
        self.gripper(True)
        self.move(np.r_[p[:2], Z + lift], _rz(g_yaw) @ DOWN, 0.9)
        turn = 0.0 if dest_yaw is None else dest_yaw - oyaw
        if dest_yaw is not None and period:
            turn = ((turn + period / 2) % period) - period / 2
        z = place_z if place_z is not None else rest_z + 0.004
        self.move(np.r_[dest_xy, Z + lift], _rz(g_yaw + turn) @ DOWN, 1.5)
        self.move(np.r_[dest_xy, z], _rz(g_yaw + turn) @ DOWN, 1.0)
        self.gripper(False, 0.7)
        self.move(np.r_[dest_xy, Z + lift], _rz(g_yaw + turn) @ DOWN, 0.8)

    def start(self):
        g = self.g
        self.closed = False
        self._send()
        while g.state in ("waiting", "countdown") and not g.finished:
            self.run_ticks(1)
        self.engage()

    def finish_level(self, key):
        g = self.g
        self.grip = 0.0
        self._send()
        while g.level.key == key and not g.finished:
            self.run_ticks(1)


def play_cubes(bot: PickBot):
    """Every step of a cube-tasks level, the way a person would."""
    from TEST_TELEOP import screen_scene as SS
    g = bot.g
    lv, key = g.level, g.level.key
    bot.start()
    for st in lv.phase.steps:
        if g.level.key != key:
            break
        mk = g.tc.markers[st.marker]
        o = g.tc.objects[st.object]
        pos_of = lambda n: g._obj(n)[:2]
        if st.type == "place_in_zone":
            bot.pick_place(st.object, pos_of, mk.pos, half=o.half if o.shape == "box" else None, rest_z=SS.rest_z(st.object))
        elif st.type == "place_in_slot":
            bot.pick_place(st.object, pos_of, mk.pos, dest_yaw=mk.yaw, period=mk.p["symmetry"], half=o.half,
                           rest_z=SS.rest_z(st.object))
        elif st.type == "place_in_bin":
            bot.pick_place(st.object, pos_of, mk.pos, place_z=SC.TABLE_Z + mk.p["wall_height"] + 0.045, lift=0.14,
                           half=o.half if o.shape == "box" else None)
        print(f"    {st.label}: {'done' if g.done.get(st.id) or g.level.key != key else 'NOT done'}")
    bot.finish_level(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="wire", choices=("wire", "cubes", "tower"))
    ap.add_argument("--clumsy", action="store_true")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--replay", action="store_true", help="replay the input log and compare the scores")
    a = ap.parse_args()
    cfg = C.load()
    out = tempfile.mkdtemp(prefix="buzzwire-bot-")
    if a.game == "wire":
        cls = Game
    elif a.game == "cubes":
        from .cubes import CubesGame as cls
    else:
        from .tower import TowerGame as cls
    g = cls(cfg, levels=a.levels, out_dir=out, player="bot")
    t0 = time.perf_counter()
    if a.game == "wire":
        bot = Bot(g, clumsy=a.clumsy)
        while not g.finished:
            bot.play_level()
    else:
        bot = PickBot(g)
        while not g.finished:
            print(f"  {g.level.title}")
            if a.game == "cubes":
                play_cubes(bot)
            else:
                from .tower import play_tower
                play_tower(bot)
    wall = time.perf_counter() - t0
    for k, r in g.results.items():
        if a.game == "wire":
            print(f"  {k:9s} {'DONE' if r['completed'] else 'not done'}{' (hung)' if r['hung'] else ''}  time {r['time_s']:6.1f}s  touches {r['touches']:2d} "
                  f"({r['touch_time_s']:.2f}s)  progress {r['progress']:.2f}  drops {r['drops']}  robot hits {r['robot_hits']}  "
                  f"jerk {r['rms_jerk']}  score {r['score']}")
        else:
            extra = {x: r[x] for x in ("steps_done", "steps_total", "height_cm", "best_height_cm", "placed", "falls") if x in r}
            print(f"  {k:9s} {'DONE' if r['completed'] else 'not done'}  time {r['time_s']:6.1f}s  grasps {r['grasps']}/{r['grasp_attempts']} "
                  f"drops {r['drops']} {extra}  score {r['score']}")
    print("  scores:", g.scores())
    sim = sum(r["time_s"] for r in g.results.values())
    print(f"  wall time {wall:.1f} s for about {sim + 3 * len(g.results):.0f} s of play")
    if a.replay:
        log = json.loads(open(f"{out}/inputs.json").read())
        from .base import replay as base_replay
        g2 = base_replay(cls, cfg, log, levels=a.levels)
        same = g2.scores() == g.scores() and all(g2.results[k] == g.results[k] for k in g.results)
        print(f"  replay: {g2.scores()}  -> {'IDENTICAL' if same else 'DIFFERENT'}")
    return g


if __name__ == "__main__":
    main()
