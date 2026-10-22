"""A bot plays the screening test through the same messages a phone sends, in simulation time:
every phase of config/screening.toml, step by step (reach, pick and place into zones,
outlines and bins, take a book out of its holder and lay it flat, pour from a cup and set it
upright, hang rings on pegs, guide the buzz-wire loop along its wire). Checks that every task can be done and prints the metrics and scores. Pictures: --pics DIR.

    python -m TEST_TELEOP.screen_bot
    python -m TEST_TELEOP.screen_bot --skip shapes      # leave a task undone (tests partial scores)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import TEST_TELEOP  # noqa: E402,F401  (registers the task "screen" and its game)
from TEST_TELEOP import screen_scene as SS  # noqa: E402
from vrteleop import scene as S  # noqa: E402
from vrteleop.server import TeleopSim, build_parser  # noqa: E402

HAND0 = np.array([0.0, -0.25, 1.1])        # where the "phone" is when the bot starts
ROUNDS = 2                                  # rounds the bot plays in a phase with repeat = true


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def do_book(bot, n, holder, zone, tilt=+1, turn=-1):
    """Take book n out of its holder by its top edge, turn it a quarter turn and tilt the gripper
    sideways so the book lies flat, and set it down in zone. tilt +1: the fingers point to +y
    (the book sticks out away from the arm, the wrist stays on the arm's side: the arm needn't
    stretch); -1 (toward -y) reaches the mat only just, the arm fully stretched. Tilted forward
    instead, the wrist hits the table 16 cm up."""
    Z = S.TABLE_Z
    g, cfg = bot.game, SS.CFG
    # the book: fingers across its thickness (world x): the hand turned 90 deg
    rx, ry = cfg.markers[holder].pos
    top = Z + 2 * cfg.objects[n].half[0]               # standing: its top edge
    Rg = _rz(np.pi / 2)
    bot.move_pose((rx, ry, Z + 0.22), Rg, 1.5)
    bot.move_pose((rx, ry, top - 0.015), Rg, 1.0)
    bot.gripper(True)
    bot.move_pose((rx, ry, Z + 0.27), Rg, 1.2)
    print(f"  {n} lifted: held {g.held[n]}")
    Rflat = _rx(tilt * np.pi / 2) @ _rz(turn * np.pi / 2) @ Rg
    fingers = _rx(tilt * np.pi / 2) @ np.array([0, 0, -1.0])     # where the fingers point now
    mx, my = cfg.markers[zone].pos
    bot.move_pose((mx, my, Z + 0.25), Rflat, 2.0)
    p, _, _ = g._obj(n)
    off = p - g.ee()                                   # the book's middle, seen from the gripper
    bot.move_pose((mx - off[0], my - off[1], Z + 0.25), Rflat, 0.8)
    bot.move_pose((mx - off[0], my - off[1], Z + 0.075), Rflat, 1.2)
    bot.gripper(False, 0.7)
    # back out along the fingers and up, without touching the book
    back = -0.06 * fingers
    bot.move_pose((mx - off[0] + back[0], my - off[1] + back[1], Z + 0.09), Rflat, 0.8)
    bot.move_pose((mx - off[0] + back[0], my - off[1] + back[1], Z + 0.20), Rflat, 0.8)
    bot.advance(0.5)
    p, _, v = g._obj(n)
    print(f"  {n}: {g.metrics[g.phase.key].get('done_steps')}  (at {np.round(p - [mx, my, Z], 3)} from the "
          f"zone centre, normal z {abs(g._thin_axis(n)[2]):.2f}, held {g.held[n]}, speed {v:.2f})")


def do_pour(bot, cup, container, pour_angle=np.radians(130)):
    """Cup taken by its rim (inner finger inside), tipped over the container, set down upright."""
    Z = S.TABLE_Z
    g, cfg = bot.game, SS.CFG
    o = cfg.objects[cup]
    cx, cy = o.start
    wall = o.half[0] - o.cup_wall / 2
    rim = Z + 2 * o.half[1]
    bot.move_pose((cx + 0.03, cy + 0.04, Z + 0.25), np.eye(3), 1.5)
    bot.move_pose((cx, cy + wall, Z + 0.22), np.eye(3), 1.2)
    bot.move_pose((cx, cy + wall, rim - 0.005), np.eye(3), 1.0)
    bot.gripper(True)
    bot.move_pose((cx, cy + wall, Z + 0.26), np.eye(3), 1.2)
    print(f"  {cup} lifted: held {g.held[cup]}")
    bx, by = cfg.markers[container].pos
    bot.move_pose((bx, by + wall, Z + 0.24), np.eye(3), 1.5)
    bot.move_pose((bx, by + wall + 0.02, Z + 0.24), _rx(pour_angle), 2.0)     # tip it over
    bot.advance(1.5)
    print(f"  poured: {g.metrics[g.phase.key].get('poured')} into the {container}")
    bot.move_pose((bx, by + wall, Z + 0.24), np.eye(3), 1.5)
    dx, dy = cx + 0.04, cy + 0.06                       # set it down near where it was
    bot.move_pose((dx, dy, Z + 0.20), np.eye(3), 1.5)
    bot.move_pose((dx, dy, Z + 0.09), np.eye(3), 1.0)
    bot.gripper(False, 0.7)
    bot.move_pose((dx, dy, Z + 0.25), np.eye(3), 1.0)
    bot.advance(1.0)
    ph = g.phase
    print(f"  steps: {g.metrics[ph.key].get('done_steps') if ph.kind == 'task' else 'phase finished'}")


def _turn(a, b):
    """The rotation (matrix) that turns unit vector a onto b along the shortest way."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    k = np.cross(a, b)
    s_, c = np.linalg.norm(k), float(a @ b)
    if s_ < 1e-9:
        return np.eye(3)
    k /= s_
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s_ * K + (1 - c) * K @ K


def do_ring_flat(bot, n, peg):
    """Ring n lies flat; the peg stands straight up: take the ring by its rim (where the tube runs
    along x, the fingers closing across it along y), lift it over the peg's tip, lower it straight
    down, let go."""
    Z = S.TABLE_Z
    g, cfg = bot.game, SS.CFG
    o, mk = cfg.objects[n], cfg.markers[peg]
    c, _ = bot.obj(n)
    R = o.half[0]                                        # (the tube's middle, from the centre)
    grab = np.array([c[0], c[1] + R, c[2]])
    bot.move_to(grab + [0, 0, 0.12], 1.2, yaw=0.0)
    bot.move_to(grab, 1.0)
    bot.gripper(True)
    top = Z + mk.p["length"] + 0.06
    bot.move_to(np.array([grab[0], grab[1], top]), 1.0)
    print(f"  {n} lifted: held {g.held[n]}")
    over = np.array([mk.pos[0], mk.pos[1] + R, top])
    bot.move_to(over, 1.6)
    bot.move_to(np.array([over[0], over[1], Z + 0.013]), 1.6)   # (all the way down: no drop)
    ph = g.phase
    bot.gripper(False, 0.7)
    if g.phase is ph:
        bot.move_to(np.array([over[0], over[1], Z + 0.15]), 1.0)   # (straight up: the fingers off the ring)
    bot.advance(1.0)


def do_ring(bot, n, peg):
    """Ring n stands on its edge (its axis along world x): take it by the top of its tube, the
    fingers across the tube; turn it so its hole lines up with the peg; bring it to just past the
    peg's tip and push it onto the peg; let go; pull the fingers back out along the gripper."""
    Z = S.TABLE_Z
    g, cfg = bot.game, SS.CFG
    o, mk = cfg.objects[n], cfg.markers[peg]
    c, _ = bot.obj(n)
    top = c + np.array([0, 0, o.half[0]])
    Rg = _rz(np.pi / 2)                                  # fingers closing along world x
    bot.move_pose((top[0], top[1], Z + 0.30), Rg, 1.5)
    bot.move_pose((top[0], top[1], top[2] - 0.002), Rg, 1.2)
    bot.gripper(True)
    bot.move_pose((top[0], top[1], Z + 0.32), Rg, 1.2)
    c, _ = bot.obj(n)
    off0 = c - bot.ee_target()                           # ring centre from the pinch point
    print(f"  {n} lifted: held {g.held[n]}")
    u = np.asarray(mk.p["dir"])
    b = u if u[0] >= 0 else -u                           # (either way through: the shorter turn)
    Q = _turn([1, 0, 0], b)
    R1 = Q @ Rg
    off = Q @ off0
    base = np.array([mk.pos[0], mk.pos[1], Z + max(mk.p["height"], 0.008)])
    tip = base + mk.p["length"] * u
    approach = Q @ np.array([0, 0, -1.0])                # where the fingers point
    if u[2] > 0.9:                                       # straight up: lower the ring over it
        c_a, c_in = tip + np.array([0, 0, 0.05]), tip - np.array([0, 0, 0.045])
    else:                                                # tilted: from past the tip, along the peg
        c_a, c_in = tip + 0.05 * u, base + 0.45 * mk.p["length"] * u
    bot.move_pose(c_a - off + np.array([0, 0, 0.10]), R1, 2.0)     # turned, above the way in
    bot.move_pose(c_a - off, R1, 1.2)
    bot.move_pose(c_in - off, R1, 1.5)
    ph = g.phase
    bot.gripper(False, 0.7)
    for goal, R in ((c_in - off - 0.07 * approach, R1),                # fingers out
                    (c_in - off - 0.07 * approach + np.array([0, 0, 0.10]), Rg)):
        if g.phase is not ph:                            # (that was the last ring: the task is over)
            break
        bot.move_pose(goal, R, 1.0)
    bot.grip = 0.0                                       # (MOVE let go)
    bot.advance(1.5)
    print(f"  {n}: on its peg {g._on_peg(n, peg)}" + ("" if g.phase.kind != "task" else f"  ({g.metrics[g.phase.key]['steps_done']}/{len(g.phase.steps)})"))


def do_stack(bot, st, upto=None):
    """Tower: take each block from the supply as it appears and set it on the one below."""
    Z = S.TABLE_Z
    g, cfg = bot.game, SS.CFG
    ph, tw = g.phase, g.stack
    plat = tw["plat"]
    top = Z + 2 * plat.p["half"][2]
    for i in range(len(st.objects) if upto is None else upto):
        for _ in range(40):                                 # (wait for the next block)
            if len(tw["out"]) > i or g.phase is not ph:
                break
            bot.advance(0.25)
        if g.phase is not ph or len(tw["out"]) <= i:
            break
        n = tw["out"][i]
        h = cfg.objects[n].half[2]
        z = top + 2 * h * i + h + 0.004                      # (its centre, on the one below)
        bot.pick_place(n, plat.pos, place_z=z, lift=z - Z + 0.09)
        print(f"  {n}: tower {g.metrics[ph.key]['height'] * 100:.0f} cm (best {g.metrics[ph.key]['best_height'] * 100:.0f} cm)")
    bot.grip = 0.0
    bot.advance(3.0)


def do_wire(bot, speed=0.05):
    """Guide the loop along the wire at `speed` [m/s], square to the wire, clutch held."""
    from TEST_TELEOP.screen_game import wand_frame, _mat_of
    g = bot.game
    w = g.wire
    W, rot = w["W"], g.phase.rotation
    tel = bot.sim.arms.teleop[bot.side]
    bot.grip = 1.0
    bot.advance(0.05)
    start, hand0 = bot.ee_target(), bot.hand.copy()
    Rs0 = _mat_of(tel.target_quat)
    Rw0 = Rs0 @ w["R_rel"]
    c0 = start + Rs0 @ w["p_rel"]
    s0, s_end = w["s"], W.length - w["mk"].p["sleeve"] / 2
    n = int((s_end - s0) / (speed * 0.02))
    for k in range(n + 1):
        c, t = W.at(s0 + (s_end - s0) * k / max(1, n))
        R1 = wand_frame(t, rot) @ Rw0.T                 # the hand's turn since the start
        Rs = R1 @ Rs0
        pinch = c - Rs @ w["p_rel"]
        bot.rot = R1
        bot.hand = hand0 + (pinch - start) / bot.scale
        bot.advance(0.02)
        if g.wire is None:
            break
    bot.advance(0.5)


class Bot:
    def __init__(self, extra_args=(), pics=None):
        args = build_parser().parse_args(["--http", "--no-preview", "--stream-cams", "", "--task", "screen",
                                          "--data-dir", tempfile.mkdtemp(prefix="screen-bot-"),
                                          "--operator", "bot", *extra_args])
        self.sim = TeleopSim(args)
        self.m, self.d, self.game = self.sim.m, self.sim.d, self.sim.task_game
        self.side = SS.CFG.arm.side
        self.ctrl_dt = 1.0 / args.control_hz
        self.hand, self.yaw, self.rot = HAND0.copy(), 0.0, None
        self.grip, self.stick, self.epoch = 0.0, 0.0, 0
        self.pics = Path(pics) if pics else None
        self.renderer = mujoco.Renderer(self.m, 360, 480) if pics else None
        self.wall0 = time.perf_counter()

    def send(self):
        if self.rot is not None:                       # full rotation (task 3)
            q4 = np.zeros(4)
            mujoco.mju_mat2Quat(q4, self.rot.flatten())
            q = list(q4)
        else:
            q = [np.cos(self.yaw / 2), 0, 0, np.sin(self.yaw / 2)]
        self.sim.latest_input = {"type": "input", "head": None, "left": None, "right": None,
                                 self.side: {"pose": list(self.hand) + q, "grip": self.grip,
                                             "stick": [0.0, self.stick], "epoch": self.epoch}}
        self.sim.latest_input_t = time.perf_counter()

    def advance(self, dur):
        t_end = self.d.time + dur
        while self.d.time < t_end - 1e-9:
            self.send()
            self.sim.control(self.ctrl_dt)
            if self.sim.recorder.active:
                self.sim.record_frame()
            for _ in range(int(round(self.ctrl_dt / self.m.opt.timestep))):
                mujoco.mj_step(self.m, self.d)

    def wait_phase(self, key, timeout=20.0):
        t_end = self.d.time + timeout
        while self.game.phase.key != key and self.d.time < t_end:
            self.advance(0.1)
        while (self.game.intro_until is not None or self.game.ready) and self.d.time < t_end:
            if self.game.ready:
                self.game.go()
            self.advance(0.1)
        assert self.game.phase.key == key, f"expected phase {key}, got {self.game.phase.key}"

    # hand motion with the clutch held: the gripper target moves by the same amount x the
    # motion scale (so the hand moves 1/scale as far)
    def ee_target(self):
        return self.sim.arms.teleop[self.side].target_pos.copy()

    @property
    def scale(self):
        return self.sim.arms.teleop[self.side].pos_scale

    def move_to(self, goal, dur=1.0, yaw=None):
        """Move the gripper target to goal (world) and turn to yaw, clutch held."""
        self.grip = 1.0
        self.advance(0.05)
        start, hand0, yaw0 = self.ee_target(), self.hand.copy(), self.yaw
        yaw1 = yaw0 if yaw is None else yaw
        n = max(1, int(dur / 0.02))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            self.hand = hand0 + (np.asarray(goal) - start) / self.scale * a
            self.yaw = yaw0 + (yaw1 - yaw0) * a
            self.advance(0.02)
        self.advance(0.25)

    def move_pose(self, goal, R1, dur=1.0):
        """Move the gripper target to goal and turn the hand (all three axes) to R1 (a world-frame
        rotation from the hand's starting orientation), clutch held."""
        self.grip = 1.0
        self.advance(0.05)
        R0 = self.rot if self.rot is not None else _rz(self.yaw)
        q0, q1 = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(q0, R0.flatten())
        mujoco.mju_mat2Quat(q1, np.asarray(R1, float).flatten())
        if np.dot(q0, q1) < 0:
            q1 = -q1
        start, hand0 = self.ee_target(), self.hand.copy()
        n = max(1, int(dur / 0.02))
        for k in range(n):
            a = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)
            q = q0 * (1 - a) + q1 * a                      # (n)lerp: fine for these smooth turns
            q /= np.linalg.norm(q)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, q)
            self.rot = R.reshape(3, 3)
            self.hand = hand0 + (np.asarray(goal) - start) / self.scale * a
            self.advance(0.02)
        self.advance(0.3)

    def gripper(self, close: bool, dur=0.9):
        self.stick = 1.0 if close else -1.0
        self.advance(dur)
        self.stick = 0.0

    def obj(self, n):
        a = self.sim.info.obj_qpos_adr[n]
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def pick_place(self, n, dest_xy, dest_yaw=None, period=None, place_z=None, lift=0.12):
        """Grab object n from above (fingers closing across its narrow side), put it down at
        dest_xy, turned to dest_yaw (or the same turn modulo period: a square every 90 deg)."""
        p, q = self.obj(n)
        o = SS.CFG.objects[n]
        oyaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        # the fingers close along world y at home (hand yaw 0); turn them to close across the
        # object: across its local x if that's its narrow side, its local y if that is, either
        # for squares and round things
        hx, hy = (o.half[0], o.half[1]) if o.shape == "box" else (1, 1)
        if hx < hy - 1e-6:
            g_yaw = (oyaw % np.pi) - np.pi / 2
        elif hy < hx - 1e-6:
            g_yaw = ((oyaw + np.pi / 2) % np.pi) - np.pi / 2
        else:
            g_yaw = ((oyaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        top = np.r_[p[:2], S.TABLE_Z + lift]
        self.move_to(top, 1.2, yaw=g_yaw)
        self.move_to(np.r_[p[:2], max(p[2], S.TABLE_Z + 0.022)], 1.0)
        self.gripper(True)
        self.move_to(np.r_[p[:2], S.TABLE_Z + lift], 0.8)
        turn = 0.0 if dest_yaw is None else dest_yaw - oyaw
        if dest_yaw is not None and period:     # the shortest turn that matches
            turn = ((turn + period / 2) % period) - period / 2
        z = place_z if place_z is not None else SS.rest_z(n) + 0.004
        self.move_to(np.r_[dest_xy, S.TABLE_Z + lift], 1.4, yaw=g_yaw + turn)
        self.move_to(np.r_[dest_xy, z], 1.0)
        self.gripper(False, 0.7)
        self.move_to(np.r_[dest_xy, S.TABLE_Z + lift], 0.8)

    def pic(self, name):
        if not self.renderer:
            return
        from PIL import Image
        self.pics.mkdir(parents=True, exist_ok=True)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.5, -0.25, S.TABLE_Z]
        cam.distance, cam.azimuth, cam.elevation = 0.9, 180, -55
        self.renderer.update_scene(self.d, camera=cam)
        Image.fromarray(self.renderer.render()).save(self.pics / f"{name}.png")


def play_phase(bot, ph):
    """Do every step of a task phase the way a person would."""
    g, cfg = bot.game, SS.CFG
    types = {st.type for st in ph.steps}
    for st in ph.steps:
        mk = cfg.markers.get(st.marker) if st.marker else None
        if st.type == "reach":
            bot.move_to(g._target(ph.steps.index(st)), 1.5)
        elif st.type == "place_in_zone":
            bot.pick_place(st.object, mk.pos)
            print(f"  {st.object}: in the zone {g._in_zone(st.object, st.marker)}")
        elif st.type == "place_in_slot":
            bot.pick_place(st.object, mk.pos, dest_yaw=mk.yaw, period=mk.p["symmetry"])
            print(f"  {st.object}: placed {g._slot_ok(st.object, st.marker)}  ({g.metrics[ph.key]['steps_done']}/{len(ph.steps)})")
        elif st.type == "place_in_bin":
            bot.pick_place(st.object, mk.pos, place_z=S.TABLE_Z + mk.p["wall_height"] + 0.045, lift=0.14)
            print(f"  {st.object}: in bin {g._bin_of(st.object)}")
        elif st.type == "remove_from":
            lay = next((s for s in ph.steps if s.type == "lay_flat" and s.object == st.object), None)
            if lay is None:
                raise SystemExit(f"the bot can only take '{st.object}' out of its holder to lay it flat")
            do_book(bot, st.object, st.marker, lay.marker)
        elif st.type == "pour_into":
            cup = cfg.objects[st.objects[0]].start_in
            if cup is None:
                raise SystemExit(f"the bot can only pour '{st.objects[0]}' from a cup they start in")
            do_pour(bot, cup, st.marker)
        elif st.type == "hang_on":
            if cfg.objects[st.object].pitch == 0 and cfg.markers[st.marker].p["tilt"] == 0:
                do_ring_flat(bot, st.object, st.marker)
            else:
                do_ring(bot, st.object, st.marker)
        elif st.type == "stack":
            do_stack(bot, st)
        elif st.type == "follow_wire":
            key = ph.key
            do_wire(bot)
            mt = g.metrics[key]
            print(f"  wire: progress {mt['progress']}, touches {mt['touches']}, touching {mt['touch_time_s']} s, "
                  f"completed {mt['completed']}")
        elif st.type in ("lay_flat", "set_upright") and ("remove_from" in types or "pour_into" in types):
            pass                                   # (done with the step above)
        else:
            raise SystemExit(f"the bot doesn't know how to do a '{st.type}' step on its own")
        if g.phase is not ph:
            break                                  # (the phase ended: done or out of time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[], help="phases to leave undone (their keys, e.g. shapes sort)")
    ap.add_argument("--pics", default=None)
    a = ap.parse_args()
    bot = Bot(pics=a.pics)
    g = bot.game
    bot.advance(0.5)
    for ph in g.phases:
        if ph.kind != "task":
            continue
        bot.wait_phase(ph.key, timeout=ph.limit + 60)
        bot.pic(f"{ph.key}_start")
        print(f"{ph.title} ({ph.key})")
        if ph.key not in a.skip and ph.trials:
            for k in range(ph.trials):                      # (each trial: pinch, play, wait for its end)
                if g.phase is not ph:
                    break
                bot.wait_phase(ph.key, timeout=10)
                print(f"  -- trial {k + 1}/{ph.trials}")
                play_phase(bot, ph)
                t_end = bot.d.time + ph.limit
                while g.phase is ph and g.trial == k and bot.d.time < t_end:
                    bot.advance(0.25)
                bot.grip, bot.rot, bot.yaw = 0.0, None, 0.0
        elif ph.key not in a.skip:
            play_phase(bot, ph)
            for _ in range(ROUNDS - 1):                     # (repeat: a few more rounds)
                if not ph.repeat or g.phase is not ph:
                    break
                bot.advance(1.0)
                print(f"  -- round {g.metrics[ph.key]['rounds'] + 1}")
                play_phase(bot, ph)
        bot.pic(f"{ph.key}_end")
        bot.grip, bot.rot, bot.yaw = 0.0, None, 0.0      # (MOVE let go: the next task starts afresh)
    bot.wait_phase(g.phases[-1].key, timeout=max(p.limit or 0 for p in g.phases) + 60)
    print(f"\nresults ({g.results_path}):")
    res = json.loads(g.results_path.read_text())
    for k, mt in res["metrics"].items():
        if "trials" in mt:
            print(f"  {k:7s} trials " + ", ".join(f"{'ok' if tr['success'] else 'FAIL'} {tr['time_s']:.1f}s "
                                                  f"bumps {tr['bumps']} -> {tr['score']}" for tr in mt["trials"]))
            continue
        print(f"  {k:7s} completed {mt['completed']!s:5} time {mt['time_s']:6.1f}s steps {mt['steps_done']}/{mt['steps_total']} "
              f"grasps {mt['grasps']}/{mt['grasp_attempts']} drops {mt['drops']} path {mt['path_m']:.2f} m "
              f"clutches {mt['clutches']} wrong bin {mt['wrong_bin']} episode {mt['episode']}")
    print("  scores:", res["scores"])
    print(f"  wall time {time.perf_counter() - bot.wall0:.0f} s for {bot.d.time:.0f} s simulated")
    return res


if __name__ == "__main__":
    main()
