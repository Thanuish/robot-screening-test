"""Robot Buzz Wire: guide a ring along a wire without touching it (one player's game, on the
server). The robot, the phone control, the physics loop, the clock and the results are the
shared core (base.py); this file is the game's own part:

    scene      the board, posts, the wire (a capsule chain), the wand (buzzwire.scene)
    start      start_held levels: the wand welded in the gripper, its ring around the wire
    guide      the arm's target limited so the ring can touch the wire but never press into it
    touches    contacts between the wand and the wire (debounced), touch time
    progress   how far along the wire the ring got; done at the far sleeve (or hung on the hook)
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from . import base
from . import scene as SC
from .base import RobotGame
from .config import Config, Level

GUIDE_DEPTH = 0.0015           # the guide lets the ring this far into the wire (so a touch registers), no further
GUIDE_MAX_TILT = np.radians(30)   # ... and keeps the ring's axis within this of the wire's direction
HANG_HOLD_S = 0.5              # the wand must rest on the end hook, let go, this long
HANG_DIST = 0.02               # ... its loop within this of the hook's rest position
HELD_DIST = 0.03               # the handle's middle this close to the pinch point + gripper closed = held
REST_SPEED = 0.02


class BuzzWireGame(RobotGame):
    KIND = "wire"
    TITLE = "Buzz Wire"

    def __init__(self, cfg: Config, levels: list[str] | None = None, out_dir: Path | None = None, player: str = ""):
        super().__init__(cfg, [lv for lv in cfg.levels if levels is None or lv.key in levels], out_dir, player)

    # ------------------------------------------------------------ the game's parts for the core
    def build_scene(self, level: Level):
        return SC.build(self.cfg, level)

    def new_metrics(self, level: Level) -> dict:
        return dict(hung=False, touches=0, touch_time_s=0.0, progress=0.0, picked_up=False, drops=0, robot_hits=0)

    def setup_level(self, lv: Level):
        self._touching = False
        self._last_touch_t = -1e9
        self._hang_since = None
        self._prev_held = False
        self.guide = None              # (start_held levels) the wand's pose in the gripper's frame
        if lv.start_held:
            self._start_in_hand()

    @property
    def touching(self) -> bool:
        return self._touching

    def hud(self) -> list:
        mt = self.metrics
        return [dict(key="touches", label="Touches", value=mt["touches"], alert=self._touching)]

    def score(self, mt: dict, lv: Level) -> float:
        return score_level(mt, lv, self.cfg.scoring)

    def level_geometry(self) -> dict:
        cfg, lv = self.cfg, self.level
        return dict(kind="wire", level=lv.key, wire=lv.wire.pts[::2].round(5).tolist(),
                    wire_length=lv.wire.length, wire_radius=cfg.wire["radius"],
                    sleeve_length=cfg.wire["sleeve_length"], sleeve_radius=cfg.wire["sleeve_radius"],
                    post_radius=cfg.wire["post_radius"], board_center=cfg.board["center"],
                    board_size=cfg.board["size"], loop_radius=lv.loop_radius,
                    loop_thickness=cfg.wand["loop_thickness"], handle_size=cfg.wand["handle_size"])

    def _start_in_hand(self, s0: float | None = None):
        """The wand in the gripper, its loop centred on the wire at arc length s0 (default: the
        start sleeve's middle): the arm is placed there by IK, the fingers close on the handle while
        the wand is held still, then it's let go of (only the fingers hold it)."""
        cfg, lv, d, m = self.cfg, self.level, self.d, self.m
        if s0 is None:
            s0 = cfg.wire["sleeve_length"] * 0.5
        centre, Rw = SC.wand_frame(lv, s0, lv.rotation == "full")
        wq = np.zeros(4)
        mujoco.mju_mat2Quat(wq, Rw.flatten())
        pinch = centre + Rw @ SC.handle_centre_local(cfg, lv)
        gq = np.zeros(4)
        mujoco.mju_mat2Quat(gq, (Rw @ SC.GRIP_IN_WAND).flatten())
        q = np.array(cfg.robot["home"], float)
        d.qvel[self.sc.arm_dof] = 0
        for _ in range(600):
            q = self.ik.step(q, pinch, gq, 0.01)
        err = float(np.linalg.norm(self.ik.fk(q)[0] - pinch))
        if err > 0.003:
            raise RuntimeError(f"level {lv.key}: the robot can't reach the wand's start ({err * 1000:.0f} mm off)")
        sc, a, dof = self.sc, self.sc.wand_qpos, self.sc.wand_dof
        d.qpos[sc.arm_qpos] = q
        d.ctrl[sc.arm_act] = q
        d.ctrl[sc.grip_act] = 255.0
        d.qpos[a:a + 3], d.qpos[a + 3:a + 7], d.qvel[dof:dof + 6] = centre, wq, 0     # the wand at its start
        mujoco.mj_forward(m, d)
        # fix the wand in the hand right where it is (a weld): it can't slip out or fall
        wb, gb = self.sc.wand_body, self.m.body(SC.GRIP + "base").id
        R1, R2 = d.xmat[wb].reshape(3, 3), d.xmat[gb].reshape(3, 3)
        rel_q, q1i = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1i, d.xquat[wb])
        mujoco.mju_mulQuat(rel_q, q1i, d.xquat[gb])
        m.eq_data[self.sc.weld, :3] = 0
        m.eq_data[self.sc.weld, 3:6] = R1.T @ (d.xpos[gb] - d.xpos[wb])
        m.eq_data[self.sc.weld, 6:10] = rel_q
        m.eq_data[self.sc.weld, 10] = 1.0
        d.eq_active[self.sc.weld] = 1
        for k in range(int(0.6 / m.opt.timestep)):     # the fingers close, the arm settles
            if k < int(0.3 / m.opt.timestep):
                d.qpos[a:a + 3], d.qpos[a + 3:a + 7], d.qvel[dof:dof + 6] = centre, wq, 0
            mujoco.mj_step(m, d)
        self.q_cmd = q
        self.teleop.reset(*self.ik.fk(q))
        self.teleop.gripper = 1.0
        self.follow.reset()
        self.metrics["picked_up"] = True
        # the guide: where the ring is relative to the gripper target (the commanded pinch pose)
        Rt = np.zeros(9)
        mujoco.mju_quat2Mat(Rt, self.teleop.target_quat)
        Rt = Rt.reshape(3, 3)
        Rwand = d.xmat[wb].reshape(3, 3)
        self.guide = (Rt.T @ (d.xpos[wb] - self.teleop.target_pos), Rt.T @ Rwand[:, 1])
        self._guide_s = s0
        self._prev_held = self.held()

    def guided(self, gp, gq):
        """The guide (a "virtual fixture"): the arm's target, limited so that the ring can touch
        the wire (GUIDE_DEPTH into it, so the touch registers) but never press into it, and its
        axis stays within GUIDE_MAX_TILT of the wire's direction. Pushing harder keeps the ring
        touching; it can't tear the wand out or push the wire through it."""
        if self.guide is None:
            return gp, gq
        lv, cfg = self.level, self.cfg
        off, ax = self.guide
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, gq)
        R = R.reshape(3, 3)
        centre = gp + R @ off
        s, _ = lv.wire.closest(centre, near=self._guide_s)
        _, t = lv.wire.at(s)
        a = R @ ax
        # tilt: turn the target so the ring's axis is within the limit of the wire's direction
        c = abs(float(np.dot(a, t)))
        ang = np.arccos(min(1.0, c))
        if ang > GUIDE_MAX_TILT:
            n = np.cross(a, t * np.sign(np.dot(a, t) or 1.0))
            nn = np.linalg.norm(n)
            if nn > 1e-9:
                dq = np.zeros(4)
                mujoco.mju_axisAngle2Quat(dq, n / nn, ang - GUIDE_MAX_TILT)
                q2 = np.zeros(4)
                mujoco.mju_mulQuat(q2, dq, gq)
                gq = q2
                R9 = np.zeros(9)
                mujoco.mju_quat2Mat(R9, gq)
                R = R9.reshape(3, 3)
                gp = centre - R @ off                    # (turned about the ring's centre)
        # position: the ring's centre at most (clearance + GUIDE_DEPTH) from the wire
        centre = gp + R @ off
        s, _ = lv.wire.closest(centre, near=self._guide_s)
        self._guide_s = s                                # (followed continuously along the wire)
        p, _ = lv.wire.at(s)
        v = centre - p
        dist = float(np.linalg.norm(v))
        allow = lv.loop_radius - cfg.wand["loop_thickness"] / 2 - cfg.wire["radius"] + GUIDE_DEPTH
        if dist > allow:
            gp = gp - v / dist * (dist - allow)
        return gp, gq

    # ------------------------------------------------------------ the game
    def wand_pose(self):
        a = self.sc.wand_qpos
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def held(self) -> bool:
        if self.sc.weld >= 0 and self.d.eq_active[self.sc.weld]:
            return True                                  # (fixed in the hand)
        p, q = self.wand_pose()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        h = p + R.reshape(3, 3) @ SC.handle_centre_local(self.cfg, self.level)
        return bool(self.teleop.gripper > 0.5 and np.linalg.norm(h - self.d.site_xpos[self.sc.ee_site]) < HELD_DIST)

    def _contacts(self):
        wand_wire = robot_wire = False
        sc = self.sc
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if (g1 in sc.wire_geoms and g2 in sc.wand_geoms) or (g2 in sc.wire_geoms and g1 in sc.wand_geoms):
                wand_wire = True
            elif (g1 in sc.wire_geoms and g2 in sc.robot_geoms) or (g2 in sc.wire_geoms and g1 in sc.robot_geoms):
                robot_wire = True
        return wand_wire, robot_wire

    def update(self, t: float):
        d, mt, lv, tel = self.d, self.metrics, self.level, self.teleop
        held = self.held()
        if held:
            mt["picked_up"] = True
        wand_wire, robot_wire = self._contacts()
        # touches: a new contact (after the debounce) is one touch; touch time while in contact
        if wand_wire:
            if not self._touching and t - self._last_touch_t > self.cfg.physics["touch_debounce"]:
                mt["touches"] += 1
            self._last_touch_t = t
            mt["touch_time_s"] = round(mt["touch_time_s"] + self.dt, 3)
        self._touching = wand_wire
        if robot_wire and not getattr(self, "_robot_touching", False):
            mt["robot_hits"] += 1
        self._robot_touching = robot_wire
        # progress: how far along the wire the loop's centre got (while it is around the wire)
        p, _ = self.wand_pose()
        s, dist = lv.wire.closest(p)
        if dist < lv.loop_radius:
            mt["progress"] = round(max(mt["progress"], s / lv.wire.length), 4)
        # drops: held before, not held now, not set down on the end hook
        end_c, _ = self.sc.rest["end"]
        if self._prev_held and not held and tel.gripper > 0.5:
            mt["drops"] += 1
        self._prev_held = held
        # finish "reach_end": the loop (held) gets to the far sleeve
        if lv.finish == "reach_end" and held and s >= lv.wire.length - 0.8 * self.cfg.wire["sleeve_length"] \
                and dist < lv.loop_radius:
            mt["progress"] = 1.0
            self.end_level(True)
            return
        # finish "hang": resting on the end hook, let go
        v = float(np.linalg.norm(d.qvel[self.sc.wand_dof:self.sc.wand_dof + 3]))
        on_hook = (lv.finish == "hang" and np.linalg.norm(p - end_c) < HANG_DIST and v < REST_SPEED
                   and tel.gripper < 0.1)
        if on_hook:
            self._hang_since = self._hang_since if self._hang_since is not None else t
            if t - self._hang_since >= HANG_HOLD_S:
                mt["hung"] = True
                mt["progress"] = 1.0
                self.end_level(True)
                return
        else:
            self._hang_since = None

def score_level(mt: dict, lv: Level, w: dict) -> float:
    """0..100 for one level (see [scoring] in the configuration)."""
    progress = 1.0 if mt["completed"] else 0.95 * mt["progress"]
    no_touch = max(0.0, 1 - mt["touches"] / w["touches_to_zero"])
    touch_t = max(0.0, 1 - mt["touch_time_s"] / w["touch_seconds_to_zero"])
    time_part = max(0.0, 1 - mt["time_s"] / lv.limit) if mt["completed"] else 0.0
    j = mt["rms_jerk"]
    smooth = 0.0 if j is None else float(np.clip((w["jerky_jerk"] - j) / (w["jerky_jerk"] - w["smooth_jerk"]), 0, 1))
    if not mt["picked_up"]:                    # nothing done: no credit for "no touches" either
        no_touch = touch_t = smooth = 0.0
    return round(w["progress"] * progress + w["no_touches"] * no_touch + w["touch_time"] * touch_t
                 + w["time"] * time_part + w["smoothness"] * smooth, 1)


def replay(cfg: Config, input_log: dict, levels: list[str] | None = None) -> BuzzWireGame:
    """Run a game again from its input log; returns the finished game (same scores)."""
    return base.replay(BuzzWireGame, cfg, input_log, levels=levels)


Game = BuzzWireGame            # (the name the bot and older code use)
