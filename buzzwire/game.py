"""Robot Buzz Wire: the simulation of one player's game, authoritative on the server.

One control tick (control_hz, 100 a second) does:
    input      the latest controller message (phone, keyboard, bot: the same format as a VR
               controller: pose, grip = clutch, stick y = close / open the gripper)
    teleop     clutch-relative mapping (vrteleop.teleop.ArmTeleop, as in the screening test):
               while MOVE is held the gripper target moves and turns like the hand
    smoothing  a low-pass filter on the target (a phone sends ~30 poses a second)
    limits     the target moves at most max_speed / max_turn
    IK         joint targets from Pinocchio (joint limits and speeds, table keep-out)
    physics    control_hz-th of a second of MuJoCo steps
    game       touches (contacts between the wand and the wire), progress, the level's state

Levels: countdown (the arm is frozen) -> play (the clock runs) -> done (the wand hangs on the
end hook, or time is up) -> the next level. Every input is logged with the tick it was applied
on, so replay() runs exactly the same game again (same physics, same score).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from vrteleop.teleop import ArmTeleop

from . import scene as SC
from .config import TABLE_Z, Config, Level
from .ik import IK

COUNTDOWN_S = 3.0
HANG_HOLD_S = 0.5              # the wand must rest on the end hook, let go, this long
HANG_DIST = 0.02               # ... its loop within this of the hook's rest position
HELD_DIST = 0.03               # the handle's middle this close to the pinch point + gripper closed = held
REST_SPEED = 0.02


def _yaw_only(q):
    """The turn about the vertical of an orientation, with the gripper pointing down."""
    w, x, y, z = q
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    R = R.reshape(3, 3)
    # the gripper's fingers close along its local y; keep that direction's heading
    fy = R[:, 1]
    yaw = np.arctan2(-fy[0], fy[1])
    down = np.array([0.0, 1.0, 0.0, 0.0])               # pointing down, yaw 0 (the home orientation)
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, down)
    return out


class Follower:
    """The arm's goal: the teleop target, smoothed and speed-limited."""

    def __init__(self, cfg: Config):
        self.tau = cfg.controls["smoothing"]
        self.vmax, self.wmax = cfg.robot["max_speed"], cfg.robot["max_turn"]
        self.s = self.g = None

    def reset(self):
        self.s = self.g = None

    def __call__(self, pos, quat, dt):
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        if self.s is None:
            self.s = self.g = (pos.copy(), quat.copy())
            return self.g
        sp, sq = self.s
        a = 1.0 if not self.tau else 1.0 - np.exp(-dt / self.tau)
        q1 = quat if np.dot(sq, quat) >= 0 else -quat
        sp = sp + (pos - sp) * a
        sq = sq * (1 - a) + q1 * a
        self.s = (sp, sq / np.linalg.norm(sq))
        gp, gq = self.g
        step = self.s[0] - gp
        n = np.linalg.norm(step)
        if n > self.vmax * dt:
            step *= self.vmax * dt / n
        q1 = self.s[1] if np.dot(gq, self.s[1]) >= 0 else -self.s[1]
        ang = 2 * np.arccos(min(1.0, abs(float(np.dot(gq, q1)))))
        if ang > self.wmax * dt:
            f = self.wmax * dt / ang
            q1 = gq * (1 - f) + q1 * f
        self.g = (gp + step, q1 / np.linalg.norm(q1))
        return self.g


class Game:
    def __init__(self, cfg: Config, levels: list[str] | None = None, out_dir: Path | None = None,
                 player: str = ""):
        self.cfg = cfg
        self.levels: list[Level] = [lv for lv in cfg.levels if levels is None or lv.key in levels]
        self.out_dir = Path(out_dir) if out_dir else None
        self.player = player
        self.dt = 1.0 / cfg.physics["control_hz"]
        self.substeps = int(round(self.dt / cfg.physics["timestep"]))
        self.results: dict[str, dict] = {}
        self.input_log: dict[str, list] = {}
        self.latest_input: dict | None = None
        self.view_yaw = 0.0
        self.li = -1
        self.finished = False
        self.next_level()

    # ------------------------------------------------------------ levels
    @property
    def level(self) -> Level:
        return self.levels[self.li]

    def next_level(self):
        self.li += 1
        if self.li >= len(self.levels):
            self.finished = True
            self.li = len(self.levels) - 1
            return
        lv = self.level
        self.sc = SC.build(self.cfg, lv)
        self.m = self.sc.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[self.sc.arm_qpos] = self.cfg.robot["home"]
        self.d.ctrl[self.sc.arm_act] = self.cfg.robot["home"]
        mujoco.mj_forward(self.m, self.d)
        self.ik = IK(self.sc, self.d)
        self.q_cmd = np.array(self.cfg.robot["home"], float)
        self.teleop = ArmTeleop(pos_scale=self.cfg.controls["motion_scale"],
                                z_min=TABLE_Z + self.cfg.robot["table_clearance"])
        self.teleop.reset(*self.ik.fk(self.q_cmd))
        self.follow = Follower(self.cfg)
        self.tick = 0
        self.state = "countdown"
        self.t_play = None
        self.touching = False
        self._last_touch_t = -1e9
        self._hang_since = None
        self._prev_held = False
        self._ee_log = []
        self.metrics = dict(level=lv.key, completed=False, hung=False, timed_out=False, time_s=0.0, touches=0,
                            touch_time_s=0.0, progress=0.0, picked_up=False, drops=0, robot_hits=0,
                            rms_jerk=None, path_m=0.0, clutches=0)
        self._was_engaged = False
        self._prev_ee = None
        self.input_log[lv.key] = []
        self._logged = None
        self.message = lv.title

    # ------------------------------------------------------------ input
    def set_input(self, msg: dict):
        """A controller message: {'right': {'pose': [x,y,z,qw,qx,qy,qz], 'grip', 'stick', 'epoch'}}
        (the phone's format; 'left' is ignored), or {'type': 'view', 'yaw': ...}."""
        self.latest_input = msg

    def _controller(self):
        msg = self.latest_input or {}
        return msg.get("right") or msg.get("left")

    # ------------------------------------------------------------ one control tick
    def step(self):
        if self.finished:
            return
        ctrl = self._controller()
        # log what is applied at this tick (only changes), for an exact replay
        key = json.dumps(ctrl, sort_keys=True) if ctrl else None
        if key != self._logged:
            self.input_log[self.level.key].append([self.tick, ctrl])
            self._logged = key
        d, sc = self.d, self.sc
        if self.state == "countdown":
            ctrl = None                                     # the arm waits for the start
            if self.tick * self.dt >= COUNTDOWN_S:
                self.state, self.t_play = "play", d.time
        if ctrl is not None and self.level.rotation == "yaw" and ctrl.get("pose"):
            ctrl = dict(ctrl, pose=list(ctrl["pose"][:3]) + list(_yaw_only(ctrl["pose"][3:7])))
        tel = self.teleop
        if not tel.engaged:
            tel.target_pos, tel.target_quat = self.ik.fk(self.q_cmd)   # glued to the arm while idle
        tel.update(ctrl, self.dt)
        if tel.engaged:
            gp, gq = self.follow(tel.target_pos, tel.target_quat, self.dt)
            self.q_cmd = self.ik.step(self.q_cmd, gp, gq, self.dt)
        else:
            self.follow.reset()
        d.ctrl[sc.arm_act] = self.q_cmd
        d.ctrl[sc.grip_act] = 255.0 * tel.gripper
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, d)
        self.tick += 1
        if self.state == "play":
            self._update_game(tel)

    # ------------------------------------------------------------ the game
    def wand_pose(self):
        a = self.sc.wand_qpos
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def held(self) -> bool:
        p, q = self.wand_pose()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        h = p + R.reshape(3, 3) @ SC.handle_centre_local(self.cfg, self.level)
        return self.teleop.gripper > 0.5 and np.linalg.norm(h - self.d.site_xpos[self.sc.ee_site]) < HELD_DIST

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

    def _update_game(self, tel):
        d, mt, lv = self.d, self.metrics, self.level
        t = d.time - self.t_play
        mt["time_s"] = round(t, 3)
        ee = d.site_xpos[self.sc.ee_site].copy()
        if tel.engaged and not self._was_engaged:
            mt["clutches"] += 1
        self._was_engaged = tel.engaged
        if self._prev_ee is not None:
            mt["path_m"] += float(np.linalg.norm(ee - self._prev_ee))
        self._prev_ee = ee
        held = self.held()
        if held:
            mt["picked_up"] = True
            self._ee_log.append(ee)
        wand_wire, robot_wire = self._contacts()
        # touches: a new contact (after the debounce) is one touch; touch time while in contact
        if wand_wire:
            if not self.touching and t - self._last_touch_t > self.cfg.physics["touch_debounce"]:
                mt["touches"] += 1
            self._last_touch_t = t
            mt["touch_time_s"] = round(mt["touch_time_s"] + self.dt, 3)
        self.touching = wand_wire
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
        # hung: resting on the end hook, let go
        v = float(np.linalg.norm(d.qvel[self.sc.wand_dof:self.sc.wand_dof + 3]))
        on_hook = np.linalg.norm(p - end_c) < HANG_DIST and v < REST_SPEED and tel.gripper < 0.1
        if on_hook:
            self._hang_since = self._hang_since if self._hang_since is not None else t
            if t - self._hang_since >= HANG_HOLD_S:
                mt["hung"] = mt["completed"] = True
                mt["progress"] = 1.0
                self._end_level()
                return
        else:
            self._hang_since = None
        if t >= lv.limit:
            mt["timed_out"] = True
            self._end_level()

    def _end_level(self):
        mt, lv = self.metrics, self.level
        mt["rms_jerk"] = rms_jerk(np.array(self._ee_log), self.dt)
        mt["score"] = score_level(mt, lv, self.cfg.scoring)
        self.results[lv.key] = dict(mt)
        self.message = ("Well done!" if mt["completed"] else "Time is up.")
        self.state = "done"
        self.write_results()
        self.next_level()

    # ------------------------------------------------------------ results
    def scores(self) -> dict:
        out = {k: r["score"] for k, r in self.results.items()}
        scored = [lv.key for lv in self.levels if lv.scored]
        if scored and all(k in out for k in scored):
            out["overall"] = round(float(np.mean([out[k] for k in scored])), 1)
        return out

    def write_results(self):
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        res = dict(player=self.player, finished=self.finished, levels=self.results, scores=self.scores(),
                   saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.out_dir / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        (self.out_dir / "inputs.json").write_text(json.dumps(self.input_log), encoding="utf-8")

    def status(self) -> dict:
        lv, mt = self.level, self.metrics
        countdown = max(0.0, COUNTDOWN_S - self.tick * self.dt) if self.state == "countdown" else 0.0
        st = dict(level=lv.key, index=self.li, count=len(self.levels), title=lv.title, text=lv.text,
                  state=self.state, countdown=round(countdown, 1), time=mt["time_s"], limit=lv.limit,
                  touches=mt["touches"], touching=self.touching, progress=mt["progress"], held=self.held(),
                  rotation=lv.rotation, finished=self.finished, message=self.message,
                  engaged=self.teleop.engaged, gripper=round(self.teleop.gripper, 2))
        # what the phone controller page reads (the screening test's format)
        st["task_game"] = dict(rotation=lv.rotation, arm="right", finished=self.finished, title=lv.title,
                               text=lv.text, phase=lv.key, kind="task")
        st["view_yaw"] = self.view_yaw
        return st


def rms_jerk(P, dt) -> float | None:
    """Root-mean-square jerk [m/s^3] of a path sampled every dt (smoothed over 50 ms first, so
    the physics' tiny vibrations don't count)."""
    if len(P) < 12:
        return None
    k = max(1, int(round(0.05 / dt)))
    kern = np.ones(k) / k
    Ps = np.stack([np.convolve(P[:, i], kern, mode="valid") for i in range(3)], axis=1)
    J = np.diff(Ps, n=3, axis=0) / dt ** 3
    return round(float(np.sqrt(np.mean(np.sum(J ** 2, axis=1)))), 3)


def score_level(mt: dict, lv: Level, w: dict) -> float:
    """0..100 for one level (see [scoring] in the configuration)."""
    progress = 1.0 if mt["hung"] else 0.95 * mt["progress"]
    no_touch = max(0.0, 1 - mt["touches"] / w["touches_to_zero"])
    touch_t = max(0.0, 1 - mt["touch_time_s"] / w["touch_seconds_to_zero"])
    time_part = max(0.0, 1 - mt["time_s"] / lv.limit) if mt["completed"] else 0.0
    j = mt["rms_jerk"]
    smooth = 0.0 if j is None else float(np.clip((w["jerky_jerk"] - j) / (w["jerky_jerk"] - w["smooth_jerk"]), 0, 1))
    if not mt["picked_up"]:                    # nothing done: no credit for "no touches" either
        no_touch = touch_t = smooth = 0.0
    return round(w["progress"] * progress + w["no_touches"] * no_touch + w["touch_time"] * touch_t
                 + w["time"] * time_part + w["smoothness"] * smooth, 1)


def replay(cfg: Config, input_log: dict, levels: list[str] | None = None) -> Game:
    """Run a game again from its input log; returns the finished Game (same scores)."""
    g = Game(cfg, levels=levels)
    while not g.finished:
        log = input_log.get(g.level.key, [])
        key = g.level.key
        i = 0
        while not g.finished and g.level.key == key:
            while i < len(log) and log[i][0] <= g.tick:
                ctrl = log[i][1]
                g.set_input({"right": ctrl} if ctrl else {})
                i += 1
            g.step()
        if g.finished:
            break
    return g
