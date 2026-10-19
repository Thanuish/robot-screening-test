"""What every robot game here shares: one player's game, simulated on the server.

One control tick (control_hz, 100 a second):
    input      the latest controller message (the phone, or a bot: the format of a VR controller:
               pose, grip = clutch / MOVE, stick y = close / open the gripper)
    teleop     clutch-relative mapping (vrteleop.teleop.ArmTeleop): while MOVE is held the
               gripper target moves and turns like the hand
    rotation   per level: "none" (the gripper keeps its angle), "yaw" (it only turns about the
               vertical), "full" (all six degrees of freedom)
    smoothing  a low-pass filter on the target (a phone sends ~30 poses a second)
    limits     the target moves at most max_speed / max_turn
    guide      (optional, per game) the target limited, e.g. so a ring can't press into a wire
    IK         joint targets from Pinocchio (joint limits and speeds, table keep-out)
    physics    control_hz-th of a second of MuJoCo steps
    game       the game's own rules (update()), the clock

Levels: waiting (the first level: until a controller sends a pose) -> countdown (the arm is
frozen) -> play (the clock runs) -> done (the game ends it, or time is up) -> the next level.
Every input is logged with the tick it was applied on, so replay() runs exactly the same game
again (same physics, same score).

A game subclasses RobotGame and supplies: build_scene(level), new_metrics(level),
setup_level(level), update(t), score(metrics, level), hud() and level_geometry().
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from vrteleop.teleop import ArmTeleop

from .config import TABLE_Z, Config
from .ik import IK

COUNTDOWN_S = 3.0
FF_SMOOTH = 0.3                 # the feed-forward velocity: this share of the way to the new one per tick (~30 ms)


def yaw_only(q):
    """The turn about the vertical of an orientation, with the gripper pointing down."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    fy = R.reshape(3, 3)[:, 1]          # the gripper's fingers close along its local y: keep its heading
    yaw = np.arctan2(-fy[0], fy[1])
    down = np.array([0.0, 1.0, 0.0, 0.0])               # pointing down, yaw 0 (the home orientation)
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, down)
    return out


class OneEuro:
    """The One Euro filter (Casiez, Roussel and Vogel, CHI 2012), the usual answer to "smooth but
    no lag" for hand tracking: a low-pass filter whose cut-off rises with the speed. A still hand
    is filtered strongly (no shaking from tracking noise); a fast one hardly at all (no lag).
        cut-off [Hz] = min_cutoff + beta x speed  (the speed itself low-passed at d_cutoff)"""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.speed = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt, distance=None):
        """x: the new value (a vector); distance(a, b): how far apart two values are (default:
        Euclidean); returns the alpha used (0..1: how far to move toward x)."""
        if self.x is None:
            self.x = x
            return 1.0
        d = float(np.linalg.norm(np.asarray(x) - np.asarray(self.x))) if distance is None else distance(self.x, x)
        a_d = self._alpha(self.d_cutoff, dt)
        self.speed += a_d * (d / dt - self.speed)
        return self._alpha(self.min_cutoff + self.beta * abs(self.speed), dt)


class Follower:
    """The arm's goal: the teleop target, smoothed, then speed-limited (as vrteleop/arms.py:
    _smoothed, then _limited).

    Smoothing ([controls] filter): "simple" (the default, as in the screening test) moves a
    fraction 1 - exp(-dt / smoothing_s) of the way toward the hand every control tick, position
    and rotation: the same delay at every speed. "one_euro" filters each new controller sample with
    the One Euro filter, then glides between samples."""

    def __init__(self, cfg: Config):
        ct = cfg.controls
        self.tau = ct["smoothing"]
        self.euro = ct.get("filter") == "one_euro"
        if self.euro:
            self.f_pos = OneEuro(ct["min_cutoff_hz"], ct["beta"])
            self.f_rot = OneEuro(ct["min_cutoff_hz"], ct["rot_beta"])
        self.vmax, self.wmax = cfg.robot["max_speed"], cfg.robot["max_turn"]
        self.s = self.g = None
        self.raw = None                 # the last controller sample (one_euro: filtered once per sample)
        self.since = 0.0                # time since that sample [s]
        self.glide = ct.get("glide", 0.025)   # [s] glide between filtered samples
        self.h = None

    def reset(self):
        self.s = self.g = self.raw = self.h = None
        self.since = 0.0
        if self.euro:
            self.f_pos.x = self.f_rot.x = None
            self.f_pos.speed = self.f_rot.speed = 0.0

    def _smooth(self, pos, quat, dt):
        sp, sq = self.s
        q1 = quat if np.dot(sq, quat) >= 0 else -quat
        if not self.euro:
            a = 1.0 if not self.tau else 1.0 - np.exp(-dt / self.tau)
            sq = sq * (1 - a) + q1 * a
            return sp + (pos - sp) * a, sq / np.linalg.norm(sq)
        self.since += dt
        if self.raw is not None and np.array_equal(pos, self.raw[0]) and np.array_equal(quat, self.raw[1]):
            return self.s                                  # (no new sample: the filtered target holds)
        sdt, self.since = max(self.since, 1e-3), 0.0
        self.raw = (pos.copy(), quat.copy())
        a = self.f_pos(pos, sdt)
        ar = self.f_rot(q1, sdt, distance=lambda u, v: 2 * np.arccos(min(1.0, abs(float(np.dot(u, v))))))
        sp = sp + (pos - sp) * a
        sq = sq * (1 - ar) + q1 * ar
        sq /= np.linalg.norm(sq)
        self.f_pos.x, self.f_rot.x = sp, sq
        return sp, sq

    def __call__(self, pos, quat, dt):
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        if self.s is None:
            self.s = self.g = (pos.copy(), quat.copy())
            self.raw = (pos.copy(), quat.copy())
            if self.euro:
                self.f_pos(pos, dt), self.f_rot(quat, dt)
            return self.g
        self.s = self._smooth(pos, quat, dt)
        if self.euro and self.glide:                       # glide between the filtered samples (no 30 Hz steps)
            a = 1.0 - np.exp(-dt / self.glide)
            hp, hq = self.h if self.h is not None else self.s
            q1 = self.s[1] if np.dot(hq, self.s[1]) >= 0 else -self.s[1]
            hq = hq * (1 - a) + q1 * a
            self.h = (hp + (self.s[0] - hp) * a, hq / np.linalg.norm(hq))
            target = self.h
        else:
            target = self.s
        gp, gq = self.g
        step = target[0] - gp
        n = np.linalg.norm(step)
        if n > self.vmax * dt:
            step *= self.vmax * dt / n
        q1 = target[1] if np.dot(gq, target[1]) >= 0 else -target[1]
        ang = 2 * np.arccos(min(1.0, abs(float(np.dot(gq, q1)))))
        if ang > self.wmax * dt:
            f = self.wmax * dt / ang
            q1 = gq * (1 - f) + q1 * f
        self.g = (gp + step, q1 / np.linalg.norm(q1))
        return self.g


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


class RobotGame:
    KIND = "base"                   # the game's key (server, menu, results)
    TITLE = ""

    def __init__(self, cfg: Config, levels: list, out_dir: Path | None = None, player: str = ""):
        self.cfg = cfg
        self.levels = levels
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

    # ------------------------------------------------------------ to be supplied by a game
    def build_scene(self, level):
        raise NotImplementedError

    def new_metrics(self, level) -> dict:
        return {}

    def setup_level(self, level):
        """After the arm is home: put the level's things in place."""

    def update(self, t: float):
        """One tick of the game's rules while playing (t: seconds since the start); call
        self.end_level(completed=True) when the level is done."""

    def score(self, mt: dict, level) -> float:
        return 0.0

    def hud(self) -> list:
        """The game's numbers for the screens: [{"key", "label", "value", "alert"}], first = main."""
        return []

    def level_geometry(self) -> dict:
        return {"kind": self.KIND}

    def guided(self, gp, gq):
        return gp, gq

    def held(self) -> bool:
        return False

    @property
    def touching(self) -> bool:
        """Shown as a buzz (red flash) on the screens."""
        return False

    # ------------------------------------------------------------ levels
    @property
    def level(self):
        return self.levels[self.li]

    def next_level(self):
        self.li += 1
        if self.li >= len(self.levels):
            self.finished = True
            self.li = len(self.levels) - 1
            return
        lv, cfg = self.level, self.cfg
        self.sc = self.build_scene(lv)
        self.m = self.sc.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[self.sc.arm_qpos] = cfg.robot["home"]
        self.d.ctrl[self.sc.arm_act] = cfg.robot["home"]
        mujoco.mj_forward(self.m, self.d)
        self.ik = IK(self.sc, self.d)
        self.q_cmd = np.array(cfg.robot["home"], float)
        self.teleop = ArmTeleop(pos_scale=cfg.controls["motion_scale"], z_min=self.ik.z_min)
        self.teleop.reset(*self.ik.fk(self.q_cmd))
        self.follow = Follower(cfg)
        self.tick = 0
        self.state = "waiting" if self.li == 0 else "countdown"
        self._count_from = 0
        self.t_play = None
        self._ee_log = []
        self._was_engaged = False
        self._prev_ee = None
        self.metrics = dict(level=lv.key, completed=False, timed_out=False, time_s=0.0, path_m=0.0, clutches=0,
                            rms_jerk=None, **self.new_metrics(lv))
        self.input_log[lv.key] = []
        self._logged = None
        self.message = lv.title
        self.setup_level(lv)
        self._quat0 = self.teleop.target_quat.copy()      # (rotation "none": the gripper keeps this)
        # velocity feed-forward ([robot] servo_feedforward): each joint motor is a PD controller
        # (force = kp (q_cmd - q) - kv qdot), which on its own trails a moving command by kv/kp
        # (0.1 s). Commanding q_cmd + (kv/kp) x the command's velocity makes it a PD controller
        # with a velocity reference, as an industrial robot's servo: it follows without trailing,
        # and it's no stiffer (no twitching)
        m, act = self.m, self.sc.arm_act
        self._ff = -m.actuator_biasprm[act, 2] / m.actuator_gainprm[act, 0] if cfg.robot["servo_feedforward"] else None
        self._q_prev, self._qd = None, np.zeros(len(act))

    def end_level(self, completed: bool):
        mt, lv = self.metrics, self.level
        mt["completed"] = bool(completed)
        mt["timed_out"] = not completed
        mt["rms_jerk"] = rms_jerk(np.array(self._ee_log), self.dt)
        mt["score"] = self.score(mt, lv)
        self.results[lv.key] = dict(mt)
        self.message = "Well done!" if completed else "Time is up."
        self.state = "done"
        self.write_results()
        self.next_level()

    # ------------------------------------------------------------ input
    def set_input(self, msg: dict):
        """A controller message: {'right': {'pose': [x,y,z,qw,qx,qy,qz], 'grip', 'stick', 'epoch'}}."""
        self.latest_input = msg

    def _controller(self):
        msg = self.latest_input or {}
        return msg.get("right") or msg.get("left")

    # ------------------------------------------------------------ one control tick
    def step(self):
        if self.finished:
            return
        lv = self.level
        ctrl = self._controller()
        key = json.dumps(ctrl, sort_keys=True) if ctrl else None    # log the changes, for an exact replay
        if key != self._logged:
            self.input_log[lv.key].append([self.tick, ctrl])
            self._logged = key
        d, sc = self.d, self.sc
        if self.state == "waiting":
            ctrl = None
            if (self._controller() or {}).get("pose"):
                self.state, self._count_from = "countdown", self.tick
        elif self.state == "countdown":
            ctrl = None                                     # the arm waits for the start
            if (self.tick - self._count_from) * self.dt >= COUNTDOWN_S:
                self.state, self.t_play = "play", d.time
        if ctrl is not None and lv.rotation == "yaw" and ctrl.get("pose"):
            ctrl = dict(ctrl, pose=list(ctrl["pose"][:3]) + list(yaw_only(ctrl["pose"][3:7])))
        if ctrl is not None and getattr(lv, "start_held", False):
            ctrl = dict(ctrl, stick=[0.0, 0.0])          # the thing is fixed in the hand: the gripper stays closed
        tel = self.teleop
        if not tel.engaged:
            tel.target_pos, tel.target_quat = self.ik.fk(self.q_cmd)   # glued to the arm while idle
        tel.update(ctrl, self.dt)
        if lv.rotation == "none":
            tel.target_quat = self._quat0.copy()         # the gripper keeps its angle
        if tel.engaged:
            gp, gq = self.follow(tel.target_pos, tel.target_quat, self.dt)
            gp, gq = self.guided(gp, gq)
            self.q_cmd = self.ik.step(self.q_cmd, gp, gq, self.dt)
        else:
            self.follow.reset()
        if self._ff is not None:
            qd = np.zeros_like(self.q_cmd) if self._q_prev is None else (self.q_cmd - self._q_prev) / self.dt
            self._qd += (qd - self._qd) * FF_SMOOTH          # (a little smoothing: no tick-to-tick jitter)
            self._q_prev = self.q_cmd.copy()
            d.ctrl[sc.arm_act] = self.q_cmd + self._ff * self._qd
        else:
            d.ctrl[sc.arm_act] = self.q_cmd
        d.ctrl[sc.grip_act] = 255.0 * tel.gripper
        for _ in range(self.substeps):
            mujoco.mj_step(self.m, d)
        self.tick += 1
        if self.state == "play":
            mt = self.metrics
            t = d.time - self.t_play
            mt["time_s"] = round(t, 3)
            ee = d.site_xpos[sc.ee_site].copy()
            if tel.engaged and not self._was_engaged:
                mt["clutches"] += 1
            self._was_engaged = tel.engaged
            if self._prev_ee is not None:
                mt["path_m"] += float(np.linalg.norm(ee - self._prev_ee))
            self._prev_ee = ee
            if self.held():
                self._ee_log.append(ee)                    # (smoothness: while holding the thing)
            key = lv.key
            self.update(t)
            if self.state == "play" and self.level.key == key and t >= lv.limit:
                self.end_level(False)

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
        res = dict(game=self.KIND, player=self.player, finished=self.finished, levels=self.results,
                   scores=self.scores(), saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        (self.out_dir / "results.json").write_text(json.dumps(res, indent=2, default=_plain), encoding="utf-8")
        (self.out_dir / "inputs.json").write_text(json.dumps(self.input_log), encoding="utf-8")

    def status(self) -> dict:
        lv, mt = self.level, self.metrics
        countdown = max(0.0, COUNTDOWN_S - (self.tick - self._count_from) * self.dt) if self.state == "countdown" else 0.0
        return dict(game=self.KIND, game_title=self.TITLE, level=lv.key, index=self.li, count=len(self.levels),
                    title=lv.title, text=lv.text, state=self.state, countdown=round(countdown, 1),
                    time=mt["time_s"], limit=lv.limit, hud=self.hud(), touching=bool(self.touching),
                    progress=float(mt.get("progress", 0.0)), steps=self.steps(), held=bool(self.held()),
                    rotation=lv.rotation, finished=self.finished, message=self.message,
                    start_held=bool(getattr(lv, "start_held", False)),
                    engaged=self.teleop.engaged, gripper=round(self.teleop.gripper, 2), view_yaw=self.view_yaw)

    def steps(self) -> list:
        """A checklist for the screens: [{"label", "done"}] (empty: none)."""
        return []


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def replay(game_cls, cfg: Config, input_log: dict, **kw):
    """Run a game again from its input log; returns the finished game (same scores)."""
    g = game_cls(cfg, **kw)
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
    return g
