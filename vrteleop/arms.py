"""The two teleoperated UR5e arms: controller input -> clutch mapping -> differential IK ->
joint and gripper commands. Knows nothing about cloth, recording or the network."""
from __future__ import annotations

import mujoco
import numpy as np

from . import scene as S
from .ik import DiffIK, QPIK
from .teleop import ArmTeleop


class FloatingIK:
    """A floating gripper's stand-in for the IK (a scene with info.floating): its "joint angles"
    are the pinch point's pose itself, [x y z qw qx qy qz]; the target goes to it directly."""

    def fk(self, q, full_qpos=None):
        q = np.asarray(q, float)
        return q[:3].copy(), q[3:] / np.linalg.norm(q[3:])

    def step(self, q, pos, quat, dt, full_qpos=None):
        return np.concatenate([np.asarray(pos, float), np.asarray(quat, float)])


def _mat(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    return R.reshape(3, 3)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


class DualArms:
    def __init__(self, info: S.SceneInfo, task: str, pos_scale: float = 1.0):
        self.info = info
        m = info.model
        self.home = {s: S.home_q(task, s) for s in S.SIDES}
        self.ik = {s: DiffIK(m, info.arm_qpos_adr[s], info.arm_dof_adr[s], info.ee_site[s], self.home[s])
                   for s in S.SIDES}
        self.teleop = {s: ArmTeleop(pos_scale=pos_scale, z_min=S.PINCH_Z_MIN[task]) for s in S.SIDES}
        self.q_cmd = {s: self.home[s].copy() for s in S.SIDES}
        # Optional speed limits for the gripper [m/s, rad/s] (None = follow the hand at once): the
        # arm then moves toward the operator's target no faster than a real robot would
        self.max_speed: float | None = None
        self.max_turn: float | None = None
        self._goal = {s: None for s in S.SIDES}      # the rate-limited target (pos, quat)
        # Optional smoothing [s] (None = off): the target follows the hand through a low-pass
        # filter with this time constant. A phone's AR tracking sends ~30 poses a second; without
        # it the arm moves in small steps
        self.smoothing: float | None = None
        self._smooth = {s: None for s in S.SIDES}    # the smoothed target (pos, quat)
        # arms that ignore the controller for now (e.g. during a game's countdown): they hold
        # still; when released, the clutch takes hold afresh (no jump)
        self.frozen: set = set()
        self._homing = {s: None for s in S.SIDES}     # an arm gliding back to a pose (start_homing)
        # a floating gripper in place of one arm (info.floating, set by a task's bind): its pose is
        # commanded through a mocap target, the hidden arm stays at home
        self.float = getattr(info, "floating", None)
        if self.float is not None:
            self._setup_floating()

    def _setup_floating(self):
        """The floating gripper starts where the arm's gripper is at home (same pose): the pinch
        point's pose relative to the gripper's mount, and that home pose."""
        f, m = self.float, self.info.model
        s = f["side"]
        d0 = mujoco.MjData(m)
        d0.qpos[self.info.arm_qpos_adr[s]] = self.home[s]
        mujoco.mj_kinematics(m, d0)
        arm_site = m.site(f"{s}_gripper_pinch").id
        home_p = d0.site_xpos[arm_site].copy()
        home_R = d0.site_xmat[arm_site].reshape(3, 3).copy()
        Rm, pm = d0.xmat[f["body"]].reshape(3, 3), d0.xpos[f["body"]]
        ps, Rs = d0.site_xpos[self.info.ee_site[s]], d0.site_xmat[self.info.ee_site[s]].reshape(3, 3)
        f["R_rel"], f["p_rel"] = Rm.T @ Rs, Rm.T @ (ps - pm)        # pinch point in the mount's frame
        self.arm_home = self.home[s].copy()
        self.home[s] = np.concatenate([home_p, _quat(home_R)])
        self.ik[s] = FloatingIK()
        self.q_cmd[s] = self.home[s].copy()

    def _floating(self, s: str) -> bool:
        return self.float is not None and s == self.float["side"]

    def _mount_pose(self, q):
        """The mount's pose that puts the pinch point at pose q."""
        f = self.float
        p, quat = FloatingIK().fk(q)
        Rm = _mat(quat) @ f["R_rel"].T
        return p - Rm @ f["p_rel"], _quat(Rm)

    def _command(self, d: mujoco.MjData, s: str):
        if self._floating(s):
            p, quat = self._mount_pose(self.q_cmd[s])
            d.mocap_pos[self.float["mocap"]], d.mocap_quat[self.float["mocap"]] = p, quat
            d.ctrl[self.info.arm_act[s]] = self.arm_home
        else:
            d.ctrl[self.info.arm_act[s]] = self.q_cmd[s]

    def place_home(self, d: mujoco.MjData, side: str):
        """Put an arm (or the floating gripper) at its home pose, at rest."""
        if self._floating(side):
            f = self.float
            p, quat = self._mount_pose(self.home[side])
            d.qpos[f["qadr"]:f["qadr"] + 7] = np.concatenate([p, quat])
            d.qvel[f["vadr"]:f["vadr"] + 6] = 0
            d.mocap_pos[f["mocap"]], d.mocap_quat[f["mocap"]] = p, quat
            d.qpos[self.info.arm_qpos_adr[side]] = self.arm_home
        else:
            d.qpos[self.info.arm_qpos_adr[side]] = self.home[side]
        d.qvel[self.info.arm_dof_adr[side]] = 0

    def use_qp_ik(self, **opts):
        """Switch every arm to the QP IK (joint limits and speeds as exact bounds, table
        keep-out); opts go to QPIK (e.g. clearance={...})."""
        m = self.info.model
        self.ik = {s: self.ik[s] if self._floating(s) else
                   QPIK(m, self.info.arm_qpos_adr[s], self.info.arm_dof_adr[s], self.info.ee_site[s],
                        self.home[s], prefix=f"{s}_", **opts) for s in S.SIDES}

    def start_homing(self, side: str, q=None, speed_deg: float = 60.0):
        """Glide the arm back to joint angles q (default: its home pose), about speed_deg per
        second on the joint that has furthest to go (0.8 to 3 s). The controller is ignored
        meanwhile; afterwards the clutch takes hold afresh (no jump). The gripper stays as it is."""
        q1 = np.asarray(self.home[side] if q is None else q, float)
        q0 = self.q_cmd[side].copy()
        if self._floating(side):                    # (a pose: the shorter way round; the time by distance)
            if np.dot(q0[3:], q1[3:]) < 0:
                q1 = np.concatenate([q1[:3], -q1[3:]])
            ang = 2 * np.arccos(min(1.0, abs(float(np.dot(q0[3:] / np.linalg.norm(q0[3:]), q1[3:])))))
            dur = float(np.clip(max(np.linalg.norm(q1[:3] - q0[:3]) / 0.25, ang / np.radians(speed_deg)), 0.8, 3.0))
            self._homing[side] = dict(q0=q0, q1=q1, t=0.0, dur=dur)
            return
        dur = float(np.clip(np.max(np.abs(q1 - q0)) / np.radians(speed_deg), 0.8, 3.0))
        self._homing[side] = dict(q0=q0, q1=q1, t=0.0, dur=dur)

    def homing(self, side: str) -> bool:
        return self._homing[side] is not None

    def hold(self, d: mujoco.MjData, side: str, q):
        """Put an arm at joint angles q (physics and command) and anchor its teleop there (a game
        that starts with the arm somewhere else than home)."""
        q = np.asarray(q, float)
        d.qpos[self.info.arm_qpos_adr[side]] = q
        d.qvel[self.info.arm_dof_adr[side]] = 0
        self.q_cmd[side] = q.copy()
        pos, quat = self.ik[side].fk(q, d.qpos)
        self.teleop[side].reset(pos, quat)
        self._goal[side] = None
        self._smooth[side] = None

    def solve(self, d: mujoco.MjData, side: str, pos, quat, iters: int = 400):
        """Joint angles that put the gripper at (pos, quat), from the current command (the IK run
        to convergence); returns (q, position error [m])."""
        q = self.q_cmd[side].copy()
        for _ in range(iters):
            q = self.ik[side].step(q, np.asarray(pos, float), np.asarray(quat, float), 0.05, full_qpos=d.qpos)
        p, _ = self.ik[side].fk(q, d.qpos)
        return q, float(np.linalg.norm(p - pos))

    def reset(self, d: mujoco.MjData):
        for s in S.SIDES:
            self.q_cmd[s] = self.home[s].copy()
            pos, quat = self.ik[s].fk(self.q_cmd[s], d.qpos)
            self.teleop[s].reset(pos, quat)
            self._goal[s] = None
            self._smooth[s] = None
            self._homing[s] = None

    def _smoothed(self, s: str, pos, quat, dt: float):
        """Move toward (pos, quat) a fraction 1 - exp(-dt / smoothing) of the way per tick."""
        if not self.smoothing:
            return pos, quat
        if self._smooth[s] is None:
            self._smooth[s] = (np.array(pos, float), np.array(quat, float))
            return self._smooth[s]
        a = 1.0 - np.exp(-dt / self.smoothing)
        sp, sq = self._smooth[s]
        q1 = np.array(quat, float)
        if np.dot(sq, q1) < 0:
            q1 = -q1
        sp = sp + (np.asarray(pos, float) - sp) * a
        sq = sq * (1 - a) + q1 * a
        self._smooth[s] = (sp, sq / np.linalg.norm(sq))
        return self._smooth[s]

    def _limited(self, s: str, pos, quat, dt: float):
        """Move the IK goal toward (pos, quat) at most max_speed / max_turn per second."""
        if self.max_speed is None and self.max_turn is None:
            return pos, quat
        if self._goal[s] is None:
            self._goal[s] = (np.array(pos, float), np.array(quat, float))
            return self._goal[s]
        gp, gq = self._goal[s]
        step = np.asarray(pos) - gp
        n = np.linalg.norm(step)
        if self.max_speed is not None and n > self.max_speed * dt:
            step *= self.max_speed * dt / n
        gp = gp + step
        q1 = np.array(quat, float)
        if np.dot(gq, q1) < 0:
            q1 = -q1
        ang = 2 * np.arccos(min(1.0, abs(float(np.dot(gq, q1)))))
        if self.max_turn is not None and ang > self.max_turn * dt:
            a = self.max_turn * dt / ang                      # (n)lerp a fraction of the way
            q1 = gq * (1 - a) + q1 * a
        gq = q1 / np.linalg.norm(q1)
        self._goal[s] = (gp, gq)
        return gp, gq

    def control(self, d: mujoco.MjData, inputs: dict | None, dt: float):
        """One control tick. inputs: the headset message ({'left': {...}, 'right': {...}}), or
        None when it is stale (the arms disengage and hold still)."""
        for s in S.SIDES:
            tel = self.teleop[s]
            h = self._homing[s]
            if h is not None:                        # gliding home: a smooth joint-space move
                h["t"] += dt
                a = min(1.0, h["t"] / h["dur"])
                a = a * a * (3 - 2 * a)
                self.q_cmd[s] = h["q0"] + (h["q1"] - h["q0"]) * a
                tel.disengage()
                tel.target_pos, tel.target_quat = self.ik[s].fk(self.q_cmd[s])
                self._goal[s] = self._smooth[s] = None
                if h["t"] >= h["dur"]:
                    self._homing[s] = None
                self._command(d, s)
                d.ctrl[self.info.grip_act[s]] = 255.0 * tel.gripper
                continue
            if not tel.engaged:
                # keep the virtual target glued to the commanded pose while idle
                tel.target_pos, tel.target_quat = self.ik[s].fk(self.q_cmd[s])
            if inputs is not None and s not in self.frozen:
                c = inputs.get(s)
                if c is not None and inputs.get("t") is not None and "t" not in c:
                    c = dict(c, t=inputs["t"])          # (the sender's clock: for the glitch guard)
                tel.update(c, dt)
            else:
                tel.disengage()
            if tel.engaged:
                pos, quat = self._smoothed(s, tel.target_pos, tel.target_quat, dt)
                pos, quat = self._limited(s, pos, quat, dt)
                self.q_cmd[s] = self.ik[s].step(self.q_cmd[s], pos, quat, dt, full_qpos=d.qpos)
            else:
                self._goal[s] = None              # (starts again from wherever the arm is)
                self._smooth[s] = None
            self._command(d, s)
            d.ctrl[self.info.grip_act[s]] = 255.0 * tel.gripper

    def gripper(self, side: str) -> float:
        """Commanded closure, 0 open .. 1 closed."""
        return self.teleop[side].gripper

    def engaged(self, side: str) -> bool:
        return self.teleop[side].engaged

    def ee_pose(self, d: mujoco.MjData, side: str):
        sid = self.info.ee_site[side]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.site_xmat[sid])
        return d.site_xpos[sid].copy(), q
