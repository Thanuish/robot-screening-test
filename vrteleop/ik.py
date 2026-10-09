"""Damped-least-squares differential IK for one UR5e arm.

The IK runs on its own MjData ("kinematic shadow"), so the commanded joint
positions evolve smoothly toward the target and the physics arm tracks them
with its position servos. If an arm is blocked by contact, the command keeps
pointing at the target and the servos push with bounded force (forcerange).
"""
from __future__ import annotations

import mujoco
import numpy as np


class DiffIK:
    def __init__(self, model: mujoco.MjModel, qpos_adr, dof_adr, site_id: int,
                 q_home: np.ndarray, damping: float = 0.03, pos_gain: float = 1.0,
                 rot_gain: float = 0.8, nullspace_gain: float = 0.3,
                 max_joint_vel: float = 3.0, iters: int = 3):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.asarray(qpos_adr)
        self.vadr = np.asarray(dof_adr)
        self.site = site_id
        self.q_home = np.asarray(q_home, float)
        self.damping = damping
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.ns_gain = nullspace_gain
        self.max_vel = max_joint_vel
        self.iters = iters
        jids = [np.where(model.jnt_qposadr == a)[0][0] for a in self.qadr]
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray, full_qpos: np.ndarray | None = None):
        """Return (pos, quat wxyz) of the EE site for arm joints q."""
        if full_qpos is not None:
            self.d.qpos[:] = full_qpos
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        pos = self.d.site_xpos[self.site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.d.site_xmat[self.site])
        return pos, quat

    def step(self, q_cmd: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray,
             dt: float) -> np.ndarray:
        """Advance q_cmd one control period toward the target pose."""
        q = q_cmd.copy()
        max_step = self.max_vel * dt
        dq_total = np.zeros(6)
        for _ in range(self.iters):
            self.d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            pos = self.d.site_xpos[self.site]
            cur_q = np.zeros(4)
            mujoco.mju_mat2Quat(cur_q, self.d.site_xmat[self.site])
            # 6D error in world frame
            err = np.zeros(6)
            err[:3] = self.pos_gain * (target_pos - pos)
            neg = np.zeros(4)
            mujoco.mju_negQuat(neg, cur_q)
            qerr = np.zeros(4)
            mujoco.mju_mulQuat(qerr, target_quat, neg)
            if qerr[0] < 0:
                qerr = -qerr
            rv = np.zeros(3)
            mujoco.mju_quat2Vel(rv, qerr, 1.0)
            err[3:] = self.rot_gain * rv
            mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, self.vadr], self._jacr[:, self.vadr]])
            JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            dq = Jpinv @ err
            # Null-space posture bias toward home (keeps elbow up, avoids wrist flips)
            N = np.eye(6) - Jpinv @ J
            dq += N @ (self.ns_gain * (self.q_home - q))
            # Rate limit the total per-period motion
            remaining = max_step - np.abs(dq_total)
            dq = np.clip(dq, -np.maximum(remaining, 0), np.maximum(remaining, 0))
            dq_total += dq
            q = np.clip(q + dq, self.lo, self.hi)
            if np.linalg.norm(err) < 1e-4:
                break
        return q
