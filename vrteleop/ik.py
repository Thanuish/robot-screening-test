"""Differential IK for one UR5e arm: damped least squares (DiffIK) and a QP (QPIK).

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
             dt: float, full_qpos: np.ndarray | None = None) -> np.ndarray:
        """Advance q_cmd one control period toward the target pose (full_qpos: unused here)."""
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


class QPIK(DiffIK):
    """Differential IK as a small quadratic program per iteration (the method of mink / Pink):

        min  |J dq - e|_W^2 + lambda^2 |dq|^2 + mu^2 |dq - k (q_home - q)|^2
        s.t. joint limits and joint speeds (exact bounds, no clipping afterwards),
             each arm shape stays at least its clearance above the table (velocity damper on
             the real shape-to-table distance, mj_geomDistance).

    Compared with DiffIK: the arm slides along the table instead of commanding into it (no
    wedged wrist), a limit stops only the joint at it (the others keep tracking), and the damping
    grows only near a singular pose (precise everywhere else, no jumps when stretched out).
    Same interface as DiffIK (fk, step)."""

    def __init__(self, model: mujoco.MjModel, qpos_adr, dof_adr, site_id: int, q_home: np.ndarray,
                 prefix: str, table_geom: str = "table_top", rot_weight: float = 1.0,
                 clearance: dict | None = None, detect: float = 0.08, damper_gain: float = 0.5,
                 damping: float = 0.002, singular_damping: float = 0.05, manip_threshold: float = 0.01,
                 posture_weight: float = 0.01, limit_margin: float = 0.02, **kw):
        super().__init__(model, qpos_adr, dof_adr, site_id, q_home, damping=damping, **kw)
        import qpsolvers                                       # (only needed for this IK)
        self._solve = qpsolvers.solve_qp
        self.W = np.diag([1.0] * 3 + [rot_weight ** 2] * 3)     # 1 rad of turning ~ rot_weight m
        self.lam2_0, self.lam2_s, self.w0 = damping ** 2, singular_damping ** 2, manip_threshold
        self.mu2 = posture_weight ** 2
        self.margin = limit_margin
        self.detect, self.xi = detect, damper_gain
        clearance = clearance or {"pad": 0.003, "gripper": 0.01, "link": 0.02}
        self.table = model.geom(table_geom).id
        # the arm's collision shapes and how far each keeps from the table; the shoulder (at the
        # base, on the table) and anything already touching the table at home are left out
        d = self.d
        d.qpos[self.qadr] = self.q_home
        mujoco.mj_kinematics(model, d)
        self.shapes = []
        fromto = np.zeros(6)
        for g in range(model.ngeom):
            name = model.body(model.geom_bodyid[g]).name
            if not name.startswith(prefix) or "shoulder" in name:
                continue
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                continue
            gn = model.geom(g).name
            kind = ("pad" if any(k in name or k in gn for k in ("pad", "follower", "coupler", "driver", "spring"))
                    else "gripper" if "gripper" in name else "link")
            if mujoco.mj_geomDistance(model, d, g, self.table, 0.05, fromto) < 0.05:
                continue
            self.shapes.append((g, model.geom_bodyid[g], clearance[kind]))
        self._jp = np.zeros((3, model.nv))
        self.failures = 0

    def step(self, q_cmd: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray,
             dt: float, full_qpos: np.ndarray | None = None) -> np.ndarray:
        """Advance q_cmd one control period toward the target pose. full_qpos: the simulation's
        state (the gripper's opening matters for the fingers' distance to the table)."""
        m, d = self.m, self.d
        if full_qpos is not None:
            d.qpos[:] = full_qpos
        q = q_cmd.copy()
        budget = np.full(6, self.max_vel * dt)                 # joint motion left this period
        fromto = np.zeros(6)
        for _ in range(self.iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.zeros(6)
            err[:3] = self.pos_gain * (target_pos - d.site_xpos[self.site])
            cur, neg, qerr, rv = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(cur, d.site_xmat[self.site])
            mujoco.mju_negQuat(neg, cur)
            mujoco.mju_mulQuat(qerr, target_quat, neg)
            if qerr[0] < 0:
                qerr = -qerr
            mujoco.mju_quat2Vel(rv, qerr, 1.0)
            err[3:] = self.rot_gain * rv
            if np.linalg.norm(err[:3]) < 1e-5 and np.linalg.norm(err[3:]) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, self.vadr], self._jacr[:, self.vadr]])
            # damping only near a singular pose (manipulability below the threshold)
            w = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
            lam2 = self.lam2_0 + (self.lam2_s * (1 - w / self.w0) ** 2 if w < self.w0 else 0.0)
            post = self.ns_gain * (self.q_home - q)
            H = J.T @ self.W @ J + (lam2 + self.mu2) * np.eye(6)
            f = -(J.T @ self.W @ err + self.mu2 * post)
            lo = np.maximum(-budget, self.lo + self.margin - q)
            hi = np.minimum(budget, self.hi - self.margin - q)
            lo = np.minimum(lo, 0.0)                           # (already past a margin: may stay)
            hi = np.maximum(hi, 0.0)
            # table: n . (Jp dq) >= -xi (dist - clearance), for shapes within `detect`
            G, h = [], []
            for g, body, clear in self.shapes:
                dist = mujoco.mj_geomDistance(m, d, g, self.table, self.detect, fromto)
                if dist >= self.detect:
                    continue
                p, pt = fromto[:3], fromto[3:]
                n = p - pt
                nn = np.linalg.norm(n)
                n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
                mujoco.mj_jac(m, d, self._jp, None, p, body)
                G.append(-(n @ self._jp[:, self.vadr]))
                h.append(max(self.xi * (dist - clear), -0.002))
            dq = None
            if G:
                dq = self._solve(H, f, np.array(G), np.array(h), lb=lo, ub=hi, solver="daqp")
            if dq is None:                                     # (no shape near / infeasible)
                if G:
                    self.failures += 1
                dq = self._solve(H, f, lb=lo, ub=hi, solver="daqp")
            if dq is None:
                break
            budget = np.maximum(budget - np.abs(dq), 0.0)
            q = np.clip(q + dq, self.lo, self.hi)
        return q
