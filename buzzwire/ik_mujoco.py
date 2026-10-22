"""The same inverse kinematics as ik.py, without Pinocchio and ProxSuite (they have no Windows
packages): the arm's kinematics from MuJoCo (the physics model itself), the small quadratic
programme solved with daqp (through qpsolvers). Same formulation and settings (ik.py's constants),
same interface (fk, step, z_min, failed). ik.py uses this one only when Pinocchio is missing.
"""
from __future__ import annotations

import mujoco
import numpy as np
import qpsolvers

from .config import TABLE_Z
from .scene import ARM, GRIP, Scene

ROT_W = 1.0                    # (as in ik.py)
DAMP = 1e-4
POSTURE = 1e-5
POSTURE_RATE = 0.05
ITERS = 3
LIMIT_MARGIN = np.radians(3)
ARM_LINKS = ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")
NEAR = 0.15


class IK:
    def __init__(self, sc: Scene, data: mujoco.MjData):
        cfg = sc.cfg
        m = sc.model
        self.m, self.live = m, data
        self.md = mujoco.MjData(m)                 # (a copy of the physics: the arm at the joints being tried)
        self.qadr, self.vadr = np.asarray(sc.arm_qpos), np.asarray(sc.arm_dof)
        self.site = sc.ee_site
        jids = [int(np.where(m.jnt_qposadr == a)[0][0]) for a in self.qadr]
        self.lo = m.jnt_range[jids, 0] + LIMIT_MARGIN
        self.hi = m.jnt_range[jids, 1] - LIMIT_MARGIN
        self.home = np.asarray(cfg.robot["home"], float)
        self.qd_max = cfg.robot["joint_speed"]
        self.table = m.geom("table_top").id
        arm_bodies = {m.body(ARM + b).id for b in ARM_LINKS}
        grip_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(GRIP)}
        self.shapes = [(g, cfg.robot["arm_clearance"] if m.geom_bodyid[g] in arm_bodies else cfg.robot["finger_clearance"])
                       for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
                       if m.geom_bodyid[g] in arm_bodies | grip_bodies]
        self._jacp, self._jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._fromto = np.zeros(6)
        self.z_min = TABLE_Z + self._pinch_floor(data, sc.ee_site)
        self.failed = 0

    def _pinch_floor(self, data, ee_site) -> float:
        lowest = min(mujoco.mj_geomDistance(self.m, data, self.table, g, 1.0, self._fromto) - c
                     for g, c in self.shapes)
        return float(data.site_xpos[ee_site][2] - TABLE_Z - lowest)

    # ---------------------------------------------------------------- kinematics
    def _pose(self, q):
        md = self.md
        md.qpos[:] = self.live.qpos
        md.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, md)
        mujoco.mj_comPos(self.m, md)

    def fk(self, q):
        """(position, quaternion wxyz) of the pinch point in the world."""
        self._pose(q)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.md.site_xmat[self.site])
        return self.md.site_xpos[self.site].copy(), quat

    def check_against(self, sc: Scene, data: mujoco.MjData, n=50, seed=0) -> float:
        return 0.0                                  # (the same model as the physics)

    # ---------------------------------------------------------------- one control tick
    def _keep_out(self):
        """(after _pose) rows C, bounds lo: distance + C dq >= clearance for shapes near the table."""
        m, md = self.m, self.md
        C, lo = [], []
        for g, clear in self.shapes:
            dist = mujoco.mj_geomDistance(m, md, self.table, g, NEAR, self._fromto)
            if dist >= NEAR:
                continue
            pt, pg = self._fromto[:3], self._fromto[3:]
            nrm = pg - pt
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if dist > 1e-6 and ln > 1e-9 else np.array([0.0, 0.0, 1.0])
            mujoco.mj_jac(m, md, self._jacp, None, pg, m.geom_bodyid[g])
            C.append(nrm @ self._jacp[:, self.vadr])
            lo.append(min(clear - dist, 0.0))           # (see ik.py: "don't move" always qualifies)
        return C, lo

    def step(self, q, target_pos, target_quat, dt):
        W = np.diag([1, 1, 1, ROT_W ** 2, ROT_W ** 2, ROT_W ** 2])
        step = self.qd_max * dt
        q0 = np.clip(np.asarray(q, float), self.lo, self.hi)
        q = q0.copy()
        tq = np.asarray(target_quat, float)
        tq = tq / np.linalg.norm(tq)
        cur, neg, qe, rv = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        for _ in range(ITERS):
            self._pose(q)
            md = self.md
            mujoco.mju_mat2Quat(cur, md.site_xmat[self.site])
            mujoco.mju_negQuat(neg, cur)
            mujoco.mju_mulQuat(qe, tq, neg)
            if qe[0] < 0:
                qe = -qe
            mujoco.mju_quat2Vel(rv, qe, 1.0)
            err = np.r_[np.asarray(target_pos, float) - md.site_xpos[self.site], rv]
            if np.linalg.norm(err) < 1e-6:
                break
            mujoco.mj_jacSite(self.m, md, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, self.vadr], self._jacr[:, self.vadr]])
            H = J.T @ W @ J + (DAMP + POSTURE) * np.eye(len(q))
            g = -J.T @ W @ err - POSTURE * POSTURE_RATE * (self.home - q)
            C, lo = self._keep_out()
            used = q - q0
            l_box = np.maximum(self.lo - q, -step - used)
            u_box = np.minimum(self.hi - q, step - used)
            l_box = np.minimum(l_box, u_box)
            G = -np.array(C) if C else None             # C dq >= lo  <=>  -C dq <= -lo
            h = -np.array(lo) if C else None
            dq = qpsolvers.solve_qp(H, g, G, h, lb=l_box, ub=u_box, solver="daqp")
            if dq is None or not np.all(np.isfinite(dq)):
                self.failed += 1
                break
            q = np.clip(q + dq, self.lo, self.hi)
        return q
