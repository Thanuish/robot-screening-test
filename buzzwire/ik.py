"""Inverse kinematics on Pinocchio: every control tick, which small joint motion brings the
gripper closest to the target, within the robot's limits?

It is a small quadratic programme (ProxSuite), solved in well under a millisecond:

    minimise   |J dq - error|^2 (position per metre and rotation per radian weigh the same)
             + damping |dq|^2 + a very weak pull toward the home posture (it must not fight
               the tracking; it only settles the elbow)
    such that  joint limits (with a margin), joint speed limits,
               the forearm and wrist stay arm_clearance (2 cm) above the table, every part of
               the gripper finger_clearance (3 mm): distances of the real collision shapes to the
               table top (MuJoCo's mj_geomDistance), so the fingers can go down to grab

Three iterations per control tick (re-linearised each time, sharing that tick's joint-speed
budget), as vrteleop/ik.py does: a large turn is followed closely, not just its first-order step.

The arm's kinematics come from the same MuJoCo model file the physics uses; the tool point is
the gripper's pinch point, measured from the physics model once. check_against() compares the
two at random poses, so they can't silently disagree.
"""
from __future__ import annotations

import mujoco
import numpy as np
try:
    import pinocchio as pin
    import proxsuite
except ImportError:            # (no Windows packages: ik_mujoco.py, the same IK on MuJoCo + daqp)
    pin = proxsuite = None

from .config import TABLE_Z
from .scene import ARM, GRIP, UR5E_XML, Scene

ROT_W = 1.0                    # rotation error weight (per radian) against position (per metre)
DAMP = 1e-4                    # damping on joint motion
POSTURE = 1e-5                 # very weak pull toward the home posture (only settles the elbow)
POSTURE_RATE = 0.05            # (a fraction of the way home per tick, as the posture's target)
ITERS = 3                      # IK iterations per control tick
MAX_ITER, MAX_ITER_IN = 200, 100   # the solver's iteration caps (it needs a few dozen)
LIMIT_MARGIN = np.radians(3)
ARM_LINKS = ("forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")   # keep arm_clearance
NEAR = 0.15                    # [m] shapes farther than this from the table are not checked


class IK:
    def __init__(self, sc: Scene, data: mujoco.MjData):
        cfg = sc.cfg
        self.model = pin.buildModelFromMJCF(str(UR5E_XML))
        m = sc.model
        # world <- Pinocchio's world, measured: the arm's base link in the physics model against
        # the same link in Pinocchio (the scene places and turns the arm)
        self.data = self.model.createData()
        pin.framesForwardKinematics(self.model, self.data, np.zeros(self.model.nq))
        bid = m.body(ARM + "base").id
        Tb = pin.SE3(data.xmat[bid].reshape(3, 3).copy(), data.xpos[bid].copy())
        self.base = Tb * self.data.oMf[self.model.getFrameId("base")].inverse()
        # the tool point: the pinch site, relative to the wrist_3 link (from the physics model)
        w3 = m.body(ARM + "wrist_3_link").id
        Tw3 = pin.SE3(data.xmat[w3].reshape(3, 3).copy(), data.xpos[w3].copy())
        Tp = pin.SE3(data.site_xmat[sc.ee_site].reshape(3, 3).copy(), data.site_xpos[sc.ee_site].copy())
        f_w3 = self.model.getFrameId("wrist_3_link")
        fr = self.model.frames[f_w3]
        self.tcp = self.model.addFrame(pin.Frame("tcp", fr.parentJoint, fr.parentFrame,
                                                 fr.placement * (Tw3.inverse() * Tp), pin.FrameType.OP_FRAME))
        self.data = self.model.createData()
        self.home = np.asarray(cfg.robot["home"], float)
        self.lo = self.model.lowerPositionLimit + LIMIT_MARGIN
        self.hi = self.model.upperPositionLimit - LIMIT_MARGIN
        self.qd_max = cfg.robot["joint_speed"]
        # keep-out: the arm's and the gripper's collision shapes against the table top, measured on a
        # copy of the physics (the fingers as they are now, the arm at the joints being tried)
        self.m, self.live = m, data
        self.md = mujoco.MjData(m)
        self.qadr, self.vadr = sc.arm_qpos, sc.arm_dof
        self.table = m.geom("table_top").id
        arm_bodies = {m.body(ARM + b).id for b in ARM_LINKS}
        grip_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(GRIP)}
        self.shapes = [(g, cfg.robot["arm_clearance"] if m.geom_bodyid[g] in arm_bodies else cfg.robot["finger_clearance"])
                       for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]
                       if m.geom_bodyid[g] in arm_bodies | grip_bodies]
        self._jacp = np.zeros((3, m.nv))
        self._fromto = np.zeros(6)
        # the lowest the pinch point can go with the gripper pointing down (for the teleop's target)
        self.z_min = TABLE_Z + self._pinch_floor(data, sc.ee_site)
        n = self.model.nv
        self.qp = proxsuite.proxqp.dense.QP(n, 0, len(self.shapes), True)   # box constraints on dq
        self.failed = 0                 # solves that didn't finish (should stay 0)

    def _pinch_floor(self, data, ee_site) -> float:
        """How high above the table the pinch point is when the gripper's lowest part touches
        finger_clearance (measured at the start pose: the gripper pointing down)."""
        m = self.m
        lowest = min(mujoco.mj_geomDistance(m, data, self.table, g, 1.0, self._fromto) - c
                     for g, c in self.shapes)
        return float(data.site_xpos[ee_site][2] - TABLE_Z - lowest)

    # ---------------------------------------------------------------- kinematics
    def fk(self, q):
        """(position, quaternion wxyz) of the pinch point in the world."""
        pin.framesForwardKinematics(self.model, self.data, q)
        T = self.base * self.data.oMf[self.tcp]
        x, y, z, w = pin.Quaternion(T.rotation).coeffs()
        return T.translation.copy(), np.array([w, x, y, z])

    def check_against(self, sc: Scene, data: mujoco.MjData, n=50, seed=0) -> float:
        """Largest disagreement [m] between Pinocchio and MuJoCo over n random arm poses."""
        rng = np.random.default_rng(seed)
        d = mujoco.MjData(sc.model)
        worst = 0.0
        for _ in range(n):
            q = rng.uniform(self.lo, self.hi)
            d.qpos[sc.arm_qpos] = q
            mujoco.mj_kinematics(sc.model, d)
            p, _ = self.fk(q)
            worst = max(worst, float(np.linalg.norm(p - d.site_xpos[sc.ee_site])))
        return worst

    # ---------------------------------------------------------------- one control tick
    def _keep_out(self, q):
        """Rows C, lower bounds lo: for each shape near the table, distance + C dq >= clearance."""
        m, md = self.m, self.md
        md.qpos[:] = self.live.qpos
        md.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, md)
        mujoco.mj_comPos(m, md)
        n = len(self.qadr)
        C, lo = np.zeros((len(self.shapes), n)), np.full(len(self.shapes), -1e9)
        for i, (g, clear) in enumerate(self.shapes):
            dist = mujoco.mj_geomDistance(m, md, self.table, g, NEAR, self._fromto)
            if dist >= NEAR:
                continue                                    # (far above: no constraint)
            pt, pg = self._fromto[:3], self._fromto[3:]
            nrm = pg - pt
            ln = np.linalg.norm(nrm)
            nrm = nrm / ln if dist > 1e-6 and ln > 1e-9 else np.array([0.0, 0.0, 1.0])
            mujoco.mj_jac(m, md, self._jacp, None, pg, m.geom_bodyid[g])
            C[i] = nrm @ self._jacp[:, self.vadr]
            # a shape already closer than its clearance (the physics pushed it, e.g. pressed onto a
            # cube) may not go deeper; the target then lifts it out. So "don't move" always meets
            # every constraint: the problem always has a solution (with "get back out within this
            # tick" it may have none, and the solver would search for seconds)
            lo[i] = min(clear - dist, 0.0)
        return C, lo

    def step(self, q, target_pos, target_quat, dt):
        """The next joint command toward the target pose (world): ITERS re-linearised steps that
        share this tick's joint-speed budget."""
        model, data = self.model, self.data
        w, x, y, z = target_quat
        Ttgt = pin.SE3(pin.Quaternion(w, x, y, z).normalized().toRotationMatrix(), np.asarray(target_pos, float))
        W = np.diag([1, 1, 1, ROT_W ** 2, ROT_W ** 2, ROT_W ** 2])
        Rb = self.base.rotation
        step = self.qd_max * dt
        q0 = np.clip(np.asarray(q, float), self.lo, self.hi)
        q = q0.copy()
        for _ in range(ITERS):
            pin.computeJointJacobians(model, data, q)
            pin.framesForwardKinematics(model, data, q)
            Tcur = self.base * data.oMf[self.tcp]
            # error in the world frame: position difference, rotation as an axis-angle vector
            err = np.r_[Ttgt.translation - Tcur.translation, pin.log3(Ttgt.rotation @ Tcur.rotation.T)]
            if np.linalg.norm(err) < 1e-6:
                break
            J = pin.getFrameJacobian(model, data, self.tcp, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J = np.vstack([Rb @ J[:3], Rb @ J[3:]])                      # into the world frame
            H = J.T @ W @ J + (DAMP + POSTURE) * np.eye(6)
            g = -J.T @ W @ err - POSTURE * POSTURE_RATE * (self.home - q)
            C, lo = self._keep_out(q)
            used = q - q0                                      # (this tick's joint motion so far)
            l_box = np.maximum(self.lo - q, -step - used)
            u_box = np.minimum(self.hi - q, step - used)
            l_box = np.minimum(l_box, u_box)
            hi = np.full(len(lo), 1e9)
            # a fresh solve every time (~0.05 ms for 6 joints): warm-starting from the previous
            # answer let one bad solve spoil the next ones (they then ran to the iteration cap)
            self.qp = proxsuite.proxqp.dense.QP(len(q), 0, len(lo), True)
            self.qp.settings.eps_abs = 1e-7
            self.qp.settings.verbose = False
            # a hard cap: it takes ~10 iterations; never let one control tick run long (the
            # default allows 10,000 x 1,500: seconds to minutes)
            self.qp.settings.max_iter = MAX_ITER
            self.qp.settings.max_iter_in = MAX_ITER_IN
            self.qp.init(H, g, None, None, C, lo, hi, l_box, u_box)
            self.qp.solve()
            dq = self.qp.results.x
            if self.qp.results.info.status != proxsuite.proxqp.PROXQP_SOLVED or not np.all(np.isfinite(dq)):
                self.failed += 1                               # (counted; the arm holds for the rest of this tick)
                break
            q = np.clip(q + dq, self.lo, self.hi)            # (exactly within the limits: the next box holds 0)
        return q


if pin is None:                                            # noqa: E305
    from .ik_mujoco import IK  # noqa: F401,F811,E402
