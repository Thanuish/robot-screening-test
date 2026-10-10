"""Inverse kinematics on Pinocchio: every control tick, which small joint motion brings the
gripper closest to the target, within the robot's limits?

It is a small quadratic programme (ProxSuite), solved in well under a millisecond:

    minimise   |J dq - twist dt|^2 (position 1, rotation 0.3 per radian)
             + damping |dq|^2 + a weak pull toward the home posture
    such that  joint limits (with a margin), joint speed limits,
               the wrist and gripper stay table_clearance above the table

The arm's kinematics come from the same MuJoCo model file the physics uses; the tool point is
the gripper's pinch point, measured from the physics model once. check_against() compares the
two at random poses, so they can't silently disagree.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin
import proxsuite

from .config import TABLE_Z
from .scene import ARM, UR5E_XML, Scene

ROT_W = 0.3                    # rotation error weight (per radian) against position (per metre)
DAMP = 1e-4                    # damping on joint motion
POSTURE = 1e-3                 # weak pull toward the home posture (resolves the elbow's freedom)
LIMIT_MARGIN = np.radians(3)
KEEP_OUT = ["wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "tcp"]   # points kept above the table


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
        self.keep = [self.model.getFrameId(n) for n in KEEP_OUT]
        self.home = np.asarray(cfg.robot["home"], float)
        self.lo = self.model.lowerPositionLimit + LIMIT_MARGIN
        self.hi = self.model.upperPositionLimit - LIMIT_MARGIN
        self.qd_max = cfg.robot["joint_speed"]
        self.z_min = TABLE_Z + cfg.robot["table_clearance"]
        n = self.model.nv
        self.qp = proxsuite.proxqp.dense.QP(n, 0, len(self.keep), True)   # box constraints on dq
        self._init = False

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
    def step(self, q, target_pos, target_quat, dt):
        """The next joint command toward the target pose (world)."""
        model, data = self.model, self.data
        pin.computeJointJacobians(model, data, q)
        pin.framesForwardKinematics(model, data, q)
        Tcur = self.base * data.oMf[self.tcp]
        w, x, y, z = target_quat
        Ttgt = pin.SE3(pin.Quaternion(w, x, y, z).normalized().toRotationMatrix(), np.asarray(target_pos, float))
        # error in the world frame: position difference, rotation as an axis-angle vector
        e_p = Ttgt.translation - Tcur.translation
        e_r = pin.log3(Ttgt.rotation @ Tcur.rotation.T)
        J = pin.getFrameJacobian(model, data, self.tcp, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Rb = self.base.rotation
        J = np.vstack([Rb @ J[:3], Rb @ J[3:]])                      # into the world frame
        W = np.diag([1, 1, 1, ROT_W ** 2, ROT_W ** 2, ROT_W ** 2])
        err = np.r_[e_p, e_r]
        H = J.T @ W @ J + (DAMP + POSTURE) * np.eye(6)
        g = -J.T @ W @ err - POSTURE * (self.home - q) * 0.05
        # keep-out: z of each point + its z-velocity >= the table clearance
        C, lo = [], []
        for f in self.keep:
            Jf = pin.getFrameJacobian(model, data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3]
            zf = (self.base * data.oMf[f]).translation[2]
            C.append((Rb @ Jf)[2])
            lo.append(self.z_min - zf)
        C, lo = np.array(C), np.array(lo)
        step = self.qd_max * dt
        l_box = np.maximum(self.lo - q, -step)
        u_box = np.minimum(self.hi - q, step)
        l_box = np.minimum(l_box, u_box)
        if not self._init:
            self.qp.settings.eps_abs = 1e-7
            self.qp.settings.verbose = False
            self.qp.init(H, g, None, None, C, lo, np.full(len(lo), 1e9), l_box, u_box)
            self._init = True
        else:
            self.qp.update(H, g, None, None, C, lo, np.full(len(lo), 1e9), l_box, u_box)
        self.qp.solve()
        dq = self.qp.results.x
        if not np.all(np.isfinite(dq)):
            dq = np.zeros(6)
        return q + dq
