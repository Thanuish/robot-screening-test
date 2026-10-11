"""The buzz-wire scene as a MuJoCo model, built from the configuration:

    room        floor and a lab table (top at z = 0.75)
    robot       one UR5e with a Robotiq 2F-85 (MuJoCo Menagerie), on the table's back edge
    board       a wooden base with two posts holding the wire's ends
    wire        a chain of thin static capsules along the level's curve; a rubber sleeve on each
                end (touching a sleeve doesn't count)
    forks       a rubber fork at each end that holds the wand's handle (the start rest and the
                end "hook")
    wand        a free body: a handle (the gripper holds it) and a loop of short capsules around
                the wire. Its frame: origin at the loop's centre, x along the handle (pointing
                away from it: the handle is at negative x), y the loop's axis (along the wire)

Touches are the contacts between the wand and the wire's capsules (see game.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .config import TABLE_Z, Config, Level

ROOT = Path(__file__).resolve().parents[1]
UR5E_XML = ROOT / "assets" / "universal_robots_ur5e" / "ur5e.xml"
GRIPPER_XML = ROOT / "assets" / "robotiq_2f85" / "2f85.xml"
ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
              "wrist_3_joint"]
ARM_ACTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
ARM = "ur_"                     # name prefixes in the model
GRIP = "ur_gripper_"
CAPSULE_STEP = 0.01             # length of one wire capsule [m]
RUBBER = [0.12, 0.12, 0.13, 1]
# the wire and the ring are hard metal: stiff contacts (time constant 2 physics steps, almost no
# give), so a strong pull can't press the wire through the ring
HARD_SOLREF = (0.004, 1.0)
HARD_SOLIMP = (0.99, 0.999, 0.0005, 0.5, 2)


@dataclass
class Scene:
    model: mujoco.MjModel
    cfg: Config
    level: object                        # the game's level (buzz wire: config.Level)
    arm_qpos: np.ndarray = None          # qpos addresses of the 6 arm joints
    arm_dof: np.ndarray = None
    arm_act: np.ndarray = None
    grip_act: int = -1
    ee_site: int = -1                    # the gripper's pinch point
    wand_qpos: int = -1                  # free joint (7)
    wand_dof: int = -1
    wand_body: int = -1
    wand_geoms: set = field(default_factory=set)
    wire_geoms: set = field(default_factory=set)
    robot_geoms: set = field(default_factory=set)
    rest: dict = field(default_factory=dict)   # "start" / "end" -> (pos, quat) of the wand at rest
    weld: int = -1                       # the equality that fixes the wand in the hand


def _box(body, name, pos, half, rgba=(0.5, 0.5, 0.5, 1), material=None, collide=True, quat=None):
    g = body.add_geom()
    g.name, g.type, g.pos, g.size, g.rgba = name, mujoco.mjtGeom.mjGEOM_BOX, list(pos), list(half), list(rgba)
    if quat is not None:
        g.quat = list(quat)
    if material:
        g.material = material
    if not collide:
        g.contype = g.conaffinity = 0
    return g


def _capsule(body, name, a, b, r, rgba, material=None):
    g = body.add_geom()
    g.name, g.type = name, mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = [*a, *b]
    g.size = [r, 0, 0]
    g.rgba = list(rgba)
    if material:
        g.material = material
    return g


def _mat_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def wand_rest(cfg: Config, level: Level, end: str):
    """(position, quaternion) of the wand resting on a fork: its loop around the sleeve at that
    end (hanging on it: the loop's inside on the sleeve's top), its handle level, pointing back
    toward the robot."""
    w, R = level.wire, level.loop_radius
    s = cfg.wire["sleeve_length"] * 0.5 if end == "start" else w.length - cfg.wire["sleeve_length"] * 0.5
    p, t = w.at(s)
    th = np.array([t[0], t[1], 0.0])
    th /= np.linalg.norm(th)
    d = np.array([-th[1], th[0], 0.0])                  # horizontal, across the wire
    if d[0] > 0:
        d = -d                                          # the handle points back, toward the robot
    x = -d                                              # (the handle lies along the wand's -x)
    y = th
    z = np.cross(x, y)
    centre = p + np.array([0, 0, cfg.wire["sleeve_radius"] - R + cfg.wand["loop_thickness"] / 2])
    return centre, _mat_quat(np.stack([x, y, z], axis=1))


def wand_frame(level: Level, s: float, full: bool):
    """(centre, rotation) of the wand ideally placed at arc length s: its loop centred on the wire
    and square to it, the handle level, pointing back toward the robot. full = False: the loop
    square to the wire's direction seen from above (the gripper only turns about the vertical)."""
    p, t = level.wire.at(s)
    th = np.array([t[0], t[1], 0.0])
    th /= np.linalg.norm(th)
    x = np.array([th[1], -th[0], 0.0])
    if x[0] < 0:
        x = -x
    y = t if full else th
    x = x - np.dot(x, y) * y
    x /= np.linalg.norm(x)
    return p, np.stack([x, y, np.cross(x, y)], axis=1)


GRIP_IN_WAND = np.diag([1.0, -1.0, -1.0])   # the gripper's orientation relative to the wand when holding it


def handle_centre_local(cfg: Config, level: Level):
    """The middle of the handle in the wand's frame (where the gripper should hold it)."""
    L = cfg.wand["handle_size"][0]
    return np.array([-(level.loop_radius + L / 2), 0.0, 0.0])


def new_spec(cfg: Config, name: str) -> mujoco.MjSpec:
    """A scene with what every game shares: the physics options, the room's floor and table, and
    the robot (UR5e + 2F-85) on the table's back edge. A game adds its own things to it."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    spec.option.timestep = cfg.physics["timestep"]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    wb = spec.worldbody
    m = spec.add_material()
    m.name, m.rgba = "wood", [0.62, 0.45, 0.28, 1]
    m = spec.add_material()
    m.name, m.rgba, m.specular, m.shininess, m.reflectance = "metal", [0.85, 0.86, 0.88, 1], 1.0, 1.0, 0.3
    m = spec.add_material()
    m.name, m.rgba, m.specular, m.shininess = "brass", [0.86, 0.68, 0.32, 1], 1.0, 0.9

    # room: floor, table, light
    fl = wb.add_geom()
    fl.name, fl.type, fl.size, fl.rgba = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [3, 3, 0.05], [0.5, 0.5, 0.5, 1]
    table = wb.add_body()
    table.name = "table"
    table.pos = [0.4, 0.0, TABLE_Z]
    _box(table, "table_top", [0, 0, -0.02], [0.55, 0.75, 0.02], [0.9, 0.9, 0.88, 1])
    light = wb.add_light()
    light.pos, light.dir, light.castshadow = [0.4, 0.2, 2.5], [0, 0, -1], True
    light.diffuse = [0.8, 0.8, 0.8]

    # robot: UR5e + 2F-85, base on the table, turned 180 deg so it reaches toward +x
    arm = mujoco.MjSpec.from_file(str(UR5E_XML))
    grip = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    for sp in (arm, grip):
        for li in list(sp.lights):
            sp.delete(li)
        for k in list(sp.keys):
            sp.delete(k)
        sp.option.cone, sp.option.impratio, sp.option.integrator = spec.option.cone, spec.option.impratio, spec.option.integrator
    arm.site("attachment_site").attach_body(grip.body("base_mount"), "gripper_", "")
    frame = wb.add_frame()
    frame.pos = [*cfg.robot["base"], TABLE_Z]
    frame.quat = [0, 0, 0, 1]
    spec.attach(arm, prefix=ARM, frame=frame)
    for b in spec.bodies:
        if b.name.startswith(ARM):
            b.gravcomp = 1.0                    # position servos track precisely
    # joint motors: Menagerie damping (kv = kp/5) halved, then k x stiffer (sqrt(k) x the damping:
    # still well damped) so the arm follows its target quicker ([robot] servo_stiffness)
    k = cfg.robot["servo_stiffness"]
    for a in spec.actuators:
        if any(a.name == ARM + n for n in ARM_ACTS):
            gp, bp = np.array(a.gainprm), np.array(a.biasprm)
            gp[0] *= k
            bp[1] *= k
            bp[2] *= 0.5 * np.sqrt(k)
            a.gainprm, a.biasprm = gp, bp
            fr = np.array(a.forcerange)
            if fr[1] > fr[0]:
                a.forcerange = fr * k
    return spec


def bind_robot(sc: Scene):
    """Fill in the robot's addresses in the compiled model (joints, motors, the pinch site, its geoms)."""
    model = sc.model
    jids = [model.joint(ARM + j).id for j in ARM_JOINTS]
    sc.arm_qpos = np.array([model.jnt_qposadr[i] for i in jids])
    sc.arm_dof = np.array([model.jnt_dofadr[i] for i in jids])
    sc.arm_act = np.array([model.actuator(ARM + a).id for a in ARM_ACTS])
    sc.grip_act = model.actuator(GRIP + "fingers_actuator").id
    sc.ee_site = model.site(GRIP + "pinch").id
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.startswith(ARM):
            sc.robot_geoms.add(g)
    return sc


def build(cfg: Config, level: Level) -> Scene:
    """The buzz-wire scene for one level."""
    spec = new_spec(cfg, f"buzzwire_{level.key}")
    wb = spec.worldbody

    # board and posts
    bx, by = cfg.board["center"]
    bs = np.array(cfg.board["size"]) / 2
    board = wb.add_body()
    board.name = "board"
    board.pos = [bx, by, TABLE_Z + bs[2]]
    _box(board, "board_top", [0, 0, 0], bs, material="wood")
    w = level.wire
    top = TABLE_Z + 2 * bs[2]
    for k, p in (("start", w.pts[0]), ("end", w.pts[-1])):
        _capsule(wb, f"post_{k}", [p[0], p[1], top], p, cfg.wire["post_radius"], [0, 0, 0, 1], "brass")

    # the wire: capsules between samples every CAPSULE_STEP, rubber sleeves on both ends
    r, sl = cfg.wire["radius"], cfg.wire["sleeve_length"]
    n = max(2, int(round(w.length / CAPSULE_STEP)))
    s_knots = np.linspace(0, w.length, n + 1)
    pts = [w.at(s)[0] for s in s_knots]
    for k in range(n):
        a, b, mid = pts[k], pts[k + 1], 0.5 * (s_knots[k] + s_knots[k + 1])
        sleeve = mid < sl or mid > w.length - sl
        g = _capsule(wb, f"{'sleeve' if sleeve else 'wire'}_{k}", a, b, cfg.wire["sleeve_radius"] if sleeve else r,
                     RUBBER if sleeve else [0, 0, 0, 1], None if sleeve else "metal")
        g.friction = [1.0, 0.01, 0.001] if sleeve else [0.3, 0.005, 0.0001]
        g.solref, g.solimp = list(HARD_SOLREF), list(HARD_SOLIMP)

    # rest forks (start rest and end hook): a crossbar under the handle's back end, a prong each side
    hw, hh = cfg.wand["handle_size"][1] / 2, cfg.wand["handle_size"][2] / 2
    rest = {}
    for end in ("start", "end"):
        c, q = wand_rest(cfg, level, end)
        Rw = np.zeros(9)
        mujoco.mju_quat2Mat(Rw, q)
        Rw = Rw.reshape(3, 3)
        hx = Rw[:, 0]                                   # along the handle (toward the loop)
        hy = Rw[:, 1]
        under = c + hx * -(level.loop_radius + cfg.wand["handle_size"][0] * 0.85) - np.array([0, 0, hh])
        fork = wb.add_body()
        fork.name = f"fork_{end}"
        h = under[2] - top                              # crossbar top at the handle's bottom
        _capsule(fork, f"fork_{end}_stem", [under[0], under[1], top], under - [0, 0, 0.004], 0.004, RUBBER)
        _capsule(fork, f"fork_{end}_bar", under - hy * (hw + 0.006) - [0, 0, 0.004],
                 under + hy * (hw + 0.006) - [0, 0, 0.004], 0.004, RUBBER)
        for sgn in (-1, 1):
            base = under + sgn * hy * (hw + 0.006)
            _capsule(fork, f"fork_{end}_prong{'ab'[sgn > 0]}", base - [0, 0, 0.004], base + [0, 0, hh * 1.6], 0.003, RUBBER)
        assert h > 0.01, "the fork would be below the board"
        rest[end] = (c, q)

    # the wand
    wd = cfg.wand
    R, t, N = level.loop_radius, wd["loop_thickness"], wd["loop_segments"]
    wand = wb.add_body()
    wand.name = "wand"
    wand.pos, wand.quat = list(rest["start"][0]), list(rest["start"][1])
    j = wand.add_freejoint()
    j.name = "wand_joint"
    L = wd["handle_size"][0]
    hg = _box(wand, "wand_handle", [-(R + L / 2), 0, 0], [L / 2, hw, hh], [0.08, 0.08, 0.09, 1])
    hg.mass, hg.friction, hg.condim = wd["handle_mass"], wd["friction"], 4
    for k in range(N):
        a0, a1 = 2 * np.pi * k / N, 2 * np.pi * (k + 1) / N
        g = _capsule(wand, f"wand_loop_{k}", [R * np.cos(a0), 0, R * np.sin(a0)], [R * np.cos(a1), 0, R * np.sin(a1)],
                     t / 2, [0, 0, 0, 1], "brass")
        g.mass, g.condim = wd["loop_mass"] / N, 3
        g.solref, g.solimp = list(HARD_SOLREF), list(HARD_SOLIMP)

    # the wand fixed in the hand (levels that start with it there): a weld, switched on by the game
    eq = spec.add_equality()
    eq.name, eq.type, eq.objtype = "wand_in_hand", mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = "wand", GRIP + "base"
    eq.active = False
    eq.solref, eq.solimp = list(HARD_SOLREF), list(HARD_SOLIMP)
    if level.start_held:
        # the weld holds the wand: the fingers' contacts with the handle would fight it (the wand
        # wobbled in the hand), so the wand and the gripper don't collide in these levels
        for b in spec.bodies:
            if b.name.startswith(GRIP):
                ex = spec.add_exclude()
                ex.bodyname1, ex.bodyname2 = "wand", b.name

    model = spec.compile()
    sc = Scene(model=model, cfg=cfg, level=level, rest=rest)
    sc.weld = model.equality("wand_in_hand").id
    bind_robot(sc)
    jw = model.joint("wand_joint").id
    sc.wand_qpos, sc.wand_dof, sc.wand_body = model.jnt_qposadr[jw], model.jnt_dofadr[jw], model.body("wand").id
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == sc.wand_body:
            sc.wand_geoms.add(g)
        elif model.geom(g).name.startswith("wire_"):
            sc.wire_geoms.add(g)
    return sc
