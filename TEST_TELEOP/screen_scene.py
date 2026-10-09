"""The screening test's scene: one arm (the right one) and three phases of objects on the
table, added to the dual-UR5e scene of vrteleop as the task "screen" (a scene plugin, see
vrteleop.scene.register_task).

The test arm is the right arm of the pair (the other one is hidden); everything is laid out in
front of the robot's head, within the arm's reach. Objects of the phases not
being played are parked on the floor behind the operator; markers and bins are mocap bodies
parked under the floor.
"""
from __future__ import annotations

import mujoco
import numpy as np

from vrteleop import scene as S
from vrteleop import textures as T
from vrteleop.scene import TABLE_Z

TASK = "screen"
ARM = "right"
# rigid objects: name -> (shape, half size, rgba, start xy, start yaw [rad])
OBJECTS = {
    # practice: pick this up and put it in the green square
    "tut_cube":     ("box", [0.02, 0.02, 0.02], [0.95, 0.85, 0.2, 1], (0.42, 0.02), 0.0),
    # task 1, shapes: each into its outline on the board (turn it to match)
    "big_square":   ("box", [0.03, 0.03, 0.025], [0.95, 0.55, 0.15, 1], (0.40, 0.13), 0.35),
    "small_square": ("box", [0.0175, 0.0175, 0.025], [0.6, 0.35, 0.85, 1], (0.43, 0.0), -0.35),
    "bar":          ("box", [0.0175, 0.045, 0.025], [0.2, 0.75, 0.8, 1], (0.40, -0.13), np.pi / 2),
    # task 2, sorting: each into the bin of its colour
    "red_ball":     ("sphere", [0.025, 0, 0], [0.85, 0.15, 0.15, 1], (0.42, -0.16), 0.0),
    "blue_cube":    ("box", [0.022, 0.022, 0.022], [0.15, 0.35, 0.9, 1], (0.40, 0.14), 0.5),
    "green_cylinder": ("cylinder", [0.02, 0.03, 0], [0.2, 0.7, 0.3, 1], (0.43, -0.01), 0.0),
    # task 3, tilt and pour (all six degrees of freedom): a book standing in a holder, to be laid
    # flat on the mat; a cup with balls, to be poured into the bowl and set down upright
    # a hardback book (15 x 12 x 2.5 cm): its local z is its thickness, x its height when standing
    "plate":        ("box", [0.06, 0.075, 0.0125], [0.55, 0.13, 0.12, 1], (0.43, 0.06), 0.0),
    "cup":          ("cup", [0.034, 0.045, 0], [0.96, 0.95, 0.9, 1], (0.42, -0.14), 0.0),
    **{f"ball_{i}": ("sphere", [0.009, 0, 0], [0.98, 0.8, 0.1, 1], (0.42, -0.14), 0.0) for i in range(5)},
}
# task 3: the book stands in its holder (its thickness along x)
PLATE_STAND_QUAT = [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]
RACK = (0.43, 0.06)
MAT = (0.60, 0.02)              # where the book is laid flat; outline half width
MAT_HALF = 0.085
CUP_WALL, CUP_BOTTOM, CUP_SEGMENTS = 0.004, 0.006, 20
BOWL = (0.62, -0.20)            # the round bowl the balls are poured into: interior radius, wall height
BOWL_HALF = 0.075
BOWL_SEGMENTS = 28
PLATE_HALF_T = 0.0125           # half the book's thickness
BOWL_WALL = 0.035
BALLS = [f"ball_{i}" for i in range(5)]


def ball_start(i: int):
    """The balls start inside the cup: four on its bottom, one on top."""
    cx, cy = OBJECTS["cup"][3]
    r = OBJECTS[BALLS[i]][1][0]
    if i < 4:
        a = np.pi / 4 + i * np.pi / 2
        return (cx + 0.013 * np.cos(a), cy + 0.013 * np.sin(a), TABLE_Z + CUP_BOTTOM + r + 0.002)
    return (cx, cy, TABLE_Z + CUP_BOTTOM + 3 * r + 0.006)
# task 1: where each piece goes (xy, yaw) and its symmetry (a square looks the same every 90 deg).
# The bar starts lengthwise along x (the fingers close across it); its slot is turned 90 deg.
SLOTS = {
    "big_square":   ((0.60, 0.13), 0.0, np.pi / 2),
    "small_square": ((0.60, 0.04), 0.0, np.pi / 2),
    "bar":          ((0.60, -0.06), 0.0, np.pi),
}
# task 2: bin centre per object; interior half width, wall height
BINS = {"red_ball": (0.62, 0.15), "blue_cube": (0.62, -0.005), "green_cylinder": (0.62, -0.16)}
BIN_HALF = 0.055
BIN_WALL = 0.04
# practice: reach targets (xyz) and the drop zone for the cube
REACH = [(0.46, 0.12, TABLE_Z + 0.11), (0.58, -0.14, TABLE_Z + 0.07), (0.40, -0.16, TABLE_Z + 0.22)]
ZONE = (0.58, 0.02)
ZONE_HALF = 0.045
PARK_Z = -2.0                   # mocap bodies out of sight under the floor
MARKERS = ("screen_goal", "screen_zone", "screen_board", "screen_bins", "screen_rack", "screen_mat",
           "screen_bowl")


def park_xy(i: int):
    """Parking spot of the i-th object (on the floor behind the operator, out of view)."""
    return (-1.3, -0.9 + 0.2 * i)


def rest_z(name: str) -> float:
    """Height of an object's origin resting on the table (its centre; the cup: its bottom)."""
    shape, size = OBJECTS[name][:2]
    if shape == "cup":
        return TABLE_Z + 0.001
    return TABLE_Z + (size[2] if shape == "box" else size[0] if shape == "sphere" else size[1]) + 0.001


def center_offset(name: str) -> float:
    """Height of an object's middle above its origin (the cup's origin is its bottom)."""
    shape, size = OBJECTS[name][:2]
    return size[1] if shape == "cup" else 0.0


def _outline(body, name, cx, cy, hx, hy, rgba, w=0.004):
    """A flat rectangular outline (visual only) on the table, centred at (cx, cy) in body frame."""
    for k, (px, py, sx, sy) in enumerate([(cx + hx, cy, w, hy + w), (cx - hx, cy, w, hy + w),
                                          (cx, cy + hy, hx + w, w), (cx, cy - hy, hx + w, w)]):
        S._add_box(body, f"{name}_{k}", [px, py, 0.0008], [sx, sy, 0.0008], rgba, collide=False)


HIDDEN_GROUP = 5               # geom group nobody draws (the viewers show groups 0-2)


def _hide_other_arm(spec: mujoco.MjSpec):
    """The test uses one arm. The other one stays in the model (the simulator expects two) but is
    invisible and touches nothing: its geoms go to a group no viewer draws, without collisions."""
    other = "left_" if ARM == "right" else "right_"
    for b in spec.bodies:
        if b.name.startswith(other):
            for g in b.geoms:
                g.group = HIDDEN_GROUP
                g.contype = g.conaffinity = 0


SERVO_STIFFNESS = 1.0          # x the arm's joint servo gains (1 = the UR5e's own; 3 made it shaky)


def _stiffer_servos(spec: mujoco.MjSpec, k: float):
    """The UR5e's position servos follow a target with ~0.1 s lag (realistic, but on top of the
    internet's delay it feels sluggish). k x the stiffness and sqrt(k) x the damping keep the
    response well damped and k^0.5 x faster."""
    for a in spec.actuators:
        if a.name.startswith(f"{ARM}_") and any(a.name.endswith(n) for n in S.ARM_ACTS):
            gp, bp = np.array(a.gainprm), np.array(a.biasprm)
            gp[0] *= k
            bp[1] *= k
            bp[2] *= np.sqrt(k)
            a.gainprm, a.biasprm = gp, bp
            fr = np.array(a.forcerange)
            if fr[1] > fr[0]:
                a.forcerange = fr * k


def add(spec: mujoco.MjSpec):
    """Add the screening objects, markers and bins to the scene; hide the unused arm."""
    _hide_other_arm(spec)
    _stiffer_servos(spec, SERVO_STIFFNESS)
    wb = spec.worldbody
    park = [0, 3.0, PARK_Z]
    # reach target: a glowing sphere
    goal = wb.add_body()
    goal.name = "screen_goal"
    goal.mocap = True
    goal.pos = park
    g = goal.add_geom()
    g.name = "screen_goal_sphere"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.022, 0, 0]
    g.rgba = [0.2, 1.0, 0.4, 0.55]
    g.contype = g.conaffinity = 0
    # practice drop zone
    zone = wb.add_body()
    zone.name = "screen_zone"
    zone.mocap = True
    zone.pos = park
    _outline(zone, "screen_zone", 0, 0, ZONE_HALF, ZONE_HALF, [0.15, 0.8, 0.35, 1], w=0.005)
    # task 1 board: an outline per piece, 6 mm larger than the piece, in the slot's orientation
    board = wb.add_body()
    board.name = "screen_board"
    board.mocap = True
    board.pos = park
    for name, ((sx, sy), yaw, _) in SLOTS.items():
        hx, hy = OBJECTS[name][1][:2]
        if abs(np.sin(yaw)) > 0.5:
            hx, hy = hy, hx
        _outline(board, f"slot_{name}", sx, sy, hx + 0.006, hy + 0.006, [0.25, 0.25, 0.28, 1])
    # task 2 bins, coloured like the object that goes in
    bins = wb.add_body()
    bins.name = "screen_bins"
    bins.mocap = True
    bins.pos = park
    h, wall = BIN_HALF, BIN_WALL
    for name, (bx, by) in BINS.items():
        rgba = list(OBJECTS[name][2][:3]) + [1]
        S._add_box(bins, f"bin_{name}_floor", [bx, by, 0.003], [h + 0.005, h + 0.005, 0.003], rgba)
        for k, (px, py, sx, sy) in enumerate([(h + 0.0025, 0, 0.0025, h + 0.005), (-h - 0.0025, 0, 0.0025, h + 0.005),
                                              (0, h + 0.0025, h + 0.005, 0.0025), (0, -h - 0.0025, h + 0.005, 0.0025)]):
            S._add_box(bins, f"bin_{name}_wall{k}", [bx + px, by + py, wall / 2], [sx, sy, wall / 2], rgba)
    # Grasp physics as in the original blocks task (no extra no-slip solver: it made grasps
    # feel odd); torsional friction (condim 6) only on the mug, so it doesn't spin when tipped
    # materials: glazed ceramic (mug, bowl) and wood (book holder)
    T.add_material(spec, "screen_ceramic", rgba=(0.95, 0.95, 0.93, 1), specular=0.6, shininess=0.8, roughness=0.25)
    T.add_material(spec, "screen_ceramic_blue", rgba=(0.55, 0.68, 0.85, 1), specular=0.6, shininess=0.8,
                   roughness=0.25)
    T.add_material(spec, "screen_wood", "wood_floor", rgba=(0.85, 0.75, 0.65, 1), specular=0.2, roughness=0.6)
    # task 3: the book holder (posts on both faces of the standing book), the mat, the bowl
    rack = wb.add_body()
    rack.name = "screen_rack"
    rack.mocap = True
    rack.pos = park
    t = PLATE_HALF_T
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        S._add_box(rack, f"rack_post{k}", [RACK[0] + sx * (t + 0.0045), RACK[1] + sy * 0.055, 0.025],
                   [0.0035, 0.0035, 0.025], material="screen_wood")
    S._add_box(rack, "rack_base", [RACK[0], RACK[1], 0.003], [0.022, 0.055, 0.003], material="screen_wood",
               collide=False)
    mat = wb.add_body()
    mat.name = "screen_mat"
    mat.mocap = True
    mat.pos = park
    _outline(mat, "screen_mat", MAT[0], MAT[1], MAT_HALF, MAT_HALF, [0.15, 0.8, 0.35, 1], w=0.005)
    bowl = wb.add_body()
    bowl.name = "screen_bowl"
    bowl.mocap = True
    bowl.pos = park
    _add_ring(bowl, "bowl", BOWL, BOWL_HALF, 0.005, BOWL_WALL, 0.004, BOWL_SEGMENTS, "screen_ceramic_blue")
    # the objects
    for i, (name, (shape, size, rgba, _, _)) in enumerate(OBJECTS.items()):
        b = wb.add_body()
        b.name = name
        px, py = park_xy(i)
        b.pos = [px, py, rest_z(name) - TABLE_Z]
        j = b.add_freejoint()
        j.name = f"{name}_joint"
        if shape == "cup":
            _add_cup(b, name, size, rgba)
            continue
        gm = b.add_geom()
        gm.name = f"{name}_geom"
        gm.type = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                   "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[shape]
        gm.size = size
        gm.rgba = rgba
        gm.mass = 0.08                        # light blocks, as in the original blocks task
        gm.condim = 4
        gm.friction = [1.2, 0.01, 0.001]
        if name in BALLS:                     # small balls that pour and settle
            gm.mass = 0.01
            gm.condim = 6
            gm.friction = [0.6, 0.005, 0.0005]
        elif shape == "sphere":               # rolling friction: the ball doesn't roll away forever
            gm.condim = 6
            gm.friction = [1.2, 0.01, 0.002]
        elif name == "plate":                 # a paperback-weight book
            gm.mass = 0.2


def _add_ring(b, name, centre, r_in, bottom, height, wall, segments, material, masses=None):
    """A hollow round vessel: a round bottom and a ring of overlapping wall slabs (no gaps for
    the balls). centre (x, y) in the body frame; the bottom's underside at z = 0."""
    cx, cy = centre
    r_mid = r_in + wall / 2
    seg = r_mid * np.tan(np.pi / segments) + 0.0015
    g = b.add_geom()
    g.name = f"{name}_bottom"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [r_in + wall, bottom / 2, 0]
    g.pos = [cx, cy, bottom / 2]
    g.material = material
    g.condim, g.friction = 6, [1.0, 0.02, 0.001]
    if masses:
        g.mass = masses[0]
    for k in range(segments):
        a = 2 * np.pi * k / segments
        g = b.add_geom()
        g.name = f"{name}_wall{k}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [wall / 2, seg, height / 2]
        g.pos = [cx + r_mid * np.cos(a), cy + r_mid * np.sin(a), bottom + height / 2]
        g.quat = [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        g.material = material
        g.condim, g.friction = 6, [1.0, 0.02, 0.001]
        if masses:
            g.mass = masses[1]


def _add_cup(b, name, size, rgba):
    """A glazed ceramic mug (~250 g): hollow, so the balls really pour out. Body origin = the
    middle of its bottom."""
    r_out, half_h = size[0], size[1]
    _add_ring(b, name, (0, 0), r_out - CUP_WALL, CUP_BOTTOM, 2 * half_h, CUP_WALL, CUP_SEGMENTS,
              "screen_ceramic", masses=(0.04, 0.005))


def bind(info: S.SceneInfo):
    """The objects are recorded as object poses (episode files)."""
    m = info.model
    info.objects = dict(OBJECTS)
    for name in OBJECTS:
        info.obj_qpos_adr[name] = m.jnt_qposadr[m.joint(f"{name}_joint").id]


def place(info: S.SceneInfo, data: mujoco.MjData, name: str, xy=None, yaw: float | None = None):
    """Put object `name` at rest at xy (default: its start), or park it (xy='park')."""
    a = info.obj_qpos_adr[name]
    dof = info.model.jnt_dofadr[info.model.joint(f"{name}_joint").id]
    quat = None
    if isinstance(xy, str):
        x, y = park_xy(list(OBJECTS).index(name))
        z = rest_z(name) - TABLE_Z
        yaw = 0.0
    elif xy is None and name == "plate":             # standing in its holder (12 cm tall)
        (x, y), z, quat = RACK, TABLE_Z + OBJECTS["plate"][1][0] + 0.001, PLATE_STAND_QUAT
    elif xy is None and name in BALLS:               # inside the cup
        x, y, z = ball_start(BALLS.index(name))
        yaw = 0.0
    else:
        x, y = OBJECTS[name][3] if xy is None else xy
        z = rest_z(name)
        yaw = OBJECTS[name][4] if yaw is None else yaw
    data.qpos[a:a + 3] = [x, y, z]
    data.qpos[a + 3:a + 7] = quat if quat is not None else S._yaw_quat(yaw)
    data.qvel[dof:dof + 6] = 0


def park_all(info: S.SceneInfo, data: mujoco.MjData):
    """Everything out of sight (the game then places what the current phase needs)."""
    for name in OBJECTS:
        place(info, data, name, "park")
    for b in MARKERS:
        data.mocap_pos[info.model.body_mocapid[info.model.body(b).id]] = [0, 3.0, PARK_Z]


# The test arm stays where the right arm of the pair is (base at y = -0.36): seen first-person
# from the robot's head (the monitor's default view, like in VR), it reaches out like your own
# right arm without hiding the table. (A TaskPlugin can move it: arm_xy, home_q.)
S.register_task(TASK, S.TaskPlugin(title="Screening test: pick and place", add=add, bind=bind,
                                   reset=park_all, pinch_z_min=TABLE_Z + 0.02))
