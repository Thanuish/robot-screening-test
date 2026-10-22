"""The screening test's scene, built from config/screening.toml (see TEST_TELEOP/config.py): one
arm, the objects and the markers of every phase, added to the dual-UR5e scene of vrteleop as
the task "screen" (a scene plugin, see vrteleop.scene.register_task).

The test arm is one arm of the pair (the other is hidden). Objects of the phases not being
played are parked on the floor behind the operator; markers (zones, outlines, bins, the bowl,
the holder) and the reach target are mocap bodies parked under the floor and moved onto the
table when their phase starts.
"""
from __future__ import annotations

import mujoco
import numpy as np

from vrteleop import scene as S
from vrteleop import textures as T
from vrteleop.scene import TABLE_Z

from . import config as C

TASK = "screen"
PARK = (0.0, 3.0, -2.0)        # mocap bodies out of sight under the floor
GOAL = "screen_goal"           # the reach target's body
HIDDEN_GROUP = 5               # geom group nobody draws (the viewers show groups 0-2)

CFG: C.Config = None           # the configuration the scene is built from (use())


def marker_body(name: str) -> str:
    return f"marker_{name}"


def park_xy(i: int):
    """Parking spot of the i-th object (on the floor behind the operator, out of view)."""
    return (-1.3, -0.9 + 0.2 * i)


def _quat(yaw: float, pitch: float = 0.0):
    """Turned by yaw about the vertical, after tipping by pitch about the object's y axis."""
    qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qp = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qy, qp)
    return q


def _half_height(o: C.Obj, quat) -> float:
    """Half the object's height when it rests in orientation quat (boxes; others: upright)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    R = R.reshape(3, 3)
    if o.shape == "box":
        return float(np.abs(R[2]) @ np.asarray(o.half))
    if o.shape == "ring":                     # lying flat: half its thickness; on its edge: its outer radius
        az = abs(R[2, 2])
        return float(az * o.half[1] + np.sqrt(max(0.0, 1 - az * az)) * o.half[0] + (1 - az) * o.half[1])
    return o.half[0] if o.shape == "sphere" else o.half[1]


def rest_z(name: str) -> float:
    """Height of an object's origin lying on the table, upright (its centre; a cup: its bottom)."""
    o = CFG.objects[name]
    if o.shape == "cup":
        return TABLE_Z + 0.001
    return TABLE_Z + (o.half[2] if o.shape == "box" else _half_height(o, [1, 0, 0, 0])) + 0.001


def center_offset(name: str) -> float:
    """Height of an object's middle above its origin (a cup's origin is its bottom)."""
    o = CFG.objects[name]
    return o.half[1] if o.shape == "cup" else 0.0


def _start_in_cup(o: C.Obj):
    """Balls start inside a cup: four on its bottom, the rest stacked in the middle."""
    cup = CFG.objects[o.start_in]
    cx, cy = cup.start
    r = o.half[0]
    i = o.index_in_group
    if i < 4:
        # four in a square on the bottom, apart by at least their diameter + 2 mm, clear of the wall
        ring = min(max(0.013, (2 * r + 0.002) / np.sqrt(2)), cup.half[0] - cup.cup_wall - r - 0.001)
        a = np.pi / 4 + i * np.pi / 2
        return (cx + ring * np.cos(a), cy + ring * np.sin(a), TABLE_Z + cup.cup_bottom + r + 0.002)
    k = i - 4
    return (cx, cy, TABLE_Z + cup.cup_bottom + (3 + 2 * k) * r + 0.006 * (k + 1))


# ---------------------------------------------------------------- building
def _outline(body, name, cx, cy, hx, hy, rgba, w):
    """A flat rectangular outline (visual only) on the table, centred at (cx, cy) in body frame;
    w = half the line width."""
    for k, (px, py, sx, sy) in enumerate([(cx + hx, cy, w, hy + w), (cx - hx, cy, w, hy + w),
                                          (cx, cy + hy, hx + w, w), (cx, cy - hy, hx + w, w)]):
        S._add_box(body, f"{name}_{k}", [px, py, 0.0008], [sx, sy, 0.0008], rgba, collide=False)


def _capsule(body, name, a, b, r, rgba=None, material=None, collide=False, group=0):
    """A capsule from point a to point b (body frame)."""
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto = [*a, *b]
    g.size = [r, 0, 0]
    if material:
        g.material = material
    elif rgba is not None:
        g.rgba = rgba
    if not collide:
        g.contype = g.conaffinity = 0
    g.group = group
    return g


def _ring_points(R, n, plane="xy"):
    a = 2 * np.pi * np.arange(n + 1) / n
    c, s = R * np.cos(a), R * np.sin(a)
    z = np.zeros_like(a)
    return np.stack([c, s, z] if plane == "xy" else [c, z, s], 1)      # "xz": a loop standing up


def _add_ring_object(b, name, o: C.Obj):
    """A ring lying flat (its axis = the body's z): short overlapping capsules around a circle."""
    P = _ring_points(o.half[0], o.ring_segments)
    for k in range(o.ring_segments):
        g = _capsule(b, f"{name}_seg{k}", P[k], P[k + 1], o.half[1], rgba=o.rgba,
                     material=_mat(o.material), collide=True)
        g.mass = o.mass / o.ring_segments
        g.condim, g.friction = o.condim, list(o.friction)


def _add_peg(body, n, mk: C.Marker):
    """A post on a small base, the peg leaving its top tilted toward the marker's +x."""
    p, rgba = mk.p, mk.rgba or [0.9, 0.9, 0.9, 1]
    h, t, L, r = p["height"], p["tilt"], p["length"], p["radius"]
    post = [0.55, 0.55, 0.58, 1]
    S._add_box(body, f"{n}_base", [0, 0, 0.004], [0.035, 0.035, 0.004], post)
    if h > 0:
        g = body.add_geom()
        g.name = f"{n}_post"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [max(r * 1.6, 0.012), h / 2, 0]
        g.pos = [0, 0, h / 2]
        g.rgba = post
    base = np.array([0.0, 0.0, max(h, 0.008)])
    tip = base + L * np.array([np.sin(t), 0.0, np.cos(t)])
    g = _capsule(body, f"{n}_peg", base, tip, r, rgba=rgba, collide=True)
    g.condim, g.friction = 3, [0.6, 0.005, 0.0001]


def wire_points(mk: C.Marker):
    """The wire's control points in the world."""
    return [[mk.pos[0] + x, mk.pos[1] + y, TABLE_Z + z] for x, y, z in mk.p["points"]]


def wand_body(name: str) -> str:
    return f"wand_{name}"


def _add_wire(wb, body, n, mk: C.Marker):
    """The wire (a chain of thin capsules, for the eye: touches are measured, not simulated),
    red rubber sleeves at both ends, posts down to a wooden base; and the wand that goes with it
    (a mocap body the game keeps in the gripper)."""
    from buzzwire.spline import Wire
    p = mk.p
    W = Wire([[x, y, z] for x, y, z in p["points"]], step=0.01)        # (marker frame)
    metal, rubber, post = [0.78, 0.8, 0.84, 1], [0.85, 0.12, 0.1, 1], [0.5, 0.5, 0.53, 1]
    for k in range(len(W.pts) - 1):
        a, b = W.pts[k], W.pts[k + 1]
        sl = p["sleeve"]
        on_sleeve = W.s[k] < sl or W.s[k + 1] > W.length - sl
        _capsule(body, f"{n}_w{k}", a, b, p["radius"] * (1.8 if on_sleeve else 1.0), rgba=rubber if on_sleeve else metal)
    for k, q in enumerate((W.pts[0], W.pts[-1])):
        _capsule(body, f"{n}_post{k}", [q[0], q[1], 0.0], [q[0], q[1], q[2]], 0.006, rgba=post)
        S._add_box(body, f"{n}_foot{k}", [q[0], q[1], 0.005], [0.03, 0.03, 0.005], [0.62, 0.45, 0.3, 1])
    # the wand: loop centre at its origin, the loop standing in its x-z plane (the wire runs along
    # its y), the handle back along -x to where the gripper holds it
    wand = wb.add_body()
    wand.name = wand_body(n)
    wand.mocap = True
    wand.pos = list(PARK)
    L = _ring_points(p["loop_r"], 28, "xz")
    for k in range(28):
        _capsule(wand, f"{n}_loop{k}", L[k], L[k + 1], p["loop_t"] / 2, rgba=[0.86, 0.68, 0.25, 1])
    _capsule(wand, f"{n}_stem", [-p["loop_r"], 0, 0], [-p["handle"] + 0.03, 0, 0], 0.003, rgba=[0.86, 0.68, 0.25, 1])
    _capsule(wand, f"{n}_grip", [-p["handle"] + 0.03, 0, 0], [-p["handle"] - 0.03, 0, 0], 0.009, rgba=[0.12, 0.12, 0.14, 1])


FLOAT = C.FLOAT_PREFIX
FLOAT_MOUNT, FLOAT_TARGET = FLOAT + "mount", FLOAT + "target"
# where the floating gripper's mount is built (the game then puts it where the arm's gripper is at
# home), pointing down: turned 180 deg about x (the 2F-85's fingers point along its +z)
FLOAT_HOME = (0.45, -0.20, TABLE_Z + 0.35)
FLOAT_QUAT = [0.0, 1.0, 0.0, 0.0]


def _add_floating_gripper(spec: mujoco.MjSpec):
    """[arm] mount = "floating": a Robotiq 2F-85 on its own, a free body held by a soft weld to an
    invisible target the hand moves (mocap); it still collides with the table and the objects and
    really grips. No arm, no inverse kinematics: the hand's target goes to the gripper directly."""
    grip = mujoco.MjSpec.from_file(str(S.GRIPPER_XML))
    grip.option.cone, grip.option.impratio, grip.option.integrator = spec.option.cone, spec.option.impratio, spec.option.integrator
    wb = spec.worldbody
    body = wb.add_body()
    body.name = FLOAT_MOUNT
    body.pos = list(FLOAT_HOME)
    body.quat = FLOAT_QUAT
    j = body.add_freejoint()
    j.name = FLOAT + "joint"
    site = body.add_site()
    site.name = FLOAT + "attach"
    site.attach_body(grip.body("base_mount"), FLOAT, "")
    tgt = wb.add_body()
    tgt.name = FLOAT_TARGET
    tgt.mocap = True
    tgt.pos = list(FLOAT_HOME)
    tgt.quat = FLOAT_QUAT
    eq = spec.add_equality()
    eq.name = FLOAT + "weld"
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1, eq.name2 = FLOAT_MOUNT, FLOAT_TARGET
    eq.solref = [0.005, 1.0]                    # (stiff, but it still gives way to the table or an object)
    eq.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    for b in spec.bodies:
        if b.name.startswith(FLOAT):
            b.gravcomp = 1.0


def _hide_arm(spec: mujoco.MjSpec, prefix: str):
    """An arm that stays in the model (the simulator expects two) but is invisible and touches
    nothing: its geoms go to a group no viewer draws, without collisions."""
    for b in spec.bodies:
        if b.name.startswith(prefix):
            for g in b.geoms:
                g.group = HIDDEN_GROUP
                g.contype = g.conaffinity = 0


def _hide_other_arm(spec: mujoco.MjSpec):
    """The test uses one arm. The other one stays in the model (the simulator expects two) but is
    invisible and touches nothing: its geoms go to a group no viewer draws, without collisions."""
    _hide_arm(spec, "left_" if CFG.arm.side == "right" else "right_")


def _stiffer_servos(spec: mujoco.MjSpec, k: float):
    """The UR5e's position servos follow a target with ~0.1 s lag (realistic, but on top of the
    internet's delay it feels sluggish). k x the stiffness and sqrt(k) x the damping keep the
    response well damped and k^0.5 x faster."""
    for a in spec.actuators:
        if a.name.startswith(f"{CFG.arm.side}_") and any(a.name.endswith(n) for n in S.ARM_ACTS):
            gp, bp = np.array(a.gainprm), np.array(a.biasprm)
            gp[0] *= k
            bp[1] *= k
            bp[2] *= np.sqrt(k)
            a.gainprm, a.biasprm = gp, bp
            fr = np.array(a.forcerange)
            if fr[1] > fr[0]:
                a.forcerange = fr * k


def _tube_mesh(spec, name, r_in, r_out, z0, z1, n=48):
    """A smooth open tube (a cup's wall, for the eye): outer and inner wall, top rim, closed at the
    bottom. Returns the mesh name."""
    a = 2 * np.pi * np.arange(n) / n
    c, s = np.cos(a), np.sin(a)
    rings = [(r_out, 0.0), (r_out, z1), (r_in, z1), (r_in, z0)]     # outer bottom/top, inner top/bottom
    vert = np.concatenate([np.stack([r * c, r * s, np.full(n, z)], 1) for r, z in rings])
    faces = []
    for k in range(n):
        k1 = (k + 1) % n
        ob0, ot0, it0, ib0 = k, n + k, 2 * n + k, 3 * n + k
        ob1, ot1, it1, ib1 = k1, n + k1, 2 * n + k1, 3 * n + k1
        faces += [(ob0, ob1, ot1), (ob0, ot1, ot0),                   # outer wall, facing out
                  (ot0, ot1, it1), (ot0, it1, it0),                   # top rim, facing up
                  (ib0, it1, ib1), (ib0, it0, it1),                   # inner wall, facing in
                  (ob0, ib1, ob1), (ob0, ib0, ib1)]                   # closed at the bottom
    mesh = spec.add_mesh()
    mesh.name = f"{name}_glass_mesh"
    mesh.uservert = vert.flatten().tolist()
    mesh.userface = np.array(faces).flatten().tolist()
    return mesh.name


def _add_ring(b, name, centre, r_in, bottom, height, wall, segments, material, masses=None, friction=None,
              spec=None, see_through=False):
    """A hollow round vessel: a round bottom and a ring of overlapping wall slabs (no gaps for
    the balls). centre (x, y) in the body frame; the bottom's underside at z = 0.
    see_through (a clear material): the slabs only collide (invisible, their overlaps would show
    through) and one smooth tube is drawn instead (needs spec)."""
    cx, cy = centre
    friction = friction or [1.0, 0.02, 0.001]
    r_mid = r_in + wall / 2
    seg = r_mid * np.tan(np.pi / segments) + 0.0015
    g = b.add_geom()
    g.name = f"{name}_bottom"
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [r_in + wall, bottom / 2, 0]
    g.pos = [cx, cy, bottom / 2]
    g.material = material
    g.condim, g.friction = 6, list(friction)
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
        if see_through:
            g.rgba = [1, 1, 1, 0]
        else:
            g.material = material
        g.condim, g.friction = 6, list(friction)
        if masses:
            g.mass = masses[1]
    if see_through:
        g = b.add_geom()
        g.name = f"{name}_glass"
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = _tube_mesh(spec, name, r_in, r_in + wall, bottom, bottom + height)
        g.pos = [cx, cy, 0]
        g.material = material
        g.contype = g.conaffinity = 0
        g.mass = 0.0


def _mat(name):
    return f"screen_{name}" if name else None


def _add_marker(wb, mk: C.Marker):
    body = wb.add_body()
    body.name = marker_body(mk.name)
    body.mocap = True
    body.pos = list(PARK)
    p, n = mk.p, mk.name
    rgba = mk.rgba or [0.25, 0.25, 0.28, 1]
    if mk.kind == "zone":
        _outline(body, n, 0, 0, p["half"][0], p["half"][1], rgba, p["line"])
    elif mk.kind == "slot":                 # the object's footprint + margin
        o = CFG.objects[p["object"]]
        _outline(body, n, 0, 0, o.half[0] + p["margin"], o.half[1] + p["margin"], rgba, p["line"])
    elif mk.kind == "bin":
        h, wt, wh = p["half"], p["wall"], p["wall_height"]
        S._add_box(body, f"{n}_floor", [0, 0, 0.003], [h + wt, h + wt, 0.003], rgba)
        for k, (px, py, sx, sy) in enumerate([(h + wt / 2, 0, wt / 2, h + wt), (-h - wt / 2, 0, wt / 2, h + wt),
                                              (0, h + wt / 2, h + wt, wt / 2), (0, -h - wt / 2, h + wt, wt / 2)]):
            S._add_box(body, f"{n}_wall{k}", [px, py, wh / 2], [sx, sy, wh / 2], rgba)
    elif mk.kind == "bowl":
        _add_ring(body, n, (0, 0), p["r_in"], p["bottom"], p["wall_height"], p["wall"], p["segments"],
                  _mat(mk.material))
    elif mk.kind == "holder":
        (dx, dy), ph, bh = p["post_spacing"], p["post_half"], p["base_half"]
        for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
            S._add_box(body, f"{n}_post{k}", [sx * dx / 2, sy * dy / 2, ph[2]], ph, material=_mat(mk.material))
        S._add_box(body, f"{n}_base", [0, 0, bh[2]], bh, material=_mat(mk.material), collide=False)
    elif mk.kind == "peg":
        _add_peg(body, n, mk)
    elif mk.kind == "wire":
        _add_wire(wb, body, n, mk)
    elif mk.kind == "platform":                     # a fixed block to build on
        h = mk.p["half"]
        g = S._add_box(body, f"{n}_top", [0, 0, h[2]], h, rgba or [0.75, 0.6, 0.45, 1], material=_mat(mk.material))
        g.friction = [1.0, 0.01, 0.001]


def add(spec: mujoco.MjSpec):
    """Add the objects, markers and the reach target to the scene; hide the unused arm."""
    _hide_other_arm(spec)
    _stiffer_servos(spec, CFG.arm.servo_stiffness)
    if CFG.arm.mount == "floating":
        _hide_arm(spec, f"{CFG.arm.side}_")
        _add_floating_gripper(spec)
    wb = spec.worldbody
    for m in CFG.materials.values():
        T.add_material(spec, _mat(m.name), m.texture, rgba=m.rgba, specular=m.specular, shininess=m.shininess,
                       reflectance=m.reflectance, roughness=m.roughness, metallic=m.metallic)
    goal = wb.add_body()
    goal.name = GOAL
    goal.mocap = True
    goal.pos = list(PARK)
    g = goal.add_geom()
    g.name = "screen_goal_sphere"
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [CFG.reach_target["radius"], 0, 0]
    g.rgba = CFG.reach_target["rgba"]
    g.contype = g.conaffinity = 0
    for mk in CFG.markers.values():
        _add_marker(wb, mk)
    # Grasp physics as in the original blocks task (no extra no-slip solver: it made grasps feel
    # odd); friction and condim per object from the configuration
    for i, (name, o) in enumerate(CFG.objects.items()):
        b = wb.add_body()
        b.name = name
        px, py = park_xy(i)
        b.pos = [px, py, rest_z(name) - TABLE_Z]
        j = b.add_freejoint()
        j.name = f"{name}_joint"
        if o.shape == "cup":
            _add_ring(b, name, (0, 0), o.half[0] - o.cup_wall, o.cup_bottom, 2 * o.half[1], o.cup_wall,
                      o.cup_segments, _mat(o.material),
                      masses=(o.mass * 2 / 7, o.mass * 5 / 7 / o.cup_segments), friction=o.friction,
                      spec=spec, see_through=bool(o.material) and CFG.materials[o.material].rgba[3] < 1)
            continue
        if o.shape == "ring":
            _add_ring_object(b, name, o)
            continue
        gm = b.add_geom()
        gm.name = f"{name}_geom"
        gm.type = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                   "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[o.shape]
        gm.size = o.half
        gm.rgba = o.rgba
        if o.material:
            gm.material = _mat(o.material)
        gm.mass = o.mass
        gm.condim = o.condim
        gm.friction = o.friction


def bind(info: S.SceneInfo):
    """The objects are recorded as object poses (episode files). A floating gripper takes the
    arm's place for the game: its pinch point, its finger motor, its finger joint."""
    m = info.model
    info.objects = dict(CFG.objects)
    for name in CFG.objects:
        info.obj_qpos_adr[name] = m.jnt_qposadr[m.joint(f"{name}_joint").id]
    if CFG.arm.mount == "floating":
        s = CFG.arm.side
        info.ee_site[s] = m.site(FLOAT + "pinch").id
        info.grip_act[s] = m.actuator(FLOAT + "fingers_actuator").id
        info.grip_qpos_adr[s] = m.jnt_qposadr[m.joint(FLOAT + "right_driver_joint").id]
        info.floating = dict(side=s, mocap=int(m.body_mocapid[m.body(FLOAT_TARGET).id]), body=m.body(FLOAT_MOUNT).id,
                             qadr=int(m.jnt_qposadr[m.joint(FLOAT + "joint").id]),
                             vadr=int(m.jnt_dofadr[m.joint(FLOAT + "joint").id]), home=FLOAT_HOME)


# ---------------------------------------------------------------- placing
def place(info: S.SceneInfo, data: mujoco.MjData, name: str, xy=None, yaw: float | None = None):
    """Put object `name` at rest at its start (xy None), at xy lying flat turned by yaw, or park
    it (xy='park')."""
    o = CFG.objects[name]
    a = info.obj_qpos_adr[name]
    dof = info.model.jnt_dofadr[info.model.joint(f"{name}_joint").id]
    if isinstance(xy, str):
        x, y = park_xy(list(CFG.objects).index(name))
        z, q = rest_z(name) - TABLE_Z, _quat(0.0)
    elif xy is None and o.start_in is not None:
        (x, y, z), q = _start_in_cup(o), _quat(0.0)
    elif xy is None:
        (x, y), q = o.start, _quat(o.yaw, o.pitch)
        z = TABLE_Z + 0.001 if o.shape == "cup" else TABLE_Z + _half_height(o, q) + 0.001
    else:
        (x, y), q = xy, _quat(o.yaw if yaw is None else yaw)
        z = rest_z(name)
    data.qpos[a:a + 3] = [x, y, z]
    data.qpos[a + 3:a + 7] = q
    data.qvel[dof:dof + 6] = 0


def show_marker(info: S.SceneInfo, data: mujoco.MjData, name: str, on: bool = True):
    mk = CFG.markers[name]
    mid = info.model.body_mocapid[info.model.body(marker_body(name)).id]
    data.mocap_pos[mid] = [*mk.pos, TABLE_Z] if on else list(PARK)
    data.mocap_quat[mid] = _quat(mk.yaw if on else 0.0)


def move_wand(info: S.SceneInfo, data: mujoco.MjData, wire: str, pos=None, quat=None):
    """The wand of a wire marker at pos / quat (world), or parked (pos None)."""
    mid = info.model.body_mocapid[info.model.body(wand_body(wire)).id]
    data.mocap_pos[mid] = list(PARK) if pos is None else pos
    data.mocap_quat[mid] = [1, 0, 0, 0] if quat is None else quat


def show_goal(info: S.SceneInfo, data: mujoco.MjData, pos=None):
    """The reach target at pos (world), or parked (None)."""
    data.mocap_pos[info.model.body_mocapid[info.model.body(GOAL).id]] = list(PARK) if pos is None else pos


def park_all(info: S.SceneInfo, data: mujoco.MjData):
    """Everything out of sight (the game then places what the current phase needs)."""
    for name in CFG.objects:
        place(info, data, name, "park")
    for name, mk in CFG.markers.items():
        show_marker(info, data, name, False)
        if mk.kind == "wire":
            move_wand(info, data, name, None)
    show_goal(info, data, None)


def use(cfg: C.Config):
    """Build the scene from cfg from now on (the next build_scene)."""
    global CFG
    CFG = cfg
    a = cfg.arm
    S.register_task(TASK, S.TaskPlugin(title="Screening test", add=add, bind=bind, reset=park_all,
                                       pinch_z_min=TABLE_Z + a.min_height,
                                       arm_xy={a.side: a.base}, home_q={a.side: a.home}))


def _broken(_spec):
    raise LOAD_ERROR


LOAD_ERROR: C.ConfigError | None = None
try:
    use(C.load())
except C.ConfigError as e:          # (reported when a scene is built: python -m TEST_TELEOP.check_config)
    LOAD_ERROR = e
    S.register_task(TASK, S.TaskPlugin(title="Screening test", add=_broken))
