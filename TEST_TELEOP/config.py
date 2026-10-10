"""The screening test's configuration: config/screening.toml (or the file in the environment
variable SCREEN_CONFIG). Everything about the test is defined there: the arm, the view, the
depth cues, the objects, the markers (zones, outlines, bins, bowl, holder), the phases and
their steps, the tolerances and the scoring. This module reads it, checks it and turns it into
plain Python objects for the scene (screen_scene.py), the game (screen_game.py), the bot and the
lobby.

Units in the file are what people use: metres, degrees, seconds, kilograms, full sizes (not the
half sizes MuJoCo uses). They are converted here (radians, half sizes).

    python -m TEST_TELEOP.check_config      # check the file, print a summary
"""
from __future__ import annotations

import difflib
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "screening.toml"

# the table (part of the room in vrteleop/scene.py): positions must lie on it
TABLE_X = (-0.15, 0.95)
TABLE_Y = (-0.75, 0.75)
ARM_REACH = 0.85                 # a UR5e reaches 85 cm from its base
TABLE_Z = 0.75                   # table top height (= vrteleop.scene.TABLE_Z)

SHAPES = ("box", "sphere", "cylinder", "cup")
MARKER_KINDS = ("zone", "slot", "bin", "bowl", "holder")
STEP_TYPES = ("reach", "place_in_zone", "place_in_slot", "place_in_bin", "remove_from", "lay_flat",
              "pour_into", "set_upright")
LATCHED = ("reach", "remove_from", "lay_flat", "pour_into", "set_upright")   # stay done once done
PLACE_STEPS = ("place_in_zone", "place_in_slot", "place_in_bin")


class ConfigError(Exception):
    def __init__(self, path, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"{path}: {len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems))


# ---------------------------------------------------------------- the parsed configuration
@dataclass
class Arm:
    model: str
    side: str                    # "left" / "right": which arm of the robot's pair is used
    base: tuple                  # (x, y) of its base on the table
    home: list                   # joint angles [rad]
    min_height: float            # lowest pinch point above the table [m]
    max_speed: float             # [m/s]
    max_turn: float              # [rad/s]
    servo_stiffness: float


@dataclass
class Material:
    name: str
    rgba: list
    texture: str | None
    specular: float
    shininess: float
    roughness: float | None
    reflectance: float
    metallic: float | None


@dataclass
class Obj:
    name: str
    shape: str
    half: list                   # MuJoCo geom size: box half sizes; sphere [r]; cylinder [r, half h]; cup [r, half h]
    rgba: list
    material: str | None
    start: tuple                 # (x, y)
    yaw: float                   # [rad]
    pitch: float                 # [rad] (boxes: e.g. a book standing on its edge)
    mass: float
    friction: list               # sliding, twisting, rolling
    condim: int
    graspable: bool
    start_in: str | None         # inside this cup at the start
    group: str                   # the [objects.<name>] table it came from (count > 1: name_0, name_1...)
    index_in_group: int
    cup_wall: float = 0.0
    cup_bottom: float = 0.0
    cup_segments: int = 0


@dataclass
class Marker:
    name: str
    kind: str
    pos: tuple                   # (x, y) on the table
    yaw: float
    rgba: list | None
    material: str | None
    p: dict                      # kind-specific values (metres, half sizes where noted)


@dataclass
class Step:
    id: str
    type: str
    label: str
    after: str | None
    object: str | None = None    # (one object)
    objects: list = field(default_factory=list)   # (pour_into: the objects to pour)
    target: tuple | None = None  # (reach: x, y, z in the world)
    marker: str | None = None    # zone / slot / bin / holder / container
    distance: float = 0.0        # remove_from
    count: int = 0               # pour_into

    @property
    def latched(self) -> bool:
        return self.type in LATCHED


@dataclass
class Phase:
    key: str
    kind: str                    # "pair", "task", "done"
    title: str
    text: str
    scored: bool = False
    limit: float | None = None   # [s]
    rotation: str = "yaw"
    objects: list = field(default_factory=list)   # object names (groups expanded)
    markers: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    graspable: list = field(default_factory=list)   # objects whose grasps and drops are counted


@dataclass
class Config:
    path: Path
    arm: Arm
    motion_scale: float
    view: dict                   # eye, look, fov (MuJoCo frame, as main.js wants it)
    depth_cues: dict
    timing: dict
    tol: dict
    scoring: dict
    texts: dict
    reach_target: dict
    materials: dict
    objects: dict                # name -> Obj (in file order; groups expanded)
    groups: dict                 # group name -> [object names]
    markers: dict                # name -> Marker
    phases: list                 # [Phase]

    def phase(self, key) -> Phase:
        return next(p for p in self.phases if p.key == key)

    @property
    def scored_phases(self) -> list:
        return [p for p in self.phases if p.scored]

    @property
    def task_phases(self) -> list:
        return [p for p in self.phases if p.kind == "task"]

    def public(self) -> dict:
        """What the web pages need (monitor, phone, admin)."""
        return dict(arm=self.arm.side, view=self.view, depth_cues=self.depth_cues, table_z=TABLE_Z,
                    phases=[dict(key=p.key, title=p.title, kind=p.kind, scored=p.scored) for p in self.phases],
                    pass_mark=self.scoring["pass_mark"], must_complete=self.scoring["must_complete"])


# ---------------------------------------------------------------- checking helpers
class _Checker:
    def __init__(self):
        self.problems: list[str] = []

    def err(self, where, msg):
        self.problems.append(f"{where}: {msg}")

    def keys(self, d: dict, where: str, allowed, required=()):
        if not isinstance(d, dict):
            self.err(where, "should be a table")
            return False
        for k in d:
            if k not in allowed:
                hint = difflib.get_close_matches(k, list(allowed), n=1)
                self.err(f"{where}.{k}", "unknown setting" + (f" (did you mean '{hint[0]}'?)" if hint else
                                                              f" (allowed: {', '.join(allowed)})"))
        for k in required:
            if k not in d:
                self.err(where, f"'{k}' is missing")
        return True

    def num(self, d, k, where, default=None, lo=None, hi=None, lo_open=False, integer=False):
        if k not in d:
            return default
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
            self.err(f"{where}.{k}", f"should be {'a whole number' if integer else 'a number'}, got {v!r}")
            return default
        if lo is not None and (v <= lo if lo_open else v < lo):
            self.err(f"{where}.{k}", f"should be {'more than' if lo_open else 'at least'} {lo}, got {v}")
            return default
        if hi is not None and v > hi:
            self.err(f"{where}.{k}", f"should be at most {hi}, got {v}")
            return default
        return v

    def vec(self, d, k, where, n, default=None, lo=None, lo_open=False):
        if k not in d:
            return default
        v = d[k]
        if (not isinstance(v, list) or len(v) != n
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v)):
            self.err(f"{where}.{k}", f"should be a list of {n} numbers, got {v!r}")
            return default
        if lo is not None and any((x <= lo if lo_open else x < lo) for x in v):
            self.err(f"{where}.{k}", f"all values should be {'more than' if lo_open else 'at least'} {lo}, got {v}")
            return default
        return [float(x) for x in v]

    def text(self, d, k, where, default=None, choices=None):
        if k not in d:
            return default
        v = d[k]
        if not isinstance(v, str):
            self.err(f"{where}.{k}", f"should be text in quotes, got {v!r}")
            return default
        if choices and v not in choices:
            self.err(f"{where}.{k}", f"should be one of {', '.join(choices)}; got '{v}'")
            return default
        return v

    def flag(self, d, k, where, default=None):
        if k not in d:
            return default
        if not isinstance(d[k], bool):
            self.err(f"{where}.{k}", f"should be true or false, got {d[k]!r}")
            return default
        return d[k]

    def color(self, d, k, where, default=None):
        """[r, g, b] or [r, g, b, a] (0..1), or "#rrggbb"."""
        if k not in d:
            return default
        v = d[k]
        if isinstance(v, str) and len(v) == 7 and v.startswith("#"):
            try:
                return [int(v[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]
            except ValueError:
                pass
        elif (isinstance(v, list) and len(v) in (3, 4)
              and all(isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 for x in v)):
            return [float(x) for x in v] + ([1.0] if len(v) == 3 else [])
        self.err(f"{where}.{k}", f"should be [red, green, blue] (0 to 1 each, optionally a 4th: opacity) "
                                 f"or \"#rrggbb\", got {v!r}")
        return default

    def on_table(self, xy, where, base=None):
        if xy is None:
            return
        x, y = xy
        if not (TABLE_X[0] <= x <= TABLE_X[1] and TABLE_Y[0] <= y <= TABLE_Y[1]):
            self.err(where, f"({x}, {y}) is off the table (x {TABLE_X[0]} to {TABLE_X[1]}, "
                            f"y {TABLE_Y[0]} to {TABLE_Y[1]})")
        elif base is not None and math.hypot(x - base[0], y - base[1]) > ARM_REACH:
            self.err(where, f"({x}, {y}) is out of the arm's reach ({ARM_REACH} m from its base "
                            f"at {tuple(base)})")


def _hex(rgba):
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02x}" for c in rgba[:3])


# ---------------------------------------------------------------- loading
def path() -> Path:
    p = os.environ.get("SCREEN_CONFIG", "").strip()
    return Path(p).resolve() if p else DEFAULT_PATH


def load(p: Path | str | None = None) -> Config:
    """Read and check the configuration; raises ConfigError with every problem found."""
    p = Path(p) if p else path()
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ConfigError(p, ["the file doesn't exist"]) from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(p, [f"not valid TOML: {e}"]) from None
    return parse(raw, p)


_cache: dict = {}


def cached(p: Path | str | None = None) -> Config:
    """load(), read again only when the file has changed. A broken file raises ConfigError
    (the caller decides whether to keep using the last good one)."""
    p = Path(p) if p else path()
    mtime = p.stat().st_mtime_ns if p.exists() else None
    hit = _cache.get(p)
    if hit is None or hit[0] != mtime:
        hit = (mtime, load(p))
        _cache[p] = hit
    return hit[1]


def set_values(updates: dict, p: Path | str | None = None) -> Config:
    """Change settings in the file, keeping its comments and layout:
    set_values({("scoring", "pass_mark"): 55}). The result is checked before anything is
    written; a bad value raises ConfigError and leaves the file as it was."""
    import tomlkit
    p = Path(p) if p else path()
    doc = tomlkit.parse(p.read_text(encoding="utf-8-sig"))
    for keys, value in updates.items():
        t = doc
        for k in keys[:-1]:
            t = t[k]
        t[keys[-1]] = value
    text = tomlkit.dumps(doc)
    cfg = parse(tomllib.loads(text), p)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return cfg


def parse(raw: dict, p: Path | str = "<config>") -> Config:
    c = _Checker()
    c.keys(raw, "file", ("version", "arm", "controls", "view", "depth_cues", "timing", "tolerances",
                         "scoring", "texts", "reach_target", "materials", "objects", "markers", "phases"),
           required=("arm", "objects", "phases"))
    if raw.get("version", 1) != 1:
        c.err("version", f"this program reads version 1, the file says {raw.get('version')!r}")

    # ---- arm
    a = raw.get("arm", {})
    c.keys(a, "arm", ("model", "side", "base", "home_deg", "min_height", "max_speed", "max_turn_deg",
                      "servo_stiffness"))
    side = c.text(a, "side", "arm", "right", ("left", "right"))
    arm = Arm(model=c.text(a, "model", "arm", "ur5e", ("ur5e",)), side=side,
              base=tuple(c.vec(a, "base", "arm", 2, [0.0, 0.36 if side == "left" else -0.36])),
              home=[math.radians(x) for x in c.vec(a, "home_deg", "arm", 6,
                                                   [0.7448, -92.3608, 111.956, -109.5495, -90.0002, 0.7448])],
              min_height=c.num(a, "min_height", "arm", 0.02, lo=0.0, hi=0.3),
              max_speed=c.num(a, "max_speed", "arm", 1.0, lo=0.0, hi=3.0, lo_open=True),
              max_turn=math.radians(c.num(a, "max_turn_deg", "arm", 180, lo=0.0, hi=720, lo_open=True)),
              servo_stiffness=c.num(a, "servo_stiffness", "arm", 1.0, lo=0.1, hi=10))
    if not (TABLE_X[0] <= arm.base[0] <= TABLE_X[1] and TABLE_Y[0] <= arm.base[1] <= TABLE_Y[1]):
        c.err("arm.base", f"{arm.base} is off the table")

    ctl = raw.get("controls", {})
    c.keys(ctl, "controls", ("motion_scale",))
    motion_scale = c.num(ctl, "motion_scale", "controls", 1.5, lo=0.1, hi=10)

    # ---- view and depth cues
    v = raw.get("view", {})
    c.keys(v, "view", ("eye", "look_at", "fov_deg"))
    view = dict(eye=c.vec(v, "eye", "view", 3, [-0.30, 0.0, 1.52]),
                look=c.vec(v, "look_at", "view", 3, [0.47, -0.02, 0.75]),
                fov=c.num(v, "fov_deg", "view", 62, lo=20, hi=120))
    if view["eye"] == view["look"]:
        c.err("view", "eye and look_at are the same point")
    dc = raw.get("depth_cues", {})
    c.keys(dc, "depth_cues", ("enabled", "grab_height", "color", "grab_color"))
    depth_cues = dict(enabled=c.flag(dc, "enabled", "depth_cues", True),
                      grab_height=c.num(dc, "grab_height", "depth_cues", 0.06, lo=0.0, hi=0.5),
                      color=_hex(c.color(dc, "color", "depth_cues", [1.0, 0.824, 0.29, 1])),
                      grab_color=_hex(c.color(dc, "grab_color", "depth_cues", [0.298, 0.816, 0.49, 1])))

    # ---- timing, tolerances, scoring, texts
    t = raw.get("timing", {})
    c.keys(t, "timing", ("countdown_s", "pair_hold_s"))
    timing = dict(countdown=c.num(t, "countdown_s", "timing", 3.0, lo=0, hi=30),
                  pair_hold=c.num(t, "pair_hold_s", "timing", 1.0, lo=0, hi=30))
    tl = raw.get("tolerances", {})
    tol_keys = dict(reach=0.03, slot_distance=0.016, slot_angle_deg=15.0, rest_speed=0.03, rest_height=0.008,
                    held_lift=0.02, held_distance=0.08, container_margin=0.01, flat_within_deg=25.842,
                    upright_within_deg=19.948)
    c.keys(tl, "tolerances", tuple(tol_keys))
    tol = {k: c.num(tl, k, "tolerances", dv, lo=0.0, lo_open=True, hi=90 if k.endswith("_deg") else 1.0)
           for k, dv in tol_keys.items()}
    tol["slot_angle"] = math.radians(tol.pop("slot_angle_deg"))
    tol["flat_cos"] = math.cos(math.radians(tol.pop("flat_within_deg")))
    tol["upright_cos"] = math.cos(math.radians(tol.pop("upright_within_deg")))
    sc = raw.get("scoring", {})
    c.keys(sc, "scoring", ("steps", "time", "grasp", "drops_to_zero", "pass_mark", "must_complete"))
    scoring = dict(steps=c.num(sc, "steps", "scoring", 60, lo=0), time=c.num(sc, "time", "scoring", 20, lo=0),
                   grasp=c.num(sc, "grasp", "scoring", 20, lo=0),
                   drops_to_zero=c.num(sc, "drops_to_zero", "scoring", 3, lo=0, lo_open=True),
                   pass_mark=c.num(sc, "pass_mark", "scoring", 50, lo=0, hi=100),
                   must_complete=sc.get("must_complete", ["reach"]))
    if not (isinstance(scoring["must_complete"], list) and all(isinstance(x, str) for x in scoring["must_complete"])):
        c.err("scoring.must_complete", "should be a list of phase keys in quotes, e.g. [\"reach\"]")
        scoring["must_complete"] = []
    if all(isinstance(scoring[k], (int, float)) for k in ("steps", "time", "grasp")) \
            and abs(scoring["steps"] + scoring["time"] + scoring["grasp"] - 100) > 1e-6:
        c.err("scoring", f"steps + time + grasp should add up to 100 (a task's score is out of 100), "
                         f"they add up to {scoring['steps'] + scoring['time'] + scoring['grasp']}")
    tx = raw.get("texts", {})
    text_keys = dict(well_done="Well done!", time_up="Time is up.", next="Next:", phone_connected="Phone connected")
    c.keys(tx, "texts", tuple(text_keys))
    texts = {k: c.text(tx, k, "texts", dv) for k, dv in text_keys.items()}
    rt = raw.get("reach_target", {})
    c.keys(rt, "reach_target", ("diameter", "color"))
    reach_target = dict(radius=c.num(rt, "diameter", "reach_target", 0.044, lo=0.005, hi=0.2) / 2,
                        rgba=c.color(rt, "color", "reach_target", [0.2, 1.0, 0.4, 0.55]))

    # ---- materials
    materials = {}
    for name, m in (raw.get("materials") or {}).items():
        w = f"materials.{name}"
        if not c.keys(m, w, ("color", "texture", "specular", "shininess", "roughness", "reflectance", "metallic")):
            continue
        materials[name] = Material(name=name, rgba=c.color(m, "color", w, [1, 1, 1, 1]),
                                   texture=c.text(m, "texture", w, None, ("wood_floor", "laminate")),
                                   specular=c.num(m, "specular", w, 0.3, lo=0, hi=1),
                                   shininess=c.num(m, "shininess", w, 0.3, lo=0, hi=1),
                                   roughness=c.num(m, "roughness", w, None, lo=0, hi=1),
                                   reflectance=c.num(m, "reflectance", w, 0.0, lo=0, hi=1),
                                   metallic=c.num(m, "metallic", w, None, lo=0, hi=1))

    def mat_ref(d, w):
        name = c.text(d, "material", w, None)
        if name is not None and name not in materials:
            hint = difflib.get_close_matches(name, list(materials), n=1)
            c.err(f"{w}.material", f"no material '{name}' in [materials]" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
            return None
        return name

    # ---- objects
    objects, groups = {}, {}
    obj_keys = {"box": ("size",), "sphere": ("diameter",), "cylinder": ("diameter", "height"),
                "cup": ("diameter", "height", "wall", "bottom", "segments")}
    common = ("shape", "color", "material", "start", "start_in", "yaw_deg", "pitch_deg", "mass", "friction",
              "condim", "graspable", "count")
    for gname, o in (raw.get("objects") or {}).items():
        w = f"objects.{gname}"
        if not isinstance(o, dict):
            c.err(w, "should be a table")
            continue
        shape = c.text(o, "shape", w, None, SHAPES)
        if shape is None:
            if "shape" not in o:
                c.err(w, f"'shape' is missing (one of {', '.join(SHAPES)})")
            continue
        c.keys(o, w, common + obj_keys[shape])
        if shape == "box":
            s = c.vec(o, "size", w, 3, None, lo=0.002)
            half = [x / 2 for x in s] if s else [0.02] * 3
            if s is None and "size" not in o:
                c.err(w, "'size' is missing: [length x, width y, height z] in metres")
        else:
            dia = c.num(o, "diameter", w, None, lo=0.004, hi=0.5)
            if dia is None and "diameter" not in o:
                c.err(w, "'diameter' is missing")
            r = (dia or 0.04) / 2
            if shape == "sphere":
                half = [r, 0, 0]
            else:
                h = c.num(o, "height", w, None, lo=0.004, hi=0.5)
                if h is None and "height" not in o:
                    c.err(w, "'height' is missing")
                half = [r, (h or 0.04) / 2, 0]
        count = c.num(o, "count", w, 1, lo=1, hi=50, integer=True)
        start_in = c.text(o, "start_in", w, None)
        start = c.vec(o, "start", w, 2, None)
        if start is None and start_in is None:
            c.err(w, "'start' is missing: [x, y] on the table (or start_in = \"<cup>\")")
        pitch = math.radians(c.num(o, "pitch_deg", w, 0.0, lo=-180, hi=180))
        if pitch and shape != "box":
            c.err(f"{w}.pitch_deg", "only boxes can start tipped over")
        mat = mat_ref(o, w)
        base = dict(shape=shape, half=half, rgba=c.color(o, "color", w, [0.7, 0.7, 0.7, 1]), material=mat,
                    start=tuple(start) if start else (0.0, 0.0), yaw=math.radians(c.num(o, "yaw_deg", w, 0.0)),
                    pitch=pitch, mass=c.num(o, "mass", w, 0.08, lo=0.001, hi=5.0),
                    friction=c.vec(o, "friction", w, 3, [1.2, 0.01, 0.001], lo=0.0),
                    condim=c.num(o, "condim", w, 6 if shape in ("sphere", "cup") else 4, integer=True),
                    graspable=c.flag(o, "graspable", w, True), start_in=start_in, group=gname)
        if base["condim"] not in (1, 3, 4, 6):
            c.err(f"{w}.condim", f"should be 1, 3, 4 or 6, got {base['condim']}")
        if shape == "cup":
            base.update(cup_wall=c.num(o, "wall", w, 0.004, lo=0.001, hi=0.05),
                        cup_bottom=c.num(o, "bottom", w, 0.006, lo=0.001, hi=0.05),
                        cup_segments=c.num(o, "segments", w, 20, lo=6, hi=64, integer=True))
            if base["cup_wall"] * 2 >= half[0] * 2:
                c.err(f"{w}.wall", "the wall is thicker than the cup is wide")
        names = [gname] if count == 1 else [f"{gname}_{i}" for i in range(count)]
        groups[gname] = names
        for i, n in enumerate(names):
            if n in objects:
                c.err(w, f"the name '{n}' is used twice")
            objects[n] = Obj(name=n, index_in_group=i, **base)
    for o in objects.values():
        w = f"objects.{o.group}"
        if o.start_in is not None:
            cup = objects.get(o.start_in)
            if cup is None or cup.shape != "cup":
                c.err(f"{w}.start_in", f"'{o.start_in}' should be an object with shape = \"cup\"")
            elif o.shape != "sphere":
                c.err(f"{w}.start_in", "only balls (spheres) can start inside a cup")
            else:
                o.start = cup.start
        elif o.index_in_group == 0:
            c.on_table(o.start, f"{w}.start", arm.base)

    # ---- markers
    markers = {}
    kind_keys = {"zone": ("size", "line", "color"),
                 "slot": ("object", "margin", "symmetry_deg", "line", "color"),
                 "bin": ("inner_size", "wall_height", "wall", "color"),
                 "bowl": ("inner_diameter", "wall_height", "wall", "bottom", "segments", "material"),
                 "holder": ("post_spacing", "post_size", "base_size", "material")}
    for name, mk in (raw.get("markers") or {}).items():
        w = f"markers.{name}"
        if not isinstance(mk, dict):
            c.err(w, "should be a table")
            continue
        kind = c.text(mk, "kind", w, None, MARKER_KINDS)
        if kind is None:
            if "kind" not in mk:
                c.err(w, f"'kind' is missing (one of {', '.join(MARKER_KINDS)})")
            continue
        c.keys(mk, w, ("kind", "pos", "yaw_deg") + kind_keys[kind])
        pos = c.vec(mk, "pos", w, 2, None)
        if pos is None:
            if "pos" not in mk:
                c.err(w, "'pos' is missing: [x, y] on the table")
            pos = [0.0, 0.0]
        else:
            c.on_table(pos, f"{w}.pos", arm.base)
        pr = {}
        if kind == "zone":
            sz = c.vec(mk, "size", w, 2, [0.09, 0.09], lo=0.005)
            pr.update(half=[x / 2 for x in sz], line=c.num(mk, "line", w, 0.01, lo=0.001, hi=0.05) / 2)
        elif kind == "slot":
            ob = c.text(mk, "object", w, None)
            if ob is None or ob not in objects:
                c.err(f"{w}.object", f"should name the object that goes in this outline, got {ob!r}")
            elif objects[ob].shape != "box":
                c.err(f"{w}.object", "outlines (slots) are for boxes")
            pr.update(object=ob, margin=c.num(mk, "margin", w, 0.006, lo=0, hi=0.1),
                      symmetry=math.radians(c.num(mk, "symmetry_deg", w, 360, lo=1, hi=360)),
                      line=c.num(mk, "line", w, 0.008, lo=0.001, hi=0.05) / 2)
        elif kind == "bin":
            pr.update(half=c.num(mk, "inner_size", w, 0.11, lo=0.02, hi=0.5) / 2,
                      wall_height=c.num(mk, "wall_height", w, 0.04, lo=0.005, hi=0.3),
                      wall=c.num(mk, "wall", w, 0.005, lo=0.001, hi=0.05))
        elif kind == "bowl":
            pr.update(r_in=c.num(mk, "inner_diameter", w, 0.15, lo=0.02, hi=0.6) / 2,
                      wall_height=c.num(mk, "wall_height", w, 0.035, lo=0.005, hi=0.3),
                      wall=c.num(mk, "wall", w, 0.004, lo=0.001, hi=0.05),
                      bottom=c.num(mk, "bottom", w, 0.005, lo=0.001, hi=0.05),
                      segments=c.num(mk, "segments", w, 28, lo=6, hi=64, integer=True))
        elif kind == "holder":
            pr.update(post_spacing=c.vec(mk, "post_spacing", w, 2, [0.034, 0.11], lo=0.0),
                      post_half=[x / 2 for x in c.vec(mk, "post_size", w, 3, [0.007, 0.007, 0.05], lo=0.001)],
                      base_half=[x / 2 for x in c.vec(mk, "base_size", w, 3, [0.044, 0.11, 0.006], lo=0.001)])
        markers[name] = Marker(name=name, kind=kind, pos=tuple(pos),
                               yaw=math.radians(c.num(mk, "yaw_deg", w, 0.0)),
                               rgba=c.color(mk, "color", w, None), material=mat_ref(mk, w) if kind in ("bowl", "holder") else None,
                               p=pr)

    # ---- phases
    phases = []
    raw_phases = raw.get("phases") or []
    if not isinstance(raw_phases, list):
        c.err("phases", "should be a list of [[phases]] tables")
        raw_phases = []
    for i, ph in enumerate(raw_phases):
        w = f"phases[{i + 1}]" + (f" ({ph.get('key')})" if isinstance(ph, dict) and ph.get("key") else "")
        if not isinstance(ph, dict):
            c.err(w, "should be a table")
            continue
        kind = c.text(ph, "kind", w, "task", ("pair", "task", "done"))
        allowed = ("key", "kind", "title", "text")
        if kind == "task":
            allowed += ("scored", "time_limit_s", "rotation", "objects", "markers", "steps")
        c.keys(ph, w, allowed, required=("key", "title", "text") + (("time_limit_s", "steps") if kind == "task" else ()))
        key = c.text(ph, "key", w, f"phase{i}")
        if key and not key.replace("_", "").isalnum():
            c.err(f"{w}.key", "use letters, digits and _ only")
        P = Phase(key=key, kind=kind, title=c.text(ph, "title", w, key), text=c.text(ph, "text", w, ""))
        if kind == "task":
            P.scored = c.flag(ph, "scored", w, False)
            P.limit = float(c.num(ph, "time_limit_s", w, 120.0, lo=5, hi=3600))
            P.rotation = c.text(ph, "rotation", w, "yaw", ("yaw", "full"))
            for o in ph.get("objects", []):
                if o in groups:
                    P.objects += groups[o]
                else:
                    hint = difflib.get_close_matches(str(o), list(groups), n=1)
                    c.err(f"{w}.objects", f"no object '{o}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
            for mname in ph.get("markers", []):
                if mname not in markers:
                    hint = difflib.get_close_matches(str(mname), list(markers), n=1)
                    c.err(f"{w}.markers", f"no marker '{mname}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))
                else:
                    P.markers.append(mname)
            P.graspable = [n for n in P.objects if objects[n].graspable]
            P.steps = _steps(c, ph.get("steps", []), w, P, objects, groups, markers, arm)
            _check_overlaps(c, w, P, objects)
        phases.append(P)
    keys = [p.key for p in phases]
    for k in set(keys):
        if keys.count(k) > 1:
            c.err("phases", f"the key '{k}' is used by {keys.count(k)} phases")
    if phases and phases[0].kind != "pair":
        c.err("phases", "the first phase should be kind = \"pair\" (connect the phone)")
    if phases and phases[-1].kind != "done":
        c.err("phases", "the last phase should be kind = \"done\"")
    if sum(p.kind == "pair" for p in phases) > 1 or sum(p.kind == "done" for p in phases) > 1:
        c.err("phases", "only one \"pair\" and one \"done\" phase")
    if phases and not any(p.scored for p in phases):
        c.err("phases", "no phase is scored = true")
    for k in scoring["must_complete"]:
        if k not in keys:
            c.err("scoring.must_complete", f"no phase '{k}'")

    if c.problems:
        raise ConfigError(p, c.problems)
    return Config(path=Path(p), arm=arm, motion_scale=motion_scale, view=view, depth_cues=depth_cues,
                  timing=timing, tol=tol, scoring=scoring, texts=texts, reach_target=reach_target,
                  materials=materials, objects=objects, groups=groups, markers=markers, phases=phases)


def _steps(c: _Checker, raw_steps, w, P: Phase, objects, groups, markers, arm) -> list[Step]:
    out: list[Step] = []
    if not isinstance(raw_steps, list) or not raw_steps:
        c.err(f"{w}.steps", "should be a list of steps, e.g. [{ type = \"reach\", target = [0.5, 0.1, 0.1] }]")
        return out
    fields = {"reach": ("target",), "place_in_zone": ("object", "zone"), "place_in_slot": ("object", "slot"),
              "place_in_bin": ("object", "bin"), "remove_from": ("object", "holder", "distance"),
              "lay_flat": ("object", "zone"), "pour_into": ("objects", "container", "count"),
              "set_upright": ("object",)}
    marker_kind = {"zone": ("zone",), "slot": ("slot",), "bin": ("bin",), "holder": ("holder",),
                   "container": ("bowl", "bin")}
    for i, s in enumerate(raw_steps):
        ws = f"{w}.steps[{i + 1}]"
        if not isinstance(s, dict):
            c.err(ws, "should be { type = \"...\", ... }")
            continue
        typ = c.text(s, "type", ws, None, STEP_TYPES)
        if typ is None:
            if "type" not in s:
                c.err(ws, f"'type' is missing (one of {', '.join(STEP_TYPES)})")
            continue
        req = fields[typ]
        c.keys(s, ws, ("type", "id", "label", "after") + req, required=tuple(k for k in req if k != "distance"))
        st = Step(id=c.text(s, "id", ws, f"step{i + 1}"), type=typ, label=c.text(s, "label", ws, ""),
                  after=c.text(s, "after", ws, None))
        if "object" in req:
            st.object = c.text(s, "object", ws, None)
            if st.object is not None and st.object not in P.objects:
                c.err(f"{ws}.object", f"'{st.object}' isn't one of this phase's objects ({', '.join(P.objects) or 'none'})")
                st.object = None
        for mk in ("zone", "slot", "bin", "holder", "container"):
            if mk in req:
                st.marker = c.text(s, mk, ws, None)
                if st.marker is not None:
                    if st.marker not in P.markers:
                        c.err(f"{ws}.{mk}", f"'{st.marker}' isn't one of this phase's markers ({', '.join(P.markers) or 'none'})")
                        st.marker = None
                    elif markers[st.marker].kind not in marker_kind[mk]:
                        c.err(f"{ws}.{mk}", f"'{st.marker}' is a {markers[st.marker].kind}, not a {' or '.join(marker_kind[mk])}")
                        st.marker = None
        if typ == "reach":
            tg = c.vec(s, "target", ws, 3, None)
            if tg is not None:
                c.on_table(tg[:2], f"{ws}.target", arm.base)
                if not 0.0 < tg[2] < 0.6:
                    c.err(f"{ws}.target", f"the height above the table (3rd number) should be 0 to 0.6 m, got {tg[2]}")
                st.target = (tg[0], tg[1], TABLE_Z + tg[2])
        if typ == "place_in_slot" and st.marker and st.object and markers[st.marker].p["object"] != st.object:
            c.err(f"{ws}.slot", f"the outline '{st.marker}' is made for '{markers[st.marker].p['object']}', not '{st.object}'")
        if typ == "remove_from":
            st.distance = c.num(s, "distance", ws, 0.07, lo=0.01, hi=1.0)
        if typ == "pour_into":
            g = c.text(s, "objects", ws, None)
            if g is not None and g not in groups:
                c.err(f"{ws}.objects", f"no object group '{g}'")
            elif g is not None:
                st.objects = groups[g]
                missing = [n for n in st.objects if n not in P.objects]
                if missing:
                    c.err(f"{ws}.objects", f"'{g}' isn't in this phase's objects")
            st.count = c.num(s, "count", ws, 1, lo=1, integer=True)
            if st.objects and st.count > len(st.objects):
                c.err(f"{ws}.count", f"{st.count} needed but there are only {len(st.objects)}")
        if not st.label:
            st.label = _default_label(st)
        out.append(st)
    ids = [s.id for s in out]
    for s in out:
        if ids.count(s.id) > 1:
            c.err(f"{w}.steps", f"the id '{s.id}' is used twice")
        if s.after is not None and (s.after not in ids or ids.index(s.after) >= ids.index(s.id)):
            c.err(f"{w}.steps ({s.id})", f"after = '{s.after}' should name an earlier step's id")
        if s.after is not None and not s.latched:
            c.err(f"{w}.steps ({s.id})", f"'after' only works with {', '.join(LATCHED)}")
    types = {s.type for s in out}
    if "reach" in types and len(types) > 1:
        c.err(f"{w}.steps", "reach steps can't be mixed with other steps in one phase")
    return out


def _default_label(st: Step) -> str:
    o = (st.object or "").replace("_", " ")
    return {"reach": "touch the target", "place_in_zone": f"{o} in the zone", "place_in_slot": f"{o} in its outline",
            "place_in_bin": f"{o} in its bin", "remove_from": f"{o} out of the holder", "lay_flat": f"{o} flat",
            "pour_into": "pour", "set_upright": f"{o} upright"}[st.type]


def _bound_r(o: Obj) -> float:
    return math.hypot(o.half[0], o.half[1]) if o.shape == "box" else o.half[0]


def _check_overlaps(c: _Checker, w, P: Phase, objects):
    """Objects of one phase must not start inside each other (balls in a cup are fine)."""
    obs = [objects[n] for n in P.objects if objects[n].start_in is None]
    for i, a in enumerate(obs):
        for b in obs[i + 1:]:
            d = math.hypot(a.start[0] - b.start[0], a.start[1] - b.start[1])
            if d < _bound_r(a) + _bound_r(b) - 0.002:
                c.err(w, f"'{a.name}' and '{b.name}' start overlapping ({d * 100:.1f} cm apart)")

