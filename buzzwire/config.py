"""Robot Buzz Wire's configuration: config/buzzwire.toml (or the file in BUZZWIRE_CONFIG). This
module reads it, checks it (every problem at once, with hints) and turns it into plain Python
values in SI units (radians, half sizes where MuJoCo wants them).
"""
from __future__ import annotations

import difflib
import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .spline import Wire

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "config" / "buzzwire.toml"
TABLE_Z = 0.75
TABLE_X, TABLE_Y = (-0.15, 0.95), (-0.75, 0.75)
ROBOTS = {"ur5e": {"reach": 0.85}}
GRIPPERS = ("robotiq_2f85",)


class ConfigError(Exception):
    def __init__(self, path, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"{path}: {len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass
class Level:
    key: str
    title: str
    text: str
    scored: bool
    limit: float
    rotation: str
    points: np.ndarray           # control points in the world
    loop_radius: float
    start_held: bool = True      # the wand starts in the gripper, its loop around the wire's start
    finish: str = "reach_end"    # "reach_end": done at the far sleeve; "hang": set it on the end hook, let go
    wire: Wire = field(repr=False, default=None)


@dataclass
class Config:
    path: Path
    robot: dict
    controls: dict
    physics: dict
    view: dict
    board: dict
    wire: dict
    wand: dict
    scoring: dict
    levels: list
    games: dict = field(default_factory=dict)    # the other games' settings: "cubes", "tower"

    def level(self, key) -> Level:
        return next(lv for lv in self.levels if lv.key == key)

    def public(self) -> dict:
        """What the web pages need."""
        return dict(view=self.view, table_z=TABLE_Z,
                    levels=[dict(key=lv.key, title=lv.title, scored=lv.scored) for lv in self.levels])


class _Checker:
    def __init__(self):
        self.problems: list[str] = []

    def err(self, where, msg):
        self.problems.append(f"{where}: {msg}")

    def keys(self, d, where, allowed, required=()):
        if not isinstance(d, dict):
            self.err(where, "should be a table")
            return False
        for k in d:
            if k not in allowed:
                hint = difflib.get_close_matches(k, list(allowed), n=1)
                self.err(f"{where}.{k}", "unknown setting" + (f" (did you mean '{hint[0]}'?)" if hint
                                                              else f" (allowed: {', '.join(allowed)})"))
        for k in required:
            if k not in d:
                self.err(where, f"'{k}' is missing")
        return True

    def num(self, d, k, where, default, lo=None, hi=None, integer=False):
        if k not in d:
            return default
        v = d[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
            self.err(f"{where}.{k}", f"should be {'a whole number' if integer else 'a number'}, got {v!r}")
            return default
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            self.err(f"{where}.{k}", f"should be between {lo} and {hi}, got {v}")
            return default
        return v

    def vec(self, d, k, where, n, default, lo=None):
        if k not in d:
            return default
        v = d[k]
        if (not isinstance(v, list) or len(v) != n
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v)):
            self.err(f"{where}.{k}", f"should be a list of {n} numbers, got {v!r}")
            return default
        if lo is not None and any(x < lo for x in v):
            self.err(f"{where}.{k}", f"all values should be at least {lo}, got {v}")
            return default
        return [float(x) for x in v]

    def text(self, d, k, where, default, choices=None):
        if k not in d:
            return default
        v = d[k]
        if not isinstance(v, str) or (choices and v not in choices):
            self.err(f"{where}.{k}", f"should be one of {', '.join(choices)}; got {v!r}" if choices
                     else f"should be text in quotes, got {v!r}")
            return default
        return v

    def flag(self, d, k, where, default):
        if k not in d:
            return default
        if not isinstance(d[k], bool):
            self.err(f"{where}.{k}", f"should be true or false, got {d[k]!r}")
            return default
        return d[k]


def path() -> Path:
    p = os.environ.get("BUZZWIRE_CONFIG", "").strip()
    return Path(p).resolve() if p else DEFAULT_PATH


def load(p=None) -> Config:
    p = Path(p) if p else path()
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ConfigError(p, ["the file doesn't exist"]) from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(p, [f"not valid TOML: {e}"]) from None
    return parse(raw, p)


def parse(raw: dict, p="<config>") -> Config:
    c = _Checker()
    c.keys(raw, "file", ("version", "robot", "controls", "physics", "view", "board", "wire", "wand",
                         "scoring", "levels", "cubes", "tower", "flow"), required=("levels",))
    if raw.get("version", 1) != 1:
        c.err("version", f"this program reads version 1, the file says {raw.get('version')!r}")

    r = raw.get("robot", {})
    c.keys(r, "robot", ("model", "gripper", "base", "home_deg", "max_speed", "max_turn_deg", "joint_speed_deg",
                        "table_clearance", "servo_stiffness"))
    robot = dict(model=c.text(r, "model", "robot", "ur5e", tuple(ROBOTS)),
                 gripper=c.text(r, "gripper", "robot", "robotiq_2f85", GRIPPERS),
                 base=c.vec(r, "base", "robot", 2, [0.0, 0.0]),
                 home=np.radians(c.vec(r, "home_deg", "robot", 6, [0.0, -100.0, 110.0, -100.0, -90.0, 0.0])),
                 max_speed=c.num(r, "max_speed", "robot", 0.3, lo=0.01, hi=3.0),
                 max_turn=math.radians(c.num(r, "max_turn_deg", "robot", 90, lo=1, hi=720)),
                 joint_speed=math.radians(c.num(r, "joint_speed_deg", "robot", 120, lo=1, hi=720)),
                 table_clearance=c.num(r, "table_clearance", "robot", 0.03, lo=0.0, hi=0.3),
                 servo_stiffness=c.num(r, "servo_stiffness", "robot", 1.0, lo=0.2, hi=10.0))
    ct = raw.get("controls", {})
    c.keys(ct, "controls", ("motion_scale", "smoothing_s", "filter", "min_cutoff_hz", "beta", "rot_beta", "glide_s",
                                     "jitter_buffer_s", "jitter_buffer_max_s"))
    controls = dict(motion_scale=c.num(ct, "motion_scale", "controls", 1.0, lo=0.1, hi=10),
                    smoothing=c.num(ct, "smoothing_s", "controls", 0.12, lo=0.0, hi=1.0),
                    filter=c.text(ct, "filter", "controls", "simple", ("simple", "one_euro")),
                    min_cutoff_hz=c.num(ct, "min_cutoff_hz", "controls", 1.0, lo=0.05, hi=30),
                    beta=c.num(ct, "beta", "controls", 20.0, lo=0.0, hi=1000),
                    rot_beta=c.num(ct, "rot_beta", "controls", 2.0, lo=0.0, hi=1000),
                    glide=c.num(ct, "glide_s", "controls", 0.025, lo=0.0, hi=0.5),
                    jitter_buffer=c.num(ct, "jitter_buffer_s", "controls", 0.03, lo=0.0, hi=0.5),
                    jitter_buffer_max=c.num(ct, "jitter_buffer_max_s", "controls", 0.15, lo=0.0, hi=0.5))
    ph = raw.get("physics", {})
    c.keys(ph, "physics", ("timestep", "control_hz", "touch_debounce_s"))
    physics = dict(timestep=c.num(ph, "timestep", "physics", 0.002, lo=0.0005, hi=0.01),
                   control_hz=c.num(ph, "control_hz", "physics", 100, lo=10, hi=1000),
                   touch_debounce=c.num(ph, "touch_debounce_s", "physics", 0.05, lo=0.0, hi=1.0))
    steps = 1 / (physics["control_hz"] * physics["timestep"])
    if abs(steps - round(steps)) > 1e-6:
        c.err("physics", f"1 / (control_hz x timestep) should be a whole number of physics steps, got {steps:.3f}")
    v = raw.get("view", {})
    c.keys(v, "view", ("eye", "look_at", "fov_deg", "arm_opacity"))
    view = dict(eye=c.vec(v, "eye", "view", 3, [-0.75, -0.15, 1.95]), look=c.vec(v, "look_at", "view", 3, [0.5, 0.0, 0.8]),
                fov=c.num(v, "fov_deg", "view", 45, lo=20, hi=120),
                arm_opacity=c.num(v, "arm_opacity", "view", 0.35, lo=0.05, hi=1.0))
    b = raw.get("board", {})
    c.keys(b, "board", ("center", "size"))
    board = dict(center=c.vec(b, "center", "board", 2, [0.5, 0.0]), size=c.vec(b, "size", "board", 3, [0.16, 0.56, 0.02], lo=0.005))
    w = raw.get("wire", {})
    c.keys(w, "wire", ("radius", "sleeve_length", "sleeve_radius", "post_radius"))
    wire = dict(radius=c.num(w, "radius", "wire", 0.0015, lo=0.0005, hi=0.01),
                sleeve_length=c.num(w, "sleeve_length", "wire", 0.045, lo=0.01, hi=0.2),
                sleeve_radius=c.num(w, "sleeve_radius", "wire", 0.004, lo=0.001, hi=0.02),
                post_radius=c.num(w, "post_radius", "wire", 0.006, lo=0.002, hi=0.03))
    if wire["sleeve_radius"] <= wire["radius"]:
        c.err("wire.sleeve_radius", "should be thicker than the wire")
    wd = raw.get("wand", {})
    c.keys(wd, "wand", ("loop_radius", "loop_thickness", "loop_segments", "handle_size", "handle_mass", "loop_mass",
                        "friction"))
    wand = dict(loop_radius=c.num(wd, "loop_radius", "wand", 0.02, lo=0.006, hi=0.08),
                loop_thickness=c.num(wd, "loop_thickness", "wand", 0.004, lo=0.001, hi=0.02),
                loop_segments=c.num(wd, "loop_segments", "wand", 24, lo=8, hi=64, integer=True),
                handle_size=c.vec(wd, "handle_size", "wand", 3, [0.12, 0.014, 0.014], lo=0.005),
                handle_mass=c.num(wd, "handle_mass", "wand", 0.04, lo=0.001, hi=1.0),
                loop_mass=c.num(wd, "loop_mass", "wand", 0.01, lo=0.001, hi=1.0),
                friction=c.vec(wd, "friction", "wand", 3, [1.5, 0.02, 0.002], lo=0.0))
    s = raw.get("scoring", {})
    sk = dict(progress=40, no_touches=25, touch_time=15, time=10, smoothness=10, touches_to_zero=10,
              touch_seconds_to_zero=5.0, smooth_jerk=5.0, jerky_jerk=60.0, pass_mark=50)
    c.keys(s, "scoring", tuple(sk))
    scoring = {k: c.num(s, k, "scoring", dv, lo=0) for k, dv in sk.items()}
    total = sum(scoring[k] for k in ("progress", "no_touches", "touch_time", "time", "smoothness"))
    if abs(total - 100) > 1e-6:
        c.err("scoring", f"progress + no_touches + touch_time + time + smoothness should add up to 100, got {total}")
    if scoring["jerky_jerk"] <= scoring["smooth_jerk"]:
        c.err("scoring", "jerky_jerk should be larger than smooth_jerk")
    for k in ("touches_to_zero", "touch_seconds_to_zero"):
        if scoring[k] <= 0:
            c.err(f"scoring.{k}", "should be more than 0")

    levels = []
    raw_levels = raw.get("levels") or []
    reach = ROBOTS.get(robot["model"], {"reach": 0.85})["reach"]
    bc = np.array([*board["center"], TABLE_Z + board["size"][2]])
    for i, lv in enumerate(raw_levels if isinstance(raw_levels, list) else []):
        wl = f"levels[{i + 1}]" + (f" ({lv.get('key')})" if isinstance(lv, dict) and lv.get("key") else "")
        if not c.keys(lv, wl, ("key", "title", "text", "scored", "time_limit_s", "rotation", "points", "loop_radius",
                               "start_held", "finish"),
                      required=("key", "title", "points")):
            continue
        pts = lv.get("points")
        if (not isinstance(pts, list) or len(pts) < 2
                or any(not isinstance(q, list) or len(q) != 3 or any(not isinstance(x, (int, float)) for x in q) for q in pts)):
            c.err(f"{wl}.points", "should be a list of at least 2 points, each [forward, left, up]")
            continue
        P = np.array(pts, float) + bc
        L = Level(key=c.text(lv, "key", wl, f"level{i}"), title=c.text(lv, "title", wl, ""), text=c.text(lv, "text", wl, ""),
                  scored=c.flag(lv, "scored", wl, True), limit=float(c.num(lv, "time_limit_s", wl, 120, lo=10, hi=3600)),
                  rotation=c.text(lv, "rotation", wl, "yaw", ("none", "yaw", "full")), points=P,
                  loop_radius=c.num(lv, "loop_radius", wl, wand["loop_radius"], lo=0.006, hi=0.08),
                  start_held=c.flag(lv, "start_held", wl, True),
                  finish=c.text(lv, "finish", wl, "reach_end", ("reach_end", "hang")))
        for k, q in enumerate(pts):
            x, y, z = P[k]
            if not (TABLE_X[0] <= x <= TABLE_X[1] and TABLE_Y[0] <= y <= TABLE_Y[1]):
                c.err(f"{wl}.points[{k + 1}]", f"{q} is off the table")
            elif math.hypot(x - robot["base"][0], y - robot["base"][1]) > reach - 0.15:
                c.err(f"{wl}.points[{k + 1}]", f"{q} is too far from the robot (keep within {reach - 0.15:.2f} m of its base)")
            if q[2] < L.loop_radius + 0.03:
                c.err(f"{wl}.points[{k + 1}]", f"{q}: too low, the loop wouldn't clear the board (up >= {L.loop_radius + 0.03:.3f})")
        L.wire = Wire(P)
        if L.wire.min_bend_radius() < 1.5 * L.loop_radius:
            c.err(f"{wl}.points", f"the wire bends too tightly for the loop (tightest bend {L.wire.min_bend_radius() * 100:.1f} cm; "
                                  f"keep it above {1.5 * L.loop_radius * 100:.1f} cm)")
        if L.wire.length < 2 * wire["sleeve_length"] + 0.05:
            c.err(f"{wl}.points", "the wire is too short")
        levels.append(L)
    keys = [lv.key for lv in levels]
    for k in set(keys):
        if keys.count(k) > 1:
            c.err("levels", f"the key '{k}' is used twice")
    if not levels:
        c.err("levels", "at least one [[levels]] entry is needed")
    elif not any(lv.scored for lv in levels):
        c.err("levels", "no level is scored = true")
    games = dict(cubes=_cubes(c, raw.get("cubes", {})), tower=_tower(c, raw.get("tower", {}), robot))
    games["flow"] = _flow(c, raw.get("flow", {}), levels, games)
    if c.problems:
        raise ConfigError(p, c.problems)
    return Config(path=Path(p), robot=robot, controls=controls, physics=physics, view=view, board=board,
                  wire=wire, wand=wand, scoring=scoring, levels=levels, games=games)


def _cubes(c: _Checker, raw: dict) -> dict:
    """[cubes]: which tasks of config/screening.toml, and how low the gripper may go."""
    c.keys(raw, "cubes", ("phases", "table_clearance"))
    phases = raw.get("phases", ["pick", "shapes", "sort"])
    if not (isinstance(phases, list) and phases and all(isinstance(x, str) for x in phases)):
        c.err("cubes.phases", "should be a list of phase keys from config/screening.toml, e.g. [\"pick\", \"shapes\"]")
        phases = ["pick", "shapes", "sort"]
    return dict(phases=phases, table_clearance=c.num(raw, "table_clearance", "cubes", 0.015, lo=0.0, hi=0.2))


def _color(c: _Checker, v, where):
    if isinstance(v, str) and len(v) == 7 and v.startswith("#"):
        try:
            return [int(v[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]
        except ValueError:
            pass
    c.err(where, f"should be a colour \"#rrggbb\", got {v!r}")
    return [0.7, 0.7, 0.7, 1.0]


def _flow(c: _Checker, raw: dict, wire_levels: list, games: dict) -> dict:
    """[flow]: the candidate's test: practice tasks, how often, the assessment's tasks."""
    c.keys(raw, "flow", ("practice", "max_practice_rounds", "assessment", "card_seconds", "default_controls"))
    known = {"wire": [lv.key for lv in wire_levels], "tower": [lv["key"] for lv in games["tower"]["levels"]],
             "cubes": None}                       # (cubes: the phases of config/screening.toml, checked when used)

    def tasks(key, default):
        out = []
        for t in raw.get(key, default):
            if not isinstance(t, str) or ":" not in t:
                c.err(f"flow.{key}", f"{t!r}: should be \"game:level\", e.g. \"wire:curves\"")
                continue
            g, lv = t.split(":", 1)
            if g not in known:
                c.err(f"flow.{key}", f"{t!r}: no game '{g}' (wire, cubes, tower)")
            elif known[g] is not None and lv not in known[g]:
                c.err(f"flow.{key}", f"{t!r}: game '{g}' has no level '{lv}' ({', '.join(known[g])})")
            else:
                out.append((g, lv))
        return out
    return dict(practice=tasks("practice", ["wire:straight", "tower:practice"]),
                max_practice_rounds=c.num(raw, "max_practice_rounds", "flow", 3, lo=1, hi=20, integer=True),
                assessment=tasks("assessment", ["cubes:shapes", "cubes:sort", "wire:curves", "tower:cubes"]),
                card_seconds=c.num(raw, "card_seconds", "flow", 8, lo=0, hi=60),
                default_controls=c.text(raw, "default_controls", "flow", "motion", ("motion", "touch", "choice")))


def _tower(c: _Checker, raw: dict, robot: dict) -> dict:
    """[tower]: the stacking game: the platform, where blocks appear, the levels' blocks."""
    c.keys(raw, "tower", ("table_clearance", "platform_center", "platform_size", "supply", "levels",
                          "block_mass", "friction"))
    out = dict(table_clearance=c.num(raw, "table_clearance", "tower", 0.015, lo=0.0, hi=0.2),
               platform_center=c.vec(raw, "platform_center", "tower", 2, [0.52, 0.06]),
               platform_size=c.vec(raw, "platform_size", "tower", 3, [0.14, 0.14, 0.02], lo=0.005),
               supply=c.vec(raw, "supply", "tower", 2, [0.42, -0.18]),
               block_mass=c.num(raw, "block_mass", "tower", 0.06, lo=0.005, hi=1.0),
               friction=c.vec(raw, "friction", "tower", 3, [1.0, 0.01, 0.001], lo=0.0), levels=[])
    for name, xy in (("platform_center", out["platform_center"]), ("supply", out["supply"])):
        if math.hypot(xy[0] - robot["base"][0], xy[1] - robot["base"][1]) > ROBOTS[robot["model"]]["reach"] - 0.2:
            c.err(f"tower.{name}", f"{xy} is too far from the robot")
    for i, lv in enumerate(raw.get("levels", []) or []):
        w = f"tower.levels[{i + 1}]"
        if not c.keys(lv, w, ("key", "title", "text", "scored", "time_limit_s", "rotation", "target_height", "blocks"),
                      required=("key", "title", "blocks")):
            continue
        blocks = []
        for j, b in enumerate(lv.get("blocks", [])):
            wb = f"{w}.blocks[{j + 1}]"
            if not c.keys(b, wb, ("size", "color", "count"), required=("size",)):
                continue
            size = c.vec(b, "size", wb, 3, [0.05, 0.05, 0.05], lo=0.015)
            if max(size[:2]) > 0.16 or min(size[:2]) > 0.075:
                c.err(f"{wb}.size", "the gripper opens 8.5 cm: one side seen from above must be at most 7.5 cm")
            col = _color(c, b.get("color", "#d9d9d9"), f"{wb}.color")
            for _ in range(c.num(b, "count", wb, 1, lo=1, hi=30, integer=True)):
                blocks.append(dict(size=size, rgba=col))
        if not blocks:
            c.err(w, "no blocks")
        out["levels"].append(dict(key=c.text(lv, "key", w, f"tower{i}"), title=c.text(lv, "title", w, ""),
                                  text=c.text(lv, "text", w, ""), scored=c.flag(lv, "scored", w, True),
                                  limit=float(c.num(lv, "time_limit_s", w, 180, lo=10, hi=3600)),
                                  rotation=c.text(lv, "rotation", w, "yaw", ("none", "yaw", "full")),
                                  target=c.num(lv, "target_height", w, 0.2, lo=0.02, hi=1.0), blocks=blocks))
    if not out["levels"]:
        c.err("tower", "at least one [[tower.levels]] entry is needed")
    return out
