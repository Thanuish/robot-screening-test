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
                         "scoring", "levels"), required=("levels",))
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
    c.keys(ct, "controls", ("motion_scale", "smoothing_s"))
    controls = dict(motion_scale=c.num(ct, "motion_scale", "controls", 1.0, lo=0.1, hi=10),
                    smoothing=c.num(ct, "smoothing_s", "controls", 0.1, lo=0.0, hi=1.0))
    ph = raw.get("physics", {})
    c.keys(ph, "physics", ("timestep", "control_hz", "touch_debounce_s"))
    physics = dict(timestep=c.num(ph, "timestep", "physics", 0.002, lo=0.0005, hi=0.01),
                   control_hz=c.num(ph, "control_hz", "physics", 100, lo=10, hi=1000),
                   touch_debounce=c.num(ph, "touch_debounce_s", "physics", 0.05, lo=0.0, hi=1.0))
    steps = 1 / (physics["control_hz"] * physics["timestep"])
    if abs(steps - round(steps)) > 1e-6:
        c.err("physics", f"1 / (control_hz x timestep) should be a whole number of physics steps, got {steps:.3f}")
    v = raw.get("view", {})
    c.keys(v, "view", ("eye", "look_at", "fov_deg"))
    view = dict(eye=c.vec(v, "eye", "view", 3, [1.15, -0.30, 1.22]), look=c.vec(v, "look_at", "view", 3, [0.47, 0.0, 0.87]),
                fov=c.num(v, "fov_deg", "view", 42, lo=20, hi=120))
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
    if c.problems:
        raise ConfigError(p, c.problems)
    return Config(path=Path(p), robot=robot, controls=controls, physics=physics, view=view, board=board,
                  wire=wire, wand=wand, scoring=scoring, levels=levels)
