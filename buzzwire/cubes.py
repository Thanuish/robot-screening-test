"""Cube tasks: the screening test's pick-and-place tasks on the shared robot core, with the new
graphics and phone controller. The tasks themselves come from config/screening.toml (the same
objects, outlines, bins, tolerances and scoring as the screening test), by default:

    pick      practice: the yellow cube into the green square
    shapes    task 1: each shape into its outline, turned to match
    sort      task 2: each object into the bin of its colour

The wand-free version of the old test: the gripper starts open; GRAB / RELEASE on the phone close
and open it. Score per task as in the screening test (TEST_TELEOP.screen_game.score_task):
steps completed, time left, grasp efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from TEST_TELEOP import screen_scene as SS
from TEST_TELEOP.screen_game import _ang_err, _yaw, score_task
from vrteleop import textures as T

from . import base
from . import scene as SC
from .base import RobotGame
from .config import TABLE_Z, Config


@dataclass
class CubeLevel:
    key: str
    title: str
    text: str
    scored: bool
    limit: float
    rotation: str
    phase: object                # TEST_TELEOP.config.Phase
    start_held: bool = False


class CubesGame(RobotGame):
    KIND = "cubes"
    TITLE = "Cube Tasks"

    def __init__(self, cfg: Config, levels: list[str] | None = None, out_dir: Path | None = None, player: str = ""):
        self.tc = SS.CFG                                  # the screening test's configuration
        gcfg = cfg.games["cubes"]
        lvls = []
        for key in gcfg["phases"]:
            if levels is not None and key not in levels:
                continue
            ph = self.tc.phase(key)
            lvls.append(CubeLevel(key=ph.key, title=ph.title, text=ph.text, scored=ph.scored, limit=ph.limit,
                                  rotation=ph.rotation, phase=ph))
        super().__init__(cfg, lvls, out_dir, player)

    # ------------------------------------------------------------ scene
    def build_scene(self, lv: CubeLevel):
        spec = SC.new_spec(self.cfg, f"cubes_{lv.key}")
        T.add_texture(spec, "wood_floor", T.wood_floor())   # (the test's wooden things use it)
        SS.add(spec)                                       # every object and marker of the test, parked
        model = spec.compile()
        sc = SC.Scene(model=model, cfg=self.cfg, level=lv)
        SC.bind_robot(sc)
        self.info = SimpleNamespace(model=model, obj_qpos_adr={
            n: model.jnt_qposadr[model.joint(f"{n}_joint").id] for n in self.tc.objects})
        self.dof = {n: model.jnt_dofadr[model.joint(f"{n}_joint").id] for n in self.tc.objects}
        return sc

    def new_metrics(self, lv: CubeLevel) -> dict:
        return dict(steps_done=0, steps_total=len(lv.phase.steps), grasp_attempts=0, grasps=0, drops=0, wrong_bin=0,
                    progress=0.0, events=[])

    def setup_level(self, lv: CubeLevel):
        ph, d = lv.phase, self.d
        SS.park_all(self.info, d)
        for n in ph.objects:
            SS.place(self.info, d, n)
        for mk in ph.markers:
            SS.show_marker(self.info, d, mk)
        mujoco.mj_forward(self.m, d)
        self.holding = {n: False for n in ph.graspable}
        self.done = {st.id: False for st in ph.steps}
        self._was_done = {}
        self.in_bin = {st.object: None for st in ph.steps if st.type == "place_in_bin"}
        self._prev_cmd = 0.0
        self._credited = True

    # ------------------------------------------------------------ state of things
    def _obj(self, n):
        a = self.info.obj_qpos_adr[n]
        q = self.d.qpos
        return q[a:a + 3], q[a + 3:a + 7], float(np.linalg.norm(self.d.qvel[self.dof[n]:self.dof[n] + 3]))

    def _axes(self, n):
        _, q, _ = self._obj(n)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)

    def _resting_height(self, n):
        o = self.tc.objects[n]
        if o.shape == "box":
            k = int(np.argmax(np.abs(self._axes(n)[2])))
            return TABLE_Z + o.half[k] + 0.001
        return SS.rest_z(n)

    def _local(self, p, mk):
        dx, dy = p[0] - mk.pos[0], p[1] - mk.pos[1]
        c, s = np.cos(mk.yaw), np.sin(mk.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _in_zone(self, n, zone):
        mk = self.tc.markers[zone]
        x, y = self._local(self._obj(n)[0], mk)
        return abs(x) < mk.p["half"][0] and abs(y) < mk.p["half"][1]

    def _slot_ok(self, n, slot):
        mk, tol = self.tc.markers[slot], self.tc.tol
        p, q, _ = self._obj(n)
        return (np.hypot(p[0] - mk.pos[0], p[1] - mk.pos[1]) < tol["slot_distance"]
                and _ang_err(_yaw(q), mk.yaw, mk.p["symmetry"]) < tol["slot_angle"])

    def _bin_of(self, n):
        p, _, _ = self._obj(n)
        for name in self.level.phase.markers:
            mk = self.tc.markers[name]
            if mk.kind != "bin" or p[2] >= TABLE_Z + mk.p["wall_height"] + self.tc.tol["container_margin"]:
                continue
            x, y = self._local(p, mk)
            if abs(x) < mk.p["half"] and abs(y) < mk.p["half"]:
                return name
        return None

    def _resting(self, n, speed):
        p, _, _ = self._obj(n)
        return (speed < self.tc.tol["rest_speed"] and abs(p[2] - SS.rest_z(n)) < self.tc.tol["rest_height"]
                and not self.holding.get(n, False))

    def _goal_ok(self, n):
        for st in self.level.phase.steps:
            if st.object == n and ((st.type == "place_in_zone" and self._in_zone(n, st.marker))
                                   or (st.type == "place_in_slot" and self._slot_ok(n, st.marker))
                                   or (st.type == "place_in_bin" and self._bin_of(n) == st.marker)):
                return True
        return False

    def held(self) -> bool:
        return any(getattr(self, "holding", {}).values())

    # ------------------------------------------------------------ the rules, every tick
    def update(self, t: float):
        mt, tol, tel = self.metrics, self.tc.tol, self.teleop
        cmd = tel.gripper
        if cmd >= 0.5 > self._prev_cmd:                  # GRAB: a grasp attempt (it can count one grasp)
            mt["grasp_attempts"] += 1
            self._credited = False
        self._prev_cmd = cmd
        ee = self.d.site_xpos[self.sc.ee_site]
        for n in self.holding:                           # grasps and drops
            p, _, _ = self._obj(n)
            mid = p + SS.center_offset(n) * self._axes(n)[:, 2]
            near = np.linalg.norm(mid - ee)
            held = p[2] > self._resting_height(n) + tol["held_lift"] and near < tol["held_distance"]
            if held and not self.holding[n] and not self._credited:
                mt["grasps"] += 1
                self._credited = True
                mt["events"].append([round(t, 2), "grasp", n])
            elif self.holding[n] and not held and cmd >= 0.95 and not self._goal_ok(n) and near >= tol["held_distance"]:
                mt["drops"] += 1
                mt["events"].append([round(t, 2), "drop", n])
            self.holding[n] = held
        for st in self.level.phase.steps:                # the steps
            n = st.object
            if st.type == "place_in_zone":
                self.done[st.id] = self._in_zone(n, st.marker) and self._resting(n, self._obj(n)[2])
            elif st.type == "place_in_slot":
                self.done[st.id] = self._slot_ok(n, st.marker) and self._resting(n, self._obj(n)[2])
            elif st.type == "place_in_bin":
                b = None if self.holding.get(n) else self._bin_of(n)
                if b is not None and b != self.in_bin[n] and b != st.marker:
                    mt["wrong_bin"] += 1
                self.in_bin[n] = b
                self.done[st.id] = b == st.marker
        for st in self.level.phase.steps:                # (when each step got done / undone, for the results)
            if self.done[st.id] != self._was_done.get(st.id, False):
                mt["events"].append([round(t, 2), "step" if self.done[st.id] else "undone", st.label])
                self._was_done[st.id] = self.done[st.id]
        mt["steps_done"] = sum(self.done.values())
        mt["progress"] = mt["steps_done"] / max(1, mt["steps_total"])
        if mt["steps_done"] == mt["steps_total"]:
            self.end_level(True)

    # ------------------------------------------------------------ results and screens
    def score(self, mt: dict, lv: CubeLevel) -> float:
        return score_task(mt, lv.limit, self.tc.scoring)

    def hud(self) -> list:
        mt = self.metrics
        return [dict(key="placed", label="Placed", value=f"{mt['steps_done']}/{mt['steps_total']}", alert=False),
                dict(key="drops", label="Drops", value=mt["drops"], alert=False)]

    def steps(self) -> list:
        return [dict(label=st.label, done=bool(self.done.get(st.id))) for st in self.level.phase.steps]

    def level_geometry(self) -> dict:
        return dict(kind="cubes", level=self.level.key)


def replay(cfg: Config, input_log: dict, levels: list[str] | None = None) -> CubesGame:
    return base.replay(CubesGame, cfg, input_log, levels=levels)
