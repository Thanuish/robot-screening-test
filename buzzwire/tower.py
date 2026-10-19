"""Tower: stack blocks as high as you can, with the robot. Blocks appear one at a time at the
supply spot (a new one when the spot is empty); take each to the wooden platform and stack it.

    height    the top of the blocks resting on the platform (or on each other), above the
              platform; the best height that stood still for a second counts
    falls     a block that was on the tower and ends up on the table
    grasps    grasp attempts (GRAB) and the ones that lifted a block; drops

The level ends when every block has been placed and everything is still, or when time is up.
Score 0..100: 60 x best height / target height + 20 x (1 - falls / 3) + 20 x grasp efficiency.
Settings: [tower] in config/buzzwire.toml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import base
from . import scene as SC
from .base import RobotGame
from .config import TABLE_Z, Config

REST_SPEED = 0.03              # [m/s] a block slower than this is still
STABLE_S = 1.0                 # a height counts once everything has been still this long
HELD_LIFT = 0.015              # a block this far above where it rested, near the gripper = held
HELD_DIST = 0.06
SUPPLY_CLEAR = 0.07            # the supply spot is empty when no block is within this (horizontally)
PARK_Z = -1.0


@dataclass
class TowerLevel:
    key: str
    title: str
    text: str
    scored: bool
    limit: float
    rotation: str
    target: float
    blocks: list = field(default_factory=list)
    start_held: bool = False


class TowerGame(RobotGame):
    KIND = "tower"
    TITLE = "Tower"

    def __init__(self, cfg: Config, levels: list[str] | None = None, out_dir: Path | None = None, player: str = ""):
        self.tw = cfg.games["tower"]
        lvls = [TowerLevel(**lv) for lv in self.tw["levels"] if levels is None or lv["key"] in levels]
        super().__init__(cfg, lvls, out_dir, player)

    # ------------------------------------------------------------ scene
    def build_scene(self, lv: TowerLevel):
        tw = self.tw
        spec = SC.new_spec(self.cfg, f"tower_{lv.key}")
        wb = spec.worldbody
        (px, py), (sx, sy, sz) = tw["platform_center"], tw["platform_size"]
        plat = wb.add_body()
        plat.name = "tower_platform"
        plat.pos = [px, py, TABLE_Z + sz / 2]
        g = SC._box(plat, "tower_platform", [0, 0, 0], [sx / 2, sy / 2, sz / 2], material="wood")
        g.friction = list(tw["friction"])
        pad = wb.add_body()                                 # the supply spot: a flat ring of tape
        pad.name = "tower_supply"
        pad.pos = [*tw["supply"], TABLE_Z]
        for k, (dx, dy, hx, hy) in enumerate([(0.035, 0, 0.003, 0.038), (-0.035, 0, 0.003, 0.038),
                                              (0, 0.035, 0.038, 0.003), (0, -0.035, 0.038, 0.003)]):
            SC._box(pad, f"tower_supply_{k}", [dx, dy, 0.0008], [hx, hy, 0.0008], [0.95, 0.75, 0.15, 1], collide=False)
        for i, b in enumerate(lv.blocks):
            body = wb.add_body()
            body.name = f"block_{i}"
            body.pos = [-1.2, -1.0 + 0.15 * i, -PARK_Z]
            j = body.add_freejoint()
            j.name = f"block_{i}_joint"
            gm = SC._box(body, f"block_{i}", [0, 0, 0], [x / 2 for x in b["size"]], b["rgba"])
            gm.mass, gm.condim, gm.friction = self.tw["block_mass"], 4, list(tw["friction"])
        model = spec.compile()
        sc = SC.Scene(model=model, cfg=self.cfg, level=lv)
        SC.bind_robot(sc)
        self.adr = [model.jnt_qposadr[model.joint(f"block_{i}_joint").id] for i in range(len(lv.blocks))]
        self.dof = [model.jnt_dofadr[model.joint(f"block_{i}_joint").id] for i in range(len(lv.blocks))]
        self.half = [np.array(b["size"]) / 2 for b in lv.blocks]
        self.top = TABLE_Z + sz
        return sc

    def new_metrics(self, lv: TowerLevel) -> dict:
        return dict(placed=0, blocks=len(lv.blocks), height_cm=0.0, best_height_cm=0.0, falls=0, grasp_attempts=0,
                    grasps=0, drops=0, progress=0.0, events=[])

    def setup_level(self, lv: TowerLevel):
        d = self.d
        for i in range(len(lv.blocks)):                    # all parked, out of sight
            a = self.adr[i]
            d.qpos[a:a + 3] = [-1.2, -1.0 + 0.15 * i, self.half[i][2] + 0.001]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.spawned = 0
        self.state_of = ["parked"] * len(lv.blocks)        # parked, supply, moving, tower, table
        self.holding = [False] * len(lv.blocks)
        self.rest_h = [0.0] * len(lv.blocks)
        self._spawn()
        mujoco.mj_forward(self.m, d)
        self._prev_cmd, self._credited = 0.0, True
        self._still_since, self._stable_h = None, 0.0

    def _spawn(self):
        """The next block onto the supply spot (if any are left)."""
        if self.spawned >= len(self.level.blocks):
            return
        i = self.spawned
        a, dof = self.adr[i], self.dof[i]
        self.d.qpos[a:a + 3] = [*self.tw["supply"], TABLE_Z + self.half[i][2] + 0.001]
        self.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.d.qvel[dof:dof + 6] = 0
        self.state_of[i] = "supply"
        self.spawned += 1

    # ------------------------------------------------------------ state of things
    def block(self, i):
        a = self.adr[i]
        q = self.d.qpos
        return q[a:a + 3], q[a + 3:a + 7], float(np.linalg.norm(self.d.qvel[self.dof[i]:self.dof[i] + 3]))

    def _half_up(self, i):
        """Half the block's height as it lies now."""
        _, q, _ = self.block(i)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return float(np.abs(R.reshape(3, 3)[2]) @ self.half[i])

    def _over_platform(self, p):
        (px, py), (sx, sy, _) = self.tw["platform_center"], self.tw["platform_size"]
        return abs(p[0] - px) < sx / 2 + 0.02 and abs(p[1] - py) < sy / 2 + 0.02

    def held(self) -> bool:
        return any(getattr(self, "holding", []))

    # ------------------------------------------------------------ the rules, every tick
    def update(self, t: float):
        mt, tel, lv = self.metrics, self.teleop, self.level
        cmd = tel.gripper
        if cmd >= 0.5 > self._prev_cmd:
            mt["grasp_attempts"] += 1
            self._credited = False
        self._prev_cmd = cmd
        ee = self.d.site_xpos[self.sc.ee_site]
        all_still = True
        height = 0.0
        for i in range(self.spawned):
            p, _, v = self.block(i)
            hz = self._half_up(i)
            bottom = p[2] - hz
            near = np.linalg.norm(p - ee) < HELD_DIST + max(self.half[i])
            still = v < REST_SPEED
            if still and not self.holding[i]:
                self.rest_h[i] = p[2]
            held = near and cmd > 0.5 and p[2] > self.rest_h[i] + HELD_LIFT
            if held and not self.holding[i]:
                if not self._credited:
                    mt["grasps"] += 1
                    self._credited = True
                    mt["events"].append([round(t, 2), "grasp", i])
                if self.state_of[i] == "tower":
                    self.state_of[i] = "moving"               # (taken off the tower again)
            elif self.holding[i] and not held and cmd >= 0.95 and not near:
                mt["drops"] += 1
                mt["events"].append([round(t, 2), "drop", i])
            self.holding[i] = held
            if not still or held:
                all_still = False
                continue
            # where it rests: on the platform / the tower, or on the table
            on_tower = self._over_platform(p) and bottom > self.top - 0.006
            if on_tower:
                if self.state_of[i] != "tower":
                    self.state_of[i] = "tower"
                height = max(height, p[2] + hz - self.top)
            elif bottom < TABLE_Z + 0.006 and np.hypot(p[0] - self.tw["supply"][0], p[1] - self.tw["supply"][1]) > SUPPLY_CLEAR:
                if self.state_of[i] == "tower":
                    mt["falls"] += 1
                    mt["events"].append([round(t, 2), "fall", i])
                self.state_of[i] = "table"
        # the next block when the supply spot is empty
        sx, sy = self.tw["supply"]
        if all(np.hypot(self.block(i)[0][0] - sx, self.block(i)[0][1] - sy) > SUPPLY_CLEAR or self.holding[i]
               for i in range(self.spawned)) and not self.held():
            self._spawn()
        mt["placed"] = sum(s == "tower" for s in self.state_of)
        mt["height_cm"] = round(float(height) * 100, 1)
        # the best height that stood still for a second
        if all_still:
            self._still_since = self._still_since if self._still_since is not None else t
            if t - self._still_since >= STABLE_S:
                mt["best_height_cm"] = max(mt["best_height_cm"], mt["height_cm"])
        else:
            self._still_since = None
        mt["progress"] = min(1.0, mt["best_height_cm"] / 100 / lv.target)
        # done: every block out, nothing held, all still for a second, the last one not on the supply spot
        if (self.spawned == len(lv.blocks) and all_still and self._still_since is not None
                and t - self._still_since >= STABLE_S and all(s in ("tower", "table") for s in self.state_of)):
            self.end_level(True)

    # ------------------------------------------------------------ results and screens
    def score(self, mt: dict, lv: TowerLevel) -> float:
        height = min(1.0, mt["best_height_cm"] / 100 / lv.target)
        falls = max(0.0, 1 - mt["falls"] / 3)
        eff = mt["grasps"] / mt["grasp_attempts"] * max(0.0, 1 - mt["drops"] / 3) if mt["grasp_attempts"] else 0.0
        if mt["best_height_cm"] == 0:
            falls = 0.0                                     # (nothing built: no credit for no falls)
        return round(60 * height + 20 * falls + 20 * min(1.0, eff), 1)

    def hud(self) -> list:
        mt = self.metrics
        return [dict(key="height", label="Height", value=f"{mt['height_cm']:.0f} cm", alert=False),
                dict(key="blocks", label="Blocks", value=f"{mt['placed']}/{mt['blocks']}", alert=False),
                dict(key="falls", label="Falls", value=mt["falls"], alert=mt["falls"] > 0)]

    def level_geometry(self) -> dict:
        return dict(kind="tower", level=self.level.key, target_cm=round(self.level.target * 100),
                    platform_center=self.tw["platform_center"], platform_size=self.tw["platform_size"])


def play_tower(bot):
    """Stack every block in the middle of the platform, each squared up on the one below."""
    g = bot.g
    lv, key = g.level, g.level.key
    bot.start()
    px, py = g.tw["platform_center"]
    for i in range(len(lv.blocks)):
        while g.level.key == key and (g.spawned <= i or g.state_of[i] != "supply"):
            bot.run_ticks(1)
        if g.level.key != key:
            break
        top = g.top + sum(2 * g.half[j][2] for j in range(i) if g.state_of[j] == "tower")
        h = g.half[i]
        period = np.pi / 2 if abs(h[0] - h[1]) < 1e-6 else np.pi
        bot.pick_place(i, lambda n: g.block(n)[:2], (px, py), dest_yaw=0.0, period=period,
                       place_z=top + h[2] + 0.004, lift=max(0.12, top - SC.TABLE_Z + 2 * h[2] + 0.06), half=h)
        print(f"    block {i}: height {g.metrics['height_cm']} cm, on the tower {g.metrics['placed']}")
    bot.finish_level(key)


def replay(cfg: Config, input_log: dict, levels: list[str] | None = None) -> TowerGame:
    return base.replay(TowerGame, cfg, input_log, levels=levels)
