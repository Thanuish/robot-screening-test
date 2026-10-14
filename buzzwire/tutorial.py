"""The tutorial: the controls taught one at a time, each step checked before the next one shows.
The scene is the cube tasks' practice (a yellow cube, a green square); the steps:

    move      push the left stick: the gripper moves (10 cm)
    updown    right stick up and down: the gripper rises and lowers (5 cm)
    turn      right stick sideways: the gripper turns (30 degrees)
    above     the open gripper right above the yellow cube
    lower     lowered until the fingers are around the cube
    grab      GRAB, and the cube lifted
    place     the cube in the green square, RELEASE

The tutorial has no time limit and no score; how long each step took is recorded.
"""
from __future__ import annotations

import mujoco
import numpy as np

from TEST_TELEOP import screen_scene as SS

from .cubes import CubesGame
from .config import Config

STEPS = [
    ("move", "Move", "Push the LEFT stick: the gripper moves. Forward on the stick = away from you, into the screen."),
    ("updown", "Up and down", "Push the RIGHT stick up and down: the gripper rises and lowers."),
    ("turn", "Turn", "Push the RIGHT stick left or right: the gripper turns."),
    ("above", "Above the cube", "Move the open gripper right above the yellow cube (watch its shadow on the table)."),
    ("lower", "Lower it", "Lower the gripper with the right stick until its fingers are on both sides of the cube."),
    ("grab", "Grab and lift", "Tap GRAB on your phone, then lift the cube with the right stick."),
    ("place", "Put it down", "Carry the cube over the green square, lower it and tap RELEASE."),
]


class TutorialGame(CubesGame):
    KIND = "tutorial"
    TITLE = "Tutorial"

    def __init__(self, cfg: Config, out_dir=None, player: str = ""):
        super().__init__(cfg, levels=["pick"], out_dir=out_dir, player=player)

    def setup_level(self, lv):
        lv.limit = 1e9                                     # (no time limit)
        lv.scored = False
        lv.title = "Tutorial: the controls"
        lv.text = "Follow the steps. Each one is ticked off when you've done it."
        super().setup_level(lv)
        self.tstep = 0
        self.tdone = {k: None for k, _, _ in STEPS}        # step -> seconds when done
        self._start = None

    def update(self, t: float):
        ee = self.d.site_xpos[self.sc.ee_site].copy()
        if self._start is None:
            self._start = (ee.copy(), self._yaw(), ee[2], ee[2])
        p0, yaw0, zlo, zhi = self._start
        self._start = (p0, yaw0, min(zlo, ee[2]), max(zhi, ee[2]))
        cube, _, _ = self._obj("tut_cube")
        half = self.tc.objects["tut_cube"].half
        near = np.hypot(ee[0] - cube[0], ee[1] - cube[1]) < 0.025
        checks = {
            "move": lambda: np.hypot(ee[0] - p0[0], ee[1] - p0[1]) > 0.10,
            "updown": lambda: self._start[3] - self._start[2] > 0.05,
            "turn": lambda: abs((self._yaw() - yaw0 + np.pi) % (2 * np.pi) - np.pi) > np.radians(30),
            "above": lambda: near and self.teleop.gripper < 0.5,
            "lower": lambda: near and ee[2] < cube[2] + half[2],
            "grab": lambda: self.holding.get("tut_cube", False),
            "place": lambda: False,                        # (the cube task's own check, below)
        }
        while self.tstep < len(STEPS):
            key = STEPS[self.tstep][0]
            if key == "place" or not checks[key]():
                break
            self.tdone[key] = round(t, 1)
            self.tstep += 1
            self.message = "✓ " + STEPS[self.tstep - 1][1]
        super().update(t)                                  # grasps, drops, the cube in the square

    def end_level(self, completed: bool):
        if completed and self.tstep == len(STEPS) - 1:     # the cube is in the square: the last step
            self.tdone["place"] = self.metrics["time_s"]
            self.tstep = len(STEPS)
            self.metrics["tutorial_steps"] = dict(self.tdone)
            super().end_level(True)
        elif not completed:
            super().end_level(False)
        # (the cube landed in the square before the earlier steps were done: keep going)

    def _yaw(self):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.teleop.target_quat)
        fy = R.reshape(3, 3)[:, 1]
        return float(np.arctan2(-fy[0], fy[1]))

    # ------------------------------------------------------------ screens
    def steps(self) -> list:
        return [dict(label=title, done=self.tdone.get(k) is not None) for k, title, _ in STEPS]

    def hud(self) -> list:
        return [dict(key="step", label="Step", value=f"{min(self.tstep + 1, len(STEPS))}/{len(STEPS)}", alert=False)]

    def status(self) -> dict:
        st = super().status()
        st["instruction"] = STEPS[self.tstep][2] if self.tstep < len(STEPS) else ""
        st["progress"] = self.tstep / len(STEPS)
        return st

    def score(self, mt, lv) -> float:
        return 0.0

    def held(self) -> bool:
        return super().held()


def play_tutorial(bot):
    """The tutorial's steps, the way a person would do them (for the checks)."""
    from .bot import DOWN, _rz
    g = bot.g
    key = g.level.key
    bot.start()
    tel = g.teleop
    p0 = tel.target_pos.copy()
    bot.move(p0 + [0.12, 0, 0], DOWN, 1.2)                                # move
    bot.move(p0 + [0.12, 0, 0.07], DOWN, 0.8)                             # up and down
    bot.move(p0 + [0.12, 0, 0.0], _rz(0.7) @ DOWN, 1.0)                   # turn
    cube = g._obj("tut_cube")[0]
    zone = g.tc.markers["zone"]
    bot.pick_place("tut_cube", lambda n: g._obj(n)[:2], zone.pos, half=g.tc.objects["tut_cube"].half,
                   rest_z=SS.rest_z("tut_cube"))
    bot.finish_level(key)
