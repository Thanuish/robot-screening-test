"""Ring on pegs: three rings stand on their edge in small wooden holders in a row in front of the
robot; straight ahead of each, further out, is a peg of the same colour on a grey post. The pegs
lean at different angles, so each ring needs a different tilt of the hand. Take a ring by the top
of its tube, lift it, tilt it so its hole lines up with the peg, slide it on from the tip, let go.

    on its peg   the ring's centre c on the peg's axis: with B the peg's base and u its direction,
                 t = (c - B).u between -1 cm and the peg's length, and the distance from the axis
                 |c - B - t u| < hole radius - peg radius + on_peg_margin
    done         on its peg, not held, slower than rest_speed (counts while it stays so; a ring
                 that falls off un-counts)
    held         lifted held_lift above where it would rest, its centre within held_distance of
                 the pinch point (where it rests depends on how it lies: see rest_height)
    grasps       a GRAB that lifted a ring (at most one per GRAB); drops: a ring that left the
                 hand while GRAB was still closed, and not onto its peg
    rounds       all three done and nothing held: the round counts, the rings jump back into
                 their holders, the next round starts; until the time is up (or end_after_rounds)

Checked check_hz times a second. Score 0..100: 80 x min(1, rings hung / (3 x target_rounds)) +
20 x grasp efficiency (grasps / GRAB presses, minus 1/3 per drop). Settings: [rings] in
config/buzzwire.toml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import base
from . import scene as SC
from .base import RobotGame
from .config import TABLE_Z, Config

PARK = [-1.2, -1.0]
PEG_FLOOR = 0.008               # [m] a peg without a post starts this far above the table


@dataclass
class RingsLevel:
    key: str
    title: str
    text: str
    scored: bool
    limit: float
    rotation: str
    target_rounds: int
    end_after_rounds: int
    start_held: bool = False


def _rot(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def peg_dir(r) -> np.ndarray:
    """The peg's direction: tilted r['tilt'] from straight up toward r['yaw']."""
    t, y = r["tilt"], r["yaw"]
    return np.array([np.sin(t) * np.cos(y), np.sin(t) * np.sin(y), np.cos(t)])


def peg_base(r) -> np.ndarray:
    return np.array([*r["peg"], TABLE_Z + (r["height"] if r["height"] > 0 else PEG_FLOOR)])


# the ring stands on its edge: its axis (local z) along the world's x (pitched 90 degrees)
RING_START_QUAT = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


class RingsGame(RobotGame):
    KIND = "rings"
    TITLE = "Ring on pegs"

    def __init__(self, cfg: Config, levels: list[str] | None = None, out_dir: Path | None = None, player: str = ""):
        self.rc = cfg.games["rings"]
        rc = self.rc
        self.r_out = rc["outer_diameter"] / 2
        self.r_mid = self.r_out - rc["thickness"] / 2
        self.hole = self.r_mid - rc["thickness"] / 2
        self.pegs = [(peg_base(r), peg_dir(r)) for r in rc["rings"]]
        lvls = [RingsLevel(**lv) for lv in rc["levels"] if levels is None or lv["key"] in levels]
        super().__init__(cfg, lvls, out_dir, player)

    # ------------------------------------------------------------ scene
    def build_scene(self, lv: RingsLevel):
        rc = self.rc
        spec = SC.new_spec(self.cfg, f"rings_{lv.key}")
        wb = spec.worldbody
        grey = [0.78, 0.8, 0.83, 1]
        t = rc["thickness"]
        n = rc["segments"]
        for i, r in enumerate(rc["rings"]):
            # the holder: 4 posts, two each side of the ring's plane (the ring's axis is along x)
            hb = wb.add_body()
            hb.name = f"holder_{i}"
            hb.pos = [*r["ring"], TABLE_Z]
            (px, py, ph), (sa, sb) = rc["holder_post"], rc["holder_spacing"]
            for k, (dx, dy) in enumerate([(-sa / 2, -sb / 2), (-sa / 2, sb / 2), (sa / 2, -sb / 2), (sa / 2, sb / 2)]):
                SC._box(hb, f"holder_{i}_post{k}", [dx, dy, ph / 2], [px / 2, py / 2, ph / 2], material="wood")
            bx, by, bz = rc["holder_base"]
            SC._box(hb, f"holder_{i}_base", [0, 0, bz / 2], [bx / 2, by / 2, bz / 2], material="wood", collide=False)
            # the peg: a base plate, a post, the peg leaning out of its top
            pb = wb.add_body()
            pb.name = f"pegpost_{i}"
            pb.pos = [*r["peg"], TABLE_Z]
            ax, ay, at = rc["peg_plate"]
            SC._box(pb, f"pegplate_{i}", [0, 0, at / 2], [ax / 2, ay / 2, at / 2], grey, material="metal")
            if r["height"] > 0:
                g = pb.add_geom()
                g.name, g.type = f"pegpost_{i}", mujoco.mjtGeom.mjGEOM_CYLINDER
                g.size = [max(1.6 * rc["peg_radius"], 0.012), r["height"] / 2, 0]
                g.pos, g.rgba, g.material = [0, 0, r["height"] / 2], grey, "metal"
            B, u = self.pegs[i]
            g = SC._capsule(wb, f"peg_{i}", B, B + rc["peg_length"] * u, rc["peg_radius"], r["rgba"])
            g.friction, g.condim = list(rc["peg_friction"]), 3
            # the ring: a torus of short overlapping capsules (so it really hangs and collides)
            body = wb.add_body()
            body.name = f"ring_{i}"
            body.pos = [*PARK, 0.2 + 0.15 * i]
            j = body.add_freejoint()
            j.name = f"ring_{i}_joint"
            step = 2 * np.pi / n
            ext = 0.25 * step                                  # (each capsule a little longer: they overlap)
            for k in range(n):
                a0, a1 = k * step - ext, (k + 1) * step + ext
                p0 = [self.r_mid * np.cos(a0), self.r_mid * np.sin(a0), 0]
                p1 = [self.r_mid * np.cos(a1), self.r_mid * np.sin(a1), 0]
                g = SC._capsule(body, f"ring_{i}_{k}", p0, p1, t / 2, r["rgba"])
                g.mass, g.condim, g.friction = rc["mass"] / n, 6, list(rc["friction"])
        model = spec.compile()
        sc = SC.Scene(model=model, cfg=self.cfg, level=lv)
        SC.bind_robot(sc)
        self.adr = [model.jnt_qposadr[model.joint(f"ring_{i}_joint").id] for i in range(len(rc["rings"]))]
        self.dof = [model.jnt_dofadr[model.joint(f"ring_{i}_joint").id] for i in range(len(rc["rings"]))]
        return sc

    def new_metrics(self, lv: RingsLevel) -> dict:
        return dict(rounds=0, hung=0, hung_now=0, round_times=[], grasp_attempts=0, grasps=0, drops=0,
                    progress=0.0, events=[])

    def setup_level(self, lv: RingsLevel):
        n = len(self.rc["rings"])
        self.holding = [False] * n
        self.done = [False] * n
        self._prev_cmd, self._credited = 0.0, True
        self._round_t0 = 0.0
        self._every = max(1, round(self.cfg.physics["control_hz"] / self.rc["check_hz"]))
        self._k = 0
        self._reset_rings()
        mujoco.mj_forward(self.m, self.d)

    def _reset_rings(self):
        """Every ring back in its holder, standing on its edge."""
        d = self.d
        for i, r in enumerate(self.rc["rings"]):
            a, dof = self.adr[i], self.dof[i]
            d.qpos[a:a + 3] = [*r["ring"], TABLE_Z + self.r_out + 0.0005]
            d.qpos[a + 3:a + 7] = RING_START_QUAT
            d.qvel[dof:dof + 6] = 0
        self.done = [False] * len(self.rc["rings"])

    # ------------------------------------------------------------ state of things
    def ring(self, i):
        """(centre, rotation matrix, speed) of ring i."""
        a = self.adr[i]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[a + 3:a + 7])
        return self.d.qpos[a:a + 3].copy(), R.reshape(3, 3), float(np.linalg.norm(self.d.qvel[self.dof[i]:self.dof[i] + 3]))

    def rest_height(self, R) -> float:
        """How high above the table a ring's centre is when it rests, lying as it does now:
        az = |axis . z|; flat (az 1): half the thickness; on its edge (az 0): the outer radius."""
        t = self.rc["thickness"]
        az = min(1.0, abs(float(R[2, 2])))
        return az * t / 2 + np.sqrt(max(0.0, 1 - az * az)) * self.r_mid + (1 - az) * t / 2

    def on_peg(self, i, c=None) -> bool:
        c = self.ring(i)[0] if c is None else c
        B, u = self.pegs[i]
        t = float((c - B) @ u)
        if not (-0.01 <= t <= self.rc["peg_length"]):
            return False
        return float(np.linalg.norm(c - B - t * u)) < self.hole - self.rc["peg_radius"] + self.rc["on_peg_margin"]

    def held(self) -> bool:
        return any(getattr(self, "holding", []))

    # ------------------------------------------------------------ the rules
    def update(self, t: float):
        self._k += 1
        if self._k % self._every:
            return
        mt, rc, lv = self.metrics, self.rc, self.level
        cmd = self.teleop.gripper
        if cmd >= 0.5 > self._prev_cmd:                     # GRAB: an attempt (it can count one grasp)
            mt["grasp_attempts"] += 1
            self._credited = False
        self._prev_cmd = cmd
        ee = self.d.site_xpos[self.sc.ee_site]
        names = [r["name"] for r in rc["rings"]]
        for i in range(len(rc["rings"])):
            c, R, v = self.ring(i)
            held = c[2] - TABLE_Z > self.rest_height(R) + rc["held_lift"] and np.linalg.norm(c - ee) < rc["held_distance"]
            onp = self.on_peg(i, c)
            if held and not self.holding[i] and not self._credited:
                mt["grasps"] += 1
                self._credited = True
                mt["events"].append([round(t, 2), "grasp", names[i]])
            elif self.holding[i] and not held and cmd >= 0.95 and not onp:
                mt["drops"] += 1
                mt["events"].append([round(t, 2), "drop", names[i]])
            self.holding[i] = held
            done = onp and not held and v < rc["rest_speed"]
            if done != self.done[i]:
                mt["events"].append([round(t, 2), "hung" if done else "off", names[i]])
            self.done[i] = done
        mt["hung_now"] = sum(self.done)
        if all(self.done) and not self.held():                # a round
            mt["rounds"] += 1
            mt["hung"] += len(self.done)
            mt["round_times"].append(round(t - self._round_t0, 2))
            mt["events"].append([round(t, 2), "round", mt["rounds"]])
            self._round_t0 = t
            self._reset_rings()
            mt["hung_now"] = 0
            self.message = f"Round {mt['rounds'] + 1}!"
            if lv.end_after_rounds and mt["rounds"] >= lv.end_after_rounds:
                self.end_level(True)
                return
        mt["progress"] = min(1.0, (mt["hung"] + mt["hung_now"]) / (len(rc["rings"]) * lv.target_rounds))

    # ------------------------------------------------------------ results and screens
    def score(self, mt: dict, lv: RingsLevel) -> float:
        n = len(self.rc["rings"])
        rings = min(1.0, (mt["hung"] + mt["hung_now"]) / (n * lv.target_rounds))
        eff = mt["grasps"] / mt["grasp_attempts"] * max(0.0, 1 - mt["drops"] / 3) if mt["grasp_attempts"] else 0.0
        return round(80 * rings + 20 * min(1.0, eff), 1)

    def status(self) -> dict:
        st = super().status()
        st["title"] = f"{self.level.title} · round {self.metrics['rounds'] + 1}"
        return st

    def hud(self) -> list:
        mt = self.metrics
        return [dict(key="round", label="Round", value=mt["rounds"] + 1, alert=False),
                dict(key="hung", label="Rings hung", value=mt["hung"] + mt["hung_now"], alert=False),
                dict(key="drops", label="Drops", value=mt["drops"], alert=mt["drops"] > 0)]

    def steps(self) -> list:
        """The rings hung this round (shown as three ticks)."""
        return [dict(label=f"{r['name'].capitalize()} ring", done=bool(d))
                for r, d in zip(self.rc["rings"], getattr(self, "done", []))]

    def level_geometry(self) -> dict:
        return dict(kind="rings", level=self.level.key, r_mid=self.r_mid, tube=self.rc["thickness"] / 2,
                    view=dict(eye=self.rc["view_eye"], look=self.rc["view_look"]),
                    rings=[dict(name=r["name"], rgba=r["rgba"]) for r in self.rc["rings"]])


def _rotation_between(a, b):
    """The shortest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def play_rings(bot, rounds: int = 2):
    """Hang every ring on its peg, `rounds` rounds (then wait for the level to end)."""
    from .bot import DOWN, _rz
    g = bot.g
    lv, key = g.level, g.level.key
    rc = g.rc
    bot.start()
    R0 = _rz(np.pi / 2) @ DOWN                         # fingers closing across the tube at the ring's top
    Z = TABLE_Z

    def changed(r0):
        return g.finished or g.level.key != key or g.metrics["rounds"] != r0

    for _ in range(rounds):
        r0 = g.metrics["rounds"]
        for i in range(len(rc["rings"])):
            if changed(r0):
                break
            c, R, _ = g.ring(i)
            top = c + np.array([0, 0, g.r_mid])           # the tube's middle at the ring's top
            bot.move(top + [0, 0, 0.10], R0, 1.4)
            bot.move(top, R0, 1.0)
            bot.gripper(True)
            bot.move(np.r_[top[:2], Z + 0.32], R0, 1.2)
            if changed(r0):
                break
            c, R, _ = g.ring(i)
            p = g.d.site_xpos[g.sc.ee_site].copy()
            off = c - p                                     # the ring's centre from the pinch point
            B, u = g.pegs[i]
            v = u if u[0] >= 0 else -u
            a = R[:, 2] if R[:, 2] @ v >= 0 else -R[:, 2]
            Q = _rotation_between(a, v)
            R1 = Q @ R0
            pinch_for = lambda centre: centre - Q @ off
            tip = B + rc["peg_length"] * u
            for centre, dur in ((tip + 0.05 * u + [0, 0, 0.10], 2.0), (tip + 0.05 * u, 1.2),
                                (B + 0.45 * rc["peg_length"] * u, 1.5)):
                bot.move(pinch_for(centre), R1, dur)
                if changed(r0):
                    break
            if changed(r0):
                break
            bot.gripper(False, 0.8)
            back = g.teleop.target_pos - 0.07 * R1[:, 2]      # pull the fingers back along the gripper's axis
            if changed(r0):
                break
            bot.move(back, R1, 0.8)
            if changed(r0):
                break
            bot.move(back + [0, 0, 0.10], R1, 0.8)
        while not changed(r0):                              # (the round counts when they're all still)
            bot.run_ticks(1)
            if g.state != "play":
                break
        print(f"    round {g.metrics['rounds']}: {g.metrics['hung']} rings hung, round times {g.metrics['round_times']}")
        if g.finished or g.level.key != key:
            return
    bot.finish_level(key)


def replay(cfg: Config, input_log: dict, levels: list[str] | None = None) -> RingsGame:
    return base.replay(RingsGame, cfg, input_log, levels=levels)
