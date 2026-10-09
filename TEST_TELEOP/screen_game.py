"""The screening test: a candidate controls one robot arm with a phone and does three
manipulation tasks after a short practice. It measures how well they understand the mapping
from their hand to the robot: time, steps completed and efficiency (grasps that work, drops,
how directly they move).

Phases (all times in simulation seconds, so a slow computer doesn't penalise anyone):

    pair       connect the phone (scan the QR code, start the controller)
    reach      practice: touch three glowing targets with the gripper          (not scored)
    pick       practice: put the yellow cube into the green square              (not scored)
    shapes     task 1: put each shape into its outline on the board (turn it)   (scored)
    sort       task 2: put each object into the bin of its colour               (scored)
    tilt       task 3: lay the book flat, pour the balls into the bowl         (scored)
    done

Practice and tasks 1-2 use the phone's position and its turn left / right (the gripper keeps
pointing down). Task 3 uses all six degrees of freedom: the phone's tilt and twist also tilt and
twist the gripper ("rotation": "full" in the phase; the phone reads it from the status).

Each phase is recorded as an episode (data/.../episode_XXXX.hdf5). results.json in the session
folder holds the metrics and scores; the lobby (TEST_TELEOP/lobby.py) reads it.

Score per task, 0..100:  60 x steps completed / steps
                       + 20 x time left / time limit       (only if the task was completed)
                       + 20 x grasp efficiency             (grasps that lifted something /
                                                            grasp attempts, minus 1/3 per drop)
The overall score is the mean of the scored tasks. The practice reach measures path efficiency
(straight-line distance / distance moved), the clearest sign of understanding the mapping.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from vrteleop import scene as S
from vrteleop import server

from . import screen_scene as SS

SIDE = SS.ARM
INTRO_S = 3.0              # countdown before each phase
HELD_LIFT = 0.02           # an object this far above its resting height near the gripper is held
HELD_DIST = 0.08           # ... and its centre within this distance of the pinch point
REACH_TOL = 0.03           # gripper this close to a target = reached
SLOT_TOL = 0.016           # task 1: piece centre this close to its slot ...
SLOT_YAW_TOL = np.radians(15)   # ... and turned to within this angle
REST_SPEED = 0.03          # [m/s] an object slower than this is at rest
MAX_SPEED = 1.0            # [m/s] the gripper's top speed (about a UR5e's; only quick flicks are capped)
MAX_TURN = np.radians(180)  # [rad/s] its top turning speed

PHASES = [
    dict(key="pair", title="Connect your phone", scored=False, limit=None,
         text="Scan the QR code with your Android phone, open the link in Chrome and tap "
              "START CONTROLLER. Hold the phone upright, screen facing you, like a camera."),
    dict(key="reach", title="Practice 1: move", scored=False, limit=150.0,
         text="Hold MOVE and move your phone: the gripper follows your hand. Touch the green "
              "ball with the gripper. Let go of MOVE to reposition your hand."),
    dict(key="pick", title="Practice 2: pick and place", scored=False, limit=150.0,
         text="Bring the open gripper down around the yellow cube, tap GRAB, lift it, put it "
              "into the green square and tap RELEASE. Turn your phone to turn the gripper."),
    dict(key="shapes", title="Task 1: shapes", scored=True, limit=240.0,
         text="Put each shape into its outline on the board. Turn the shapes to match."),
    dict(key="sort", title="Task 2: sort by colour", scored=True, limit=180.0,
         text="Put each object into the bin of its colour."),
    dict(key="tilt", title="Task 3: tilt and pour", scored=True, limit=300.0, rotation="full",
         text="Now the gripper also tilts and twists with your phone. Take the book out of its "
              "holder, turn it flat and lay it on the green mat. Then pour the balls from the cup "
              "into the bowl and set the cup down upright."),
    dict(key="done", title="Finished", scored=False, limit=None,
         text="Thank you! You can close this page."),
]
PHASE_OBJECTS = {"pick": ["tut_cube"], "shapes": list(SS.SLOTS), "sort": list(SS.BINS),
                 "tilt": ["plate", "cup", *SS.BALLS]}
PHASE_MARKERS = {"pick": ["screen_zone"], "shapes": ["screen_board"], "sort": ["screen_bins"],
                 "tilt": ["screen_rack", "screen_mat", "screen_bowl"]}
GRASPABLE = {"tilt": ["plate", "cup"]}     # (the balls are poured, not grasped)
TILT_STEPS = ["book out of the holder", "book flat on the mat", "balls in the bowl", "cup upright again"]
FLAT_COS = 0.9             # task 3: plate normal within ~25 deg of vertical = flat
UPRIGHT_COS = 0.94         # ... cup axis within ~20 deg of vertical = upright
BALLS_NEEDED = 4           # ... of 5 balls in the bowl


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _ang_err(a, b, period):
    d = (a - b) % period
    return min(d, period - d)


def _jsonable(o):
    """numpy numbers in the metrics -> plain Python for results.json."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def score_task(m: dict, limit: float) -> float:
    """0..100 for one scored task (see the module docstring)."""
    steps = m["steps_done"] / max(1, m["steps_total"])
    time_part = max(0.0, 1 - m["time_s"] / limit) if m["completed"] else 0.0
    if m["grasp_attempts"]:
        eff = m["grasps"] / m["grasp_attempts"] * max(0.0, 1 - m["drops"] / 3)
    else:
        eff = 0.0
    return round(60 * steps + 20 * time_part + 20 * min(1.0, eff), 1)


class ScreenGame:
    def __init__(self, sim):
        self.sim = sim                  # vrteleop.server.TeleopSim
        self.m, self.d, self.info = sim.m, sim.d, sim.info
        m = self.m
        self.mocap = {b: m.body_mocapid[m.body(b).id] for b in SS.MARKERS}
        self.dof = {n: m.jnt_dofadr[m.joint(f"{n}_joint").id] for n in SS.OBJECTS}
        self.results_path = Path(sim.session.dir) / "results.json"
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        # a real robot's pace: the gripper follows the phone at most this fast (a flick of the
        # wrist doesn't whip the arm around)
        sim.arms.max_speed, sim.arms.max_turn = MAX_SPEED, MAX_TURN
        # Testing one part: SCREEN_ONLY=<phase key> (e.g. tilt) runs only pairing, that phase,
        # and the end; the lobby sets it for its "test only" codes
        only = os.environ.get("SCREEN_ONLY", "").strip()
        keys = [p["key"] for p in PHASES]
        if only and only not in keys:
            raise ValueError(f"SCREEN_ONLY={only!r}: choose from {', '.join(keys[1:-1])}")
        self.only = only or None
        self.phases = [p for p in PHASES if not only or p["key"] in ("pair", only, "done")]
        self.reset()

    # ------------------------------------------------------------ phases
    def reset(self):
        self.i = 0
        self.metrics: dict[str, dict] = {}
        self.intro_until = None
        self.t0 = None
        self.paired_since = None
        self._next_check = 0.0
        self.finished = False
        self.message = ""
        self._setup_phase()

    @property
    def phase(self) -> dict:
        return self.phases[self.i]

    def _new_metrics(self, steps_total):
        return dict(time_s=0.0, completed=False, timed_out=False, steps_done=0, steps_total=steps_total,
                    grasp_attempts=0, grasps=0, drops=0, path_m=0.0, turn_deg=0.0, clutches=0,
                    wrong_bin=0, episode=None)

    def _setup_phase(self):
        """Place this phase's objects and markers, home the arm, start the countdown."""
        d, info, key = self.d, self.info, self.phase["key"]
        SS.park_all(info, d)
        for n in PHASE_OBJECTS.get(key, []):
            SS.place(info, d, n)
        for mk in PHASE_MARKERS.get(key, []):
            d.mocap_pos[self.mocap[mk]] = [*SS.ZONE, S.TABLE_Z] if mk == "screen_zone" else [0, 0, S.TABLE_Z]
        # the arm back home, gripper open
        s = SIDE
        d.qpos[info.arm_qpos_adr[s]] = self.sim.arms.home[s]
        d.qvel[info.arm_dof_adr[s]] = 0
        self.sim.arms.reset(d)
        self.sim.arms.teleop[s].gripper = 0.0
        mujoco.mj_forward(self.m, d)
        self.reach_k = 0
        self.reach_log = []
        self._reach_t = self._reach_path = 0.0
        self._reach_from = d.site_xpos[info.ee_site[s]].copy()
        if key == "reach":
            self._show_target()
        self.held = {n: False for n in GRASPABLE.get(key, PHASE_OBJECTS.get(key, []))}
        self.tilt_done = {s: False for s in TILT_STEPS}
        self.in_bin = {n: None for n in PHASE_OBJECTS.get(key, [])}
        self._prev_cmd = 0.0
        self._prev_engaged = False
        self._prev_target = None
        self._prev_quat = None
        self.t0 = None
        self.intro_until = d.time + INTRO_S if self.phase["limit"] else None
        if self.phase["limit"]:
            steps = {"reach": len(SS.REACH), "pick": 1, "tilt": len(TILT_STEPS)}.get(
                key, len(PHASE_OBJECTS.get(key, [])))
            self.metrics[key] = self._new_metrics(steps)

    def _show_target(self):
        mid = self.mocap["screen_goal"]
        if self.reach_k < len(SS.REACH):
            self.d.mocap_pos[mid] = SS.REACH[self.reach_k]
        else:
            self.d.mocap_pos[mid] = [0, 3.0, SS.PARK_Z]

    def _start_phase(self):
        self.t0 = self.d.time
        self.intro_until = None
        self.sim.recorder.start(self.d.time)

    def _end_phase(self, completed: bool):
        key = self.phase["key"]
        mt = self.metrics[key]
        mt["completed"] = completed
        mt["timed_out"] = not completed
        mt["time_s"] = round(self.d.time - self.t0, 2)
        if key == "reach":
            mt["reach"] = self.reach_log
        if self.sim.recorder.active:
            self.sim.save_episode(success=completed)
            mt["episode"] = self.sim.last_saved
        self.message = ("Well done!" if completed else "Time is up.") + " Next: " + self.phases[self.i + 1]["title"]
        self.i += 1
        self._setup_phase()
        if self.phase["key"] == "done":
            self.finished = True
            SS.park_all(self.info, self.d)
            mujoco.mj_forward(self.m, self.d)
        self.write_results()

    # ------------------------------------------------------------ per control tick
    def update(self, dt: float):
        d, key = self.d, self.phase["key"]
        if key == "done":
            return
        if key == "pair":
            if self.phone_connected():
                self.paired_since = self.paired_since or d.time
                if d.time - self.paired_since > 1.0:
                    self.message = "Phone connected"
                    self.i += 1
                    self._setup_phase()
                    self.write_results()
            else:
                self.paired_since = None
            return
        if self.intro_until is not None:
            if d.time >= self.intro_until:
                self._start_phase()
            return
        mt = self.metrics[key]
        tel = self.sim.arms.teleop[SIDE]
        # motion while engaged, clutch presses, grasp attempts (every tick: cheap)
        if tel.engaged:
            if self._prev_target is not None:
                step = float(np.linalg.norm(tel.target_pos - self._prev_target))
                mt["path_m"] += step
                self._reach_path += step
            if self._prev_quat is not None:
                dq = np.zeros(4)
                mujoco.mju_mulQuat(dq, tel.target_quat, np.array([1, -1, -1, -1]) * self._prev_quat)
                mt["turn_deg"] += float(np.degrees(2 * np.arccos(min(1.0, abs(dq[0])))))
            self._prev_target = tel.target_pos.copy()
            self._prev_quat = tel.target_quat.copy()
            if not self._prev_engaged:
                mt["clutches"] += 1
        else:
            self._prev_target = None
            self._prev_quat = None
        self._prev_engaged = tel.engaged
        cmd = self.sim.arms.gripper(SIDE)
        if cmd >= 0.5 > self._prev_cmd:
            mt["grasp_attempts"] += 1
            self._attempt_credited = False           # this attempt can count one grasp
        self._prev_cmd = cmd
        if d.time < self._next_check:
            return
        self._next_check = d.time + 0.05            # the rest at 20 Hz
        elapsed = d.time - self.t0
        mt["time_s"] = round(elapsed, 2)
        done = getattr(self, f"_check_{key}")(mt, cmd)
        if done:
            self._end_phase(True)
        elif elapsed >= self.phase["limit"]:
            self._end_phase(False)

    # ------------------------------------------------------------ goal checks
    def ee(self):
        return self.d.site_xpos[self.info.ee_site[SIDE]]

    def phone_connected(self) -> bool:
        inp = self.sim.latest_input or {}
        fresh = (time.perf_counter() - self.sim.latest_input_t) < self.sim.args.input_timeout
        return bool(fresh and (inp.get(SIDE) or {}).get("pose"))

    def _obj(self, n):
        a = self.info.obj_qpos_adr[n]
        q = self.d.qpos
        return q[a:a + 3], q[a + 3:a + 7], float(np.linalg.norm(self.d.qvel[self.dof[n]:self.dof[n] + 3]))

    def _track_holding(self, mt, cmd, goal_ok):
        """Grasps (an object lifted near the gripper: at most one per grasp attempt) and drops
        (an object that leaves the hand while the gripper is still commanded fully closed;
        letting go on purpose opens it first, and setting it down keeps it at the fingers)."""
        ee = self.ee()
        for n in self.held:
            p, _, _ = self._obj(n)
            mid = p + SS.center_offset(n) * self._axis_z(n)      # the cup's middle, also tilted
            held = p[2] > self._resting_height(n) + HELD_LIFT and np.linalg.norm(mid - ee) < HELD_DIST
            t = round(self.d.time - self.t0, 2)
            if held and not self.held[n] and not getattr(self, "_attempt_credited", True):
                mt["grasps"] += 1
                self._attempt_credited = True
                mt.setdefault("events", []).append([t, "grasp", n])
            elif (self.held[n] and not held and cmd >= 0.95 and not goal_ok(n)
                  and np.linalg.norm(mid - ee) >= HELD_DIST):     # (set down in the hand is no drop)
                mt["drops"] += 1
                mt.setdefault("events", []).append([t, "drop", n, round(float(p[2] - S.TABLE_Z), 3),
                                                    round(float(np.linalg.norm(mid - ee)), 3)])
            self.held[n] = held

    def _resting(self, n, speed):
        p, _, _ = self._obj(n)
        return speed < REST_SPEED and abs(p[2] - SS.rest_z(n)) < 0.008 and not self.held[n]

    def _check_reach(self, mt, cmd):
        if np.linalg.norm(self.ee() - SS.REACH[self.reach_k]) < REACH_TOL:
            straight = float(np.linalg.norm(np.array(SS.REACH[self.reach_k]) - self._reach_from))
            t = self.d.time - self.t0
            self.reach_log.append(dict(time_s=round(t - self._reach_t, 2), path_m=round(self._reach_path, 3),
                                       straight_m=round(straight, 3),
                                       efficiency=round(min(1.0, straight / max(self._reach_path, 1e-6)), 3)))
            self._reach_t, self._reach_path = t, 0.0
            self._reach_from = self.ee().copy()
            self.reach_k += 1
            mt["steps_done"] = self.reach_k
            self._show_target()
        return self.reach_k >= len(SS.REACH)

    def _check_pick(self, mt, cmd):
        def in_zone(n):
            p, _, _ = self._obj(n)
            return max(abs(p[0] - SS.ZONE[0]), abs(p[1] - SS.ZONE[1])) < SS.ZONE_HALF
        self._track_holding(mt, cmd, in_zone)
        _, _, v = self._obj("tut_cube")
        ok = in_zone("tut_cube") and self._resting("tut_cube", v)
        mt["steps_done"] = int(ok)
        return ok

    def _slot_ok(self, n):
        p, q, _ = self._obj(n)
        (sx, sy), syaw, period = SS.SLOTS[n]
        return (np.hypot(p[0] - sx, p[1] - sy) < SLOT_TOL
                and _ang_err(_yaw(q), syaw, period) < SLOT_YAW_TOL)

    def _check_shapes(self, mt, cmd):
        self._track_holding(mt, cmd, self._slot_ok)
        placed = [n for n in self.held if self._slot_ok(n) and self._resting(n, self._obj(n)[2])]
        mt["steps_done"] = len(placed)
        return len(placed) == len(self.held)

    def _bin_of(self, n):
        p, _, _ = self._obj(n)
        for b, (bx, by) in SS.BINS.items():
            if (abs(p[0] - bx) < SS.BIN_HALF and abs(p[1] - by) < SS.BIN_HALF
                    and p[2] < S.TABLE_Z + SS.BIN_WALL + 0.01):
                return b
        return None

    def _check_sort(self, mt, cmd):
        self._track_holding(mt, cmd, lambda n: self._bin_of(n) == n)
        ok = 0
        for n in self.held:
            b = None if self.held[n] else self._bin_of(n)
            if b is not None and b != self.in_bin[n] and b != n:
                mt["wrong_bin"] += 1
            self.in_bin[n] = b
            ok += b == n
        mt["steps_done"] = ok
        return ok == len(self.held)

    def _axis_z(self, n):
        """The object's local z axis in the world (plate: its normal; cup: its axis)."""
        _, q, _ = self._obj(n)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)[:, 2]

    def _plate_standing(self):
        return abs(self._axis_z("plate")[2]) < 0.5

    def _resting_height(self, n):
        """Height of n's origin when it rests on the table as it is now (a standing plate rests
        on its edge, half its width up)."""
        if n == "plate" and self._plate_standing():
            return S.TABLE_Z + SS.OBJECTS["plate"][1][0] + 0.001
        return SS.rest_z(n)

    def _in_bowl(self, n):
        p, _, _ = self._obj(n)
        bx, by = SS.BOWL
        return np.hypot(p[0] - bx, p[1] - by) < SS.BOWL_HALF and p[2] < S.TABLE_Z + SS.BOWL_WALL + 0.01

    def _check_tilt(self, mt, cmd):
        self._track_holding(mt, cmd, lambda n: False)
        done = self.tilt_done
        p, _, v = self._obj("plate")
        rx, ry = SS.RACK
        if not done["book out of the holder"] and np.hypot(p[0] - rx, p[1] - ry) > 0.07:
            done["book out of the holder"] = True
        if (done["book out of the holder"] and not done["book flat on the mat"]
                and abs(self._axis_z("plate")[2]) > FLAT_COS and not self.held["plate"]
                and max(abs(p[0] - SS.MAT[0]), abs(p[1] - SS.MAT[1])) < SS.MAT_HALF and v < REST_SPEED):
            done["book flat on the mat"] = True
        mt["balls_in_bowl"] = int(sum(bool(self._in_bowl(b)) for b in SS.BALLS))
        if mt["balls_in_bowl"] >= BALLS_NEEDED:
            done["balls in the bowl"] = True
        c, _, cv = self._obj("cup")
        if (done["balls in the bowl"] and not self.held["cup"] and cv < REST_SPEED
                and self._axis_z("cup")[2] > UPRIGHT_COS and c[2] < S.TABLE_Z + 0.01):
            done["cup upright again"] = True
        mt["tilt_steps"] = [s for s in TILT_STEPS if done[s]]
        mt["steps_done"] = sum(done.values())
        return all(done.values())

    # ------------------------------------------------------------ results
    def scores(self) -> dict:
        out = {}
        scored = [p for p in self.phases if p["scored"]]
        for p in scored:
            if p["key"] in self.metrics:
                out[p["key"]] = score_task(self.metrics[p["key"]], p["limit"])
        if scored and all(p["key"] in out for p in scored):
            out["overall"] = round(float(np.mean([out[p["key"]] for p in scored])), 1)
        r = self.metrics.get("reach", {}).get("reach") or []
        if r:
            out["reach_efficiency"] = round(float(np.mean([x["efficiency"] for x in r])), 3)
        return out

    def write_results(self):
        res = dict(candidate=self.sim.args.operator, started_at=self.started_at, only=self.only,
                   finished=self.finished, phase=self.phase["key"],
                   finished_at=time.strftime("%Y-%m-%d %H:%M:%S") if self.finished else None,
                   scores=self.scores(), metrics=self.metrics)
        tmp = self.results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2, default=_jsonable), encoding="utf-8")
        tmp.replace(self.results_path)

    def status(self) -> dict:
        p, d = self.phase, self.d
        mt = self.metrics.get(p["key"], {})
        intro = max(0.0, self.intro_until - d.time) if self.intro_until is not None else 0.0
        return dict(phase=p["key"], index=self.i, count=len(self.phases), title=p["title"], text=p["text"],
                    scored=p["scored"], limit=p["limit"], intro=round(intro, 1),
                    rotation=p.get("rotation", "yaw"),
                    time=round(d.time - self.t0, 1) if self.t0 is not None else 0.0,
                    steps_done=mt.get("steps_done", 0), steps_total=mt.get("steps_total", 0),
                    phone=self.phone_connected(), finished=self.finished, message=self.message,
                    scores=self.scores() if self.finished else None)


server.register_game(SS.TASK, ScreenGame)
