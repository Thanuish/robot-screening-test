"""The screening test: a candidate controls one robot arm with a phone and does manipulation
tasks after a short practice. It measures how well they understand the mapping from their hand
to the robot: time, steps completed and efficiency (grasps that work, drops, how directly they
move).

The phases, their objects, markers and steps, the tolerances and the scoring all come from
config/screening.toml (TEST_TELEOP/config.py). Phase kinds:

    pair       connect the phone (scan the QR code, start the controller)
    task       a practice (scored = false) or a scored task: its steps, a time limit
    done       the end

Step types (what the candidate has to do), checked 20 times a second:

    reach          touch a glowing target with the gripper (one at a time, in order)
    place_in_zone  an object in a zone, at rest              } count while they hold
    place_in_slot  in its outline, turned to match, at rest  }
    place_in_bin   in a bin                                  }
    remove_from    taken out of a holder                     } stay done once done
    lay_flat       lying on its thin side in a zone, let go  } ("after": only once that
    pour_into      enough objects in a bowl / bin            }  step is done)
    set_upright    a cup standing upright again, let go      }

All times are simulation seconds, so a slow computer doesn't penalise anyone. Each task phase is
recorded as an episode (data/.../episode_XXXX.hdf5). results.json in the session folder holds
the metrics and scores; the lobby (TEST_TELEOP/lobby.py) reads it.

Score per scored task, 0..100 (weights from [scoring]; by default 60 / 20 / 20):
      steps x steps completed / steps
    + time  x time left / time limit       (only if the task was completed)
    + grasp x grasp efficiency             (grasps that lifted something / grasp attempts,
                                            minus 1/drops_to_zero per drop)
The overall score is the mean of the scored tasks. Reach steps also measure path efficiency
(straight-line distance / distance moved), the clearest sign of understanding the mapping.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from vrteleop import scene as S
from vrteleop import server

from . import config as C
from . import screen_scene as SS


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


def score_task(m: dict, limit: float, w: dict) -> float:
    """0..100 for one scored task (see the module docstring); w = the [scoring] weights."""
    steps = m["steps_done"] / max(1, m["steps_total"])
    time_part = max(0.0, 1 - m["time_s"] / limit) if m["completed"] else 0.0
    if m["grasp_attempts"]:
        eff = m["grasps"] / m["grasp_attempts"] * max(0.0, 1 - m["drops"] / w["drops_to_zero"])
    else:
        eff = 0.0
    return round(w["steps"] * steps + w["time"] * time_part + w["grasp"] * min(1.0, eff), 1)


class ScreenGame:
    def __init__(self, sim):
        self.sim = sim                  # vrteleop.server.TeleopSim
        self.m, self.d, self.info = sim.m, sim.d, sim.info
        self.cfg: C.Config = SS.CFG
        self.side = self.cfg.arm.side
        m = self.m
        self.dof = {n: m.jnt_dofadr[m.joint(f"{n}_joint").id] for n in self.cfg.objects}
        self.results_path = Path(sim.session.dir) / "results.json"
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        # the configuration this attempt ran with, kept next to its results
        text = self.cfg.path.read_bytes()
        self.config_sha = hashlib.sha256(text).hexdigest()[:12]
        (Path(sim.session.dir) / "screening.toml").write_bytes(text)
        # a real robot's pace: the gripper follows the phone at most this fast (a flick of the
        # wrist doesn't whip the arm around)
        sim.arms.max_speed, sim.arms.max_turn = self.cfg.arm.max_speed, self.cfg.arm.max_turn
        sim.arms.teleop[self.side].pos_scale = self.cfg.motion_scale
        sim.arms.smoothing = self.cfg.smoothing or None
        # Testing one part: SCREEN_ONLY=<phase key> (e.g. tilt) runs only pairing, that phase,
        # and the end; the lobby sets it for its "test only" codes
        only = os.environ.get("SCREEN_ONLY", "").strip()
        tasks = [p.key for p in self.cfg.task_phases]
        if only and only not in tasks:
            raise ValueError(f"SCREEN_ONLY={only!r}: choose from {', '.join(tasks)}")
        self.only = only or None
        self.phases = [p for p in self.cfg.phases if not only or p.kind != "task" or p.key == only]
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
    def phase(self) -> C.Phase:
        return self.phases[self.i]

    def _new_metrics(self, steps_total):
        return dict(time_s=0.0, completed=False, timed_out=False, steps_done=0, steps_total=steps_total,
                    grasp_attempts=0, grasps=0, drops=0, path_m=0.0, turn_deg=0.0, clutches=0,
                    wrong_bin=0, episode=None)

    def _setup_phase(self):
        """Place this phase's objects and markers, home the arm, start the countdown."""
        d, info, ph = self.d, self.info, self.phase
        SS.park_all(info, d)
        for n in ph.objects:
            SS.place(info, d, n)
        for mk in ph.markers:
            SS.show_marker(info, d, mk)
        # the arm back home, gripper open
        s = self.side
        d.qpos[info.arm_qpos_adr[s]] = self.sim.arms.home[s]
        d.qvel[info.arm_dof_adr[s]] = 0
        self.sim.arms.reset(d)
        self.sim.arms.teleop[s].gripper = 0.0
        mujoco.mj_forward(self.m, d)
        self.reach_k = 0
        self.reach_log = []
        self._reach_t = self._reach_path = 0.0
        self._reach_from = d.site_xpos[info.ee_site[s]].copy()
        self.reaching = bool(ph.steps) and ph.steps[0].type == "reach"
        if self.reaching:
            self._show_target()
        self.held = {n: False for n in ph.graspable}
        self.done = {st.id: False for st in ph.steps}
        self.in_bin = {st.object: None for st in ph.steps if st.type == "place_in_bin"}
        self._prev_cmd = 0.0
        self._prev_engaged = False
        self._prev_target = None
        self._prev_quat = None
        self.t0 = None
        self.intro_until = d.time + self.cfg.timing["countdown"] if ph.kind == "task" else None
        if ph.kind == "task":
            self.metrics[ph.key] = self._new_metrics(len(ph.steps))

    def _show_target(self):
        steps = self.phase.steps
        SS.show_goal(self.info, self.d, steps[self.reach_k].target if self.reach_k < len(steps) else None)

    def _start_phase(self):
        self.t0 = self.d.time
        self.intro_until = None
        self.sim.recorder.start(self.d.time)

    def _end_phase(self, completed: bool):
        key = self.phase.key
        mt = self.metrics[key]
        mt["completed"] = completed
        mt["timed_out"] = not completed
        mt["time_s"] = round(self.d.time - self.t0, 2)
        if self.reaching:
            mt["reach_log"] = self.reach_log
        if self.sim.recorder.active:
            self.sim.save_episode(success=completed)
            mt["episode"] = self.sim.last_saved
        tx = self.cfg.texts
        self.message = f"{tx['well_done'] if completed else tx['time_up']} {tx['next']} {self.phases[self.i + 1].title}"
        self.i += 1
        self._setup_phase()
        if self.phase.kind == "done":
            self.finished = True
            SS.park_all(self.info, self.d)
            mujoco.mj_forward(self.m, self.d)
        self.write_results()

    # ------------------------------------------------------------ per control tick
    def update(self, dt: float):
        d, ph = self.d, self.phase
        if ph.kind == "done":
            return
        if ph.kind == "pair":
            if self.phone_connected():
                self.paired_since = self.paired_since or d.time
                if d.time - self.paired_since > self.cfg.timing["pair_hold"]:
                    self.message = self.cfg.texts["phone_connected"]
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
        mt = self.metrics[ph.key]
        tel = self.sim.arms.teleop[self.side]
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
        cmd = self.sim.arms.gripper(self.side)
        if cmd >= 0.5 > self._prev_cmd:
            mt["grasp_attempts"] += 1
            self._attempt_credited = False           # this attempt can count one grasp
        self._prev_cmd = cmd
        if d.time < self._next_check:
            return
        self._next_check = d.time + 0.05            # the rest at 20 Hz
        elapsed = d.time - self.t0
        mt["time_s"] = round(elapsed, 2)
        done = self._check_reach(mt) if self.reaching else self._check_steps(mt, cmd)
        if done:
            self._end_phase(True)
        elif elapsed >= ph.limit:
            self._end_phase(False)

    # ------------------------------------------------------------ state of things
    def ee(self):
        return self.d.site_xpos[self.info.ee_site[self.side]]

    def phone_connected(self) -> bool:
        inp = self.sim.latest_input or {}
        fresh = (time.perf_counter() - self.sim.latest_input_t) < self.sim.args.input_timeout
        return bool(fresh and (inp.get(self.side) or {}).get("pose"))

    def _obj(self, n):
        a = self.info.obj_qpos_adr[n]
        q = self.d.qpos
        return q[a:a + 3], q[a + 3:a + 7], float(np.linalg.norm(self.d.qvel[self.dof[n]:self.dof[n] + 3]))

    def _axes(self, n):
        """The object's local axes (columns) in the world."""
        _, q, _ = self._obj(n)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)

    def _axis_z(self, n):
        """The object's local z axis in the world (a cup: its axis)."""
        return self._axes(n)[:, 2]

    def _thin_axis(self, n):
        """A box's thinnest direction (a book: its normal); other shapes: their z axis."""
        o = self.cfg.objects[n]
        k = int(np.argmin(o.half)) if o.shape == "box" else 2
        return self._axes(n)[:, k]

    def _resting_height(self, n):
        """Height of n's origin when it rests on the table as it is now (a box on the face
        that's most nearly down: a book standing in its holder rests on its edge)."""
        o = self.cfg.objects[n]
        if o.shape == "box":
            k = int(np.argmax(np.abs(self._axes(n)[2])))
            return S.TABLE_Z + o.half[k] + 0.001
        return SS.rest_z(n)

    def _local(self, p, mk: C.Marker):
        """p (world) in a marker's frame on the table: (x, y) from its centre, along its sides."""
        dx, dy = p[0] - mk.pos[0], p[1] - mk.pos[1]
        c, s = np.cos(mk.yaw), np.sin(mk.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _in_zone(self, n, zone):
        mk = self.cfg.markers[zone]
        p, _, _ = self._obj(n)
        x, y = self._local(p, mk)
        return abs(x) < mk.p["half"][0] and abs(y) < mk.p["half"][1]

    def _slot_ok(self, n, slot):
        mk = self.cfg.markers[slot]
        p, q, _ = self._obj(n)
        return (np.hypot(p[0] - mk.pos[0], p[1] - mk.pos[1]) < self.cfg.tol["slot_distance"]
                and _ang_err(_yaw(q), mk.yaw, mk.p["symmetry"]) < self.cfg.tol["slot_angle"])

    def _inside(self, n, mk: C.Marker):
        """n inside a bin or bowl (below its rim)."""
        p, _, _ = self._obj(n)
        if p[2] >= S.TABLE_Z + mk.p["wall_height"] + self.cfg.tol["container_margin"]:
            return False
        if mk.kind == "bowl":
            return np.hypot(p[0] - mk.pos[0], p[1] - mk.pos[1]) < mk.p["r_in"]
        x, y = self._local(p, mk)
        return abs(x) < mk.p["half"] and abs(y) < mk.p["half"]

    def _bin_of(self, n):
        """The bin of this phase that n is in, or None."""
        for mk in self.phase.markers:
            if self.cfg.markers[mk].kind == "bin" and self._inside(n, self.cfg.markers[mk]):
                return mk
        return None

    def _resting(self, n, speed):
        p, _, _ = self._obj(n)
        return (speed < self.cfg.tol["rest_speed"] and abs(p[2] - SS.rest_z(n)) < self.cfg.tol["rest_height"]
                and not self.held.get(n, False))

    def _goal_ok(self, n):
        """n is where a place step wants it (a drop there is no drop: it was put down)."""
        for st in self.phase.steps:
            if st.object != n:
                continue
            if ((st.type == "place_in_zone" and self._in_zone(n, st.marker))
                    or (st.type == "place_in_slot" and self._slot_ok(n, st.marker))
                    or (st.type == "place_in_bin" and self._bin_of(n) == st.marker)):
                return True
        return False

    def _track_holding(self, mt, cmd):
        """Grasps (an object lifted near the gripper: at most one per grasp attempt) and drops
        (an object that leaves the hand while the gripper is still commanded fully closed;
        letting go on purpose opens it first, and setting it down keeps it at the fingers)."""
        ee, tol = self.ee(), self.cfg.tol
        for n in self.held:
            p, _, _ = self._obj(n)
            mid = p + SS.center_offset(n) * self._axis_z(n)      # a cup's middle, also tilted
            held = p[2] > self._resting_height(n) + tol["held_lift"] and np.linalg.norm(mid - ee) < tol["held_distance"]
            t = round(self.d.time - self.t0, 2)
            if held and not self.held[n] and not getattr(self, "_attempt_credited", True):
                mt["grasps"] += 1
                self._attempt_credited = True
                mt.setdefault("events", []).append([t, "grasp", n])
            elif (self.held[n] and not held and cmd >= 0.95 and not self._goal_ok(n)
                  and np.linalg.norm(mid - ee) >= tol["held_distance"]):   # (set down in the hand is no drop)
                mt["drops"] += 1
                mt.setdefault("events", []).append([t, "drop", n, round(float(p[2] - S.TABLE_Z), 3),
                                                    round(float(np.linalg.norm(mid - ee)), 3)])
            self.held[n] = held

    # ------------------------------------------------------------ goal checks
    def _check_reach(self, mt):
        steps = self.phase.steps
        target = steps[self.reach_k].target
        if np.linalg.norm(self.ee() - target) < self.cfg.tol["reach"]:
            straight = float(np.linalg.norm(np.array(target) - self._reach_from))
            t = self.d.time - self.t0
            self.reach_log.append(dict(time_s=round(t - self._reach_t, 2), path_m=round(self._reach_path, 3),
                                       straight_m=round(straight, 3),
                                       efficiency=round(min(1.0, straight / max(self._reach_path, 1e-6)), 3)))
            self._reach_t, self._reach_path = t, 0.0
            self._reach_from = self.ee().copy()
            self.done[steps[self.reach_k].id] = True
            self.reach_k += 1
            mt["steps_done"] = self.reach_k
            self._show_target()
        return self.reach_k >= len(steps)

    def _check_steps(self, mt, cmd):
        self._track_holding(mt, cmd)
        tol, done = self.cfg.tol, self.done
        for st in self.phase.steps:
            n = st.object
            if st.type == "place_in_zone":
                done[st.id] = self._in_zone(n, st.marker) and self._resting(n, self._obj(n)[2])
            elif st.type == "place_in_slot":
                done[st.id] = self._slot_ok(n, st.marker) and self._resting(n, self._obj(n)[2])
            elif st.type == "place_in_bin":
                b = None if self.held.get(n) else self._bin_of(n)
                if b is not None and b != self.in_bin[n] and b != st.marker:
                    mt["wrong_bin"] += 1
                self.in_bin[n] = b
                done[st.id] = b == st.marker
            elif done[st.id] or (st.after is not None and not done[st.after]):
                continue                                           # (latched, or waiting)
            elif st.type == "remove_from":
                p, _, _ = self._obj(n)
                h = self.cfg.markers[st.marker].pos
                done[st.id] = bool(np.hypot(p[0] - h[0], p[1] - h[1]) > st.distance)
            elif st.type == "lay_flat":
                _, _, v = self._obj(n)
                done[st.id] = bool(abs(self._thin_axis(n)[2]) > tol["flat_cos"] and not self.held.get(n)
                                   and self._in_zone(n, st.marker) and v < tol["rest_speed"])
            elif st.type == "pour_into":
                mk = self.cfg.markers[st.marker]
                mt["poured"] = int(sum(bool(self._inside(b, mk)) for b in st.objects))
                done[st.id] = mt["poured"] >= st.count
            elif st.type == "set_upright":
                p, _, v = self._obj(n)
                done[st.id] = bool(not self.held.get(n) and v < tol["rest_speed"]
                                   and self._axis_z(n)[2] > tol["upright_cos"]
                                   and p[2] < SS.rest_z(n) + tol["rest_height"])
        mt["done_steps"] = [st.label for st in self.phase.steps if done[st.id]]
        mt["steps_done"] = sum(done.values())
        return all(done.values())

    # ------------------------------------------------------------ results
    def scores(self) -> dict:
        out = {}
        scored = [p for p in self.phases if p.scored]
        for p in scored:
            if p.key in self.metrics:
                out[p.key] = score_task(self.metrics[p.key], p.limit, self.cfg.scoring)
        if scored and all(p.key in out for p in scored):
            out["overall"] = round(float(np.mean([out[p.key] for p in scored])), 1)
        logs = [x for mt in self.metrics.values() for x in mt.get("reach_log", [])]
        if logs:
            out["reach_efficiency"] = round(float(np.mean([x["efficiency"] for x in logs])), 3)
        return out

    def write_results(self):
        res = dict(candidate=self.sim.args.operator, started_at=self.started_at, only=self.only,
                   finished=self.finished, phase=self.phase.key,
                   finished_at=time.strftime("%Y-%m-%d %H:%M:%S") if self.finished else None,
                   config_sha=self.config_sha, scores=self.scores(), metrics=self.metrics)
        tmp = self.results_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2, default=_jsonable), encoding="utf-8")
        tmp.replace(self.results_path)

    def status(self) -> dict:
        """What the candidate's pages show (no scores: those are for the admin page only)."""
        p, d = self.phase, self.d
        mt = self.metrics.get(p.key, {})
        intro = max(0.0, self.intro_until - d.time) if self.intro_until is not None else 0.0
        return dict(phase=p.key, kind=p.kind, index=self.i, count=len(self.phases), title=p.title, text=p.text,
                    scored=p.scored, limit=p.limit, intro=round(intro, 1), rotation=p.rotation, arm=self.side,
                    time=round(d.time - self.t0, 1) if self.t0 is not None else 0.0,
                    steps_done=mt.get("steps_done", 0), steps_total=mt.get("steps_total", 0),
                    steps=[dict(label=st.label, done=bool(self.done.get(st.id))) for st in p.steps],
                    phone=self.phone_connected(), finished=self.finished, message=self.message)


server.register_game(SS.TASK, ScreenGame)
