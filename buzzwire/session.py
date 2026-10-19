"""A candidate's test, from the personal link to the thank-you ([flow] in config/buzzwire.toml):

    welcome     name and consent (the computer screen)
    tutorial    the controls, step by step (tutorial.py); the phone is connected at its start
    practice    short tasks with hints; then
    ready       "Start the assessment" or "Practise again" (at most max_practice_rounds rounds)
    card        before each assessment task: a full-screen card with the task (card_seconds)
    assessment  the task itself: the screen shows only a thin timer, no instructions, no score
    done        "Thank you": no scores (those go to results.json for the admin)

Everything is saved in the candidate's folder: results.json (each stage, times, practice rounds,
the phone, every task's metrics and score) and one input log per game (an exact replay).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import Config
from .cubes import CubesGame
from .tower import TowerGame
from .tutorial import TutorialGame

GAME_CLASSES = {"cubes": CubesGame, "tower": TowerGame}
GAME_NAMES = {"cubes": "Cube Tasks", "tower": "Tower"}


class Session:
    def __init__(self, cfg: Config, code: str, name: str, out_dir: Path, device: str = ""):
        self.cfg, self.code, self.name = cfg, code, name
        self.flow = cfg.games["flow"]
        self.dir = Path(out_dir)
        self.stage = "welcome"
        self.game = None
        self.task = None               # (game, level) being played
        self.queue: list = []          # the stage's tasks still to come
        self.round = 0                 # practice rounds started
        self.task_i = 0                # assessment: which task (0-based)
        self.card_until = None
        self.started = time.time()
        self.res = dict(code=code, name=name, device=device, controls=None,
                        started=time.strftime("%Y-%m-%d %H:%M:%S"), finished=None, stage="welcome",
                        tutorial=None, practice=[], assessment={}, scores={}, stage_times={})

    # ------------------------------------------------------------ the flow
    def begin(self, name: str):
        """Welcome done (consent given): the tutorial."""
        if self.stage != "welcome":
            return
        self.name = name or self.name
        self.res["name"] = self.name
        self.res["consent"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._stage("tutorial")
        self.game = TutorialGame(self.cfg, out_dir=self.dir / "tutorial", player=self.name)
        self.task = ("tutorial", "pick")

    def choose(self, choice: str):
        """At "ready": "assess" or "practice"."""
        if self.stage != "ready":
            return
        if choice == "practice" and self.round < self.flow["max_practice_rounds"]:
            self._practice_round()
        elif choice == "assess":
            self.queue = list(self.flow["assessment"])
            self.task_i = -1
            self._stage("assessment")
            self._next_assessment_task()

    def _practice_round(self):
        self.round += 1
        self._stage("practice")
        self.queue = list(self.flow["practice"])
        self.res["practice"].append(dict(round=self.round, started=round(time.time() - self.started, 1), tasks={}))
        self._next_practice_task()

    def _start(self, kind: str, level: str, folder: str):
        self.task = (kind, level)
        self.game = GAME_CLASSES[kind](self.cfg, levels=[level], out_dir=self.dir / folder, player=self.name)

    def _next_practice_task(self):
        if not self.queue:
            self.game, self.task = None, None
            self._stage("ready")
            return
        kind, level = self.queue.pop(0)
        self._start(kind, level, f"practice{self.round}_{kind}_{level}")

    def _next_assessment_task(self):
        if not self.queue:
            self.game, self.task = None, None
            self._stage("done")
            self.res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            scored = [v["score"] for v in self.res["assessment"].values() if v.get("score") is not None]
            if scored:
                self.res["scores"]["overall"] = round(float(np.mean(scored)), 1)
            return
        kind, level = self.queue.pop(0)
        self.task_i += 1
        self._start(kind, level, f"assessment{self.task_i + 1}_{kind}_{level}")
        self.card_until = time.time() + self.flow["card_seconds"]   # (the card first; the game waits)

    def _stage(self, s: str):
        self.stage = s
        self.res["stage"] = s
        self.res["stage_times"][s] = round(time.time() - self.started, 1)
        self.save()

    # ------------------------------------------------------------ every control tick
    def tick(self):
        g = self.game
        if g is None:
            return
        if self.stage == "assessment" and self.card_until is not None:
            if time.time() < self.card_until:
                return                                     # (the card: nothing moves yet)
            self.card_until = None
        g.step()
        if not g.finished:
            return
        key = f"{self.task[0]}:{self.task[1]}"
        result = next(iter(g.results.values()), {})
        if self.stage == "tutorial":
            self.res["tutorial"] = result
            self._practice_round()
        elif self.stage == "practice":
            self.res["practice"][-1]["tasks"][key] = result
            self._next_practice_task()
        elif self.stage == "assessment":
            self.res["assessment"][key] = result
            self.res["scores"][key] = result.get("score")
            self._next_assessment_task()
        self.save()

    # ------------------------------------------------------------ screens and results
    def status(self) -> dict:
        """The flow's part of the status (the game's own part is added by the server)."""
        st = dict(candidate=self.code, stage=self.stage, name=self.name, round=self.round,
                  max_rounds=self.flow["max_practice_rounds"],
                  can_practice=self.round < self.flow["max_practice_rounds"])
        if self.stage == "assessment":
            n = len(self.flow["assessment"])
            st.update(task_index=self.task_i, task_count=n, card=self.card_until is not None,
                      card_left=round(max(0.0, (self.card_until or 0) - time.time()), 1))
            if self.game is not None:
                lv = self.game.level
                st.update(card_title=f"Task {self.task_i + 1} of {n}: {GAME_NAMES[self.task[0]]}",
                          card_subtitle=lv.title, card_text=lv.text, card_limit=lv.limit)
        return st

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.res["elapsed_s"] = round(time.time() - self.started, 1)
        tmp = self.dir / "results.tmp"
        tmp.write_text(json.dumps(self.res, indent=2, default=_plain), encoding="utf-8")
        tmp.replace(self.dir / "results.json")


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")
