"""Recording pipeline: where episodes go and what is stored next to them.

Every server run is one *session*. Layout:

    data/
      jeans/                                   one folder per task
        index.csv                              one row per saved episode (all sessions)
        2026-09-28_14-05-12/                   one folder per session (server run)
          session.json                         who / when / settings of this session
          episode_0000.hdf5                    the demonstration (see recorder.py)
          episode_0000.json                    summary: success, duration, fold coverage, ...
          episode_0000.mp4                     first-person video: what the operator saw in the
                                               headset (rendered in the background)

The HDF5 file is the data; the JSON and the CSV make it easy to browse, filter and
count episodes without opening HDF5 files; the MP4 lets you check a demo at a glance.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_FIELDS = ["session", "episode", "path", "saved_at", "operator", "task", "success",
                "duration_s", "frames", "final_coverage", "final_height_m", "notes"]


class Session:
    def __init__(self, data_root: Path, task: str, name: str | None = None,
                 operator: str = "", settings: dict | None = None, preview: bool = True,
                 preview_camera: str = "operator"):
        self.task = task
        self.task_dir = Path(data_root) / task
        self.name = name or _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir = self.task_dir / self.name
        self.operator = operator
        self.preview = preview
        self.preview_camera = preview_camera
        self.dir.mkdir(parents=True, exist_ok=True)
        info = {
            "session": self.name,
            "task": task,
            "operator": operator,
            "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "machine": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "settings": settings or {},
        }
        try:
            import mujoco
            info["mujoco"] = mujoco.__version__
        except ImportError:
            pass
        path = self.dir / "session.json"
        if not path.exists():               # resuming a named session keeps its original info
            path.write_text(json.dumps(info, indent=2))
        self._previews: list[subprocess.Popen] = []

    def relpath(self, p: Path) -> str:
        try:
            return str(Path(p).relative_to(self.task_dir.parent))
        except ValueError:
            return str(p)

    def episode_saved(self, h5_path: Path, summary: dict):
        """Write the JSON sidecar, append to index.csv and start the preview render."""
        h5_path = Path(h5_path)
        summary = {"session": self.name, "episode": h5_path.stem, "path": self.relpath(h5_path),
                   "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
                   "operator": self.operator, "task": self.task, **summary}
        h5_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        index = self.task_dir / "index.csv"
        new = not index.exists()
        with index.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: summary.get(k, "") for k in INDEX_FIELDS})
        if self.preview:
            self._start_preview(h5_path)

    def _start_preview(self, h5_path: Path):
        """Render a preview video in a low-priority background process (never blocks the sim)."""
        self._previews = [p for p in self._previews if p.poll() is None]
        cmd = [sys.executable, str(ROOT / "scripts" / "replay.py"), str(h5_path),
               "--video", str(h5_path.with_suffix(".mp4")), "--camera", self.preview_camera,
               "--width", "960", "--height", "720"]
        try:
            low = ({"preexec_fn": lambda: os.nice(15)} if hasattr(os, "nice")
                   else {"creationflags": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)})
            self._previews.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **low))
        except OSError as e:
            print(f"[pipeline] preview render failed to start: {e}")
