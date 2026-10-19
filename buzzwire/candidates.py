"""Candidates and their personal links (data/buzzwire/candidates.json; not in git).

    python -m buzzwire.candidates new "Jane Doe" jane@example.com [--controls touch]
    python -m buzzwire.candidates controls <code> motion|touch|choice
    python -m buzzwire.candidates list

Each candidate's controls (set by the admin): "motion" (move the phone: tracked on Android, tilt
steering on iPhone), "touch" (two joysticks) or "choice" (the candidate picks on the phone).
The default is [flow] default_controls in config/buzzwire.toml. The admin page (/admin) does the same.

The link (https://<public address>/c/<code>) is opened on the candidate's computer; the phone
then scans the QR code on the screen (/p/<code> opens the iPhone or Android controller).
"""
from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "buzzwire"
FILE = DATA / "candidates.json"
TUNNEL = ROOT / "data" / "screening" / "tunnel.json"
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"       # no 0/O, 1/I
CONTROLS = ("motion", "touch", "choice")
CONTROL_NAMES = {"motion": "Move the phone", "touch": "Touch joysticks", "choice": "Candidate chooses"}


def load() -> dict:
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save(c: dict):
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(c, indent=2), encoding="utf-8")
    tmp.replace(FILE)


def get(code: str) -> dict | None:
    return load().get((code or "").strip().upper())


def default_controls() -> str:
    try:
        from . import config as C
        return C.load().games["flow"]["default_controls"]
    except Exception:
        return "motion"


def add(name: str, email: str = "", controls: str | None = None) -> str:
    controls = controls or default_controls()
    if controls not in CONTROLS:
        raise ValueError(f"controls must be one of {', '.join(CONTROLS)}")
    c = load()
    while True:
        code = "".join(secrets.choice(ALPHABET) for _ in range(8))
        if code not in c:
            break
    c[code] = dict(name=name, email=email, created=time.strftime("%Y-%m-%d %H:%M"), status="invited", attempts=0,
                   controls=controls)
    save(c)
    return code


def update(code: str, **kw):
    c = load()
    if code in c:
        c[code].update(kw)
        save(c)


def controls_of(code: str) -> str:
    c = get(code)
    return (c or {}).get("controls") or default_controls()


def latest_results(code: str) -> dict | None:
    """The candidate's most recent attempt's results.json (or None)."""
    runs = sorted((folder(code)).glob("*/results.json")) if folder(code).exists() else []
    if not runs:
        return None
    try:
        return json.loads(runs[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def folder(code: str) -> Path:
    return DATA / "candidates" / code


def public_url() -> str:
    try:
        return json.loads(TUNNEL.read_text())["url"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "http://localhost:8000"


def main():
    a = sys.argv[1:]
    controls = None
    if "--controls" in a:
        i = a.index("--controls")
        controls = a[i + 1] if i + 1 < len(a) else None
        a = a[:i] + a[i + 2:]
    if a[:1] == ["new"] and len(a) >= 2:
        code = add(a[1], a[2] if len(a) > 2 else "", controls)
        print(f"{a[1]}: code {code}, controls: {CONTROL_NAMES[controls_of(code)]}\n  link: {public_url()}/c/{code}")
    elif a[:1] == ["controls"] and len(a) == 3 and a[2] in CONTROLS:
        if get(a[1]) is None:
            print(f"no candidate {a[1]}")
        else:
            update(a[1].upper(), controls=a[2])
            print(f"{a[1].upper()}: controls {CONTROL_NAMES[a[2]]}")
    elif a[:1] == ["list"]:
        for code, c in load().items():
            print(f"{code}  {c['status']:10s}  {CONTROL_NAMES[c.get('controls') or default_controls()]:18s}  {c['name']}  {c.get('email', '')}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
