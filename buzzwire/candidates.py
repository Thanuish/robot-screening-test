"""Candidates and their personal links (data/buzzwire/candidates.json; not in git).

    python -m buzzwire.candidates new "Jane Doe" jane@example.com     # a code and the link to send
    python -m buzzwire.candidates list

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


def add(name: str, email: str = "") -> str:
    c = load()
    while True:
        code = "".join(secrets.choice(ALPHABET) for _ in range(8))
        if code not in c:
            break
    c[code] = dict(name=name, email=email, created=time.strftime("%Y-%m-%d %H:%M"), status="invited", attempts=0)
    save(c)
    return code


def update(code: str, **kw):
    c = load()
    if code in c:
        c[code].update(kw)
        save(c)


def folder(code: str) -> Path:
    return DATA / "candidates" / code


def public_url() -> str:
    try:
        return json.loads(TUNNEL.read_text())["url"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "http://localhost:8000"


def main():
    a = sys.argv[1:]
    if a[:1] == ["new"] and len(a) >= 2:
        code = add(a[1], a[2] if len(a) > 2 else "")
        print(f"{a[1]}: code {code}\n  link: {public_url()}/c/{code}")
    elif a[:1] == ["list"]:
        for code, c in load().items():
            print(f"{code}  {c['status']:10s}  {c['name']}  {c.get('email', '')}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
