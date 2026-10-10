"""The screening lobby: the one web server candidates connect to (through a tunnel such as
Cloudflare Tunnel for remote candidates). It hands out invite codes, runs one simulator per
candidate (TEST_TELEOP.worker --task screen, on a local port), relays the candidate's monitor
page and phone to it, and shows the results to you.

    python -m TEST_TELEOP.lobby batch "October A" --n 15   # create 15 invite codes
    python -m TEST_TELEOP.lobby serve --tunnel             # start (or: start_screening.bat)

Candidate:  /                 enter invite code + name, consent
            /play/<code>      the monitor page: the robot, the instructions, a QR code
            /m/<code>         the phone controller (Chrome on Android, WebXR motion tracking)
You:        /admin            results, ranking, pass / fail, new invite codes (this PC only)

The test itself (phases, objects, scoring, pass mark): config/screening.toml (see
TEST_TELEOP/config.py). Data: data/screening/candidates.json (codes, names, status),
data/screening/settings.json (how many at a time, idle timeout), and per candidate
data/screening/<batch>/<code>/screen/attempt/ (results.json, the configuration it ran with,
one episode file per phase, the simulator's log).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import io
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
from aiohttp import WSMsgType, web

from vrteleop import scene as S

from . import config as C

ROOT = S.ROOT
WEB_DIR = ROOT / "vrteleop" / "web"                  # the 3D viewer (main.js) and three.js
TEST_WEB = Path(__file__).resolve().parent / "web"   # this test's pages
DATA = ROOT / "data" / "screening"
CANDIDATES = DATA / "candidates.json"
SETTINGS = DATA / "settings.json"
DEFAULT_SETTINGS = {"max_concurrent": 15, "idle_timeout_min": 20}
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"     # no 0/O, 1/I


# ---------------------------------------------------------------- storage
def set_data_dir(path: Path):
    global DATA, CANDIDATES, SETTINGS
    DATA = path.resolve()
    CANDIDATES = DATA / "candidates.json"
    SETTINGS = DATA / "settings.json"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load(path: Path, default):
    try:                        # utf-8-sig: also files saved by Notepad / PowerShell (with a BOM)
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        print(f"[lobby] can't read {path}: {e}")
        return default


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def settings() -> dict:
    st = _load(SETTINGS, {})
    return {k: st.get(k, v) for k, v in DEFAULT_SETTINGS.items()}


def cfg() -> C.Config:
    """The test's configuration (read again when the file changes). If it was edited into a
    broken state, the last good one is used and the problems are shown on the admin page."""
    try:
        c = C.cached()
        _cfg_state.update(good=c, error=None)
    except C.ConfigError as e:
        if _cfg_state["error"] != str(e):
            print(f"[lobby] {e}")
        _cfg_state["error"] = str(e)
        if _cfg_state["good"] is None:
            raise
    return _cfg_state["good"]


_cfg_state = {"good": None, "error": None}


def candidates() -> dict:
    return _load(CANDIDATES, {})


def new_batch(name: str, n: int, only: str | None = None) -> list[str]:
    """n invite codes. only: a phase key (e.g. "tilt"): a test code that runs just that part."""
    if only and only not in [p.key for p in cfg().task_phases]:
        raise ValueError(f"no task phase '{only}'")
    cands = candidates()
    codes = []
    for _ in range(n):
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
            if code not in cands:
                break
        cands[code] = {"batch": name, "created": time.strftime("%Y-%m-%d %H:%M"), "name": "",
                       "status": "new", "started": None, "finished": None, "only": only or None}
        codes.append(code)
    _save(CANDIDATES, cands)
    return codes


def session_dir(code: str, c: dict) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in c["batch"]).strip() or "batch"
    return DATA / safe / code


def results_of(code: str, c: dict) -> dict | None:
    p = session_dir(code, c) / "screen" / "attempt" / "results.json"
    return _load(p, None)


def passed(res: dict | None) -> bool | None:
    """Overall >= the pass mark and every must_complete phase completed (config [scoring])."""
    if not res or not res.get("finished") or res.get("only"):     # (a test of one part: no verdict)
        return None
    sc, c = res.get("scores", {}), cfg()
    ok = sc.get("overall", 0) >= c.scoring["pass_mark"]
    for k in c.scoring["must_complete"]:
        ok = ok and bool(res.get("metrics", {}).get(k, {}).get("completed"))
    return ok


# ---------------------------------------------------------------- tunnel
def tunnel_file() -> Path:
    """Where a separate tunnel process writes its public address (DATA can change: --data)."""
    return DATA / "tunnel.json"


def open_tunnel(port: int):
    """Start a Cloudflare quick tunnel to the lobby: a public https:// address, no account
    needed. A quick tunnel gets a new random address every time it starts, so it runs on while
    the lobby restarts (`python -m TEST_TELEOP.lobby tunnel` in its own window)."""
    import shutil
    exe = shutil.which("cloudflared")
    if exe is None:
        print("  [tunnel] cloudflared not found. Install it:  winget install Cloudflare.cloudflared")
        return None
    return subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")


def watch_tunnel(proc, on_url):
    """Read cloudflared's log until it ends; report the public address once it appears."""
    import re
    seen = False
    for line in proc.stderr:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m and not seen:
            seen = True
            on_url(m.group(0))
            print(f"\n  [tunnel] candidates open:  {m.group(0)}\n", flush=True)


def run_tunnel(port: int):
    """The tunnel as its own process: writes its address to data/screening/tunnel.json, which the
    lobby shows; the lobby can then be restarted without the address changing."""
    proc = open_tunnel(port)
    if proc is None:
        sys.exit(1)
    try:
        watch_tunnel(proc, lambda u: _save(tunnel_file(), {"url": u, "since": time.strftime("%Y-%m-%d %H:%M:%S")}))
    finally:
        proc.terminate()
        tunnel_file().unlink(missing_ok=True)


# ---------------------------------------------------------------- workers
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Worker:
    """One simulator process (TEST_TELEOP.worker --task screen) for one candidate."""

    def __init__(self, code: str, c: dict):
        self.code = code
        self.port = _free_port()
        self.dir = session_dir(code, c)
        self.dir.mkdir(parents=True, exist_ok=True)
        log = open(self.dir / "simulator.log", "ab")
        env = dict(os.environ, SCREEN_ONLY=c.get("only") or "")      # a test code: one part only
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "TEST_TELEOP.worker", "--http", "--host", "127.0.0.1",
             "--port", str(self.port), "--task", "screen", "--no-preview", "--stream-cams", "",
             # 60 pose updates a second (~100 KB/s per candidate): less delay on top of the
             # internet's (the monitor shows the arm one update behind, interpolated)
             "--stream-hz", "60", "--data-dir", str(self.dir), "--session", "attempt",
             "--operator", c.get("name") or code, "--exit-when-idle", "25"],
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
        self.clients = 0
        self.last_seen = time.time()
        self.started = time.time()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc.poll() is None

    async def ready(self, session: aiohttp.ClientSession, timeout=60.0) -> bool:
        t_end = time.time() + timeout
        while time.time() < t_end and self.alive():
            try:
                async with session.get(self.url + "/api/status", timeout=aiohttp.ClientTimeout(total=2)) as r:
                    if r.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(0.5)
        return False

    def stop(self):
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ---------------------------------------------------------------- pages
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
 body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
        background:#15171c; color:#e8e8e8; font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 .card {{ background:#1f2229; border:1px solid #333; border-radius:14px; padding:28px 32px; max-width:560px; margin:16px; }}
 h1 {{ font-size:22px; margin:0 0 12px; }} p {{ color:#c4c4c4; }}
 input[type=text] {{ width:100%; box-sizing:border-box; padding:10px 12px; margin:6px 0 14px; font:inherit;
        background:#111; color:#eee; border:1px solid #444; border-radius:8px; }}
 button {{ background:#2f6fde; color:#fff; border:0; border-radius:8px; padding:11px 20px; font:inherit; cursor:pointer; }}
 .err {{ color:#ff8080; }} label {{ color:#c4c4c4; }} small {{ color:#888; }}
</style></head><body><div class="card">{body}</div></body></html>"""


def page(title, body, status=200):
    return web.Response(text=PAGE.format(title=html.escape(title), body=body), content_type="text/html",
                        status=status, headers={"Cache-Control": "no-store"})


JOIN_FORM = """<h1>Robot control test</h1>
<p>You will control a robot arm in a simulation with your phone and do three short manipulation
tasks, after a practice round. It takes about 20 minutes.</p>
<p><b>You need:</b> this computer (keep this page open on its screen) and an <b>Android phone</b>
with Chrome. You will scan a QR code with the phone.</p>
{error}
<form method="post" action="/join">
 <label>Invite code</label><input type="text" name="code" value="{code}" autocomplete="off" required>
 <label>Your name</label><input type="text" name="name" value="{name}" required>
 <p><label><input type="checkbox" name="consent" required> I agree that my control of the robot
 is recorded and scored for this application. The recording is used only to evaluate the test.</label></p>
 <button type="submit">Start</button>
</form>"""


def make_app(args) -> web.Application:
    workers: dict[str, Worker] = {}
    @web.middleware
    async def no_cache_scripts(request, handler):
        """Pages and scripts are revalidated on every load, so a fix reaches phones at once
        (only three.js, which never changes, may be cached)."""
        resp = await handler(request)
        if request.path.startswith(("/test-static/", "/static/")) and "/vendor/" not in request.path:
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app = web.Application(middlewares=[no_cache_scripts])
    http: dict = {}

    async def on_startup(_):
        http["s"] = aiohttp.ClientSession()
        app["reaper"] = asyncio.create_task(reaper())
        if args.tunnel:
            start_tunnel()

    async def on_cleanup(_):
        app["reaper"].cancel()
        for w in workers.values():
            w.stop()
        if app.get("tunnel") is not None:
            app["tunnel"].terminate()
        await http["s"].close()

    tunnel_state = {"url": None}           # the in-process tunnel's public address, once known

    def start_tunnel():
        import threading
        proc = open_tunnel(args.port)
        if proc is None:
            return
        app["tunnel"] = proc
        threading.Thread(target=lambda: watch_tunnel(proc, lambda u: tunnel_state.update(url=u)),
                         daemon=True).start()

    def public_url():
        """The in-process tunnel's address, or the one a separate `tunnel` process wrote."""
        if tunnel_state["url"]:
            return tunnel_state["url"]
        info = _load(tunnel_file(), None)
        return info.get("url") if info else None

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    def local_only(handler):
        async def wrapped(request):
            if request.remote not in ("127.0.0.1", "::1") or "cf-connecting-ip" in request.headers \
                    or "x-forwarded-for" in request.headers:
                return web.Response(status=403, text="the admin pages only open on the screening PC")
            return await handler(request)
        return wrapped

    def public_base(request) -> str:
        proto = request.headers.get("X-Forwarded-Proto", request.scheme)
        return f"{proto}://{request.host}"

    def lookup(code: str):
        code = (code or "").strip().upper()
        return code, candidates().get(code)

    async def ensure_worker(code: str, c: dict) -> Worker:
        w = workers.get(code)
        if w is not None and w.alive():
            return w
        st = settings()
        busy = sum(1 for x in workers.values() if x.alive())
        if busy >= st["max_concurrent"]:
            raise web.HTTPServiceUnavailable(text="All test places are busy right now. Please try again in a few minutes.")
        w = Worker(code, c)
        workers[code] = w
        if not await w.ready(http["s"]):
            raise web.HTTPInternalServerError(text="The simulator didn't start. Please tell the organiser.")
        return w

    async def reaper():
        """Stop simulators of finished or abandoned tests."""
        while True:
            await asyncio.sleep(10)
            st = settings()
            cands = candidates()
            changed = False
            for code, w in list(workers.items()):
                c = cands.get(code)
                res = results_of(code, c) if c else None
                if res and res.get("finished") and c.get("status") != "done":
                    c["status"], c["finished"] = "done", res.get("finished_at")
                    changed = True
                idle = w.clients == 0 and time.time() - w.last_seen > 60 * st["idle_timeout_min"]
                done_idle = res and res.get("finished") and w.clients == 0 and time.time() - w.last_seen > 30
                if not w.alive() or idle or done_idle:
                    w.stop()
                    del workers[code]
                    if c and c.get("status") == "playing" and not (res and res.get("finished")):
                        c["status"] = "abandoned"
                        changed = True
            if changed:
                _save(CANDIDATES, cands)

    # ---- candidate pages
    async def index(request):
        return page("Robot control test", JOIN_FORM.format(error="", code=html.escape(request.query.get("code", "")), name=""))

    async def join(request):
        form = await request.post()
        code, c = lookup(form.get("code", ""))
        name = (form.get("name") or "").strip()[:80]

        def again(msg):
            return page("Robot control test", JOIN_FORM.format(error=f'<p class="err">{html.escape(msg)}</p>',
                                                              code=html.escape(code), name=html.escape(name)))
        if c is None:
            return again("This invite code doesn't exist. Check it and try again.")
        if not form.get("consent"):
            return again("Please agree to the recording to take the test.")
        if c["status"] in ("done", "abandoned"):
            return page("Robot control test", "<h1>Already taken</h1><p>This invite code has been used. "
                                              "Contact the organiser if you think this is a mistake.</p>")
        if c["status"] == "new":
            cands = candidates()
            cands[code].update(name=name, status="playing", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            _save(CANDIDATES, cands)
            c = cands[code]
        await ensure_worker(code, c)
        raise web.HTTPFound(f"/play/{code}")

    def static_page(file, code):
        text = (TEST_WEB / file).read_text(encoding="utf-8")
        text = (text.replace("__CODE__", code).replace("__BOOT__", str(app["boot"]))
                .replace("__CONFIG__", json.dumps(cfg().public())))
        return web.Response(text=text, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def play(request):
        code, c = lookup(request.match_info["code"])
        if c is None or c["status"] not in ("playing",):
            raise web.HTTPFound("/")
        await ensure_worker(code, c)
        return static_page("screen.html", code)

    async def phone(request):
        code, c = lookup(request.match_info["code"])
        if c is None or c["status"] != "playing":
            return page("Robot control test", "<h1>Link not active</h1><p>Open the test on your computer first.</p>")
        return static_page("phone.html", code)

    async def qr(request):
        import segno
        code, c = lookup(request.match_info["code"])
        if c is None:
            raise web.HTTPNotFound()
        buf = io.BytesIO()
        segno.make(f"{public_base(request)}/m/{code}", error="m").save(buf, kind="svg", scale=8, border=2,
                                                                        dark="#000", light="#fff")
        return web.Response(body=buf.getvalue(), content_type="image/svg+xml")

    async def phone_url(request):
        code, _ = lookup(request.match_info["code"])
        return web.json_response({"url": f"{public_base(request)}/m/{code}"})

    # ---- relay to the candidate's simulator
    async def scene_json(request):
        code, c = lookup(request.match_info["code"])
        w = workers.get(code)
        if c is None or w is None:
            raise web.HTTPNotFound()
        async with http["s"].get(w.url + "/scene.json", auto_decompress=False) as r:
            body = await r.read()
            return web.Response(body=body, headers={"Content-Type": "application/json",
                                                    "Content-Encoding": r.headers.get("Content-Encoding", ""),
                                                    "Cache-Control": "no-cache"})

    async def ws_relay(request):
        code, c = lookup(request.match_info["code"])
        w = workers.get(code)
        if c is None or w is None or not w.alive():
            raise web.HTTPNotFound()
        down = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
        await down.prepare(request)
        w.clients += 1
        try:
            async with http["s"].ws_connect(w.url + "/ws", max_msg_size=0, heartbeat=15) as up:
                async def to_sim():
                    async for msg in down:
                        if msg.type == WSMsgType.TEXT:
                            await up.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await up.send_bytes(msg.data)
                async def to_browser():
                    async for msg in up:
                        if msg.type == WSMsgType.TEXT:
                            await down.send_str(msg.data)
                        elif msg.type == WSMsgType.BINARY:
                            await down.send_bytes(msg.data)
                tasks = [asyncio.create_task(to_sim()), asyncio.create_task(to_browser())]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in tasks:
                    t.cancel()
        except (aiohttp.ClientError, ConnectionResetError):
            pass
        finally:
            w.clients -= 1
            w.last_seen = time.time()
        return down

    # ---- admin (this PC only)
    async def admin(_):
        return web.FileResponse(TEST_WEB / "admin.html", headers={"Cache-Control": "no-store"})

    def rows():
        c = cfg()
        practice = [p.key for p in c.task_phases if not p.scored]
        out = []
        for code, cand in candidates().items():
            res = results_of(code, cand)
            sc = (res or {}).get("scores", {})
            mt = (res or {}).get("metrics", {})
            r = {"code": code, "batch": cand["batch"], "name": cand.get("name", ""), "status": cand["status"],
                 "phase": (res or {}).get("phase"), "started": cand.get("started"),
                 "finished": cand.get("finished") or (res or {}).get("finished_at"),
                 "practice_done": all(mt.get(k, {}).get("completed") for k in practice),
                 "reach_efficiency": sc.get("reach_efficiency")}
            for p in c.scored_phases:           # a score and steps column per scored task
                r[p.key] = sc.get(p.key)
                r[f"{p.key}_steps"] = f"{mt[p.key]['steps_done']}/{mt[p.key]['steps_total']}" if p.key in mt else ""
            r.update({"overall": sc.get("overall"), "pass": passed(res), "config": (res or {}).get("config_sha"),
                      "folder": _rel(session_dir(code, cand)),
                      "running": code in workers and workers[code].alive(), "only": cand.get("only")})
            out.append(r)
        return out

    async def admin_api(_):
        c = cfg()
        return web.json_response({"settings": settings(), "candidates": rows(), "config": c.public(),
                                  "config_path": _rel(c.path), "config_error": _cfg_state["error"],
                                  "public_url": public_url(), "tunnel": bool(args.tunnel)})

    async def admin_csv(_):
        buf = io.StringIO()
        r = rows()
        if r:
            wr = csv.DictWriter(buf, fieldnames=list(r[0]))
            wr.writeheader()
            wr.writerows(r)
        return web.Response(text=buf.getvalue(), content_type="text/csv",
                            headers={"Content-Disposition": 'attachment; filename="screening_results.csv"'})

    async def admin_batch(request):
        body = await request.json()
        try:
            codes = new_batch(str(body.get("name") or time.strftime("%Y-%m-%d")), int(body.get("n", 15)),
                              body.get("only") or None)
        except ValueError as e:
            raise web.HTTPBadRequest(text=str(e))
        return web.json_response({"codes": codes})

    async def admin_settings(request):
        """Lobby settings (settings.json) and the pass rule, which lives in the test's
        configuration ([scoring] pass_mark, must_complete: saved into the file, comments kept)."""
        body = await request.json()
        st = settings()
        for k in DEFAULT_SETTINGS:
            if k in body:
                st[k] = type(DEFAULT_SETTINGS[k])(body[k])
        _save(SETTINGS, st)
        upd = {}
        if "pass_mark" in body:
            v = float(body["pass_mark"])
            upd[("scoring", "pass_mark")] = int(v) if v.is_integer() else v
        if "must_complete" in body:
            upd[("scoring", "must_complete")] = [str(x) for x in body["must_complete"]]
        if upd:
            try:
                C.set_values(upd, cfg().path)
            except C.ConfigError as e:
                raise web.HTTPBadRequest(text="\n".join(e.problems))
        return web.json_response({"settings": st, "config": cfg().public()})

    async def admin_reset(request):
        """Let a candidate take the test again (keeps the old attempt's files, renamed)."""
        code, c = lookup(request.match_info["code"])
        if c is None:
            raise web.HTTPNotFound()
        if code in workers:
            workers.pop(code).stop()
        d = session_dir(code, c) / "screen" / "attempt"
        if d.exists():
            d.rename(d.with_name(f"attempt_old_{time.strftime('%Y%m%d_%H%M%S')}"))
        cands = candidates()
        cands[code].update(status="new", started=None, finished=None)
        _save(CANDIDATES, cands)
        return web.json_response({"ok": True})

    app["boot"] = int(time.time())
    app.router.add_get("/", index)
    app.router.add_post("/join", join)
    app.router.add_get("/play/{code}", play)
    app.router.add_get("/m/{code}", phone)
    app.router.add_get("/qr/{code}.svg", qr)
    app.router.add_get("/phone-url/{code}", phone_url)
    app.router.add_get("/s/{code}/scene.json", scene_json)
    app.router.add_get("/s/{code}/ws", ws_relay)
    app.router.add_get("/admin", local_only(admin))
    app.router.add_get("/admin/api", local_only(admin_api))
    app.router.add_get("/admin/results.csv", local_only(admin_csv))
    app.router.add_post("/admin/batch", local_only(admin_batch))
    app.router.add_post("/admin/settings", local_only(admin_settings))
    app.router.add_post("/admin/reset/{code}", local_only(admin_reset))
    app.router.add_static("/static/", WEB_DIR)
    app.router.add_static("/test-static/", TEST_WEB)
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch", help="create invite codes")
    b.add_argument("name", help="batch name, e.g. 'October A'")
    b.add_argument("--n", type=int, default=15)
    s = sub.add_parser("serve", help="start the lobby")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 (default): reachable through a tunnel only; 0.0.0.0: also on the local network")
    s.add_argument("--tunnel", action="store_true",
                   help="also start a Cloudflare quick tunnel (needs cloudflared) and show its public address")
    b.add_argument("--only", default=None, help="test codes that run only this part, e.g. tilt")
    tn = sub.add_parser("tunnel", help="run the public tunnel on its own (its address survives lobby restarts)")
    tn.add_argument("--port", type=int, default=8000)
    for p in (b, s, tn):
        p.add_argument("--data", default=str(DATA), help="folder for codes, settings and results")
    a = ap.parse_args(argv)
    set_data_dir(Path(a.data))
    if a.cmd == "tunnel":
        print(f"  Public tunnel to http://localhost:{a.port} (keep this window open)...", flush=True)
        run_tunnel(a.port)
        return
    if a.cmd == "batch":
        codes = new_batch(a.name, a.n, a.only)
        print(f"{len(codes)} invite codes for batch '{a.name}' (saved in {_rel(CANDIDATES)}):")
        for c in codes:
            print("  ", c)
        return
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    if not SETTINGS.exists():
        _save(SETTINGS, DEFAULT_SETTINGS)
    try:
        c = cfg()
    except C.ConfigError as e:
        print(f"\n  The test's configuration has problems; fix them and start again:\n{e}\n")
        sys.exit(1)
    print(f"  Test configuration: {_rel(c.path)} ({len(c.task_phases)} phases)")
    print(f"\n  Screening lobby: http://localhost:{a.port}   (results: http://localhost:{a.port}/admin)")
    if a.tunnel:
        print("  Starting the public tunnel (its address appears below and on the admin page)...")
    else:
        print("  Candidates need a public https:// address: the tunnel (a separate window:")
        print("     python -m TEST_TELEOP.lobby tunnel), or start this with --tunnel\n")
    web.run_app(make_app(a), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
