"""Robot Buzz Wire: the development server (one game, fixed addresses). The simulation runs here
in real time; browsers only show it and send controller input.

    python -m buzzwire.server [--port 8000] [--levels straight curves]

    /            the games: a menu (Buzz Wire, Cube Tasks, Tower), the 3D view, the HUD, the QR code
    /phone       the Android phone controller (touch joysticks, or motion tracking)
    /iphone      the iPhone controller (web/apple/: touch joysticks, or tilt steering)
    /scene.json  the current level's geometry (rebuilt per level)
    /ws          poses (binary, ~50 a second), status (JSON, 10 a second); input in

A public https address for the phone: the Cloudflare tunnel to this port
(python -m TEST_TELEOP.lobby tunnel writes data/screening/tunnel.json, shown on the page).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import WSMsgType, web

from vrteleop.webscene import export_scene

from . import config as C
from .cubes import CubesGame
from .game import BuzzWireGame
from .tower import TowerGame
from . import candidates as CAND
from .session import Session

NOT_FOUND = ("<!doctype html><meta name=viewport content='width=device-width'><body style='font:16px system-ui;"
             "background:#0b0d12;color:#eee;display:grid;place-items:center;height:100vh;margin:0'>"
             "<div style='max-width:420px;text-align:center'><h2>This link isn't valid</h2><p>Please check the "
             "link in your invitation e-mail, or contact us.</p></div>")

GAMES = {   # key: (class, title, one line for the menu)
    "wire": (BuzzWireGame, "Buzz Wire", "Slide the ring along the wire without touching it."),
    "cubes": (CubesGame, "Cube Tasks", "Pick up cubes and shapes: into squares, outlines and bins."),
    "tower": (TowerGame, "Tower", "Stack blocks as high as you can without the tower falling."),
}

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
THREE = WEB / "vendor"                 # three.js r170 and its addons (MIT, see vendor/three/LICENSE)
DATA = ROOT / "data" / "buzzwire"
TUNNEL = ROOT / "data" / "screening" / "tunnel.json"
POSE_EVERY = 2                 # control ticks between pose packets (100 Hz / 2 = 50 a second)


def _plain(o):
    """numpy numbers and booleans in a message -> plain Python (json can't encode numpy's)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def dumps(obj) -> str:
    return json.dumps(obj, default=_plain)
STATUS_EVERY = 10


class Server:
    def __init__(self, args):
        self.args = args
        self.cfg = C.load()
        self.clients: set[web.WebSocketResponse] = set()
        self.counts: dict = {}                # per connection: messages seen (for the log)
        self.phone_seen = 0.0                 # when the phone last sent its pose
        self.phone_msg = 0.0                  # when the phone last sent anything (with or without a pose)
        self.rtf = 1.0                        # simulated seconds per real second (measured by run())
        self.scene_id = -1                    # bumped with every new scene (browsers reload it)
        self._level = None                    # (game, level, scene) last sent to the browsers
        self._game = None                     # the game chosen in the menu (None: the menu)
        self.session = None                   # a candidate's test (session.py); its game is shown instead
        self.view_yaw = 0.0
        self.controls = "touch"               # what the phone says it uses: motion, tilt, touch
        self.kind = None
        if args.game:
            self.new_game(args.game)

    def new_game(self, kind: str | None = None):
        """Start a game (kind: wire, cubes, tower; None: the current one again)."""
        kind = kind or self.kind
        if kind not in GAMES:
            return
        out = DATA / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{kind}"
        levels = self.args.levels if kind == (self.args.game or kind) else None
        self.session = None
        self._game = GAMES[kind][0](self.cfg, levels=levels, out_dir=out, player=self.args.player)
        self.kind = kind
        self.game.view_yaw = self.view_yaw            # (the screen's camera heading, for the phone's "forward")
        self._export()
        print(f"[game] {GAMES[kind][1]}", flush=True)

    def to_menu(self):
        self._game = None
        self.session = None

    @property
    def game(self):
        """The game being played: the candidate's current task, or the one chosen in the menu."""
        return self.session.game if self.session is not None else self._game

    def join(self, code: str, device: str = "") -> bool:
        """A candidate opened their personal link: their test (continued if it's already running)."""
        c = CAND.get(code)
        if c is None:
            return False
        code = code.strip().upper()
        if self.session is None or self.session.code != code:
            self.session = Session(self.cfg, code, c["name"], CAND.folder(code) / time.strftime("%Y-%m-%d_%H-%M-%S"))
            CAND.update(code, status="started", attempts=c.get("attempts", 0) + 1,
                        last_started=time.strftime("%Y-%m-%d %H:%M"))
            print(f"[candidate] {code} {c['name']}: the test starts", flush=True)
        return True

    def _export(self):
        g = self.game
        self.bodies, self.scene_gz = export_scene(g.m)
        self.scene_id += 1
        self._level = (id(g), g.li, id(g.sc))

    # ---------------------------------------------------------------- the real-time loop
    async def run(self):
        """The simulation in real time. An error in one tick is logged and the loop goes on (it
        must never stop silently: the robot would freeze while the pages still load)."""
        errors = 0
        while True:
            try:
                await self._run()
            except asyncio.CancelledError:
                raise
            except Exception:
                errors += 1
                print(f"[error] the simulation loop failed ({errors}); restarting it:\n{traceback.format_exc()}", flush=True)
                if errors >= 3:
                    print("[error] three failures: back to the menu", flush=True)
                    self.to_menu()
                    errors = 0
                await asyncio.sleep(0.1)

    async def _run(self):
        dt = 1.0 / self.cfg.physics["control_hz"]
        t0, tick0 = time.perf_counter(), 0
        rtf_w, rtf_t, self.rtf = time.perf_counter(), 0, 1.0
        while True:
            g = self.game
            if g is None:                             # the menu, or a candidate between tasks: only the status
                await self.broadcast(dumps(self.status()))
                await asyncio.sleep(0.1)
                t0, tick0, rtf_t, rtf_w = time.perf_counter(), 0, 0, time.perf_counter()
                continue
            if (id(g), g.li, id(g.sc)) != self._level:
                self._export()                        # a new level or game: new geometry for the browsers
            if self.session is not None:
                stage = self.session.stage
                self.session.tick()                   # (steps the task's game; moves the test along)
                if self.session.stage != stage:
                    print(f"[candidate] {self.session.code}: {self.session.stage}", flush=True)
                    if self.session.stage == "done":
                        CAND.update(self.session.code, status="done", finished=time.strftime("%Y-%m-%d %H:%M"))
                if self.game is not g:
                    continue
            else:
                g.step()
            tick0 += 1
            if tick0 % POSE_EVERY == 0:
                await self.broadcast(self.pose_packet())
            if tick0 % STATUS_EVERY == 0:
                now = time.perf_counter()
                if now - rtf_w > 1.0:
                    self.rtf = (tick0 - rtf_t) * dt / (now - rtf_w)
                    rtf_w, rtf_t = now, tick0
                await self.broadcast(dumps(self.status()))
            # keep to real time (if behind, catch up without sleeping)
            wait = t0 + tick0 * dt - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            elif wait < -0.25:
                t0, tick0 = time.perf_counter(), 0   # (far behind, e.g. after a level build: resync)
                rtf_t = 0
                rtf_w = time.perf_counter()

    def pose_packet(self) -> bytes:
        g, d = self.game, self.game.d
        tel = g.teleop
        ghost = list(tel.target_pos) + list(tel.target_quat) if tel.engaged else [np.nan] * 7
        head = np.array([d.time, self.scene_id, len(self.bodies), 1.0 if g.touching else 0.0], np.float32)
        P = np.concatenate([d.xpos[self.bodies], d.xquat[self.bodies]], axis=1).astype(np.float32).ravel()
        return head.tobytes() + P.tobytes() + np.array(ghost, np.float32).tobytes()

    def status(self) -> dict:
        now = time.perf_counter()
        phone = now - self.phone_seen < 1.5
        # the phone is there but doesn't know where it is (its camera sees too little)
        lost = now - self.phone_msg < 1.5 and not phone
        common = dict(type="status", phone=phone, tracking_lost=lost, scene_id=self.scene_id, view_yaw=self.view_yaw,
                      controls=self.controls)
        if self.game is not None and hasattr(self.game, "controls"):
            self.game.controls = self.controls        # (the tutorial words its steps for these controls)
        if self.session is not None:
            ss = self.session
            st = self.game.status() if self.game is not None else dict(state=ss.stage, hud=[], steps=[])
            if ss.stage == "assessment":
                st.update(hud=[], steps=[], instruction="", message="")   # (nothing on the screen but the timer)
            st.update(ss.status())
            st.update(common, rtf=round(self.rtf, 2))
            return st
        if self.game is None:
            return dict(common, state="menu", rtf=1.0,
                        games=[dict(key=k, title=v[1], text=v[2]) for k, v in GAMES.items()])
        st = self.game.status()
        st.update(common, rtf=round(self.rtf, 2), scores=None)
        return st

    def level_geometry(self) -> dict:
        """The current level's shapes for the browser to draw finely (the physics uses simple ones)."""
        if self.game is None:
            return dict(kind="none", scene_id=self.scene_id, table_z=C.TABLE_Z)
        return dict(self.game.level_geometry(), scene_id=self.scene_id, table_z=C.TABLE_Z)

    async def broadcast(self, msg):
        dead = []
        for ws in self.clients:
            try:
                if isinstance(msg, bytes):
                    await ws.send_bytes(msg)
                else:
                    await ws.send_str(msg)
            except (ConnectionResetError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    # ---------------------------------------------------------------- web
    def app(self) -> web.Application:
        app = web.Application()

        @web.middleware
        async def no_cache(request, handler):
            resp = await handler(request)
            if "/vendor/" not in request.path and "/assets/" not in request.path:
                resp.headers["Cache-Control"] = "no-cache"
            return resp
        app.middlewares.append(no_cache)

        async def index(_):
            return web.FileResponse(WEB / "index.html")

        async def phone(_):
            return web.FileResponse(WEB / "phone.html")

        async def level(_):
            return web.json_response(self.level_geometry())

        async def scene(_):
            return web.Response(body=self.scene_gz, headers={"Content-Type": "application/json",
                                                             "Content-Encoding": "gzip"})

        async def config(_):
            return web.json_response(dict(self.cfg.public(), phone_url=self.phone_url(),
                                          iphone_url=self.phone_url("/iphone")))

        async def qr(request):
            import segno
            buf = io.BytesIO()
            f = request.query.get("for")
            page = f"/p/{request.query.get('c', '')}" if f == "candidate" else "/iphone" if f == "iphone" else "/phone"
            segno.make(self.phone_url(page) or "no public address", error="m").save(buf, kind="svg", scale=6, border=2)
            return web.Response(body=buf.getvalue(), content_type="image/svg+xml")

        async def ws_handler(request):
            ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
            await ws.prepare(request)
            self.clients.add(ws)
            ua = request.headers.get("User-Agent", "")
            who = "phone" if "Android" in ua else "iphone" if ("iPhone" in ua or "iPad" in ua) else "browser"
            print(f"[ws] {who} connected ({len(self.clients)} open)", flush=True)
            await ws.send_str(dumps(self.status()))
            self.counts[ws] = {"in": 0, "fps": 0}
            try:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        continue
                    try:
                        m = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    try:
                        await self.handle(ws, who, m)
                    except Exception:                    # one bad message must not drop the connection
                        print(f"[error] message {str(m)[:120]}:\n{traceback.format_exc()}", flush=True)
            finally:
                self.clients.discard(ws)
                self.counts.pop(ws, None)
                print(f"[ws] {who} disconnected", flush=True)
            return ws

        app.router.add_get("/", index)
        app.router.add_get("/phone", phone)
        app.router.add_get("/iphone", lambda _: web.FileResponse(WEB / "apple" / "index.html"))

        async def candidate_page(request):            # the personal link: the game screen in test mode
            if CAND.get(request.match_info["code"]) is None:
                return web.Response(status=404, content_type="text/html", text=NOT_FOUND)
            return web.FileResponse(WEB / "index.html")

        async def phone_for(request):                 # the phone's link: the right controller for it
            code = request.match_info["code"]
            ua = request.headers.get("User-Agent", "")
            page = "/iphone" if ("iPhone" in ua or "iPad" in ua or "Macintosh" in ua and "Mobile" in ua) else "/phone"
            raise web.HTTPFound(f"{page}?c={code}")
        app.router.add_get("/c/{code}", candidate_page)
        app.router.add_get("/p/{code}", phone_for)
        app.router.add_get("/scene.json", scene)
        app.router.add_get("/config.json", config)
        app.router.add_get("/level.json", level)
        app.router.add_get("/qr.svg", qr)
        app.router.add_get("/ws", ws_handler)
        app.router.add_static("/static/", WEB)
        app.router.add_static("/vendor/", THREE)
        app.router.add_static("/assets/", WEB / "assets")
        app.router.add_static("/apple/", WEB / "apple")          # the iPhone controller

        async def start(_):
            app["loop"] = asyncio.create_task(self.run())

        async def stop(_):
            app["loop"].cancel()
        app.on_startup.append(start)
        app.on_cleanup.append(stop)
        return app

    async def handle(self, ws, who: str, m: dict):
        """One message from a browser or the phone."""
        t, n = m.get("type"), self.counts.setdefault(ws, {"in": 0, "fps": 0})
        if t == "input":
            if self.game is not None:                # (the menu: no robot to move)
                self.game.set_input(m)
            if self.session is not None and who in ("phone", "iphone") and not self.session.res.get("device"):
                self.session.res["device"] = "iPhone" if who == "iphone" else "Android"
            self.phone_msg = time.perf_counter()
            if (m.get("right") or {}).get("pose"):
                self.phone_seen = time.perf_counter()
            n["in"] += 1
            if n["in"] in (1, 100) or n["in"] % 1000 == 0:
                c = m.get("right") or {}
                print(f"[input] {who}: {n['in']} messages; MOVE {c.get('grip')} GRAB {c.get('stick')} "
                      f"{'menu' if self.game is None else self.game.state}", flush=True)
        elif t == "view":
            self.view_yaw = float(m.get("yaw", 0.0))
            if self.game is not None:
                self.game.view_yaw = self.view_yaw
        elif t == "fps":                             # the game screen's frame rate, for the log
            n["fps"] += 1
            if n["fps"] % 6 == 1:
                print(f"[fps] {who}: {m.get('fps')} fps, quality {m.get('quality')}, {m.get('gpu', '')}", flush=True)
        elif t == "ping":
            await ws.send_str(json.dumps({"type": "pong", "t": m.get("t")}))
        elif t == "restart":
            self.new_game()
        elif t == "select":
            self.new_game(str(m.get("game")))
        elif t == "menu":
            self.to_menu()
        elif t == "controls":                        # the phone: which controls it uses (motion, tilt, touch)
            self.controls = str(m.get("mode", ""))[:10]
            if self.session is not None:
                self.session.res["controls"] = self.controls
            print(f"[controls] {who}: {self.controls}", flush=True)
        elif t == "join":                            # a candidate's computer screen
            ok = self.join(str(m.get("code", "")))
            await ws.send_str(json.dumps({"type": "joined", "ok": ok}))
        elif t == "begin" and self.session is not None:
            self.session.begin(str(m.get("name", ""))[:80])
        elif t == "ready" and self.session is not None:
            self.session.choose(str(m.get("choice", "")))

    def phone_url(self, page: str = "/phone") -> str | None:
        try:
            return json.loads(TUNNEL.read_text())["url"] + page
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--game", default=None, choices=list(GAMES), help="start this game (default: the menu)")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--player", default="dev")
    a = ap.parse_args()
    srv = Server(a)
    print(f"\n  Robot Buzz Wire: http://localhost:{a.port}   phone: {srv.phone_url() or '(no tunnel)'}\n", flush=True)
    web.run_app(srv.app(), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
