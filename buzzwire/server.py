"""Robot Buzz Wire: the development server (one game, fixed addresses). The simulation runs here
in real time; browsers only show it and send controller input.

    python -m buzzwire.server [--port 8000] [--levels straight curves]

    /            the games: a menu (Buzz Wire, Cube Tasks, Tower), the 3D view, the HUD, the QR code
    /phone       the phone controller (the only controller: hold to move, grab / release)
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
        self.phone_seen = 0.0                 # when the phone last sent its pose
        self.rtf = 1.0                        # simulated seconds per real second (measured by run())
        self.scene_id = -1                    # bumped with every new scene (browsers reload it)
        self.game = None                      # None: the menu
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
        self.game = GAMES[kind][0](self.cfg, levels=levels, out_dir=out, player=self.args.player)
        self.kind = kind
        self._export()
        print(f"[game] {GAMES[kind][1]}", flush=True)

    def to_menu(self):
        self.game = None

    def _export(self):
        g = self.game
        self.bodies, self.scene_gz = export_scene(g.m)
        self.scene_id += 1
        self._level = (g.li, id(g.sc))

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
            if g is None:                             # the menu: only the status, 10 times a second
                await self.broadcast(dumps(self.status()))
                await asyncio.sleep(0.1)
                t0, tick0, rtf_t, rtf_w = time.perf_counter(), 0, 0, time.perf_counter()
                continue
            if (g.li, id(g.sc)) != self._level:
                self._export()                        # a new level: new geometry for the browsers
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
        phone = time.perf_counter() - self.phone_seen < 1.5
        if self.game is None:
            return dict(type="status", state="menu", phone=phone, scene_id=self.scene_id, rtf=1.0,
                        games=[dict(key=k, title=v[1], text=v[2]) for k, v in GAMES.items()])
        st = self.game.status()
        st.update(type="status", scene_id=self.scene_id, rtf=round(self.rtf, 2), scores=None, phone=phone)
        return st

    def level_geometry(self) -> dict:
        """The current level's shapes for the browser to draw finely (the physics uses simple ones)."""
        if self.game is None:
            return dict(kind="menu", scene_id=self.scene_id, table_z=C.TABLE_Z)
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
            return web.json_response(dict(self.cfg.public(), phone_url=self.phone_url()))

        async def qr(_):
            import segno
            buf = io.BytesIO()
            segno.make(self.phone_url() or "no public address", error="m").save(buf, kind="svg", scale=6, border=2)
            return web.Response(body=buf.getvalue(), content_type="image/svg+xml")

        async def ws_handler(request):
            ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
            await ws.prepare(request)
            self.clients.add(ws)
            who = "phone" if "Android" in request.headers.get("User-Agent", "") else "browser"
            print(f"[ws] {who} connected ({len(self.clients)} open)", flush=True)
            await ws.send_str(dumps(self.status()))
            n_in = 0
            fps_n = 0
            try:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        continue
                    try:
                        m = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    t = m.get("type")
                    if t == "input":
                        self.game.set_input(m)
                        if (m.get("right") or {}).get("pose"):
                            self.phone_seen = time.perf_counter()
                        n_in += 1
                        if n_in in (1, 100) or n_in % 1000 == 0:
                            c = m.get("right") or {}
                            print(f"[input] {who}: {n_in} messages; MOVE {c.get('grip')} GRAB {c.get('stick')} "
                                  f"state {self.game.state}", flush=True)
                    elif t == "view":
                        self.game.view_yaw = float(m.get("yaw", 0.0))
                    elif t == "fps":                       # the game screen's frame rate, for the log
                        fps_n += 1
                        if fps_n % 6 == 1:
                            print(f"[fps] {who}: {m.get('fps')} fps, quality {m.get('quality')}, "
                                  f"{m.get('gpu', '')}", flush=True)
                    elif t == "ping":
                        await ws.send_str(json.dumps({"type": "pong", "t": m.get("t")}))
                    elif t == "restart":
                        self.new_game()
                    elif t == "select":
                        self.new_game(str(m.get("game")))
                    elif t == "menu":
                        self.to_menu()
            finally:
                self.clients.discard(ws)
                print(f"[ws] {who} disconnected", flush=True)
            return ws

        app.router.add_get("/", index)
        app.router.add_get("/phone", phone)
        app.router.add_get("/scene.json", scene)
        app.router.add_get("/config.json", config)
        app.router.add_get("/level.json", level)
        app.router.add_get("/qr.svg", qr)
        app.router.add_get("/ws", ws_handler)
        app.router.add_static("/static/", WEB)
        app.router.add_static("/vendor/", THREE)
        app.router.add_static("/assets/", WEB / "assets")

        async def start(_):
            app["loop"] = asyncio.create_task(self.run())

        async def stop(_):
            app["loop"].cancel()
        app.on_startup.append(start)
        app.on_cleanup.append(stop)
        return app

    def phone_url(self) -> str | None:
        try:
            return json.loads(TUNNEL.read_text())["url"] + "/phone"
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
