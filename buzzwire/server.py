"""Robot Buzz Wire: the development server (one game, fixed addresses). The simulation runs here
in real time; browsers only show it and send controller input.

    python -m buzzwire.server [--port 8000] [--levels straight curves]

    /            the game (3D view, HUD, keyboard control, the QR code for the phone)
    /phone       the phone controller (the screening test's, unchanged)
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
from pathlib import Path

import mujoco
import numpy as np
from aiohttp import WSMsgType, web

from vrteleop.webscene import export_scene

from . import config as C
from .game import Game

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).resolve().parent / "web"
THREE = ROOT / "vrteleop" / "web" / "vendor"
PHONE = ROOT / "TEST_TELEOP" / "web"
DATA = ROOT / "data" / "buzzwire"
TUNNEL = ROOT / "data" / "screening" / "tunnel.json"
POSE_EVERY = 2                 # control ticks between pose packets (100 Hz / 2 = 50 a second)
STATUS_EVERY = 10


class Server:
    def __init__(self, args):
        self.args = args
        self.cfg = C.load()
        self.clients: set[web.WebSocketResponse] = set()
        self.new_game()

    def new_game(self):
        out = DATA / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.game = Game(self.cfg, levels=self.args.levels, out_dir=out, player=self.args.player)
        self.scene_id = -1
        self._export()

    def _export(self):
        g = self.game
        self.bodies, self.scene_gz = export_scene(g.m)
        self.scene_id += 1
        self._level = (g.li, id(g.sc))

    # ---------------------------------------------------------------- the real-time loop
    async def run(self):
        g = self.game
        dt = g.dt
        t0, tick0 = time.perf_counter(), 0
        rtf_w, rtf_t, self.rtf = time.perf_counter(), 0, 1.0
        while True:
            g = self.game
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
                await self.broadcast(json.dumps(self.status()))
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
        st = self.game.status()
        st.update(type="status", scene_id=self.scene_id, rtf=round(self.rtf, 2), scores=None)
        # the phone page's fields (screening format): time left, countdown
        st["task_game"].update(limit=st["limit"], time=st["time"], intro=st["countdown"])
        return st

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
            if "/vendor/" not in request.path:
                resp.headers["Cache-Control"] = "no-cache"
            return resp
        app.middlewares.append(no_cache)

        async def index(_):
            return web.FileResponse(WEB / "index.html")

        async def phone(_):
            text = (PHONE / "phone.html").read_text(encoding="utf-8")
            text = text.replace("window.TELEOP_BASE = '/s/__CODE__';", "window.TELEOP_BASE = '';")
            text = text.replace("/test-static/phone.js?v=__BOOT__", f"/phone-static/phone.js?v={int(time.time())}")
            return web.Response(text=text, content_type="text/html")

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
            await ws.send_str(json.dumps(self.status()))
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
                    elif t == "view":
                        self.game.view_yaw = float(m.get("yaw", 0.0))
                    elif t == "ping":
                        await ws.send_str(json.dumps({"type": "pong", "t": m.get("t")}))
                    elif t == "restart":
                        self.new_game()
            finally:
                self.clients.discard(ws)
            return ws

        app.router.add_get("/", index)
        app.router.add_get("/phone", phone)
        app.router.add_get("/scene.json", scene)
        app.router.add_get("/config.json", config)
        app.router.add_get("/qr.svg", qr)
        app.router.add_get("/ws", ws_handler)
        app.router.add_static("/static/", WEB)
        app.router.add_static("/vendor/", THREE)
        app.router.add_static("/phone-static/", PHONE)

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
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--player", default="dev")
    a = ap.parse_args()
    srv = Server(a)
    print(f"\n  Robot Buzz Wire: http://localhost:{a.port}   phone: {srv.phone_url() or '(no tunnel)'}\n", flush=True)
    web.run_app(srv.app(), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
