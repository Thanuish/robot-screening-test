// Absolute hand control, TeleBench style ([controls] hand_mode = "absolute"): first a "home" is
// set (the hand held still in a circle on the camera picture); from then on where the hand is in
// the picture, compared with home, *is* where the gripper is over the table:
//
//   sideways / up-down   the palm's centre in the picture (x gain sideways, y gain up / down)
//   forward / back       the hand's size in the picture (log of size / size at home): the hand
//                        toward the camera = the gripper further from you, into the table
//   pinch                (thumb and index tips together) closes the gripper; open to let go
//   no turning           the gripper keeps pointing down, turned as at the start
//
// No clutch, no fist: a lost hand just holds the gripper where it is. Each trial starts with a
// pinch held 0.6 s ("Pinch to begin"), which also re-centres home a little (posture drift). A
// fixed delay (lag) is added to every move. The same MediaPipe tracker as core.js (preload()).
import { preload, sharedTracker, OneEuro, SETTINGS } from './core.js';

export const ABS = {
  // home: the palm must be in this box of the picture (0..1) and this size (wrist -> middle
  // knuckle, in picture heights) to set it; it is then kept inside the safe box
  accX: [0.35, 0.65], accY: [0.34, 0.71], accS: [0.095, 0.148],
  safeX: [0.30, 0.70], safeY: [0.29, 0.76], safeS: [0.091, 0.154],
  pad: { cx: 0.50, cy: 0.52, rx: 0.11, ry: 0.13 },    // the circle drawn on the picture
  homeHold: 1.5,                // [s] held still in the circle
  homeGrace: 0.2,               // [s] a moment outside doesn't start the hold again
  pinchOn: 0.30, pinchOff: 0.46, pinchSmooth: 0.5,    // thumb-index / palm size, with hysteresis
  goHold: 0.6,                  // [s] the pinch held to start a trial
  fxy: [1.3, 0.9], fz: [0.9, 0.5],   // One Euro (minCutoff, beta) sideways/up and forward
  edge: 0.12,                   // the palm this close to the picture's edge: a warning
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** opts: send(msg), status() (the simulator's last status), onFrame(state), cfg (CFG.hand:
 *  centre, height, gains, workspace, lag, table_z), log(frames) (optional). */
export class AbsoluteHand {
  constructor(opts) {
    this.opts = opts;
    this.cfg = opts.cfg;
    this.armed = false;
    this.neutral = null;                 // home: {x, y, s}
    this.home = { active: true, dwell: 0, grace: 0, acc: null, hint: '' };
    this.fx = new OneEuro({ minCutoff: ABS.fxy[0], beta: ABS.fxy[1], dCutoff: 1 });
    this.fy = new OneEuro({ minCutoff: ABS.fxy[0], beta: ABS.fxy[1], dCutoff: 1 });
    this.fz = new OneEuro({ minCutoff: ABS.fz[0], beta: ABS.fz[1], dCutoff: 1 });
    this.pinch = false;
    this.pinchS = 1;
    this.goHold = 0;
    this.gripBlocked = false;            // (after "pinch to begin": the pinch must open before it grips)
    this.reAcc = null;
    this.queue = [];                     // the delay line
    this.frames = [];
    this.pos = null;
    this.log = [];
    this.lastLog = null;
    this.lastT = null;
  }

  /** with a playing <video>: its frames go to the tracker (core.js's), the newest only */
  async start(video) {
    this.video = video;
    await preload();
    const sh = sharedTracker();
    sh.onResult = (m) => this._result(m);
    this.running = true;
    const canvas = document.createElement('canvas');
    const pump = () => {
      if (!this.running) return;
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(pump); else requestAnimationFrame(pump);
      if (sh.busy || video.readyState < 2 || !video.videoWidth) return;
      sh.busy = true;
      const k = Math.min(1, SETTINGS.maxFrame / Math.max(video.videoWidth, video.videoHeight));
      const w = Math.round(video.videoWidth * k), h = Math.round(video.videoHeight * k);
      const t = performance.now();
      const send = (bmp) => sh.worker.postMessage({ type: 'frame', bitmap: bmp, t, pose: false }, [bmp]);
      createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' }).then(send).catch(() => {
        canvas.width = w; canvas.height = h;                 // (older browsers: through a canvas)
        canvas.getContext('2d').drawImage(video, 0, 0, w, h);
        createImageBitmap(canvas).then(send).catch(() => { sh.busy = false; });
      });
    };
    pump();
  }

  stop() { this.running = false; }
  arm(on) { this.armed = on; }
  /** set home again (the candidate asked to: the next frames in the circle) */
  rehome() { this.neutral = null; this.home = { active: true, dwell: 0, grace: 0, acc: null, hint: '' }; }

  _result(m) {
    const now = performance.now(), t = now / 1000;
    const dt = this.lastT === null ? 1 / 30 : Math.min(0.1, t - this.lastT);
    this.lastT = t;
    this.frames.push(now);
    while (this.frames.length && now - this.frames[0] > 1000) this.frames.shift();
    const v = this.video, W = v.videoWidth, H = v.videoHeight;
    const st = this.opts.status() || {};
    const g = st.task_game || {};
    const state = { tracked: false, lm: m.lm, pinch: this.pinch, homeSet: !!this.neutral, home: null,
                    edge: false, goHold: 0, fps: this.frames.length, latency: Math.round(now - m.t),
                    gesture: 'none', driving: false, gripping: false };
    const lm = m.lm;
    if (lm) {
      state.tracked = true;
      // the palm: its centre (wrist and knuckles) in the picture, its size wrist -> middle knuckle
      // (in picture heights: the same numbers on a 4:3 or a 16:9 camera)
      let x = 0, y = 0;
      for (const i of [0, 5, 9, 13, 17]) { x += lm[i].x / 5; y += lm[i].y / 5; }
      const s = Math.hypot((lm[9].x - lm[0].x) * W / H, lm[9].y - lm[0].y);
      const ratio = Math.hypot((lm[4].x - lm[8].x) * W / H, lm[4].y - lm[8].y) / Math.max(s, 1e-4);
      this.pinchS += ABS.pinchSmooth * (ratio - this.pinchS);
      this.pinch = this.pinch ? this.pinchS < ABS.pinchOff : this.pinchS < ABS.pinchOn;
      state.pinch = this.pinch;
      state.pinchRatio = this.pinchS;
      state.edge = x < ABS.edge || x > 1 - ABS.edge || y < ABS.edge || y > 1 - ABS.edge;
      state.palm = { x, y, s };
      if (!this.neutral) this._homeStep({ x, y, s }, dt, state);
      else {
        // where the gripper goes: sideways, up-down from the palm's place against home; forward
        // from the log of its size against home (filtered: One Euro)
        const n = this.neutral, c = this.cfg, ws = c.workspace, [g0, g1, g2] = c.gains;
        const px = this.fx.filter([x], t)[0], py = this.fy.filter([y], t)[0];
        const lz = this.fz.filter([Math.log(Math.max(s, 1e-4) / n.s)], t)[0];
        const side = clamp((n.x - px) * g0, -ws.side, ws.side);      // (the picture isn't mirrored: hand to your right = x smaller)
        const up = clamp(c.height + (n.y - py) * g1, ws.low, ws.high);
        const fwd = clamp(lz * g2, -ws.depth, ws.depth);
        // the robot's frame: x forward (away from you), y to the left, z up
        this.pos = [c.centre[0] + fwd, c.centre[1] - side, c.table_z + up];
        state.saturated = Math.abs(side) >= ws.side || up <= ws.low || up >= ws.high || Math.abs(fwd) >= ws.depth;
      }
      state.homeSet = !!this.neutral;
      // pinch to begin (a trial waiting): held 0.6 s, with the hand near home; meanwhile home
      // follows the hand a little (bounded: posture drift, not a new home)
      if (g.ready && this.neutral && this.armed) {
        if (this.pinch) {
          this.goHold += dt;
          this.reAcc = this.reAcc || { n: 0, x: 0, y: 0, s: 0 };
          Object.assign(this.reAcc, { n: this.reAcc.n + 1, x: this.reAcc.x + x, y: this.reAcc.y + y, s: this.reAcc.s + s });
        } else { this.goHold = 0; this.reAcc = null; }
        if (this.goHold >= ABS.goHold) {
          const r = this.reAcc, n = this.neutral;
          if (r && r.n >= 8) {
            n.x = clamp(n.x + clamp(r.x / r.n - n.x, -0.05, 0.05), ABS.safeX[0], ABS.safeX[1]);
            n.y = clamp(n.y + clamp(r.y / r.n - n.y, -0.05, 0.05), ABS.safeY[0], ABS.safeY[1]);
            n.s = clamp(n.s * clamp((r.s / r.n) / n.s, 0.94, 1.06), ABS.safeS[0], ABS.safeS[1]);
          }
          this.goHold = 0; this.reAcc = null;
          this.gripBlocked = true;
          this.opts.send({ type: 'go' });
        }
      } else { this.goHold = 0; this.reAcc = null; }
      if (!this.pinch) this.gripBlocked = false;
      state.goHold = this.goHold / ABS.goHold;
      state.gesture = this.pinch ? 'pinch' : 'open';
      state.gripping = this.pinch && !this.gripBlocked && !g.ready;
      state.driving = this.armed && !!this.neutral;
    }
    state.home = this.neutral ? null : { ...this.home };
    this._send(state, t);
    this._log(m, state, t);
    this.opts.onFrame?.(state);
  }

  /** setting home: guide the hand into the circle at the right size, hold it there */
  _homeStep(p, dt, state) {
    const h = this.home;
    const inX = p.x >= ABS.accX[0] && p.x <= ABS.accX[1], inY = p.y >= ABS.accY[0] && p.y <= ABS.accY[1];
    const inS = p.s >= ABS.accS[0] && p.s <= ABS.accS[1];
    if (inX && inY && inS) {
      h.grace = 0;
      h.acc = h.acc || { n: 0, x: 0, y: 0, s: 0 };
      Object.assign(h.acc, { n: h.acc.n + 1, x: h.acc.x + p.x, y: h.acc.y + p.y, s: h.acc.s + p.s });
      h.dwell += dt;
      h.hint = 'Hold still…';
      if (h.dwell >= ABS.homeHold) {
        const a = h.acc;
        this.neutral = { x: clamp(a.x / a.n, ABS.safeX[0], ABS.safeX[1]), y: clamp(a.y / a.n, ABS.safeY[0], ABS.safeY[1]),
                         s: clamp(a.s / a.n, ABS.safeS[0], ABS.safeS[1]) };
        for (const f of [this.fx, this.fy, this.fz]) f.reset();
        h.active = false;
        h.hint = 'Home set. Steer from this spot.';
      }
      return;
    }
    if (h.dwell > 0 && h.grace < ABS.homeGrace) { h.grace += dt; return; }
    h.dwell = 0; h.acc = null;
    h.hint = !inS ? (p.s < ABS.accS[0] ? 'Come a little closer' : 'Sit back a little')
      : !inX ? (p.x < ABS.accX[0] ? 'Move your hand to your left' : 'Move your hand to your right')   // (the picture isn't mirrored: small x = your right)
      : (p.y < ABS.accY[0] ? 'Lower your hand a little' : 'Raise your hand a little');
  }

  /** the controller input, through the delay line */
  _send(state, t) {
    if (!this.armed) return;
    const side = (this.opts.status() || {}).task_game?.arm || 'right';
    const ctrl = state.tracked && this.pos
      ? { pose: [...this.pos, 1, 0, 0, 0], abs: true, grip: 1, stick: [0, state.gripping ? 1 : -1], buttons: [], epoch: 0 }
      : null;                                         // (no hand: the gripper stays where it is)
    this.queue.push({ due: t + (this.cfg.lag || 0), msg: { type: 'input', t, head: null, left: null, right: null, [side]: ctrl, tracker: 'hand' } });
    while (this.queue.length && this.queue[0].due <= t + 1e-3) this.opts.send(this.queue.shift().msg);
  }

  _log(m, state, t) {
    if (!this.opts.log) return;
    const r4 = (v) => Math.round(v * 1e4) / 1e4;
    this.log.push({ t: r4(t), lat: state.latency, g: state.gesture, d: state.driving ? 1 : 0,
                    lm: m.lm ? m.lm.map(p => [r4(p.x), r4(p.y)]) : null,
                    wl: m.wl ? m.wl.map(p => [r4(p.x), r4(p.y), r4(p.z)]) : null,
                    pos: this.pos ? this.pos.map(r4) : null, W: this.video.videoWidth, H: this.video.videoHeight });
    if (this.lastLog === null) this.lastLog = t;
    if (t - this.lastLog >= 2) { this.opts.log(this.log); this.log = []; this.lastLog = t; }
  }
}
