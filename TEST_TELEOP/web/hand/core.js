// Hand control: a camera (a computer's webcam or any phone's) watches the candidate's hand, and
// the hand drives the robot arm. MediaPipe's Hand Landmarker (Google, Apache 2.0) finds the hand's
// 21 points in each frame, in a background thread of this browser; the video never leaves the
// device, only the hand's pose is sent, as the same controller input as a VR controller or the
// phone (so the server, the arm and the tasks don't change).
//
//   open hand   drives the robot (the clutch is held): the gripper follows the hand's motion
//               and turns (in tasks with full rotation it also tilts)
//   fist        pauses: move the hand without moving the robot (like lifting a mouse), open again
//   pinch       (thumb and index finger together) closes the gripper; open the pinch to let go
//               (gripMode 'pinch'); with gripMode 'key' a key toggles the gripper instead
//               (toggleGrip(): the computer's SPACE bar)
//
// Frames. The camera looks at the person from in front (the screen, or a phone standing there):
// camera x = right in its image (the person's left), y = down, z = away from the camera. MuJoCo:
// x forward (into the screen), y left, z up.
export const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
export const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export const SETTINGS = {
  hfov: 60,                    // [deg] the camera's field of view, a guess (an error only scales the motion a little)
  pinchOn: 0.32, pinchOff: 0.5, // thumb-index tips / palm length: closed below, open above (hysteresis: no flicker)
  // One Euro filter (adaptive: steady when the hand is still, little lag when it moves)
  pos: { minCutoff: 0.45, beta: 5, dCutoff: 1.0 },    // position [m]
  depth: { minCutoff: 0.3, beta: 2.5, dCutoff: 1.0 },  // distance from the camera (the noisiest) [m]
  rot: { minCutoff: 0.5, beta: 0.4, dCutoff: 1.0 },    // orientation (per rad/s)
  maxFrame: 640,               // frames are scaled to this long side for the tracker (faster, same result)
};

// ------------------------------------------------------------ small math
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mulMV = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const mulMM = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
export const CAM_TO_MJ = [[0, 0, -1], [1, 0, 0], [0, -1, 0]];   // MuJoCo vector = this x camera vector
export function matToQuat(R) {        // -> [w, x, y, z]
  const tr = R[0][0] + R[1][1] + R[2][2];
  let w, x, y, z;
  if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); w = s / 4; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s; }
  else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) { const s = 2 * Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]); w = (R[2][1] - R[1][2]) / s; x = s / 4; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s; }
  else if (R[1][1] > R[2][2]) { const s = 2 * Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]); w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = s / 4; z = (R[1][2] + R[2][1]) / s; }
  else { const s = 2 * Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]); w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = s / 4; }
  return [w, x, y, z];
}

// ------------------------------------------------------------ One Euro filter (Casiez et al. 2012)
const alpha = (cutoff, dt) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };
export class OneEuro {
  constructor({ minCutoff, beta, dCutoff }) { this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff; this.reset(); }
  reset() { this.x = null; this.dx = null; this.t = null; }
  /** x: an array (a vector); speed(dx): its speed, for the adaptive cutoff */
  filter(x, t, speed = (d) => Math.hypot(...d)) {
    if (this.x === null || t - this.t > 0.5) { this.x = x.slice(); this.dx = x.map(() => 0); this.t = t; return this.x; }
    const dt = Math.max(1e-3, t - this.t);
    this.t = t;
    const a_d = alpha(this.dCutoff, dt);
    this.dx = x.map((v, i) => this.dx[i] + a_d * ((v - this.x[i]) / dt - this.dx[i]));
    const a = alpha(this.minCutoff + this.beta * speed(this.dx), dt);
    this.x = x.map((v, i) => this.x[i] + a * (v - this.x[i]));
    return this.x;
  }
}

// ------------------------------------------------------------ the hand's pose and gesture
// the bones of the palm: their real length (MediaPipe's 3D hand, metres) against their length in
// the image gives the hand's distance from the camera
const PALM = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]];
export const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                      [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

/** One frame: lm (21 image points 0..1), wl (21 3D points [m]), W x H the image, f the focal
 *  length in image widths. In the camera frame: pos (the palm's centre [m]), D (its distance),
 *  R (hand axes as columns: a = wrist -> middle knuckle, b = index -> pinky knuckle, n = palm
 *  normal), pinch (thumb-index tips / palm length), curled (how many of middle, ring, pinky are
 *  folded in: 3 = a fist). */
export function handPose(lm, wl, W, H, f) {
  const P = (i) => [lm[i].x, lm[i].y * H / W];
  const Q = (i) => [wl[i].x, wl[i].y, wl[i].z];
  let real = 0, img = 0;
  for (const [a, b] of PALM) {
    real += len(sub(Q(b), Q(a)));
    img += Math.hypot(P(b)[0] - P(a)[0], P(b)[1] - P(a)[1]);
  }
  const D = f * real / Math.max(img, 1e-4);
  const c = [0, 5, 9, 13, 17].map(P).reduce((s, p) => [s[0] + p[0] / 5, s[1] + p[1] / 5], [0, 0]);
  const pos = [(c[0] - 0.5) * D / f, (c[1] - 0.5 * H / W) * D / f, D];
  const a = norm(sub(Q(9), Q(0)));
  let b = norm(sub(Q(17), Q(5)));
  const n = norm(cross(a, b));
  b = cross(n, a);
  const R = [[a[0], b[0], n[0]], [a[1], b[1], n[1]], [a[2], b[2], n[2]]];
  const palm = Math.max(len(sub(Q(9), Q(0))), 1e-3);
  const pinch = len(sub(Q(4), Q(8))) / palm;
  // a finger is folded in when its tip is closer to the wrist than its knuckle is (plus a little)
  const folded = (tip, knuckle) => len(sub(Q(tip), Q(0))) < 1.15 * len(sub(Q(knuckle), Q(0)));
  let curled = 0;
  for (const [tip, knuckle] of [[12, 9], [16, 13], [20, 17]]) if (folded(tip, knuckle)) curled++;
  const indexFolded = folded(8, 5);
  return { pos, D, R, a, n, pinch, curled, indexFolded };
}

// ------------------------------------------------------------ the tracker (background thread)
let shared = null;                   // one tracker per page, loaded once (preload() at page open)
/** Start loading the tracker now (the ~17 MB engine and model take a few seconds the first time;
 *  afterwards the browser has them). Resolves with {delegate} or rejects. */
export function preload() {
  if (shared) return shared.ready;
  const worker = new Worker(new URL('./worker.js', import.meta.url));
  const s = shared = { worker, busy: false, onResult: null, delegate: null };
  s.ready = new Promise((resolve, reject) => {
    worker.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'ready') { s.delegate = m.delegate; resolve(m); }
      else if (m.type === 'result') { s.busy = false; s.onResult?.(m); }
      else if (m.type === 'error') { s.busy = false; if (!s.delegate) reject(new Error(m.message)); else console.warn('hand tracker:', m.message); }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'the hand tracker could not start'));
  });
  worker.postMessage({ type: 'init', mp: MP, model: MODEL });
  return s.ready;
}

/** Hand control. opts:
 *    send(msg)      to the simulator (the controller input)
 *    status()       the simulator's last status (arm, view_yaw, task_game.rotation)
 *    onFrame(state) after each frame: {tracked, lm, gesture: 'open' | 'fist' | 'pinch' | 'none',
 *                   driving, gripping, depth, fps, latency (camera -> result, ms)}
 *  start(video) with a playing <video>; arm(true) to start sending (after the checklist). */
export class HandControl {
  constructor(opts) {
    this.opts = opts;
    this.gripMode = opts.gripMode || 'pinch';     // 'pinch': the pinch grips; 'key': toggleGrip() does
    this.armed = false;
    this.fist = false;
    this.pinched = false;
    this.epoch = 1;
    this.lastRot = 'yaw';
    this.lastVY = 0;
    this.fPos = new OneEuro(SETTINGS.pos);
    this.fDepth = new OneEuro(SETTINGS.depth);
    this.fRot = new OneEuro(SETTINGS.rot);
    this.frames = [];
  }

  async start(video) {
    this.video = video;
    await preload();
    shared.onResult = (m) => this._result(m);
    this.running = true;
    const canvas = document.createElement('canvas');
    this._pump = () => {
      if (!this.running) return;
      if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(this._pump);
      else requestAnimationFrame(this._pump);
      if (shared.busy || video.readyState < 2 || !video.videoWidth) return;   // (only the newest frame: no queue)
      shared.busy = true;
      const k = Math.min(1, SETTINGS.maxFrame / Math.max(video.videoWidth, video.videoHeight));
      const w = Math.round(video.videoWidth * k), h = Math.round(video.videoHeight * k);
      const t = performance.now();
      const send = (bmp) => shared.worker.postMessage({ type: 'frame', bitmap: bmp, t }, [bmp]);
      createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' }).then(send).catch(() => {
        canvas.width = w; canvas.height = h;                 // (older browsers: through a canvas)
        canvas.getContext('2d').drawImage(video, 0, 0, w, h);
        createImageBitmap(canvas).then(send).catch(() => { shared.busy = false; });
      });
    };
    this._pump();
  }

  stop() { this.running = false; if (shared) shared.onResult = null; }
  /** gripMode 'key': close the gripper if open, open it if closed */
  toggleGrip() { this.pinched = !this.pinched; return this.pinched; }
  arm(on) { this.armed = on; this.epoch++; }

  _result(m) {
    const now = performance.now(), t = now / 1000;
    this.frames.push(now);
    while (this.frames.length && now - this.frames[0] > 1000) this.frames.shift();
    const v = this.video, W = v.videoWidth, H = v.videoHeight;
    const st = this.opts.status() || {};
    const state = { tracked: false, lm: m.lm, gesture: 'none', driving: false, gripping: this.pinched,
                    depth: null, fps: this.frames.length, latency: Math.round(now - m.t) };
    let pose = null;
    if (m.lm && m.wl) {
      const f = 0.5 / Math.tan((SETTINGS.hfov / 2) * Math.PI / 180);
      const hp = handPose(m.lm, m.wl, W, H, f);
      state.tracked = true;
      // gestures (hysteresis): a fist (the index folded in too) pauses; a pinch grips and keeps
      // driving (in a pinch the index bends toward the thumb and the others may curl: no fist)
      // (the index folded into the palm decides: then it is a fist whatever the thumb does)
      this.fist = hp.indexFolded && (this.fist ? hp.curled >= 2 : hp.curled >= 3);
      if (!this.fist && this.gripMode === 'pinch')
        this.pinched = this.pinched ? hp.pinch < SETTINGS.pinchOff : hp.pinch < SETTINGS.pinchOn;
      state.gesture = this.fist ? 'fist' : this.pinched ? 'pinch' : 'open';
      state.gripping = this.pinched;
      // smoothed position (depth filtered on its own: it is the noisiest)
      const D = this.fDepth.filter([hp.D], t)[0];
      const camPos = this.fPos.filter([hp.pos[0] * D / hp.D, hp.pos[1] * D / hp.D, D], t);
      state.depth = D;
      // orientation: the hand's frame in MuJoCo axes, smoothed as a quaternion
      const Rmj = mulMM(CAM_TO_MJ, hp.R);
      let q = matToQuat(Rmj);
      if (this.fRot.x && q[0] * this.fRot.x[0] + q[1] * this.fRot.x[1] + q[2] * this.fRot.x[2] + q[3] * this.fRot.x[3] < 0) q = q.map(x => -x);
      q = this.fRot.filter(q, t, (d) => 2 * Math.hypot(d[1], d[2], d[3]));
      const qn = Math.hypot(...q);
      q = q.map(x => x / qn);
      const vy = st.view_yaw ?? 0;
      if (Math.abs(vy - this.lastVY) > 0.005) { this.lastVY = vy; this.epoch++; }
      const rotation = st.task_game?.rotation || 'yaw';
      if (rotation !== this.lastRot) { this.lastRot = rotation; this.epoch++; }   // (re-anchor: no jump)
      const mj0 = mulMV(CAM_TO_MJ, camPos);
      const cv = Math.cos(vy), sv = Math.sin(vy);
      const mj = [cv * mj0[0] - sv * mj0[1], sv * mj0[0] + cv * mj0[1], mj0[2]];
      const zr = [[cv, -sv, 0], [sv, cv, 0], [0, 0, 1]];
      if (rotation === 'full') {
        q = matToQuat(mulMM(zr, quatToMat(q)));
      } else {
        // heading only (about the vertical): where the palm faces, or (hand flat) the fingers
        const R = quatToMat(q);
        const nm = [R[0][2], R[1][2], R[2][2]], am = [R[0][0], R[1][0], R[2][0]];
        const hv = Math.hypot(nm[0], nm[1]) > 0.35 ? nm : am;
        const h = (Math.atan2(hv[1], hv[0]) + vy) / 2;
        q = [Math.cos(h), 0, 0, Math.sin(h)];
      }
      pose = [...mj, ...q];
      state.driving = this.armed && !this.fist;
    } else {
      this.fist = false;
    }
    this.opts.onFrame?.(state);
    if (!this.armed) return;
    const side = st.task_game?.arm || 'right';
    const right = pose ? { pose, grip: state.driving ? 1 : 0, stick: [0, this.pinched ? 1 : -1], buttons: [], epoch: this.epoch } : null;
    this.opts.send({ type: 'input', t, head: null, left: null, right: null, [side]: right, tracker: 'hand' });
  }
}

function quatToMat([w, x, y, z]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}

/** Open a camera (the front one on a phone: it stands in front of the person like a webcam). */
export async function openCamera(facing = 'user') {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: { facingMode: facing, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 60, max: 60 } } });
  const video = document.createElement('video');
  video.setAttribute('playsinline', ''); video.muted = true; video.srcObject = stream;
  await video.play();
  return { video, stream, stop: () => stream.getTracks().forEach(t => t.stop()) };
}
