// Hand control: a camera (a computer's webcam or any phone's) watches the candidate's hand, and
// the hand drives the robot arm. MediaPipe's Hand Landmarker (Google, Apache 2.0) finds the hand's
// 21 points in each frame, in a background thread of this browser; the video never leaves the
// device, only the hand's pose is sent, as the same controller input as a VR controller or the
// phone (so the server, the arm and the tasks don't change).
//
//   open hand   drives the robot (the clutch is held): the gripper follows the hand's motion
//   twist       the hand turned like a doorknob (palm still toward the camera) turns the gripper
//               left / right: measured in the flat image, where it is steady
//   tilt        in tasks with full rotation the hand's whole orientation counts (pose3d.js: the
//               palm's rigid points fitted in 3D and in the image, the forearm from MediaPipe Pose
//               when the palm turns edge-on): fingertips toward / away from the screen tilt the
//               gripper forward / back, the palm turned left / right tilts it sideways, the twist
//               still turns it. setTilt() (W/S, A/D) adds to that from the keyboard
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
// MediaPipe Pose (the light model): the elbow and wrist, for the forearm's direction
export const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
import { solveHandPose, PalmModel, PALM_IDS, forearm, ForearmFusion, OneEuroRot, BacklashRot } from './pose3d.js';

export const SETTINGS = {
  hfov: 60,                    // [deg] the camera's field of view, a guess (an error only scales the motion a little)
  pinchOn: 0.32, pinchOff: 0.5, // thumb-index tips / palm length: closed below, open above (hysteresis: no flicker)
  // One Euro filter (adaptive: steady when the hand is still, little lag when it moves)
  pos: { minCutoff: 0.45, beta: 5, dCutoff: 1.0 },    // position [m]
  depth: { minCutoff: 0.15, beta: 2.0, dCutoff: 1.0 },  // distance from the camera (the noisiest) [m]
  rot: { minCutoff: 0.5, beta: 0.4, dCutoff: 1.0 },    // orientation (per rad/s)
  refDepth: { minCutoff: 0.15, beta: 0.5, dCutoff: 1.0 },   // the reference distance that scales left/right and up/down
  // a dead band ("backlash") after the filters: while the hand stays within this of where the
  // output is, nothing moves (a still hand holds the robot perfectly still); beyond it the output
  // follows from the band's edge (smoothly, no jump)
  deadPos: [0.003, 0.003, 0.012],   // [m] camera x (left/right), y (up/down), z (toward/away: the noisiest)
  deadRot: 2.0,                // [deg]
  twist: { minCutoff: 0.6, beta: 0.6, dCutoff: 1.0 },   // the doorknob turn (per rad/s)
  tiltRate: 60,                // [deg/s] how fast the tilt keys tilt the gripper
  // after a fist (or a lost hand) the robot drives again only once the hand has been open this
  // many frames (~0.25 s): opening the hand changes its estimated centre, size and twist, and
  // the filters would carry that change into the robot; at that moment they start afresh
  settleFrames: 12,
  // frames are scaled to this long side for the tracker. The tracker looks at the hand in a
  // 224 x 224 crop of the frame: a hand ~150 px tall in a 640 px frame is blown up (blurry
  // points); at 1280 it has the detail. (The models' cost doesn't depend on this.)
  maxFrame: 1280,
  // the hand's full orientation (tasks with full rotation): its filter and dead band
  rot3d: { minCutoff: 0.3, beta: 0.5, dCutoff: 1.0 },
  deadRot3d: 2.5,              // [deg]
  forearm: true,               // track the forearm too (MediaPipe Pose, a second model: a little slower)
  poseEvery: 3,                // run it on every n-th frame (the forearm moves slowly)
  // closing or opening a pinch bends the palm a little (its arch): the pose is held this many
  // frames, then taken up again from where the hand is (OAT's grasp stabilization)
  gripHoldFrames: 8,
  logEvery: 2.0,               // [s] with opts.log: the raw tracking (points only) sent this often
};
// the hand's frame (the camera's, palm toward it) -> the gripper's (pointing down): the palm's
// normal toward the camera is the gripper's down, the fingers (up) its forward. So the doorknob
// twist turns the gripper about the vertical, as in the tasks with only that turn
const HAND_TO_MJ = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];

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

// ------------------------------------------------------------ dead band (backlash)
/** A vector that moves only when the input leaves a band around it (half-widths per axis: an
 *  ellipsoid), and then just enough to keep the input on the band's edge: tiny wobbles are
 *  ignored, motion is kept. */
export class Backlash {
  constructor(widths) { this.w = widths; this.x = null; }
  reset() { this.x = null; }
  filter(v) {
    if (this.x === null) { this.x = v.slice(); return this.x; }
    const d = v.map((u, i) => u - this.x[i]);
    const n = Math.hypot(...d.map((u, i) => u / this.w[i]));     // (1 = on the band's edge)
    if (n > 1) this.x = this.x.map((u, i) => u + d[i] * (1 - 1 / n));
    return this.x;
  }
}
/** The same for an orientation (quaternion w, x, y, z): it turns only past `deg`. */
export class BacklashQuat {
  constructor(deg) { this.rad = deg * Math.PI / 180; this.q = null; }
  reset() { this.q = null; }
  filter(q) {
    if (this.q === null) { this.q = q.slice(); return this.q; }
    let dot = q[0] * this.q[0] + q[1] * this.q[1] + q[2] * this.q[2] + q[3] * this.q[3];
    if (dot < 0) { q = q.map(x => -x); dot = -dot; }
    const ang = 2 * Math.acos(Math.min(1, dot));
    if (ang > this.rad) {                                   // (slerp toward q, keeping `rad` behind it)
      const f = 1 - this.rad / ang, th = ang / 2;
      const a = Math.sin((1 - f) * th) / Math.sin(th), b = Math.sin(f * th) / Math.sin(th);
      const r = this.q.map((x, i) => a * x + b * q[i]);
      const n = Math.hypot(...r);
      this.q = r.map(x => x / n);
    }
    return this.q;
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
  worker.postMessage({ type: 'init', mp: MP, model: MODEL, poseModel: SETTINGS.forearm ? POSE_MODEL : null });
  return s.ready;
}

/** The page's tracker ({worker, busy, onResult}) for another controller (absolute.js). */
export function sharedTracker() { return shared; }

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
    this.fRef = new OneEuro(SETTINGS.refDepth);
    this.fTwist = new OneEuro(SETTINGS.twist);
    this.twist = null;                            // the doorknob turn [rad] (unwrapped, then held in a dead band)
    this.twistRaw = null;
    this.pitch = 0; this.roll = 0;                // from the tilt keys [rad]
    this.tilt = { pitch: 0, roll: 0 };            // the keys held: -1, 0, +1
    this.lastT = null;
    this.bPos = new Backlash(SETTINGS.deadPos);
    this.bRot = new BacklashQuat(SETTINGS.deadRot);
    this.frames = [];
    // the full 3D pose (pose3d.js)
    this.palm = new PalmModel();                   // this person's palm, learned while it faces the camera
    this.sol = null;                               // the last frame's solution
    this.fusion = new ForearmFusion();
    this.fRot3 = new OneEuroRot(SETTINGS.rot3d);
    this.bRot3 = new BacklashRot(SETTINGS.deadRot3d);
    this.lastPose = null;
    this.holdFrames = 0;                           // grip changes: frames the pose is still held
    this.frameNo = 0;
    this.log = [];
    this.lastLog = null;
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
      const pose = SETTINGS.forearm && (this.frameNo++ % SETTINGS.poseEvery === 0);
      const send = (bmp) => shared.worker.postMessage({ type: 'frame', bitmap: bmp, t, pose }, [bmp]);
      createImageBitmap(video, { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' }).then(send).catch(() => {
        canvas.width = w; canvas.height = h;                 // (older browsers: through a canvas)
        canvas.getContext('2d').drawImage(video, 0, 0, w, h);
        createImageBitmap(canvas).then(send).catch(() => { shared.busy = false; });
      });
    };
    this._pump();
  }

  stop() { this.running = false; if (shared) shared.onResult = null; }
  /** the tilt keys held now: pitch +1 forward / -1 back, roll +1 right / -1 left */
  setTilt(pitch, roll) { this.tilt = { pitch, roll }; }
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
      if (m.pose) this.pose = m.pose;                 // (the forearm: from the last frame it ran on)
      // the full 3D pose; a bad solution (a confused tracker) is dropped and the last one kept
      let sol = null;
      try { sol = solveHandPose(m.lm, m.wl, W, H, f, this.palm, this.sol); } catch (e) { sol = null; }
      if (sol && (!(sol.t[2] > 0.12 && sol.t[2] < 2.5) || !(sol.err < 25))) sol = null;
      if (sol) this.sol = sol;
      else if (this.sol) sol = this.sol;
      // gestures (hysteresis): a fist (the index folded in too) pauses; a pinch grips and keeps
      // driving (in a pinch the index bends toward the thumb and the others may curl: no fist)
      // (the index folded into the palm decides: then it is a fist whatever the thumb does)
      this.fist = hp.indexFolded && (this.fist ? hp.curled >= 2 : hp.curled >= 3);
      if (!this.fist && this.gripMode === 'pinch')
        this.pinched = this.pinched ? hp.pinch < SETTINGS.pinchOff : hp.pinch < SETTINGS.pinchOn;
      state.gesture = this.fist ? 'fist' : this.pinched ? 'pinch' : 'open';
      state.gripping = this.pinched;
      // learn the palm (open hand only: a fist or a pinch bends it)
      if (sol && state.gesture === 'open') {
        const img = PALM_IDS.map(i => [(m.lm[i].x - 0.5) / f * sol.t[2], (m.lm[i].y - 0.5) * (H / W) / f * sol.t[2], 0]);
        this.palm.add(m.wl, img, sol.facing);
      }
      // a grip change: hold the pose a moment, then take it up again from the hand (a new epoch)
      if (this.pinched !== this.wasPinched && this.wasPinched !== undefined) this.holdFrames = SETTINGS.gripHoldFrames;
      this.wasPinched = this.pinched;
      // settle (see SETTINGS.settleFrames): then the filters restart from the hand's pose now and
      // the clutch anchors there (a new epoch), so opening the hand can't move the robot
      this.openFrames = this.fist ? 0 : (this.openFrames || 0) + 1;
      const settled = this.openFrames >= SETTINGS.settleFrames;
      if (this.openFrames === 1) {                     // (the hand just opened: the filters start afresh ...
        for (const fl of [this.fPos, this.fDepth, this.fRef, this.fTwist]) fl.reset();
        this.bPos.reset();
        this.twist = null;
        this.twistRaw = null;
      }
      if (settled && !this.wasSettled) this.epoch++;    // (... and have settled when the clutch anchors)
      this.wasSettled = settled;
      // position: left/right and up/down from where the hand is in the image, scaled by a steady
      // reference distance (the per-frame distance, from the palm's size, is the noisiest number:
      // scaling by it shook the sideways motion too); forward/back from the distance, filtered
      // on its own; then the dead band (a still hand: no motion at all)
      // (from the 3D solution: the distance from the palm's size *at its tilt*, so tilting the palm
      // doesn't read as moving it away; the point is the wrist, which the hand turns about, so
      // turning the hand doesn't move the gripper)
      const pt = sol ? sol.wrist : null;
      const Draw = pt ? pt[2] : hp.D;
      const dirX = pt ? pt[0] / pt[2] : hp.pos[0] / hp.D, dirY = pt ? pt[1] / pt[2] : hp.pos[1] / hp.D;
      const D = this.fDepth.filter([Draw], t)[0];
      const Dref = this.fRef.filter([Draw], t)[0];
      const camPos = this.fPos.filter([dirX * Dref, dirY * Dref, D], t);
      const held = this.bPos.filter(camPos);
      state.depth = D;
      // the twist: the angle of the wrist -> middle knuckle line in the image (0 = fingers up).
      // In the camera's image the person's clockwise turn goes the other way, so this angle is
      // already the robot's turn: clockwise (seen from behind the hand) = clockwise from above
      const ang = Math.atan2((m.lm[9].x - m.lm[0].x) * W, -(m.lm[9].y - m.lm[0].y) * H);
      if (this.twistRaw === null) this.twistRaw = ang;
      this.twistRaw += Math.atan2(Math.sin(ang - this.twistRaw), Math.cos(ang - this.twistRaw));   // (unwrapped)
      const tw = this.fTwist.filter([this.twistRaw], t)[0];
      const band = SETTINGS.deadRot * Math.PI / 180;
      if (this.twist === null) this.twist = tw;
      else if (Math.abs(tw - this.twist) > band) this.twist = tw - Math.sign(tw - this.twist) * band;
      // the tilt keys (only in tasks with full rotation)
      const dt = this.lastT === null ? 0 : Math.min(0.1, t - this.lastT);
      this.lastT = t;
      const vy = st.view_yaw ?? 0;
      if (Math.abs(vy - this.lastVY) > 0.005) { this.lastVY = vy; this.epoch++; }
      const rotation = st.task_game?.rotation || 'yaw';
      if (rotation !== this.lastRot) { this.lastRot = rotation; this.epoch++; }   // (re-anchor: no jump)
      const mj0 = mulMV(CAM_TO_MJ, held);
      const cv = Math.cos(vy), sv = Math.sin(vy);
      const mj = [cv * mj0[0] - sv * mj0[1], sv * mj0[0] + cv * mj0[1], mj0[2]];
      const yaw = this.twist + vy;
      let q;
      // the forearm (when Pose found it) and the hand's rotation, fused; filtered, dead band
      const fa = sol && this.pose ? forearm(this.pose, m.lm[0], W, H, f, sol.t[2]) : null;
      state.forearm = fa ? { conf: fa.conf, weight: this.fusion.weight } : null;
      let Rh = null;
      if (sol) {
        Rh = this.fusion.update(sol.R, sol.facing, fa, dt || 1 / 30);
        if (this.openFrames === 1) { this.fRot3.reset(); this.bRot3.reset(); }
        Rh = this.bRot3.filter(this.fRot3.filter(Rh, t));
        state.axes = projectAxes(Rh, sol.centre, W, H, f);
        state.facing = sol.facing;
      }
      if (rotation === 'full' && Rh) {
        const r = SETTINGS.tiltRate * Math.PI / 180 * dt;
        this.pitch += this.tilt.pitch * r;
        this.roll += this.tilt.roll * r;
        const cv2 = Math.cos(vy), sv2 = Math.sin(vy), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
        const cr = Math.cos(this.roll), sr = Math.sin(this.roll);
        // the hand's orientation in the robot's frame (the screen's heading turned in), then the
        // keys' extra tilt forward/back (about the robot's left axis) and sideways (its forward axis)
        const Rz = [[cv2, -sv2, 0], [sv2, cv2, 0], [0, 0, 1]];
        const Ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]];
        const Rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]];
        const Rmj = mulMM(HAND_TO_MJ, mulMM(Rh, transpose(HAND_TO_MJ)));
        q = matToQuat(mulMM(Rz, mulMM(Ry, mulMM(Rx, Rmj))));
      } else {
        this.pitch = 0; this.roll = 0;
        q = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
      }
      pose = [...mj, ...q];
      if (this.holdFrames > 0 && this.lastPose) {
        pose = this.lastPose;
        if (--this.holdFrames === 0) this.epoch++;     // (taken up again from here: no jump)
      }
      this.lastPose = pose;
      state.driving = this.armed && settled;
    } else {
      this.fist = false;
      this.openFrames = 0;                            // (a lost hand settles again when it is back)
      this.wasSettled = false;
      this.sol = null;
      this.fusion.reset();
    }
    this._log(m, state, t);
    this.opts.onFrame?.(state);
    if (!this.armed) return;
    const side = st.task_game?.arm || 'right';
    const right = pose ? { pose, grip: state.driving ? 1 : 0, stick: [0, this.pinched ? 1 : -1], buttons: [], epoch: this.epoch } : null;
    this.opts.send({ type: 'input', t, head: null, left: null, right: null, [side]: right, tracker: 'hand' });
  }
}

/** The hand's axes drawn on the camera picture (image 0..1): fingers, across, out of the palm. */
function projectAxes(R, c, W, H, f) {
  const p = (X) => ({ x: 0.5 + f * X[0] / X[2], y: (0.5 * H / W + f * X[1] / X[2]) * W / H });
  const L = 0.06;
  return { o: p(c), axes: [0, 1, 2].map(j => p([c[0] + R[0][j] * L, c[1] + R[1][j] * L, c[2] + R[2][j] * L])) };
}
const transpose = (M) => [[M[0][0], M[1][0], M[2][0]], [M[0][1], M[1][1], M[2][1]], [M[0][2], M[1][2], M[2][2]]];
const r4 = (v) => Math.round(v * 1e4) / 1e4;

HandControl.prototype._log = function (m, state, t) {
  // the raw tracking (points, no picture), for tuning: opts.log(frames) every SETTINGS.logEvery s
  if (!this.opts.log) return;
  const pose = m.pose ? [13, 14, 15, 16].map(i => m.pose.lm[i] ? [r4(m.pose.lm[i].x), r4(m.pose.lm[i].y), r4(m.pose.lm[i].visibility ?? 0),
    r4(m.pose.wl[i].x), r4(m.pose.wl[i].y), r4(m.pose.wl[i].z)] : null) : null;
  this.log.push({ t: r4(t), lat: state.latency, g: state.gesture, d: state.driving ? 1 : 0,
                  lm: m.lm ? m.lm.map(p => [r4(p.x), r4(p.y)]) : null, wl: m.wl ? m.wl.map(p => [r4(p.x), r4(p.y), r4(p.z)]) : null,
                  pose, W: this.video.videoWidth, H: this.video.videoHeight });
  if (this.lastLog === null) this.lastLog = t;
  if (t - this.lastLog >= SETTINGS.logEvery) {
    this.opts.log(this.log);
    this.log = [];
    this.lastLog = t;
  }
};

function quatToMat([w, x, y, z]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}

/** Open a camera (the front one on a phone: it stands in front of the person like a webcam). */
export async function openCamera(facing = 'user') {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
  const video = document.createElement('video');
  video.setAttribute('playsinline', ''); video.muted = true; video.srcObject = stream;
  await video.play();
  return { video, stream, stop: () => stream.getTracks().forEach(t => t.stop()) };
}
