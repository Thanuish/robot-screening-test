// Webcam hand tracking: the candidate's hand drives the arm, no phone needed. MediaPipe's Hand
// Landmarker (Google, Apache 2.0) finds 21 points of the hand in the webcam image, in this
// browser: the video never leaves the computer, only the hand's pose is sent, in the same
// message as the phone's (vrteleop's controller input), so the server, the arm and the tasks
// don't change.
//
//   the hand moves     the gripper moves (left / right and up / down from where the hand is in
//                      the image, forward / back from how big it looks: the palm's real size is
//                      known from MediaPipe's 3D hand, so the motion is in metres)
//   the hand turns     the gripper turns the same way (in tasks with full rotation: tilts too)
//   pinch              (thumb and index finger together) closes the gripper; open = release
//   SPACE held         = MOVE (the clutch): let go to move the hand without moving the robot
//   C  re-centre,  H  HOME (the arm back to its start pose)
//
// Frames: the webcam looks at the person from the screen. Camera: x right in the image (the
// person's left), y down, z away from the camera (toward the person). MuJoCo: x forward (into
// the screen), y left, z up.
const MP = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const HFOV = 60;                        // [deg] a typical laptop webcam's field of view (an error only scales the motion)
const PINCH_ON = 0.32, PINCH_OFF = 0.5; // thumb-index tip distance / palm length: closed below, open above (hysteresis)
const SMOOTH = 0.45, SMOOTH_DEPTH = 0.25;    // per-frame low-pass (the server smooths again)
const CAM_TO_MJ = [[0, 0, -1], [1, 0, 0], [0, -1, 0]];      // MuJoCo vector = this x camera vector

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mulMV = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
const mulMM = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
function matToQuat(R) {                 // -> [w, x, y, z]
  const tr = R[0][0] + R[1][1] + R[2][2];
  let w, x, y, z;
  if (tr > 0) { const s = 2 * Math.sqrt(tr + 1); w = s / 4; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s; }
  else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) { const s = 2 * Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]); w = (R[2][1] - R[1][2]) / s; x = s / 4; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s; }
  else if (R[1][1] > R[2][2]) { const s = 2 * Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]); w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = s / 4; z = (R[1][2] + R[2][1]) / s; }
  else { const s = 2 * Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]); w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = s / 4; }
  return [w, x, y, z];
}

// the bones of the palm (wrist to the knuckles, index to pinky knuckle): their real length (3D
// hand, metres) against their length in the image gives the hand's distance from the camera
const PALM = [[0, 5], [0, 9], [0, 13], [0, 17], [5, 17]];

/** One frame's hand pose from MediaPipe's output (lm: 21 image points 0..1, wl: 21 3D points [m],
 *  W x H: the image size; f: the focal length in image widths). Returns, in the camera frame:
 *  pos (the palm's centre [m]), R (the hand's axes as columns: a = wrist -> middle knuckle, b =
 *  index -> pinky knuckle, n = the palm's normal), and pinch (thumb-index tip / palm length). */
export function handPose(lm, wl, W, H, f) {
  const P = (i) => [lm[i].x, lm[i].y * H / W];               // image point, in image widths
  const Q = (i) => [wl[i].x, wl[i].y, wl[i].z];              // 3D hand point [m]
  let real = 0, img = 0;
  for (const [a, b] of PALM) {
    real += len(sub(Q(b), Q(a)));
    img += Math.hypot(P(b)[0] - P(a)[0], P(b)[1] - P(a)[1]);
  }
  const D = f * real / Math.max(img, 1e-4);                  // the hand's distance from the camera [m]
  const c = [0, 5, 9, 13, 17].map(P).reduce((s2, p) => [s2[0] + p[0] / 5, s2[1] + p[1] / 5], [0, 0]);
  const pos = [(c[0] - 0.5) * D / f, (c[1] - 0.5 * H / W) * D / f, D];
  const a = norm(sub(Q(9), Q(0)));
  let b = norm(sub(Q(17), Q(5)));
  const n = norm(cross(a, b));
  b = cross(n, a);
  const R = [[a[0], b[0], n[0]], [a[1], b[1], n[1]], [a[2], b[2], n[2]]];
  const pinch = len(sub(Q(4), Q(8))) / Math.max(len(sub(Q(9), Q(0))), 1e-3);
  return { pos, D, R, a, n, pinch };
}
export const toMJ = (v) => mulMV(CAM_TO_MJ, v);
export const toMJR = (R) => mulMM(CAM_TO_MJ, R);

/** Start the webcam and the hand tracker. opts: send(msg) (to the simulator), status() (its last
 *  status: arm, view_yaw, task_game.rotation), preview (a canvas for the camera image and the
 *  hand), onInfo(text, ok) (what to show). Returns {stop()}. Throws if the camera is refused. */
export async function startHandTracking(opts) {
  const { HandLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false, video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } });
  const video = document.createElement('video');
  video.setAttribute('playsinline', ''); video.muted = true; video.srcObject = stream;
  await video.play();
  opts.onInfo?.('Loading the hand tracker…', false);
  const vision = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const hands = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' }, runningMode: 'VIDEO', numHands: 1,
    minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6 });

  const f = 0.5 / Math.tan((HFOV / 2) * Math.PI / 180);    // focal length, in image widths
  let space = false, pinched = false, epoch = 1, calYaw = null, recenter = true;
  let pos = null, depth = null, lastVideoT = -1, running = true, lastSend = 0, lastRot = 'yaw', lastVY = 0;
  const onKey = (down) => (e) => {
    if (e.code === 'Space') { e.preventDefault(); space = down; }
    if (down && e.code === 'KeyC') recenter = true;
    if (down && e.code === 'KeyH') opts.send({ type: 'home' });
  };
  const kd = onKey(true), ku = onKey(false);
  window.addEventListener('keydown', kd);
  window.addEventListener('keyup', ku);
  window.addEventListener('blur', () => { space = false; });

  const pctx = opts.preview?.getContext('2d');
  const step = () => {
    if (!running) return;
    requestAnimationFrame(step);
    if (video.currentTime === lastVideoT) return;
    lastVideoT = video.currentTime;
    const res = hands.detectForVideo(video, performance.now());
    const lm = res.landmarks?.[0], wl = res.worldLandmarks?.[0];
    const W = video.videoWidth, H = video.videoHeight;
    let tracked = false, info;
    let pose = null;
    if (lm && wl) {
      tracked = true;
      const hp = handPose(lm, wl, W, H, f);
      depth = depth === null ? hp.D : depth + (hp.D - depth) * SMOOTH_DEPTH;
      const camPos = [hp.pos[0] * depth / hp.D, hp.pos[1] * depth / hp.D, depth];
      pos = pos === null ? camPos : pos.map((v, i) => v + (camPos[i] - v) * SMOOTH);
      const { a, n } = hp;
      const Rmj = toMJR(hp.R);
      // pinch: thumb tip to index tip, relative to the palm's length (hysteresis: no flicker)
      pinched = pinched ? hp.pinch < PINCH_OFF : hp.pinch < PINCH_ON;
      // heading of the hand (about the vertical): where its palm faces, or (held flat) its fingers
      const nm = mulMV(CAM_TO_MJ, n), am = mulMV(CAM_TO_MJ, a);
      const hv = Math.hypot(nm[0], nm[1]) > 0.35 ? nm : am;
      const yaw = Math.atan2(hv[1], hv[0]);
      if (recenter) { calYaw = yaw; recenter = false; epoch++; }
      const st = opts.status() || {};
      const vy = st.view_yaw ?? 0;
      if (Math.abs(vy - lastVY) > 0.005) { lastVY = vy; epoch++; }
      const rotation = st.task_game?.rotation || 'yaw';
      if (rotation !== lastRot) { lastRot = rotation; epoch++; }   // (re-anchor: no jump)
      // camera -> MuJoCo, then turned by the monitor camera's heading ("forward" = into the view)
      const m = mulMV(CAM_TO_MJ, pos);
      const cv = Math.cos(vy), sv = Math.sin(vy);
      const mj = [cv * m[0] - sv * m[1], sv * m[0] + cv * m[1], m[2]];
      let q;
      if (rotation === 'full') {
        const t = vy - calYaw, ct = Math.cos(t), stn = Math.sin(t);
        q = matToQuat(mulMM([[ct, -stn, 0], [stn, ct, 0], [0, 0, 1]], Rmj));
      } else {
        const h = (yaw - calYaw + vy) / 2;
        q = [Math.cos(h), 0, 0, Math.sin(h)];
      }
      pose = [...mj, ...q];
      info = `Hand ${Math.round(depth * 100)} cm from the camera  ·  ${space ? 'MOVE (space held)' : 'hold SPACE to move'}` +
             `  ·  ${pinched ? 'PINCH: gripper closed' : 'gripper open'}`;
    } else {
      info = 'Show your hand to the webcam (palm toward the screen, the whole hand in view)';
    }
    opts.onInfo?.(info, tracked);
    // the camera image (mirrored, like a mirror) and the hand's points
    if (pctx) {
      const cw = opts.preview.width, ch = opts.preview.height;
      pctx.save();
      pctx.translate(cw, 0); pctx.scale(-1, 1);
      pctx.drawImage(video, 0, 0, cw, ch);
      if (lm) {
        pctx.fillStyle = pinched ? '#ff6b5e' : '#4cd07d';
        for (const p of lm) pctx.fillRect(p.x * cw - 2, p.y * ch - 2, 4, 4);
      }
      pctx.restore();
    }
    const now = performance.now();
    if (now - lastSend >= 16) {                                  // (~60 a second, like the phone)
      lastSend = now;
      const side = opts.status()?.task_game?.arm || 'right';
      const right = pose ? { pose, grip: space && tracked ? 1 : 0, stick: [0, pinched ? 1 : -1], buttons: [], epoch } : null;
      opts.send({ type: 'input', t: now / 1000, head: null, left: null, right: null, [side]: right, tracker: 'hand' });
    }
  };
  requestAnimationFrame(step);
  return {
    stop() {
      running = false;
      window.removeEventListener('keydown', kd);
      window.removeEventListener('keyup', ku);
      stream.getTracks().forEach(t => t.stop());
      hands.close();
    },
  };
}
