// Phone controller for the screening test: an AR session (WebXR, Chrome on Android with
// ARCore) tracks the phone's position and rotation in the room; every frame its pose is sent to
// the simulator as the arm's "controller" (the right one by default), the same message a VR
// controller sends.
//   MOVE (hold)   = the grip button (clutch): the gripper follows the phone's motion
//   GRAB/RELEASE  = gripper closed / open (sent as the thumbstick, which the server integrates)
//   RE-CENTER     = "the direction I'm pointing now is forward (into the screen)"
// Only the turn about the vertical axis is used, so the gripper always points down.
const $ = (id) => document.getElementById(id);
const BASE = window.TELEOP_BASE;

// ------------------------------------------------------------ network
let ws = null, wsOpen = false, status = {}, wasTouching = false;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${BASE}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsOpen = true; };
  ws.onclose = () => { wsOpen = false; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;          // robot poses for the monitor: not needed here
    const msg = JSON.parse(ev.data);
    if (msg.type === 'pong') { rtt = performance.now() - msg.t; return; }
    status = msg;
    showStatus();
    // buzz wire: one short buzz when the loop touches the wire (short: shaking the phone
    // disturbs its motion tracking)
    const touching = !!msg.task_game?.wire?.touching;
    if (touching && !wasTouching && navigator.vibrate) navigator.vibrate(30);
    wasTouching = touching;
  };
}
connect();
// round trip phone -> simulator -> phone, shown on the screen (the robot on the monitor lags
// the phone by about this much plus the arm's own response)
let rtt = null;
setInterval(() => { if (wsOpen) ws.send(JSON.stringify({ type: 'ping', t: performance.now() })); }, 1000);

function showStatus() {
  const g = status.task_game;
  if (!g) return;
  let t = g.round ? `${g.title} (round ${g.round})` : g.title;
  if (g.limit && g.intro === 0) t += `  ·  ${Math.max(0, Math.ceil(g.limit - g.time))} s left`;
  if (g.intro > 0) t += `  ·  starts in ${Math.ceil(g.intro)}`;
  if (g.finished) t = 'Finished, thank you! You can close this page.';
  if (g.rotation === 'full' && !g.finished) t += '  ·  TILT ON: tilting and twisting the phone tilts the gripper';
  if (g.homing) t = 'Going home…  ' + t;
  $('phase').textContent = t;
}

// ------------------------------------------------------------ buttons
let moving = false, closed = false, epoch = 0, recenterReq = true;
const move = $('move'), grab = $('grab');
// MOVE = a finger anywhere on the screen except the GRAB and RE-CENTER buttons. In an AR session
// Chrome hands screen touches to the session as "select" input (selectstart ... selectend, for
// as long as the finger is down) and may cancel the page's own touch events meanwhile, which
// made a touch-event button let go after a moment. So the session's select events hold MOVE;
// the page's touch events are a backup (held = either says the finger is down).
let xrHold = false, touchHold = false;
function refreshMove() {
  const on = xrHold || touchHold;
  if (on === moving) return;
  moving = on;
  move.classList.toggle('on', on);
  move.textContent = on ? 'MOVING' : 'HOLD ANYWHERE TO MOVE';
  if (navigator.vibrate) navigator.vibrate(on ? 18 : 8);
}
function setMove(on) { touchHold = on; refreshMove(); }
const onButton = (el) => !!(el && el.closest && el.closest('#grab, #recenter, #home'));
$('overlay').addEventListener('touchstart', (e) => {
  if (onButton(e.target)) return;
  e.preventDefault();
  setMove(true);
}, { passive: false });
document.addEventListener('touchend', (e) => { if (e.touches.length === 0) setMove(false); }, { passive: true });
document.addEventListener('touchcancel', (e) => { if (e.touches.length === 0) setMove(false); }, { passive: true });
document.addEventListener('touchmove', (e) => e.preventDefault(), { passive: false });
document.addEventListener('contextmenu', (e) => e.preventDefault());
// taps on the two buttons must not also count as AR "select" (= MOVE)
for (const b of [grab, $('recenter'), $('home')]) b.addEventListener('beforexrselect', (e) => e.preventDefault());
// mouse (testing on a computer)
$('overlay').addEventListener('mousedown', (e) => { if (!onButton(e.target)) setMove(true); });
document.addEventListener('mouseup', () => setMove(false));
// A tap on a button: on the touch itself (inside some AR runtimes, e.g. an iPhone App Clip, a tap
// in the AR session never becomes a "click"), and on click as well; one tap counts once.
function onTap(el, fn) {
  let last = 0;
  const run = (e) => {
    const now = performance.now();
    if (now - last < 400) return;
    last = now;
    if (e && e.cancelable) e.preventDefault();
    fn();
  };
  el.addEventListener('touchend', run, { passive: false });
  el.addEventListener('pointerup', run);
  el.addEventListener('click', run);
}
onTap(grab, () => {
  closed = !closed;
  grab.classList.toggle('closed', closed);
  grab.textContent = closed ? 'RELEASE' : 'GRAB';
  if (navigator.vibrate) navigator.vibrate(25);
});
onTap($('recenter'), () => { recenterReq = true; if (navigator.vibrate) navigator.vibrate(12); });
// HOME: the arm glides back to its start pose (out of an awkward, stuck pose); MOVE is ignored
// until it's there
onTap($('home'), () => {
  if (wsOpen) ws.send(JSON.stringify({ type: 'home' }));
  if (navigator.vibrate) navigator.vibrate(20);
});

// ------------------------------------------------------------ pose math
// WebXR: Y up, -Z = where the phone's back camera looked when tracking started.
// MuJoCo: Z up, +X forward (into the monitor), +Y left.
function yawOf(q) {             // heading of the phone's back camera (or top edge when held flat)
  const [x, y, z, w] = q;
  // rotate (0,0,-1) and (0,1,0) by q
  const fx = -(2 * (x * z + w * y)), fz = -(1 - 2 * (x * x + y * y));
  const ux = 2 * (x * y - w * z), uz = 2 * (y * z + w * x);
  const fh = Math.hypot(fx, fz);
  const [hx, hz] = fh > 0.35 ? [fx, fz] : [ux, uz];
  return Math.atan2(-hx, -hz);
}
let calYaw = 0, lastViewYaw = 0, lastRotation = 'yaw';

// 3x3 matrices (row-major arrays of rows) for the full-rotation mode
const mul3 = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const T3 = (A) => [0, 1, 2].map(i => [A[0][i], A[1][i], A[2][i]]);
function quatToMat([x, y, z, w]) {
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]];
}
function matToQuatWXYZ(R) {
  const tr = R[0][0] + R[1][1] + R[2][2];
  let w, x, y, z;
  if (tr > 0) {
    const s = 2 * Math.sqrt(tr + 1);
    w = s / 4; x = (R[2][1] - R[1][2]) / s; y = (R[0][2] - R[2][0]) / s; z = (R[1][0] - R[0][1]) / s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = 2 * Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]);
    w = (R[2][1] - R[1][2]) / s; x = s / 4; y = (R[0][1] + R[1][0]) / s; z = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    const s = 2 * Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]);
    w = (R[0][2] - R[2][0]) / s; x = (R[0][1] + R[1][0]) / s; y = s / 4; z = (R[1][2] + R[2][1]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]);
    w = (R[1][0] - R[0][1]) / s; x = (R[0][2] + R[2][0]) / s; y = (R[1][2] + R[2][1]) / s; z = s / 4;
  }
  return [w, x, y, z];
}
const XR_TO_MJ = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]];      // MuJoCo vector = this x WebXR vector

function send(pose, tracked) {
  if (!wsOpen) return;
  let right = null;
  if (pose) {
    const p = pose.transform.position, o = pose.transform.orientation;
    const yaw = yawOf([o.x, o.y, o.z, o.w]);
    if (recenterReq && tracked) { calYaw = yaw; recenterReq = false; epoch++; }
    // undo the calibration heading, then XR -> MuJoCo axes
    const c = Math.cos(-calYaw), s = Math.sin(-calYaw);
    const x = c * p.x + s * p.z, z = -s * p.x + c * p.z;
    const ph = [-z, -x, p.y];                        // phone frame: x = where it pointed at RE-CENTER
    // "forward" is into the monitor's view: turn by the camera's heading (the monitor reports it;
    // when it changes the server re-anchors, so the arm doesn't jump)
    const vy = status.view_yaw ?? 0;
    if (Math.abs(vy - lastViewYaw) > 0.005) { lastViewYaw = vy; epoch++; }
    const cv = Math.cos(vy), sv = Math.sin(vy);
    const mj = [cv * ph[0] - sv * ph[1], sv * ph[0] + cv * ph[1], ph[2]];
    // rotation: only the turn left / right (the gripper keeps pointing down), or in phases with
    // rotation "full" the phone's whole orientation: its tilt and twist tilt and twist the gripper
    const rotation = status.task_game?.rotation || 'yaw';
    if (rotation !== lastRotation) { lastRotation = rotation; epoch++; }   // re-anchor: no jump
    let q;
    if (rotation === 'full') {
      const Rmj = mul3(mul3(XR_TO_MJ, quatToMat([o.x, o.y, o.z, o.w])), T3(XR_TO_MJ));
      const t = vy - calYaw, ct = Math.cos(t), st = Math.sin(t);
      q = matToQuatWXYZ(mul3([[ct, -st, 0], [st, ct, 0], [0, 0, 1]], Rmj));
    } else {
      const a = (yaw - calYaw + vy) / 2;
      q = [Math.cos(a), 0, 0, Math.sin(a)];
    }
    right = { pose: [...mj, ...q],
              grip: moving && tracked ? 1 : 0, stick: [0, closed ? 1 : -1], buttons: [], epoch };
  }
  // sent as the controller of the arm the test uses ([arm] side; the other one gets nothing)
  const side = status.task_game?.arm || 'right';
  ws.send(JSON.stringify({ type: 'input', t: performance.now() / 1000, head: null, left: null, right: null,
                           [side]: right }));
}

// ------------------------------------------------------------ AR session
async function supported() {
  if (!window.isSecureContext) return 'This page must be opened over https://.';
  if (!navigator.xr) return 'This phone or browser can\'t track motion. Use Chrome on an Android phone.';
  try {
    if (!(await navigator.xr.isSessionSupported('immersive-ar')))
      return 'This phone can\'t track motion (it needs Google Play Services for AR). Please use another Android phone.';
  } catch (e) { return 'Motion tracking is not available: ' + e.message; }
  return null;
}
supported().then(err => { if (err) { $('err').textContent = err; $('start').disabled = true; } });

$('start').addEventListener('click', async () => {
  const err = await supported();
  if (err) { $('err').textContent = err; return; }
  let session;
  try {
    session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['local'], optionalFeatures: ['dom-overlay'], domOverlay: { root: $('overlay') } });
  } catch (e) { $('err').textContent = 'Could not start tracking: ' + e.message; return; }
  $('intro').classList.add('hidden');
  $('overlay').classList.remove('hidden');
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl', { xrCompatible: true, alpha: true });
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const ref = await session.requestReferenceSpace('local');
  recenterReq = true;
  // a finger on the screen (not on GRAB / RE-CENTER) is a "screen" input source: hold = MOVE
  const isScreen = (e) => e.inputSource && e.inputSource.targetRayMode === 'screen';
  session.addEventListener('selectstart', (e) => { if (isScreen(e)) { xrHold = true; refreshMove(); } });
  session.addEventListener('selectend', (e) => { if (isScreen(e)) { xrHold = false; refreshMove(); } });
  session.addEventListener('inputsourceschange', (e) => {
    if (e.removed.some(s => s.targetRayMode === 'screen') && session.inputSources.length === 0) {
      xrHold = false; refreshMove();
    }
  });
  let lastSend = 0;
  const onFrame = (t, frame) => {
    session.requestAnimationFrame(onFrame);
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    // opaque: covers the camera image (the AR session needs the camera to track the phone, but
    // nobody needs to see it; Chrome keeps the overlay page itself see-through)
    gl.clearColor(0.08, 0.09, 0.11, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const pose = frame.getViewerPose(ref);
    const tracked = !!pose && !pose.emulatedPosition;
    const delay = rtt === null ? '' : `  ·  connection delay ${Math.round(rtt)} ms`;
    $('track').textContent = (tracked ? 'Tracking OK' : 'Tracking lost: move slowly, point the camera at your desk or screen') + delay;
    $('track').className = tracked ? 'ok' : 'bad';
    if (t - lastSend >= 15) { send(pose, tracked); lastSend = t; }     // ~60 messages a second
  };
  session.requestAnimationFrame(onFrame);
  session.addEventListener('end', () => {
    $('overlay').classList.add('hidden');
    $('intro').classList.remove('hidden');
    $('start').textContent = 'RESTART CONTROLLER';
    xrHold = false;
    setMove(false);
  });
});
