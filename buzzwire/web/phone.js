// Buzz Wire phone controller. An AR session (WebXR, Chrome on Android with ARCore) tracks the
// phone's position and rotation; every frame its pose is sent to the game server as the robot's
// controller, the same message a VR controller sends (the motion core is the screening test's).
//   hold the pad (anywhere but the buttons) = MOVE (clutch): the gripper follows the phone
//   GRAB / RELEASE = close / open the gripper
//   RE-CENTER      = "the direction I'm pointing now is forward (into the screen)"
// Haptics: grab, release, start / stop moving, a buzz on every touch of the wire, level done.
const $ = (id) => document.getElementById(id);
const vibrate = (p) => { try { if (navigator.vibrate) navigator.vibrate(p); } catch (_) {} };

// ------------------------------------------------------------ network
let ws = null, wsOpen = false, status = {}, rtt = null;
let controlsMode = null;              // 'motion' or 'touch': told to the server (recorded; the tutorial's texts)
function setControls(mode) { controlsMode = mode; if (wsOpen) ws.send(JSON.stringify({ type: 'controls', mode })); }
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { wsOpen = true; if (controlsMode) ws.send(JSON.stringify({ type: 'controls', mode: controlsMode })); };
  ws.onclose = () => { wsOpen = false; showNet(); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;          // robot poses for the game screen: not needed here
    const msg = JSON.parse(ev.data);
    if (msg.type === 'pong') { rtt = performance.now() - msg.t; showNet(); return; }
    if (msg.type === 'status') { status = msg; showStatus(); }
  };
}
connect();
setInterval(() => { if (wsOpen) ws.send(JSON.stringify({ type: 'ping', t: performance.now() })); }, 1000);
function showNet() {
  $('net').classList.toggle('ok', wsOpen);
  $('netT').textContent = wsOpen ? (rtt === null ? 'Connected' : `Connected · ${Math.round(rtt)} ms`) : 'Reconnecting…';
}

// ------------------------------------------------------------ HUD + game feedback
const fmt = (s) => { s = Math.max(0, Math.ceil(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
let lastTouching = false, lastLevel = null, lastMsg = '', buzzT = 0, msgT = 0;
function showStatus() {
  const s = status;
  if (s.state === 'menu') {                          // the game menu is on the computer screen
    $('lvlK').textContent = 'Robot games';
    $('lvlT').textContent = 'Choose a game on the computer';
    $('timer').textContent = '-'; $('touches').textContent = '-';
    $('bar').style.width = '0%'; $('tilt').classList.add('hidden'); $('count').classList.add('hidden');
    $('msg').textContent = ''; $('done').classList.add('hidden');
    lastLevel = null;
    return;
  }
  $('lvlK').textContent = s.count ? `${s.game_title || ''} · Level ${s.index + 1} of ${s.count}` : 'Level';
  $('lvlT').textContent = (s.title || '').replace(/^Level \d+:\s*/, '');
  const left = s.state === 'play' ? s.limit - s.time : s.limit;
  $('timer').textContent = fmt(left || 0);
  $('timer').classList.toggle('low', s.state === 'play' && left < 20);
  const main = (s.hud && s.hud[0]) || { label: '', value: '' };
  $('touches').textContent = main.value;
  $('mainLab').textContent = main.label;
  $('bar').style.width = `${Math.round(100 * (s.progress || 0))}%`;
  $('tilt').classList.toggle('hidden', s.rotation !== 'full' || s.finished);
  const c = s.state === 'countdown' && s.countdown > 0 ? Math.ceil(s.countdown) : '';
  $('count').classList.toggle('hidden', !c);
  if (c && $('count').textContent !== String(c)) { $('count').textContent = c; vibrate(30); }
  if (!c) $('count').textContent = '';
  // a touch: buzz in the hand, flash red
  if (s.touching && !lastTouching) { vibrate([70, 30, 70]); buzzT = performance.now(); }
  if (main.alert) { $('touches').classList.add('hit'); setTimeout(() => $('touches').classList.remove('hit'), 600); }
  $('flash').style.opacity = s.touching ? 1 : 0;
  lastTouching = !!s.touching;
  if (lastLevel !== null && s.level !== lastLevel) vibrate([40, 60, 40, 60, 120]);   // level done
  // levels with the wand fixed in the robot's hand: no GRAB / RELEASE
  if (s.start_held !== grab.disabled) {
    grab.disabled = !!s.start_held;
    if (s.start_held) { closed = true; grab.classList.add('closed'); $('grabT').textContent = 'HOLDING'; }
    else setClosed(false);
  }
  lastLevel = s.level;
  let msg = '';
  if (s.state === 'waiting') msg = touchMode ? 'Push a stick to start' : warmed ? 'Hold the pad and move the phone to start' : '';
  else if (s.state === 'play' && !s.held && !s.start_held && s.game === 'wire') msg = 'Pick up the wand by its handle';
  if (s.message && s.message !== lastMsg) { lastMsg = s.message; msgT = performance.now(); }
  if (performance.now() - msgT < 2500) msg = s.message;
  $('msg').textContent = msg;
  $('done').classList.toggle('hidden', !s.finished);
}
// keep buzzing while the wand stays on the wire
setInterval(() => { if (lastTouching && performance.now() - buzzT > 250) { vibrate(60); buzzT = performance.now(); } }, 120);

// ------------------------------------------------------------ buttons
let moving = false, closed = false, epoch = 0, recenterReq = true;
const pad = $('pad'), grab = $('grab');
// MOVE = a finger anywhere on the screen except GRAB and RE-CENTER. In an AR session Chrome hands
// screen touches to the session as "select" input (selectstart ... selectend, for as long as the
// finger is down) and may cancel the page's own touch events meanwhile. So the session's select
// events hold MOVE; the page's touch events are a backup (held = either says the finger is down).
let xrHold = false, touchHold = false;
function refreshMove() {
  const on = xrHold || touchHold;
  if (on === moving) return;
  moving = on;
  pad.classList.toggle('on', on);
  $('padT').innerHTML = on ? '<span>MOVING</span>' : '<span>HOLD<small>TO MOVE</small></span>';
  vibrate(on ? 18 : 8);
}
function setMove(on) { touchHold = on; refreshMove(); }
const onButton = (el) => !!(el && el.closest && el.closest('#grab, #recenter'));
$('overlay').addEventListener('touchstart', (e) => {
  if (touchMode) return;                             // (joysticks: they handle their own touches)
  if (onButton(e.target)) return;
  e.preventDefault();
  setMove(true);
}, { passive: false });
document.addEventListener('touchend', (e) => { if (e.touches.length === 0) setMove(false); }, { passive: true });
document.addEventListener('touchcancel', (e) => { if (e.touches.length === 0) setMove(false); }, { passive: true });
document.addEventListener('touchmove', (e) => { if (!$('intro').contains(e.target)) e.preventDefault(); }, { passive: false });
document.addEventListener('touchstart', (e) => { if (touchMode && e.target.closest && e.target.closest('.stick')) e.preventDefault(); }, { passive: false });
document.addEventListener('contextmenu', (e) => e.preventDefault());
// taps on the two buttons must not also count as AR "select" (= MOVE)
for (const b of [grab, $('recenter')]) b.addEventListener('beforexrselect', (e) => e.preventDefault());
$('overlay').addEventListener('mousedown', (e) => { if (!touchMode && !onButton(e.target)) setMove(true); });   // (testing on a computer)
document.addEventListener('mouseup', () => setMove(false));
function setClosed(c) {
  closed = c;
  grab.classList.toggle('closed', closed);
  $('grabT').textContent = closed ? 'RELEASE' : 'GRAB';
}
grab.addEventListener('click', () => { setClosed(!closed); vibrate(closed ? [30, 40, 30] : 20); });
$('recenter').addEventListener('click', () => { if (!touchMode) { recenterReq = true; vibrate(15); } });

// ------------------------------------------------------------ pose math
// WebXR: Y up, -Z = where the phone's back camera looked when tracking started.
// MuJoCo: Z up, +X forward (into the monitor), +Y left.
function yawOf(q) {             // heading of the phone's back camera (or top edge when held flat)
  const [x, y, z, w] = q;
  const fx = -(2 * (x * z + w * y)), fz = -(1 - 2 * (x * x + y * y));
  const ux = 2 * (x * y - w * z), uz = 2 * (y * z + w * x);
  const fh = Math.hypot(fx, fz);
  const [hx, hz] = fh > 0.35 ? [fx, fz] : [ux, uz];
  return Math.atan2(-hx, -hz);
}
let calYaw = 0, lastViewYaw = 0, lastRotation = 'yaw';
const WARM_MS = 1500;
let warmed = false, warmMs = 0;
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
    // "forward" is into the game screen's view: turned by the camera's heading (the screen
    // reports it; when it changes the server re-anchors, so the arm doesn't jump)
    const vy = status.view_yaw ?? 0;
    if (Math.abs(vy - lastViewYaw) > 0.005) { lastViewYaw = vy; epoch++; }
    const cv = Math.cos(vy), sv = Math.sin(vy);
    const mj = [cv * ph[0] - sv * ph[1], sv * ph[0] + cv * ph[1], ph[2]];
    // rotation: only the turn left / right (the gripper keeps pointing down), or in levels with
    // rotation "full" the phone's whole orientation: its tilt and twist tilt and twist the gripper
    const rotation = status.rotation || 'yaw';
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
    right = { pose: [...mj, ...q], grip: moving && tracked && warmed ? 1 : 0, stick: [0, closed ? 1 : -1], buttons: [], epoch };
  }
  ws.send(JSON.stringify({ type: 'input', t: performance.now() / 1000, head: null, left: null, right }));
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
supported().then(err => {
  if (!err) return;                                   // (this phone can't track motion: touch controls instead)
  $('err').textContent = 'Motion control is not available on this phone (' + err + ') Use the touch controls.';
  $('start').disabled = true;
  $('startTouch').classList.add('primary');
});

$('start').addEventListener('click', async () => {
  const err = await supported();
  if (err) { $('err').textContent = err; return; }
  let session;
  try {
    session = await navigator.xr.requestSession('immersive-ar', {
      requiredFeatures: ['local'], optionalFeatures: ['dom-overlay'], domOverlay: { root: document.body } });
  } catch (e) { $('err').textContent = 'Could not start tracking: ' + e.message; return; }
  vibrate(30);
  setControls('motion');
  $('intro').classList.add('hidden');
  $('overlay').classList.remove('hidden');
  warmed = false; warmMs = 0; $('warm').classList.remove('hidden');
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
  let lastSend = 0, wasTracked = null, lastT = null;
  const onFrame = (t, frame) => {
    session.requestAnimationFrame(onFrame);
    const layer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    // opaque: covers the camera image (the AR session needs the camera to track the phone, but
    // nobody needs to see it; Chrome keeps the overlay page itself see-through)
    gl.clearColor(0.043, 0.051, 0.071, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const pose = frame.getViewerPose(ref);
    const tracked = !!pose && !pose.emulatedPosition;
    // warm-up: the tracking must hold steadily for WARM_MS before MOVE works (it locks on while the
    // phone is moved a little); a later loss of tracking just pauses the robot, no warm-up again
    const dt = lastT === null ? 0 : t - lastT; lastT = t;
    if (!warmed) {
      warmMs = tracked ? warmMs + dt : 0;
      $('warmBar').style.width = `${Math.min(100, 100 * warmMs / WARM_MS)}%`;
      if (warmMs >= WARM_MS) { warmed = true; recenterReq = true; $('warm').classList.add('hidden'); vibrate([30, 40, 60]); }
    }
    if (tracked !== wasTracked) {
      wasTracked = tracked;
      $('trk').classList.toggle('ok', tracked);
      $('trkT').textContent = tracked ? 'Tracking' : 'Tracking lost';
      $('lost').classList.toggle('hidden', tracked || !warmed);
      if (!tracked) vibrate([20, 40, 20]);
    }
    if (t - lastSend >= 15) { send(pose, tracked); lastSend = t; }     // ~60 messages a second
  };
  session.requestAnimationFrame(onFrame);
  session.addEventListener('end', () => {
    $('overlay').classList.add('hidden');
    $('intro').classList.remove('hidden');
    $('start').textContent = 'Restart controller';
    xrHold = false;
    setMove(false);
  });
});

// ------------------------------------------------------------ touch controls (no camera)
// Two joysticks drive a virtual hand, sent exactly like the motion-tracked phone's pose:
//   left stick   move forward / back / left / right ("forward" = into the game screen)
//   right stick  up / down (y) and turn (x); with TILT on (levels with full rotation):
//                tilt (y) and twist (x)
// MOVE is held while a stick is pushed; letting go of both stops the robot where it is.
const SPEED = 0.12, TURN = 70 * Math.PI / 180, DEAD = 0.12;    // [m/s], [rad/s] at full deflection
let touchMode = false, tiltOn = false;
const hand = [0, 0, 1], handQ = [1, 0, 0, 0];                   // w, x, y, z
const sticks = { stL: { x: 0, y: 0, id: null }, stR: { x: 0, y: 0, id: null } };
function qmul(a, b) {
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
function turnAbout(axis, ang) {                                   // world-frame rotation of the hand
  if (!ang) return;
  const s = Math.sin(ang / 2), q = qmul([Math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s], handQ);
  const n = Math.hypot(...q);
  for (let i = 0; i < 4; i++) handQ[i] = q[i] / n;
}
function bindStick(id) {
  const el = $(id), knob = el.querySelector('.knob'), st = sticks[id];
  const update = (e) => {
    const r = el.getBoundingClientRect(), R = Math.min(r.width, r.height) * 0.36;
    let dx = (e.clientX - (r.left + r.width / 2)) / R, dy = (e.clientY - (r.top + r.height / 2)) / R;
    const n = Math.hypot(dx, dy);
    if (n > 1) { dx /= n; dy /= n; }
    st.x = dx; st.y = -dy;                                        // (up on the screen = +y)
    knob.style.transform = `translate(${dx * R}px, ${dy * R}px)`;
  };
  el.addEventListener('pointerdown', (e) => {
    if (st.id !== null) return;
    st.id = e.pointerId; el.setPointerCapture(e.pointerId); el.classList.add('on'); update(e); vibrate(10);
  });
  el.addEventListener('pointermove', (e) => { if (e.pointerId === st.id) update(e); });
  const end = (e) => {
    if (e.pointerId !== st.id) return;
    st.id = null; st.x = st.y = 0; knob.style.transform = ''; el.classList.remove('on');
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
}
bindStick('stL'); bindStick('stR');
const shape = (v) => { const a = Math.abs(v); return a < DEAD ? 0 : Math.sign(v) * ((a - DEAD) / (1 - DEAD)) ** 2; };  // fine near the middle
let lastTouch = performance.now();
setInterval(() => {
  if (!touchMode || !wsOpen) return;
  const now = performance.now(), dt = Math.min(0.05, (now - lastTouch) / 1000);
  lastTouch = now;
  const L = { x: shape(sticks.stL.x), y: shape(sticks.stL.y) }, R = { x: shape(sticks.stR.x), y: shape(sticks.stR.y) };
  const vy = status.view_yaw ?? 0, fwd = [Math.cos(vy), Math.sin(vy), 0], left = [-Math.sin(vy), Math.cos(vy), 0];
  for (let i = 0; i < 3; i++) hand[i] += (fwd[i] * L.y - left[i] * L.x) * SPEED * dt;
  const full = status.rotation === 'full';
  if (tiltOn && full) { turnAbout(left, -R.y * TURN * dt); turnAbout(fwd, R.x * TURN * dt); }
  else { hand[2] += R.y * SPEED * dt; turnAbout([0, 0, 1], -R.x * TURN * dt); }
  const active = L.x || L.y || R.x || R.y;
  ws.send(JSON.stringify({ type: 'input', t: now / 1000, head: null, left: null,
    right: { pose: [...hand, ...handQ], grip: active ? 1 : 0, stick: [0, closed ? 1 : -1], buttons: [], epoch } }));
}, 16);
$('startTouch').addEventListener('click', () => {
  touchMode = true;
  setControls('touch');
  vibrate(30);
  $('intro').classList.add('hidden');
  $('overlay').classList.remove('hidden');
  $('pad').classList.add('hidden');
  $('sticks').classList.remove('hidden');
  $('trk').classList.add('hidden');
  $('tilt').textContent = 'Tilt level: tap TILT, then tilt and twist with the right stick';
  // RE-CENTER isn't needed with joysticks: the button becomes TILT (levels with full rotation)
  const rc = $('recenter');
  rc.lastChild.textContent = 'TILT';
  rc.addEventListener('click', (e) => {
    if (!touchMode) return;
    if (status.rotation !== 'full') { tiltOn = false; rc.classList.remove('tilt-on'); return; }
    tiltOn = !tiltOn; rc.classList.toggle('tilt-on', tiltOn);
    $('stRT').textContent = tiltOn ? 'TILT · TWIST' : 'UP / DOWN · TURN';
  });
});

// ------------------------------------------------------------ the controls the admin gave this candidate
// /phone?c=<code>: "motion" (only START: move the phone; touch only if this phone can't track),
// "touch" (only the joysticks), "choice" (both, motion first)
const TOUCH_STEPS = `
  <li><span class="num">1</span><span>Keep the game open on your <b>computer screen</b>.</span></li>
  <li><span class="num">2</span><span>Tap <b>START</b> below.</span></li>
  <li><span class="num">3</span><span><b>Left stick</b>: move the gripper (forward = into the screen). <b>Right stick</b>: up / down and turn.</span></li>
  <li><span class="num">4</span><span><b>GRAB</b> closes the gripper, <b>RELEASE</b> opens it. Push the sticks gently for fine moves.</span></li>`;
const candidate = new URLSearchParams(location.search).get('c');
if (candidate) {
  fetch(`/api/controls/${encodeURIComponent(candidate)}`).then(r => r.json()).then(async ({ controls }) => {
    const hide = (el) => el && el.classList.add('hidden');
    const fineAfter = (el) => el && el.nextElementSibling && el.nextElementSibling.classList.contains('fine') ? el.nextElementSibling : null;
    if (controls === 'touch') {
      hide($('start')); hide(fineAfter($('start'))); hide($('err'));
      $('startTouch').textContent = 'Start'; $('startTouch').classList.add('primary'); hide($('startTouchFine'));
      document.querySelector('#intro .steps').innerHTML = TOUCH_STEPS;
    } else if (controls === 'motion') {
      const err = await supported();
      if (!err) { hide($('startTouch')); hide($('startTouchFine')); }
      else $('err').textContent = 'This phone can\'t track motion, so you\'ll use touch controls: two joysticks on the screen.';
    }
  }).catch(() => {});
}
