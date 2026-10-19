// Robot Games: the iPhone controller (Safari). iPhones can't do the Android page's motion
// tracking (Safari has no WebXR AR), so this page drives the robot with:
//   touch    two joysticks: left = move (forward = into the game screen), right = up / down and
//            turn (with TILT on, in levels with full rotation: tilt and twist)
//   tilt     the left pad held + the iPhone tilted (gyroscope, DeviceOrientation): tilt it like a
//            joystick in the air to move; the right stick as in touch mode
// Both drive a virtual hand sent as the same messages as the Android phone (pose, MOVE held while
// steering, GRAB / RELEASE). iPhones don't vibrate from web pages: feedback is sound and colour.
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ network
let ws = null, wsOpen = false, status = {}, rtt = null;
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onopen = () => { wsOpen = true; showNet(); if (mode) ws.send(JSON.stringify({ type: 'controls', mode })); };
  ws.onclose = () => { wsOpen = false; showNet(); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const m = JSON.parse(ev.data);
    if (m.type === 'pong') { rtt = performance.now() - m.t; showNet(); }
    else if (m.type === 'status') { status = m; showStatus(); }
  };
}
connect();
setInterval(() => { if (wsOpen) ws.send(JSON.stringify({ type: 'ping', t: performance.now() })); }, 1000);
function showNet() {
  $('net').classList.toggle('ok', wsOpen);
  $('netT').textContent = wsOpen ? (rtt === null ? 'Connected' : `Connected · ${Math.round(rtt)} ms`) : 'Reconnecting…';
}

// ------------------------------------------------------------ sound (no vibration on iPhone)
let audio = null;
function startAudio() {             // must run inside a tap (Safari's rule); plays a silent blip to unlock
  if (audio) return;
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return;
  audio = new AC();
  tone(440, 0.01, 0.0001);
}
function tone(freq, dur, vol = 0.18, type = 'sine') {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain(), t = audio.currentTime;
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(vol, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audio.destination); o.start(t); o.stop(t + dur + 0.02);
}
const buzz = () => tone(130, 0.14, 0.22, 'square');

// ------------------------------------------------------------ HUD and feedback
const fmt = (s) => { s = Math.max(0, Math.ceil(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
let lastTouching = false, lastLevel = null, lastCount = '', lastMsg = '', msgT = 0, buzzT = 0;
function showStatus() {
  const s = status;
  if (s.state === 'menu') {
    $('lvlK').textContent = 'Robot games'; $('lvlT').textContent = 'Choose a game on the computer';
    $('timer').textContent = '-'; $('main').textContent = '-'; $('mainLab').innerHTML = '&nbsp;';
    $('bar').style.width = '0%'; $('msg').textContent = ''; $('count').textContent = ''; $('done').classList.add('hidden');
    lastLevel = null;
    return;
  }
  $('lvlK').textContent = `${s.game_title || ''} · Level ${s.index + 1} of ${s.count}`;
  $('lvlT').textContent = (s.title || '').replace(/^Level \d+:\s*/, '');
  const left = s.state === 'play' ? s.limit - s.time : s.limit;
  $('timer').textContent = fmt(left || 0);
  $('timer').classList.toggle('low', s.state === 'play' && left < 20);
  const main = (s.hud && s.hud[0]) || { label: '', value: '-' };
  $('main').textContent = main.value; $('mainLab').textContent = main.label;
  $('bar').style.width = `${Math.round(100 * (s.progress || 0))}%`;
  $('tiltHint').classList.toggle('hidden', s.rotation !== 'full' || s.finished);
  const c = s.state === 'countdown' && s.countdown > 0 ? String(Math.ceil(s.countdown)) : '';
  if (c !== lastCount) { $('count').textContent = c; if (c) tone(c === '1' ? 880 : 660, 0.12); else if (lastCount) tone(1175, 0.25); lastCount = c; }
  if (s.touching && !lastTouching) { buzz(); buzzT = performance.now(); $('main').classList.add('hit'); setTimeout(() => $('main').classList.remove('hit'), 600); }
  $('flash').style.opacity = s.touching ? 1 : 0;
  lastTouching = !!s.touching;
  if (lastLevel !== null && s.level !== lastLevel) [784, 988, 1319].forEach((f, i) => setTimeout(() => tone(f, 0.25), i * 110));
  lastLevel = s.level;
  if (s.start_held !== grab.disabled) {                 // the wand fixed in the robot's hand: no GRAB / RELEASE
    grab.disabled = !!s.start_held;
    if (s.start_held) { closed = true; grab.classList.add('closed'); grab.textContent = 'HOLDING'; }
    else setClosed(false);
  }
  let msg = s.state === 'waiting' ? 'Push a stick to start' : '';
  if (s.message && s.message !== lastMsg) { lastMsg = s.message; msgT = performance.now(); }
  if (performance.now() - msgT < 2500) msg = s.message;
  if (s.tracking_lost) msg = '';
  $('msg').textContent = msg;
  $('done').classList.toggle('hidden', !s.finished);
}
setInterval(() => { if (lastTouching && performance.now() - buzzT > 260) { buzz(); buzzT = performance.now(); } }, 120);

// ------------------------------------------------------------ GRAB / TILT buttons
const grab = $('grab');
let closed = false, tiltOn = false;
function setClosed(c) { closed = c; grab.classList.toggle('closed', c); grab.textContent = c ? 'RELEASE' : 'GRAB'; }
grab.addEventListener('click', () => { setClosed(!closed); tone(closed ? 520 : 390, 0.06, 0.15, 'triangle'); });
$('tiltBtn').addEventListener('click', () => {
  if (status.rotation !== 'full') { tone(220, 0.08, 0.1); return; }     // (only levels with full rotation)
  tiltOn = !tiltOn; $('tiltBtn').classList.toggle('on', tiltOn);
  $('stRT').textContent = tiltOn ? 'TILT · TWIST' : 'UP / DOWN · TURN';
});

// ------------------------------------------------------------ joysticks
const SPEED = 0.12, TURN = 70 * Math.PI / 180, DEAD = 0.12;    // [m/s], [rad/s] at full deflection
const hand = [0, 0, 1], handQ = [1, 0, 0, 0];                   // the virtual hand (MuJoCo frame), w x y z
const sticks = { stL: { x: 0, y: 0, id: null }, stR: { x: 0, y: 0, id: null } };
let mode = null, epoch = 0;
function qmul(a, b) {
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
function turnAbout(axis, ang) {
  if (!ang) return;
  const s = Math.sin(ang / 2), q = qmul([Math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s], handQ);
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  for (let i = 0; i < 4; i++) handQ[i] = q[i] / n;
}
function bindStick(id) {
  const el = $(id), knob = el.querySelector('.knob'), st = sticks[id];
  const place = (dx, dy, R) => { knob.style.transform = `translate(${dx * R}px, ${-dy * R}px)`; };
  const update = (e) => {
    const r = el.getBoundingClientRect(), R = Math.min(r.width, r.height) * 0.36;
    let dx = (e.clientX - (r.left + r.width / 2)) / R, dy = -(e.clientY - (r.top + r.height / 2)) / R;
    const n = Math.hypot(dx, dy);
    if (n > 1) { dx /= n; dy /= n; }
    if (id === 'stL' && mode === 'tilt') return;              // (tilt mode: the iPhone's angle moves, see below)
    st.x = dx; st.y = dy; place(dx, dy, R);
  };
  el.addEventListener('pointerdown', (e) => {
    if (st.id !== null) return;
    e.preventDefault();
    st.id = e.pointerId;
    try { el.setPointerCapture(e.pointerId); } catch (_) {}
    el.classList.add('on');
    if (id === 'stL' && mode === 'tilt') tilt.zero = { b: tilt.b, g: tilt.g };   // "level" = how it's held now
    update(e);
  });
  el.addEventListener('pointermove', (e) => { if (e.pointerId === st.id) update(e); });
  const end = (e) => {
    if (e.pointerId !== st.id) return;
    st.id = null; st.x = st.y = 0; knob.style.transform = ''; el.classList.remove('on');
  };
  el.addEventListener('pointerup', end);
  el.addEventListener('pointercancel', end);
  el._place = place;
}
bindStick('stL'); bindStick('stR');
document.addEventListener('touchmove', (e) => { if (!$('intro').contains(e.target)) e.preventDefault(); }, { passive: false });
document.addEventListener('gesturestart', (e) => e.preventDefault());     // (no pinch zoom in Safari)
document.addEventListener('dblclick', (e) => e.preventDefault());

// ------------------------------------------------------------ tilt steering (gyroscope)
const tilt = { b: 0, g: 0, zero: null, ok: false };
const TILT_FULL = 25, TILT_DEAD = 3;                            // degrees: full speed at 25, nothing below 3
function onOrientation(e) {
  if (e.beta === null) return;
  tilt.b = e.beta; tilt.g = e.gamma; tilt.ok = true;
}
async function enableTilt() {
  const DOE = window.DeviceOrientationEvent;
  if (!DOE) throw new Error('This browser gives no motion data.');
  if (typeof DOE.requestPermission === 'function') {            // iOS 13+: ask (inside the tap)
    const r = await DOE.requestPermission();
    if (r !== 'granted') throw new Error('Motion access was not allowed. Settings > Safari > Motion & Orientation Access, or use touch controls.');
  }
  window.addEventListener('deviceorientation', onOrientation);
}
function tiltStick() {
  if (sticks.stL.id === null || !tilt.zero) return { x: 0, y: 0 };
  const f = (d) => { const a = Math.abs(d); return a < TILT_DEAD ? 0 : Math.sign(d) * Math.min(1, (a - TILT_DEAD) / (TILT_FULL - TILT_DEAD)); };
  // top of the phone tipped away from you = forward; right side down = right
  return { x: f(tilt.g - tilt.zero.g), y: f(tilt.zero.b - tilt.b) };
}

// ------------------------------------------------------------ sending (60 a second)
const shape = (v) => { const a = Math.abs(v); return a < DEAD ? 0 : Math.sign(v) * ((a - DEAD) / (1 - DEAD)) ** 2; };
let lastT = performance.now();
setInterval(() => {
  if (!mode || !wsOpen) return;
  const now = performance.now(), dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;
  let L;
  if (mode === 'tilt') {
    L = tiltStick();
    const el = $('stL'), r = el.getBoundingClientRect();
    el._place(L.x, L.y, Math.min(r.width, r.height) * 0.36);
  } else L = { x: shape(sticks.stL.x), y: shape(sticks.stL.y) };
  const R = { x: shape(sticks.stR.x), y: shape(sticks.stR.y) };
  const vy = status.view_yaw || 0, fwd = [Math.cos(vy), Math.sin(vy), 0], left = [-Math.sin(vy), Math.cos(vy), 0];
  for (let i = 0; i < 3; i++) hand[i] += (fwd[i] * L.y - left[i] * L.x) * SPEED * dt;
  if (tiltOn && status.rotation === 'full') { turnAbout(left, -R.y * TURN * dt); turnAbout(fwd, R.x * TURN * dt); }
  else { hand[2] += R.y * SPEED * dt; turnAbout([0, 0, 1], -R.x * TURN * dt); }
  const active = !!(L.x || L.y || R.x || R.y || (mode === 'tilt' && sticks.stL.id !== null));
  ws.send(JSON.stringify({ type: 'input', t: now / 1000, head: null, left: null,
    right: { pose: [...hand, ...handQ], grip: active ? 1 : 0, stick: [0, closed ? 1 : -1], buttons: [], epoch } }));
}, 16);

// ------------------------------------------------------------ start
function begin(m) {
  mode = m;
  if (wsOpen) ws.send(JSON.stringify({ type: 'controls', mode: m }));     // (recorded; the tutorial's texts)
  $('intro').classList.add('hidden');
  $('ctl').classList.remove('hidden');
  if (m === 'tilt') { $('stL').classList.add('tilt'); $('stLT').textContent = 'HOLD + TILT THE PHONE'; }
  showStatus();
}
$('goTouch').addEventListener('click', () => { startAudio(); begin('touch'); });
$('goTilt').addEventListener('click', async () => {
  startAudio();
  try { await enableTilt(); begin('tilt'); }
  catch (e) { $('err').textContent = e.message; }
});

// ------------------------------------------------------------ the controls the admin gave this candidate
// /iphone?c=<code>: "motion" (only tilt steering: an iPhone can't track its position in a browser),
// "touch" (only the joysticks), "choice" (both, tilt first)
const candidate = new URLSearchParams(location.search).get('c');
if (candidate) {
  fetch(`/api/controls/${encodeURIComponent(candidate)}`).then(r => r.json()).then(({ controls }) => {
    const hide = (el) => el && el.classList.add('hidden');
    const fineAfter = (el) => el && el.nextElementSibling && el.nextElementSibling.classList.contains('fine') ? el.nextElementSibling : null;
    if (controls === 'touch') {
      hide($('goTilt')); hide(fineAfter($('goTilt')));
      const b = $('goTouch'); b.textContent = 'Start'; b.style.marginTop = 'auto';
      b.style.background = 'linear-gradient(135deg, #2fdc7f, #19a35b)'; b.style.border = '0'; b.style.fontSize = '18px'; b.style.padding = '19px';
      hide(fineAfter(b));
      const li = document.querySelectorAll('#intro .steps li');
      if (li[2]) li[2].innerHTML = '<span class="num">3</span><span><b>Left stick</b>: move the gripper (forward = into the screen). <b>Right stick</b>: up / down and turn.</span>';
      if (li[1]) li[1].innerHTML = '<span class="num">2</span><span>Tap <b>START</b>. Turn the sound on: the phone beeps when the robot touches something.</span>';
    } else if (controls === 'motion') {
      hide($('goTouch')); hide(fineAfter($('goTouch')));
    }
  }).catch(() => {});
}
