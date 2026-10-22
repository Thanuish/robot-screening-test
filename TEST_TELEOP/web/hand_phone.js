// A phone as the hand camera: it stands in front of the candidate (front camera toward them,
// like a webcam), tracks their hand and sends its pose to the simulator; the computer shows the
// robot. Works on any phone with a camera and a modern browser (no AR, no app).
import { preload, HandControl, openCamera } from '/test-static/hand/core.js';
import { drawHand, Checklist, COLORS, LABELS } from '/test-static/hand/ui.js';
import { AbsoluteHand } from '/test-static/hand/absolute.js';
// [controls] hand_mode = "absolute" (TeleBench style): home, the hand's place = the gripper's
const ABS_MODE = window.SCREEN_CONFIG?.hand?.mode === 'absolute';
const $ = (id) => document.getElementById(id);
const BASE = window.TELEOP_BASE;

// ---- the simulator
let ws = null, wsOpen = false, status = {};
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${BASE}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { wsOpen = true; };
  ws.onclose = () => { wsOpen = false; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') return;
    const m = JSON.parse(ev.data);
    if (m.type !== 'status') return;
    status = m;
    const g = m.task_game;
    if (g) $('phase').textContent = g.finished ? 'Finished, thank you!' : (g.round ? `${g.title} · round ${g.round}` : g.title);
  };
}
connect();
const send = (m) => { if (wsOpen) ws.send(JSON.stringify(m)); };

// ---- the tracker: loading starts now
const tracker = preload();
tracker.catch((e) => { $('msg').textContent = 'The hand tracker could not load: ' + e.message; $('msg').className = 'bad'; });

let cam = null, hand = null, state = null, facing = 'user';
const checklist = new Checklist($('checklist'), () => { hand.arm(true); $('homeBtn').classList.remove('hidden'); }, ABS_MODE ? {
  steps: [
    ['camera', 'Camera on'],
    ['hand', 'Sit back, raise one hand, palm to the camera'],
    ['home', 'Hold it in the circle until it fills: that is <b>home</b>'],
    ['pinch', 'Pinch 🤏 thumb and index finger: that <b>grips</b>'],
    ['release', 'Open the pinch. Ready!'],
  ],
  check: (k, st, camera) => k === 'camera' ? camera : k === 'hand' ? !!st?.tracked : k === 'home' ? !!st?.homeSet
    : k === 'pinch' ? !!st?.pinch : !!st?.tracked && !st?.pinch,
} : {});
function loop() {
  requestAnimationFrame(loop);
  if (!cam) return;
  const c = $('view');
  const r = c.getBoundingClientRect();
  const w = Math.round(r.width * Math.min(2, devicePixelRatio)), h = Math.round(r.height * Math.min(2, devicePixelRatio));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  drawHand(c, cam.video, state, { mirror: facing === 'user', dim: 0.15 });
}
requestAnimationFrame(loop);

async function startCamera() {
  if (cam) { cam.stop(); cam = null; }
  cam = await openCamera(facing);
  $('overlay').classList.add('hidden');
  if (!hand) {
    await tracker;
    const Ctl = ABS_MODE ? AbsoluteHand : HandControl;
    hand = new Ctl({
      send, status: () => status, cfg: window.SCREEN_CONFIG?.hand,
      log: (frames) => send({ type: 'track_log', frames }),   // (the raw points, for tuning)
      onFrame: (st) => {
        state = st;
        checklist.update(true, st);
        const g = st.gesture || 'none';
        const ready = g === 'open' && !st.driving;
        $('chip').style.color = !st.tracked ? COLORS.none : ready ? '#7fb4ff' : COLORS[g];
        $('chip').querySelector('b').textContent = !st.tracked ? 'NO HAND' : ABS_MODE && !st.homeSet ? 'SET HOME'
          : ready ? 'READY' : ABS_MODE && g === 'open' ? 'STEERING' : LABELS[g];
        if (ABS_MODE && hand?.armed) {                   // (a trial waiting: pinch to begin)
          const tg = status.task_game;
          $('msg').textContent = tg?.ready ? `Pinch to begin… ${Math.round(100 * (st.goHold || 0))}%` : '';
        }
        $('stats').textContent = `${st.fps} fps · ${st.latency} ms`;
      },
    });
  }
  await hand.start(cam.video);
}

$('startBtn').addEventListener('click', async () => {
  $('startBtn').disabled = true;
  $('startBtn').textContent = 'Starting…';
  try {
    await startCamera();
    $('startBtn').classList.add('hidden');
    $('flipBtn').classList.remove('hidden');
    $('msg').textContent = '';
  } catch (e) {
    $('startBtn').disabled = false;
    $('startBtn').textContent = 'Start';
    $('msg').textContent = 'Could not start the camera: ' + (e.message || e) + '. Allow the camera for this page.';
    $('msg').className = 'bad';
  }
});
$('flipBtn').addEventListener('click', async () => {
  facing = facing === 'user' ? 'environment' : 'user';
  hand?.stop();
  try { await startCamera(); } catch (e) { $('msg').textContent = 'Could not switch the camera: ' + (e.message || e); }
});
$('homeBtn').addEventListener('click', () => {
  if (ABS_MODE) hand?.rehome(); else send({ type: 'home' });     // (absolute: set home again)
  navigator.vibrate?.(20);
});
