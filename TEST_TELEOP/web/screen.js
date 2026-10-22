// Screening monitor overlay: shows the phase, instructions, timer and progress from the
// simulator's status (the 'teleop-status' events of main.js), and runs the hand control (a
// webcam watches the candidate's hand; or a phone's camera does, on its own page).
import { preload, HandControl, openCamera } from '/test-static/hand/core.js';
import { drawHand, Checklist, COLORS, LABELS } from '/test-static/hand/ui.js';
const $ = (id) => document.getElementById(id);
const code = window.TELEOP_BASE.split('/').pop();

let lastMsg = '', toastTimer = 0;
function toast(text) {
  $('toast').textContent = text;
  $('toast').style.opacity = 1;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').style.opacity = 0; }, 3500);
}

function fmt(s) {
  s = Math.max(0, Math.ceil(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// ------------------------------------------------------------ depth cues
// Seen on a flat screen, it's hard to judge how high the gripper is and where it would come down.
// So: a ring on the table straight below the gripper (where it would land), a thin line down to
// it, and both turn green when the fingertips are low enough to grab something ([depth_cues]).
const CFG = window.SCREEN_CONFIG, CUES = CFG.depth_cues;
const GRIP = CFG.gripper_prefix || `${CFG.arm}_gripper_`;   // the gripper's names (on the arm, or floating)
const TABLE_Z = CFG.table_z, GRAB_HEIGHT = CUES.grab_height;
const CUE_COLOUR = parseInt(CUES.color.slice(1), 16), GRAB_COLOUR = parseInt(CUES.grab_color.slice(1), 16);
let cues = null;
function makeCues() {
  const T = window.__teleop, THREE = T.THREE;
  if (cues) cues.grp.parent && cues.grp.parent.remove(cues.grp);
  const grp = new THREE.Group();
  T.mjRoot.add(grp);                                   // MuJoCo coordinates (z up)
  const mat = (opacity) => new THREE.MeshBasicMaterial({ color: CUE_COLOUR, transparent: true, opacity,
                                                         depthTest: false, depthWrite: false });
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.028, 48), mat(0.9));
  const dot = new THREE.Mesh(new THREE.CircleGeometry(0.005, 20), mat(0.9));
  const lineGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
  const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: CUE_COLOUR, transparent: true, opacity: 0.7,
                                                                      depthTest: false, depthWrite: false }));
  for (const o of [ring, dot, line]) { o.renderOrder = 10; o.frustumCulled = false; grp.add(o); }
  cues = { grp, ring, dot, line, colour: null };
}
function updateCues() {
  requestAnimationFrame(updateCues);
  const T = window.__teleop;
  if (!CUES.enabled || !T || !T.body) return;
  if (!cues) makeCues();
  const r = T.body(`${GRIP}right_pad`), l = T.body(`${GRIP}left_pad`);
  if (!r || !l) { cues.grp.visible = false; return; }
  const THREE = T.THREE, p = new THREE.Vector3(), q = new THREE.Vector3();
  r.getWorldPosition(p); l.getWorldPosition(q);
  p.add(q).multiplyScalar(0.5);                        // between the finger pads
  T.mjRoot.worldToLocal(p);
  const h = p.z - TABLE_Z;
  cues.grp.visible = h > -0.01 && h < 0.6;
  cues.ring.position.set(p.x, p.y, TABLE_Z + 0.0015);
  cues.dot.position.set(p.x, p.y, TABLE_Z + 0.0016);
  const pos = cues.line.geometry.attributes.position;
  pos.setXYZ(0, p.x, p.y, p.z); pos.setXYZ(1, p.x, p.y, TABLE_Z + 0.002);
  pos.needsUpdate = true;
  const colour = h < GRAB_HEIGHT ? GRAB_COLOUR : CUE_COLOUR;
  if (colour !== cues.colour) {
    for (const o of [cues.ring, cues.dot, cues.line]) o.material.color.setHex(colour);
    cues.colour = colour;
  }
}
window.addEventListener('teleop-scene', () => {      // (scene reloaded: rebuild)
  if (cues && cues.grp.parent) cues.grp.parent.remove(cues.grp);
  cues = null;
});
requestAnimationFrame(updateCues);

// ------------------------------------------------------------ view aids ([view], [side_view])
// See-through arm: the shoulder, upper arm and forearm are drawn faint so they don't hide the
// work. Table grid: faint lines every few cm, for judging distances. Side view: a second small
// view from the side at table height, following the gripper, so its height is seen directly.
const VIEW = CFG.view, SIDE = VIEW.side;
let sideCam = null, sideFrame = null, sideLook = null;
function setupViewAids() {
  const T = window.__teleop, THREE = T.THREE;
  if (VIEW.arm_opacity < 1) {
    for (const link of ['shoulder_link', 'upper_arm_link', 'forearm_link']) {
      const b = T.body(`${CFG.arm}_${link}`);
      if (!b) continue;
      b.traverse(o => {
        if (!o.isMesh) return;
        o.material = o.material.clone();               // (materials are shared with other parts)
        Object.assign(o.material, { transparent: true, opacity: VIEW.arm_opacity, depthWrite: true });  // (hides its own inside)
        o.renderOrder = 5;
        o.castShadow = false;
      });
    }
  }
  const old = T.mjRoot.getObjectByName('screen_grid');
  if (old) T.mjRoot.remove(old);                       // (scene reloaded)
  if (VIEW.table_grid > 0) {
    const step = VIEW.table_grid, pts = [];
    const x0 = -0.15, x1 = 0.95, y0 = -0.75, y1 = 0.75, z = TABLE_Z + 0.0008;   // the table top
    for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) pts.push(x, y0, z, x, y1, z);
    for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) pts.push(x0, y, z, x1, y, z);
    const geo = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    const grid = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x5a6070, transparent: true, opacity: 0.22 }));
    grid.name = 'screen_grid';
    T.mjRoot.add(grid);
  }
  if (SIDE.enabled) {
    sideCam = new THREE.PerspectiveCamera(SIDE.fov, 1.5, 0.02, 10);
    if (!sideFrame) {
      sideFrame = document.createElement('div');
      sideFrame.id = 'sideView';
      sideFrame.innerHTML = '<span>SIDE VIEW</span>';
      document.body.appendChild(sideFrame);
    }
    if (!T.afterRender.includes(drawSideView)) T.afterRender.push(drawSideView);
  }
}
function sideRect() {                                  // [x, y from the bottom, w, h] in CSS pixels
  const w = Math.round(window.innerWidth * SIDE.size), h = Math.round(w / 1.5);
  const top = 120;
  return [window.innerWidth - w - 18, window.innerHeight - top - h, w, h];
}
function drawSideView(renderer, scene) {
  const T = window.__teleop, THREE = T.THREE;
  const r = T.body(`${GRIP}right_pad`), l = T.body(`${GRIP}left_pad`);
  if (!r || !l) { sideFrame.style.display = 'none'; return; }
  const p = new THREE.Vector3(), q = new THREE.Vector3();
  r.getWorldPosition(p); l.getWorldPosition(q);
  p.add(q).multiplyScalar(0.5);
  T.mjRoot.worldToLocal(p);                            // the gripper, MuJoCo frame
  const target = new THREE.Vector3(p.x, p.y, TABLE_Z + SIDE.height);
  if (!sideLook) sideLook = target.clone();
  sideLook.lerp(target, 0.12);                         // follows smoothly (no shaking)
  const dir = SIDE.side === 'left' ? 1 : -1;
  const eye = sideLook.clone().add(new THREE.Vector3(0, dir * SIDE.distance, 0.02));
  sideCam.position.copy(T.mjRoot.localToWorld(eye.clone()));
  sideCam.lookAt(T.mjRoot.localToWorld(sideLook.clone()));
  const [x, y, w, h] = sideRect();
  sideCam.aspect = w / h;
  // only what's within ~15 cm of the gripper (across the view) is shown: things nearer the side
  // camera (another object, the book holder) are cut away instead of hiding the gripper
  sideCam.near = Math.max(0.02, SIDE.distance - 0.15);
  sideCam.updateProjectionMatrix();
  Object.assign(sideFrame.style, { display: 'block', left: `${x}px`, bottom: `${y}px`, width: `${w}px`, height: `${h}px` });
  renderer.setScissorTest(true);
  renderer.setScissor(x, y, w, h);
  renderer.setViewport(x, y, w, h);
  renderer.render(scene, sideCam);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
}
window.addEventListener('teleop-scene', setupViewAids);
if (window.__teleop) setupViewAids();

let wasHoming = false;
// ------------------------------------------------------------ hand control
// The tracker starts loading now (a few seconds the first time), the camera when the candidate
// presses the button; the checklist (hand, fist, pinch, open) must be done before the hand
// drives the robot. A phone can be the camera instead (/h/<code>): then this page only watches.
let lastStatus = null, cam = null, hand = null, handState = null;
const tracker = preload();
tracker.then(({ delegate }) => {
  $('trackerState').textContent = `Hand tracker ready (${delegate === 'GPU' ? 'graphics chip' : 'processor'})`;
  $('trackerState').className = 'ok';
}).catch((e) => {
  $('trackerState').textContent = 'The hand tracker could not load: ' + e.message + '. Check the internet connection and reload.';
  $('trackerState').className = 'bad';
});
const checklist = new Checklist($('checklist'), () => {
  hand.arm(true);                                         // (now the hand drives: the test goes on)
  $('handBox').classList.remove('hidden');
});
// a pinch (thumb and index finger together) closes the gripper while it is held; W/S and A/D tilt
// the gripper in tasks with full rotation (the hand moves and twists; it never has to tilt)
const tiltKeys = { KeyW: 0, KeyS: 0, KeyA: 0, KeyD: 0 };
function updateTilt() { hand?.setTilt(tiltKeys.KeyW - tiltKeys.KeyS, tiltKeys.KeyD - tiltKeys.KeyA); }
window.addEventListener('keydown', (e) => {
  if (e.code in tiltKeys) { tiltKeys[e.code] = 1; updateTilt(); }
});
window.addEventListener('keyup', (e) => { if (e.code in tiltKeys) { tiltKeys[e.code] = 0; updateTilt(); } });
window.addEventListener('blur', () => { for (const k in tiltKeys) tiltKeys[k] = 0; updateTilt(); });
function setChip(el, state) {
  const g = state?.gesture || 'none';
  const label = !state?.tracked ? 'NO HAND' : (g === 'open' && !state.driving ? 'READY' : LABELS[g]);
  el.style.color = !state?.tracked ? COLORS.none : (g === 'open' && !state.driving ? '#7fb4ff' : COLORS[g]);
  el.querySelector('b').textContent = label;
}
function drawLoop() {
  requestAnimationFrame(drawLoop);
  if (!cam) return;
  if (!$('start').classList.contains('hidden')) drawHand($('startCam'), cam.video, handState, { dim: 0.15 });
  if (!$('handBox').classList.contains('hidden')) drawHand($('handPreview'), cam.video, handState, { dim: 0.2 });
}
requestAnimationFrame(drawLoop);
$('camBtn').addEventListener('click', async () => {
  if (cam) return;
  $('camBtn').disabled = true;
  $('camBtn').textContent = 'Starting the camera…';
  try {
    cam = await openCamera('user');
    $('camOverlay').classList.add('hidden');
    $('startChip').querySelector('b').textContent = 'LOADING';
    await tracker;
    hand = new HandControl({
      send: (m) => window.__teleop?.send?.(m),
      status: () => lastStatus,
      gripMode: 'pinch',
      log: (frames) => window.__teleop?.send?.({ type: 'track_log', frames }),   // (the raw points, for tuning)
      onFrame: (st) => {
        handState = st;
        checklist.update(true, st);
        setChip($('startChip'), st);
        setChip($('handChip'), st);
        $('handStats').textContent = st.tracked ? `${st.fps} fps · ${st.latency} ms` : `${st.fps} fps`;
      },
    });
    await hand.start(cam.video);
    $('camBtn').classList.add('hidden');
  } catch (e) {
    $('camBtn').disabled = false;
    $('camBtn').textContent = 'Start the camera';
    $('trackerState').textContent = 'Could not start the camera: ' + (e.message || e) + '. Allow the camera for this page, or use a phone as the camera.';
    $('trackerState').className = 'bad';
    if (cam) { cam.stop(); cam = null; }
  }
});
$('phoneCamBtn').addEventListener('click', (e) => {
  e.preventDefault();
  $('phoneCamQr').classList.toggle('hidden');
});
window.addEventListener('keydown', (e) => {
  if (e.code === 'KeyH' && hand?.armed) window.__teleop?.send?.({ type: 'home' });
});
window.addEventListener('teleop-status', (ev) => {
  lastStatus = ev.detail;
  $('robotState').textContent = 'Robot connected';
  $('robotState').className = 'ok';
});
window.addEventListener('teleop-scene-error', (ev) => {
  $('robotState').textContent = 'Connecting to the robot… (retrying)';
  $('robotState').className = 'bad';
});

window.addEventListener('teleop-status', (ev) => {
  const g = ev.detail.task_game;
  if (!g) return;
  $('phaseTitle').textContent = g.round ? `${g.title}  ·  round ${g.round}` : g.title;
  $('phaseText').textContent = g.text;
  const pairing = g.kind === 'pair';
  $('start').classList.toggle('hidden', !pairing);
  if (!pairing && cam && hand?.armed) $('handBox').classList.toggle('hidden', !!g.finished);
  $('done').classList.toggle('hidden', !g.finished);
  // steps
  const steps = $('steps');
  if (steps.children.length !== g.steps_total) {
    steps.innerHTML = '';
    for (let i = 0; i < g.steps_total; i++) steps.appendChild(document.createElement('span'));
  }
  [...steps.children].forEach((s, i) => s.classList.toggle('on', i < g.steps_done));
  // tower: its height now and the best that stood, blocks left, falls
  const tw = g.tower;
  $('taskHud').classList.toggle('hidden', !tw);
  steps.classList.toggle('hidden', !!tw || !!g.wire);
  if (tw) {
    $('taskHud').innerHTML = `Tower <b>${Math.round(tw.height * 100)} cm</b> (best ${Math.round(tw.best * 100)} cm, ` +
      `goal ${Math.round(tw.target * 100)} cm)  ·  blocks left ${tw.left}  ·  falls ${tw.falls}`;
  }
  // buzz wire: touches, how far along, a red glow while touching, a hint when the loop is off
  const w = g.wire;
  $('wireHud').classList.toggle('hidden', !w);

  $('buzzFlash').classList.toggle('on', !!(w && w.touching));
  if (w) {
    $('wireTouches').textContent = `Touches: ${w.touches}`;
    $('wireTouches').classList.toggle('hit', w.touching);
    $('wireBar').firstChild.style.width = `${Math.round(100 * w.progress)}%`;
    $('wireOff').textContent = w.off && g.intro === 0 ? 'The loop is off the wire: bring it back around the wire' : '';
  }
  // timer and countdown
  if (g.limit) {
    const left = g.intro > 0 ? g.limit : g.limit - g.time;
    $('timer').textContent = fmt(left);
    $('timer').classList.toggle('low', g.intro === 0 && left < 30);
    $('practice').textContent = g.scored ? '' : 'practice: not scored';
  } else {
    $('timer').textContent = '';
    $('practice').textContent = '';
  }
  $('count').textContent = g.intro > 0 ? Math.ceil(g.intro) : '';
  if (g.homing && !wasHoming) toast('Going home…');
  wasHoming = !!g.homing;
  if (g.message && g.message !== lastMsg) toast(g.message);
  lastMsg = g.message;
});
