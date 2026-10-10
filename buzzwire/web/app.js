// Robot Buzz Wire: the player's screen. Shows the simulation streamed from the server (MuJoCo
// coordinates: z up, metres), the HUD, a ghost of the commanded target, and lets the keyboard
// act as the controller (the same message as the phone: a hand pose, MOVE, GRAB).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const cfg = await (await fetch('/config.json')).json();

// ------------------------------------------------------------ renderer, camera, lights
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x20242b);
const camera = new THREE.PerspectiveCamera(cfg.view.fov, innerWidth / innerHeight, 0.02, 30);
camera.up.set(0, 0, 1);
camera.position.set(...cfg.view.eye);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(...cfg.view.look);
controls.enableDamping = true;
controls.update();
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a2e24, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(0.9, 0.6, 2.4);
sun.target.position.set(0.45, 0, 0.75);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -0.8, right: 0.8, top: 0.8, bottom: -0.8, near: 0.5, far: 4 });
sun.shadow.bias = -0.0004;
scene.add(sun, sun.target);

// ------------------------------------------------------------ the scene from the server
let bodies = [], sceneId = -1, root = null;
const b64 = (s, T) => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new T(b.buffer); };
function material(g, mats) {
  const m = g.mat >= 0 ? mats[g.mat] : null;
  const rgba = m ? m.rgba : g.rgba;
  const metal = m && m.metallic > 0 ? m.metallic : (m && m.specular > 0.8 ? 0.9 : 0);
  const rough = m && m.roughness > 0 ? m.roughness : (m ? 1 - 0.7 * m.shininess : 0.7);
  return new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]).convertSRGBToLinear(),
    metalness: metal, roughness: Math.max(0.08, rough), transparent: rgba[3] < 1, opacity: rgba[3] });
}
function geometry(g, meshes) {
  const s = g.size;
  switch (g.type) {
    case 'box': return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case 'sphere': return new THREE.SphereGeometry(s[0], 24, 16);
    case 'capsule': return new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 12).rotateX(Math.PI / 2);
    case 'cylinder': return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2);
    case 'ellipsoid': return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case 'plane': return new THREE.PlaneGeometry(2 * (s[0] || 5), 2 * (s[1] || 5));
    case 'mesh': {
      const M = meshes[g.mesh], geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(b64(M.vert, Float32Array), 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(b64(M.normal, Float32Array), 3));
      if (M.indexed) geo.setIndex(new THREE.BufferAttribute(b64(M.face, Uint32Array), 1));
      return geo;
    }
  }
  return null;
}
async function loadScene(id) {
  const j = await (await fetch('/scene.json?v=' + id)).json();
  if (root) scene.remove(root);
  root = new THREE.Group();
  bodies = j.bodies.map(() => { const b = new THREE.Group(); root.add(b); return b; });
  for (const g of j.geoms) {
    const geo = geometry(g, j.meshes);
    if (!geo) continue;
    const mesh = new THREE.Mesh(geo, material(g, j.materials));
    mesh.position.set(...g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = g.type !== 'plane';
    mesh.receiveShadow = true;
    bodies[g.body].add(mesh);
  }
  scene.add(root);
  sceneId = id;
  $('loading').style.display = 'none';
}

// the ghost: where the controller points (the arm follows it with its speed limits)
const ghost = new THREE.Group();
ghost.add(new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0x5cf2a0, transparent: true, opacity: 0.45, depthWrite: false })));
const finger = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.06, 8).rotateX(Math.PI / 2).translate(0, 0, 0.03),
  new THREE.MeshBasicMaterial({ color: 0x5cf2a0, transparent: true, opacity: 0.45, depthWrite: false }));
ghost.add(finger);                                     // along the gripper's pointing direction
ghost.visible = false;
scene.add(ghost);

// ------------------------------------------------------------ network
let ws, status = {};
function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onclose = () => setTimeout(connect, 1000);
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') { onStatus(JSON.parse(ev.data)); return; }
    const f = new Float32Array(ev.data);
    const id = f[1], n = f[2];
    if (id !== sceneId) { if (id !== pendingScene) { pendingScene = id; loadScene(id); } return; }
    for (let i = 0; i < n && i < bodies.length; i++) {
      const o = 4 + 7 * i;
      bodies[i].position.set(f[o], f[o + 1], f[o + 2]);
      bodies[i].quaternion.set(f[o + 4], f[o + 5], f[o + 6], f[o + 3]);
    }
    const o = 4 + 7 * n;
    ghost.visible = !Number.isNaN(f[o]);
    if (ghost.visible) { ghost.position.set(f[o], f[o + 1], f[o + 2]); ghost.quaternion.set(f[o + 4], f[o + 5], f[o + 6], f[o + 3]); }
    buzz(f[3] > 0.5);
  };
}
let pendingScene = -1;
connect();
const send = (m) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); };
$('restart').onclick = () => { send({ type: 'restart' }); $('restart').blur(); };

// ------------------------------------------------------------ HUD
const fmt = (s) => { s = Math.max(0, Math.ceil(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
let lastMsg = '', toastT = 0;
function onStatus(s) {
  if (s.type !== 'status') return;
  status = s;
  $('title').textContent = s.title;
  $('text').textContent = s.finished ? 'All levels done. Thank you!' : s.text;
  $('timer').textContent = s.state === 'countdown' ? fmt(s.limit) : fmt(s.limit - s.time);
  $('nTouch').textContent = s.touches;
  $('bar').style.width = `${Math.round(100 * s.progress)}%`;
  $('count').textContent = s.state === 'countdown' && s.countdown > 0 ? Math.ceil(s.countdown) : '';
  if (s.state === 'waiting') $('text').textContent = 'Connect your phone (scan the QR code) or press a movement key to start. ' + s.text;
  $('net').textContent = `simulation speed ${s.rtf.toFixed(2)}x real time`;
  if (s.message && s.message !== lastMsg) {
    $('toast').textContent = s.message; $('toast').style.opacity = 1;
    clearTimeout(toastT); toastT = setTimeout(() => $('toast').style.opacity = 0, 2500);
  }
  lastMsg = s.message;
}
fetch('/config.json').then(r => r.json()).then(c => {
  if (c.phone_url) { $('qr').src = '/qr.svg'; $('phoneUrl').innerHTML = `<a href="${c.phone_url}">${c.phone_url}</a>`; }
  else $('phoneUrl').textContent = 'No public address (start the tunnel for the phone).';
});

// buzz: red edge flash and a buzzing sound while the wand touches the wire
let audio = null, osc = null, gain = null, buzzing = false;
addEventListener('pointerdown', () => {
  if (audio) return;
  audio = new AudioContext();
  osc = audio.createOscillator(); gain = audio.createGain();
  osc.type = 'square'; osc.frequency.value = 110; gain.gain.value = 0;
  osc.connect(gain).connect(audio.destination); osc.start();
}, { once: false });
function buzz(on) {
  if (on === buzzing) return;
  buzzing = on;
  $('flash').style.opacity = on ? 1 : 0;
  if (gain) gain.gain.setTargetAtTime(on ? 0.08 : 0, audio.currentTime, 0.01);
}

// ------------------------------------------------------------ the view's heading (for "forward")
let sentYaw = null;
function viewYaw() {
  const d = new THREE.Vector3(); camera.getWorldDirection(d);
  return Math.atan2(d.y, d.x);
}
setInterval(() => {
  const y = viewYaw();
  if (sentYaw === null || Math.abs(y - sentYaw) > 0.01) { send({ type: 'view', yaw: y }); sentYaw = y; }
}, 200);

// ------------------------------------------------------------ keyboard as the controller
// A virtual hand: the keys move and turn it ("forward" = into the view), MOVE is held while a
// movement key is down, C toggles the gripper. Sent like the phone's messages.
const keys = new Set();
const hand = new THREE.Vector3(0, 0, 1), handQ = new THREE.Quaternion();
let closed = false, lastKey = 0;
addEventListener('keydown', (e) => {
  if (e.repeat) return;
  const k = e.key.toLowerCase();
  if (k === 'c') { closed = !closed; lastKey = performance.now(); }
  keys.add(k);
});
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
addEventListener('blur', () => keys.clear());
const MOVE_KEYS = ['w', 'a', 's', 'd', 'r', 'f', 'q', 'e', 'i', 'k', 'j', 'l'];
let prev = performance.now();
function keyboard(now) {
  const dt = Math.min(0.05, (now - prev) / 1000); prev = now;
  const moving = MOVE_KEYS.some(k => keys.has(k));
  if (moving) lastKey = now;
  if (now - lastKey > 400) { $('keys').classList.remove('on'); return; }   // keyboard idle: the phone may drive
  $('keys').classList.add('on');
  const v = keys.has('shift') ? 0.03 : 0.09, w = keys.has('shift') ? 0.4 : 1.0;
  const yaw = viewYaw(), fwd = new THREE.Vector3(Math.cos(yaw), Math.sin(yaw), 0), left = new THREE.Vector3(-Math.sin(yaw), Math.cos(yaw), 0);
  const ax = (a, b) => (keys.has(a) ? 1 : 0) - (keys.has(b) ? 1 : 0);
  hand.addScaledVector(fwd, ax('w', 's') * v * dt).addScaledVector(left, ax('a', 'd') * v * dt);
  hand.z += ax('r', 'f') * v * dt;
  const rot = (axis, a) => handQ.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, a));
  rot(new THREE.Vector3(0, 0, 1), ax('q', 'e') * w * dt);
  if (status.rotation === 'full') { rot(left, ax('k', 'i') * w * dt); rot(fwd, ax('l', 'j') * w * dt); }
  send({ type: 'input', head: null, left: null,
         right: { pose: [hand.x, hand.y, hand.z, handQ.w, handQ.x, handQ.y, handQ.z], grip: moving ? 1 : 0,
                  stick: [0, closed ? 1 : -1], epoch: 0 } });
}

// ------------------------------------------------------------ frame loop
renderer.setAnimationLoop((now) => {
  keyboard(now);
  controls.update();
  renderer.render(scene, camera);
});
