// Robot Buzz Wire: the player's screen. The simulation runs on the server; this page draws it
// as realistically as an ordinary laptop allows, and shows the HUD. The phone is the only
// controller (its page is /phone).
//
// Rendering: physically based materials lit by an HDRI studio (reflections on the chrome wire,
// the brass ring and the robot), a key light with soft shadows, ACES tone mapping, MSAA.
// Presets: "standard" (the default: light enough for laptop graphics), "low" (chosen by itself
// when the frame rate drops below 45), "high" (?q=high: adds ambient occlusion and bloom).
// The robot's positions arrive ~50 times a second; the page glides between them (a 30 ms
// buffer), so the motion is smooth at any frame rate.
// The physics' simple shapes are replaced here by fine ones: the wire is one smooth tube along
// the level's curve, the loop a real torus, the board a rounded wooden block.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';

const $ = (id) => document.getElementById(id);
const cfg = await (await fetch('/config.json')).json();

// ------------------------------------------------------------ renderer, world frame, camera
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c0f);
// everything from the simulation lives in MuJoCo's frame (z up); turned so that three.js's y is up
const mj = new THREE.Group();
mj.rotation.x = -Math.PI / 2;
scene.add(mj);
mj.updateMatrixWorld(true);
const toWorld = (v) => new THREE.Vector3(...v).applyMatrix4(mj.matrixWorld);
const camera = new THREE.PerspectiveCamera(cfg.view.fov, innerWidth / innerHeight, 0.02, 40);
camera.position.copy(toWorld(cfg.view.eye));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(toWorld(cfg.view.look));
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 0.35; controls.maxDistance = 2.2;
controls.maxPolarAngle = Math.PI * 0.47;
controls.update();
const home = { pos: camera.position.clone(), target: controls.target.clone() };
renderer.domElement.addEventListener('dblclick', () => { camera.position.copy(home.pos); controls.target.copy(home.target); });

// ------------------------------------------------------------ light: HDRI studio + a soft key light
const pmrem = new THREE.PMREMGenerator(renderer);
new RGBELoader().load('/assets/studio_small_08_1k.hdr', (hdr) => {
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  scene.environment = pmrem.fromEquirectangular(hdr).texture;
  scene.environmentIntensity = 0.75;
  hdr.dispose();
});
const key = new THREE.DirectionalLight(0xfff4e6, 2.6);
key.position.copy(toWorld([1.3, -0.9, 2.6]));
key.target.position.copy(toWorld([0.45, 0, 0.8]));
key.castShadow = true;
Object.assign(key.shadow.camera, { left: -0.9, right: 0.9, top: 0.9, bottom: -0.9, near: 0.5, far: 5 });
key.shadow.bias = -0.0002; key.shadow.normalBias = 0.01; key.shadow.radius = 4;
scene.add(key, key.target);
const fill = new THREE.HemisphereLight(0xcfe0ff, 0x2a2018, 0.35);
scene.add(fill);

// ------------------------------------------------------------ materials and textures
const texLoader = new THREE.TextureLoader();
function tex(name, repeat, srgb) {
  const t = texLoader.load(`/assets/${name}`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(repeat[0], repeat[1]);
  t.anisotropy = 8;
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
function pbrSet(base, repeat) {
  return { map: tex(`${base}_diff_1k.jpg`, repeat, true), normalMap: tex(`${base}_nor_gl_1k.jpg`, repeat),
           roughnessMap: tex(`${base}_rough_1k.jpg`, repeat) };
}
const M = {
  chrome: new THREE.MeshPhysicalMaterial({ color: 0xeef0f3, metalness: 1, roughness: 0.1, envMapIntensity: 1.4,
                                          emissive: 0xff1030, emissiveIntensity: 0 }),
  brass: new THREE.MeshPhysicalMaterial({ color: 0xd9a84e, metalness: 1, roughness: 0.22, clearcoat: 0.4, clearcoatRoughness: 0.2 }),
  rubberRed: new THREE.MeshStandardMaterial({ color: 0xb3122a, roughness: 0.62 }),
  rubberBlack: new THREE.MeshStandardMaterial({ color: 0x16171a, roughness: 0.78 }),
  grip: new THREE.MeshPhysicalMaterial({ color: 0x1b1c20, roughness: 0.55, clearcoat: 0.15, sheen: 0.4, sheenColor: 0x444a55 }),
  wood: new THREE.MeshPhysicalMaterial({ ...pbrSet('wood_table_001', [1.0, 0.35]), roughness: 1, color: 0xffe9d2, clearcoat: 0.35, clearcoatRoughness: 0.35 }),
  concrete: new THREE.MeshStandardMaterial({ ...pbrSet('concrete_floor_02', [4, 4]), roughness: 1, color: 0xb8b8b8 }),
  plaster: new THREE.MeshStandardMaterial({ ...pbrSet('plastered_wall_04', [3, 1.5]), roughness: 1, color: 0xd8d4cc }),
  bench: new THREE.MeshPhysicalMaterial({ color: 0xe9eae7, roughness: 0.42, clearcoat: 0.25, clearcoatRoughness: 0.4 }),
  alu: new THREE.MeshStandardMaterial({ color: 0xc7cbd1, metalness: 1, roughness: 0.32 }),
  ceiling: new THREE.MeshStandardMaterial({ color: 0x24262b, roughness: 0.9 }),
  panel: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xfff6ea, emissiveIntensity: 2.2 }),
};
// the robot's own materials (MuJoCo Menagerie names) as real surfaces
const ROBOT = {
  ur_black: new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.5 }),
  ur_jointgray: new THREE.MeshPhysicalMaterial({ color: 0x3a3c40, metalness: 0.3, roughness: 0.45, clearcoat: 0.3 }),
  ur_linkgray: new THREE.MeshStandardMaterial({ color: 0xc9ccd1, metalness: 1, roughness: 0.3 }),
  ur_urblue: new THREE.MeshPhysicalMaterial({ color: 0x6fa8d6, metalness: 0.1, roughness: 0.35, clearcoat: 0.8, clearcoatRoughness: 0.15 }),
  ur_gripper_metal: new THREE.MeshStandardMaterial({ color: 0xb9bdc3, metalness: 1, roughness: 0.35 }),
  ur_gripper_silicone: new THREE.MeshStandardMaterial({ color: 0x1d1e21, roughness: 0.85 }),
  ur_gripper_gray: new THREE.MeshStandardMaterial({ color: 0x5a5d63, metalness: 0.5, roughness: 0.45 }),
  ur_gripper_black: new THREE.MeshStandardMaterial({ color: 0x141518, roughness: 0.55 }),
};

// ------------------------------------------------------------ the room (static)
function box(w, d, h, mat, pos, radius = 0) {
  const g = radius > 0 ? new RoundedBoxGeometry(w, d, h, 3, radius) : new THREE.BoxGeometry(w, d, h);
  const m = new THREE.Mesh(g, mat);
  m.position.set(...pos); m.castShadow = m.receiveShadow = true;
  mj.add(m); return m;
}
{
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(7, 7), M.concrete);
  floor.receiveShadow = true; mj.add(floor);
  for (const [w, d, x, y] of [[7, 0.05, 0, 3.5], [7, 0.05, 0, -3.5], [0.05, 7, 3.5, 0], [0.05, 7, -3.5, 0]]) {
    const wall = box(w, d, 3, M.plaster, [x, y, 1.5]); wall.castShadow = false;
  }
  box(7, 7, 0.05, M.ceiling, [0, 0, 3.0]).castShadow = false;
  for (const [x, y] of [[0.5, -0.8], [0.5, 0.8], [-1.5, 0], [2, 0]]) box(1.2, 0.3, 0.02, M.panel, [x, y, 2.98]).castShadow = false;
  // the lab bench: laminate top on an aluminium frame (the physics' table is its top)
  box(1.1, 1.5, 0.04, M.bench, [0.4, 0, 0.73], 0.006);
  for (const [sx, sy] of [[-1, -1], [-1, 1], [1, -1], [1, 1]]) box(0.045, 0.045, 0.71, M.alu, [0.4 + sx * 0.5, sy * 0.7, 0.355], 0.004);
  for (const sy of [-1, 1]) box(1.0, 0.03, 0.03, M.alu, [0.4, sy * 0.7, 0.15], 0.004);
}

// ------------------------------------------------------------ the level (static: board, posts, wire)
const level = new THREE.Group();
mj.add(level);
let wireMats = [];
async function buildLevel() {
  const L = await (await fetch('/level.json')).json();
  level.clear();
  const [bx, by] = L.board_center, [bw, bd, bh] = L.board_size, top = L.table_z + bh;
  // the board: a rounded wooden block
  const board = new THREE.Mesh(new RoundedBoxGeometry(bw, bd, bh, 4, 0.006), M.wood);
  board.position.set(bx, by, L.table_z + bh / 2); board.castShadow = board.receiveShadow = true;
  level.add(board);
  const pts = L.wire.map(p => new THREE.Vector3(...p));
  // posts: brass, a flange at the foot, a ball on top
  for (const p of [pts[0], pts[pts.length - 1]]) {
    const h = p.z - top;
    const post = new THREE.Mesh(new THREE.CylinderGeometry(L.post_radius, L.post_radius, h, 24).rotateX(Math.PI / 2), M.brass);
    post.position.set(p.x, p.y, top + h / 2);
    const flange = new THREE.Mesh(new THREE.CylinderGeometry(L.post_radius * 2.2, L.post_radius * 2.4, 0.006, 32).rotateX(Math.PI / 2), M.brass);
    flange.position.set(p.x, p.y, top + 0.003);
    const ball = new THREE.Mesh(new THREE.SphereGeometry(L.post_radius * 1.35, 24, 16), M.brass);
    ball.position.copy(p);
    for (const m of [post, flange, ball]) { m.castShadow = m.receiveShadow = true; level.add(m); }
  }
  // the wire: one smooth chrome tube; red rubber sleeves at both ends (touching them doesn't count)
  const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal');
  const n = Math.max(64, pts.length * 3);
  const wire = new THREE.Mesh(new THREE.TubeGeometry(curve, n, L.wire_radius, 14), M.chrome);
  wire.castShadow = true; level.add(wire);
  const sub = (a, b) => {                                     // part of the curve between arc lengths a and b
    const u = [], k = 16;
    for (let i = 0; i <= k; i++) u.push(curve.getPointAt((a + (b - a) * i / k) / L.wire_length));
    return new THREE.CatmullRomCurve3(u);
  };
  for (const [a, b] of [[0, L.sleeve_length], [L.wire_length - L.sleeve_length, L.wire_length]]) {
    const s = new THREE.Mesh(new THREE.TubeGeometry(sub(a, b), 24, L.sleeve_radius, 16), M.rubberRed);
    s.castShadow = true; level.add(s);
  }
  return L;
}

// ------------------------------------------------------------ the simulation's bodies
// The server's geometry (scene.json) gives the robot, the forks and the wand's body. Shapes this
// page draws better itself are left out (by name); the wand gets a torus loop and a rubber grip.
let bodies = [], sceneId = -1, simRoot = null, pendingScene = -1;
const SKIP = /^(floor|table_top|board_top|wire_|sleeve_|post_|wand_)/;
const b64 = (s, T) => { const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new T(b.buffer); };
function geometry(g, meshes) {
  const s = g.size;
  switch (g.type) {
    case 'box': return new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    case 'sphere': return new THREE.SphereGeometry(s[0], 24, 16);
    case 'capsule': return new THREE.CapsuleGeometry(s[0], 2 * s[1], 6, 16).rotateX(Math.PI / 2);
    case 'cylinder': return new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32).rotateX(Math.PI / 2);
    case 'ellipsoid': return new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]);
    case 'mesh': {
      const Mm = meshes[g.mesh], geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(b64(Mm.vert, Float32Array), 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(b64(Mm.normal, Float32Array), 3));
      if (Mm.indexed) geo.setIndex(new THREE.BufferAttribute(b64(Mm.face, Uint32Array), 1));
      return geo;
    }
  }
  return null;
}
function materialFor(g, mats) {
  const name = g.mat >= 0 ? mats[g.mat].name : '';
  if (ROBOT[name]) return ROBOT[name];
  if (g.name.startsWith('fork_')) return M.rubberBlack;
  const rgba = g.mat >= 0 ? mats[g.mat].rgba : g.rgba;
  return new THREE.MeshStandardMaterial({ color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: 0.6 });
}
function wand(L) {
  const R = L.loop_radius, t = L.loop_thickness, [len, w, h] = L.handle_size;
  const g = new THREE.Group();
  const loop = new THREE.Mesh(new THREE.TorusGeometry(R, t / 2, 20, 96).rotateX(Math.PI / 2), M.brass);
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(t * 0.6, t * 0.6, 0.016, 16).rotateZ(Math.PI / 2), M.brass);
  stem.position.set(-R - 0.006, 0, 0);
  const ferrule = new THREE.Mesh(new THREE.CylinderGeometry(w * 0.55, w * 0.55, 0.012, 24).rotateZ(Math.PI / 2), M.brass);
  ferrule.position.set(-R - 0.012, 0, 0);
  const handle = new THREE.Mesh(new RoundedBoxGeometry(len - 0.012, w, h, 3, Math.min(w, h) * 0.35), M.grip);
  handle.position.set(-R - 0.012 - (len - 0.012) / 2, 0, 0);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(w * 0.52, 20, 12), M.brass);
  cap.position.set(-R - len, 0, 0);
  for (const m of [loop, stem, ferrule, handle, cap]) { m.castShadow = m.receiveShadow = true; g.add(m); }
  return g;
}
async function loadScene(id) {
  const [j, L] = await Promise.all([(await fetch('/scene.json?v=' + id)).json(), buildLevel()]);
  if (simRoot) mj.remove(simRoot);
  simRoot = new THREE.Group();
  bodies = j.bodies.map((name) => {
    const b = new THREE.Group(); simRoot.add(b);
    if (name === 'wand') b.add(wand(L));
    return b;
  });
  for (const g of j.geoms) {
    if (SKIP.test(g.name) || g.type === 'plane') continue;
    const geo = geometry(g, j.meshes);
    if (!geo) continue;
    const mesh = new THREE.Mesh(geo, materialFor(g, j.materials));
    mesh.position.set(...g.pos);
    mesh.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    mesh.castShadow = mesh.receiveShadow = true;
    bodies[g.body].add(mesh);
  }
  mj.add(simRoot);
  sceneId = id;
  $('loading').style.opacity = 0;
  setTimeout(() => $('loading').classList.add('hidden'), 700);
}

// ------------------------------------------------------------ post-processing and quality
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType }));
composer.addPass(new RenderPass(scene, camera));
const gtao = new GTAOPass(scene, camera, innerWidth, innerHeight);
gtao.updateGtaoMaterial({ radius: 0.12, distanceExponent: 1.5, thickness: 1.2, scale: 1.0, samples: 12 });
gtao.blendIntensity = 0.9;
composer.addPass(gtao);
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.35, 0.5, 0.92);
composer.addPass(bloom);
composer.addPass(new OutputPass());
const smaa = new SMAAPass(innerWidth, innerHeight);
composer.addPass(smaa);
const QUALITY = {
  high: { ratio: Math.min(devicePixelRatio, 2), shadow: 4096, ao: true, bloom: true },
  standard: { ratio: Math.min(devicePixelRatio, 1.25), shadow: 2048, ao: false, bloom: false },
  low: { ratio: Math.min(devicePixelRatio, 1) * 0.8, shadow: 1024, ao: false, bloom: false },
};
const forced = new URLSearchParams(location.search).get('q');
let quality = QUALITY[forced] ? forced : 'standard';
let useComposer = false;                 // effects on: render through the composer, else directly
function applyQuality() {
  const q = QUALITY[quality];
  renderer.setPixelRatio(q.ratio);
  composer.setPixelRatio(q.ratio);
  if (key.shadow.mapSize.x !== q.shadow) {
    key.shadow.mapSize.set(q.shadow, q.shadow);
    if (key.shadow.map) { key.shadow.map.dispose(); key.shadow.map = null; }
  }
  gtao.enabled = q.ao; bloom.enabled = q.bloom;
  useComposer = q.ao || q.bloom;
  key.shadow.radius = quality === 'low' ? 2 : 4;
  resize();
}
function resize() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
}
addEventListener('resize', resize);
applyQuality();

// ------------------------------------------------------------ network
// ------------------------------------------------------------ smooth motion between packets
// Each packet: the simulation time and every body's pose. Shown 30 ms late, interpolated between
// the two packets around that moment (positions linearly, rotations by slerp).
const DELAY = 0.03;
let snaps = [], clockOff = null, lastSim = -1;
function addSnap(f) {
  const t = f[0], n = f[2];
  if (t < lastSim - 0.001) { snaps = []; clockOff = null; }     // a new level: the clock restarts
  lastSim = t;
  const off = performance.now() / 1000 - t;                       // local clock - simulation clock
  clockOff = clockOff === null ? off : Math.min(off, clockOff + 0.002);   // lower envelope: the least-delayed packet
  snaps.push({ t, n, f });
  if (snaps.length > 12) snaps.shift();
}
const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();
function applySnaps() {
  if (!snaps.length || !bodies.length) return;
  const rt = performance.now() / 1000 - clockOff - DELAY;       // the simulation moment to show
  let a = snaps[0], b = snaps[snaps.length - 1];
  for (let i = snaps.length - 1; i > 0; i--) {
    if (snaps[i - 1].t <= rt) { a = snaps[i - 1]; b = snaps[i]; break; }
  }
  const u = b.t > a.t ? Math.min(1, Math.max(0, (rt - a.t) / (b.t - a.t))) : 1;
  const n = Math.min(a.n, b.n, bodies.length);
  for (let i = 0; i < n; i++) {
    const o = 4 + 7 * i, A = a.f, B = b.f;
    bodies[i].position.set(A[o] + (B[o] - A[o]) * u, A[o + 1] + (B[o + 1] - A[o + 1]) * u, A[o + 2] + (B[o + 2] - A[o + 2]) * u);
    _qa.set(A[o + 4], A[o + 5], A[o + 6], A[o + 3]); _qb.set(B[o + 4], B[o + 5], B[o + 6], B[o + 3]);
    bodies[i].quaternion.slerpQuaternions(_qa, _qb, u);
  }
}

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
    addSnap(f);
    buzz(f[3] > 0.5);
  };
}
connect();
const send = (m) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); };
$('restart').onclick = () => { send({ type: 'restart' }); $('restart').blur(); };

// ------------------------------------------------------------ HUD
const fmt = (s) => { s = Math.max(0, Math.ceil(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
let lastMsg = '', toastT = 0, lastCount = '', lastTouches = 0;
function onStatus(s) {
  if (s.type !== 'status') return;
  status = s;
  $('lvlK').textContent = `Level ${s.index + 1} of ${s.count}`;
  $('title').textContent = s.title.replace(/^Level \d+:\s*/, '');
  $('text').textContent = s.text;
  const left = s.state === 'play' ? s.limit - s.time : s.limit;
  $('timer').textContent = fmt(left);
  $('timer').classList.toggle('low', s.state === 'play' && left < 20);
  $('touches').textContent = s.touches;
  if (s.touches > lastTouches) { $('touches').classList.add('hit'); setTimeout(() => $('touches').classList.remove('hit'), 700); }
  lastTouches = s.touches;
  $('bar').style.width = `${Math.round(100 * s.progress)}%`;
  $('phoneChip').classList.toggle('ok', s.phone);
  $('phoneT').textContent = s.phone ? 'Phone connected' : 'Phone not connected';
  $('pair').classList.toggle('hidden', s.phone || s.state !== 'waiting' || s.finished);
  $('done').classList.toggle('hidden', !s.finished);
  const c = s.state === 'countdown' && s.countdown > 0 ? String(Math.ceil(s.countdown)) : '';
  if (c !== lastCount) {
    $('count').textContent = c; $('count').classList.remove('pop'); void $('count').offsetWidth;
    if (c) { $('count').classList.add('pop'); beep(c === '1' ? 880 : 660, 0.12); }
    else if (lastCount === '1') { $('count').textContent = 'GO'; $('count').classList.add('pop'); beep(1175, 0.25);
      setTimeout(() => { if (!lastCount) $('count').textContent = ''; }, 800); }
    lastCount = c;
  }
  if (s.message && s.message !== lastMsg) {
    $('toast').textContent = s.message; $('toast').classList.add('show');
    clearTimeout(toastT); toastT = setTimeout(() => $('toast').classList.remove('show'), 2200);
    if (/well done/i.test(s.message)) chime();
  }
  lastMsg = s.message;
}
fetch('/config.json').then(r => r.json()).then(c => {
  if (c.phone_url) { $('qr').src = '/qr.svg'; $('phoneUrl').innerHTML = `or open <a href="${c.phone_url}">${c.phone_url}</a>`; }
  else $('phoneUrl').textContent = 'No public address yet: start the tunnel for the phone.';
});

// ------------------------------------------------------------ sound and the buzz
let audio = null, buzzOsc = null, buzzGain = null, buzzing = false, shake = 0;
function ensureAudio() {
  if (audio) return;
  audio = new AudioContext();
  buzzOsc = audio.createOscillator(); buzzGain = audio.createGain();
  const filt = audio.createBiquadFilter(); filt.type = 'lowpass'; filt.frequency.value = 1400;
  buzzOsc.type = 'sawtooth'; buzzOsc.frequency.value = 118; buzzGain.gain.value = 0;
  buzzOsc.connect(filt).connect(buzzGain).connect(audio.destination); buzzOsc.start();
}
addEventListener('pointerdown', ensureAudio);
function beep(freq, dur) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = 'sine'; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, audio.currentTime); g.gain.exponentialRampToValueAtTime(0.2, audio.currentTime + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + dur);
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + dur + 0.02);
}
function chime() { [784, 988, 1319].forEach((f, i) => setTimeout(() => beep(f, 0.3), i * 110)); }
function buzz(on) {
  if (on === buzzing) return;
  buzzing = on;
  $('flash').style.opacity = on ? 1 : 0;
  if (on) shake = 1;
  if (buzzGain) buzzGain.gain.setTargetAtTime(on ? 0.09 : 0, audio.currentTime, 0.01);
}

// ------------------------------------------------------------ the view's heading (phone "forward")
let sentYaw = null;
const inv = new THREE.Matrix4();
function viewYaw() {                     // the camera's heading in MuJoCo's frame
  const d = new THREE.Vector3(); camera.getWorldDirection(d);
  d.applyMatrix4(inv.copy(mj.matrixWorld).invert().setPosition(0, 0, 0));
  return Math.atan2(d.y, d.x);
}
setInterval(() => {
  const y = viewYaw();
  if (sentYaw === null || Math.abs(y - sentYaw) > 0.01) { send({ type: 'view', yaw: y }); sentYaw = y; }
}, 200);

// ------------------------------------------------------------ frame loop, frame rate, auto quality
let frames = 0, fpsT = performance.now(), slow = 0, reportT = 0;
const gpu = (() => { try { const gl = renderer.getContext(); const e = gl.getExtension('WEBGL_debug_renderer_info');
  return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : ''; } catch (_) { return ''; } })();
renderer.setAnimationLoop((now) => {
  controls.update();
  // the touched wire glows red; a small camera shake on a new touch
  const target = buzzing ? 2.5 : 0;
  M.chrome.emissiveIntensity += (target - M.chrome.emissiveIntensity) * 0.35;
  let sx = 0, sy = 0;
  if (shake > 0.01) { sx = (Math.random() - 0.5) * 0.004 * shake; sy = (Math.random() - 0.5) * 0.004 * shake; shake *= 0.85; }
  camera.position.x += sx; camera.position.y += sy;
  applySnaps();
  if (useComposer) composer.render(); else renderer.render(scene, camera);
  camera.position.x -= sx; camera.position.y -= sy;
  frames++;
  if (now - fpsT > 2000) {
    const fps = frames * 1000 / (now - fpsT);
    frames = 0; fpsT = now;
    $('fps').textContent = `${Math.round(fps)} fps · ${quality}`;
    if (!forced && sceneId >= 0) {
      slow = fps < 45 ? slow + 1 : 0;
      if (slow >= 2 && quality !== 'low') { quality = quality === 'high' ? 'standard' : 'low'; slow = 0; applyQuality(); }
    }
    if (now - reportT > 10000) { send({ type: 'fps', fps: Math.round(fps), quality, gpu }); reportT = now; }
  }
});
