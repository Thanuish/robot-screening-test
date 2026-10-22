// What the hand-control pages show: the camera image (mirrored, like a mirror) with the hand's
// skeleton, coloured by what the hand does, and the short start checklist.
import { BONES } from './core.js';

export const COLORS = { open: '#3ee08f', fist: '#ffc24a', pinch: '#ff5f6d', none: '#9aa3b2' };
export const LABELS = { open: 'DRIVING', fist: 'PAUSED', pinch: 'GRIPPING', none: 'NO HAND' };

/** Draw the camera image and the hand (state from HandControl's onFrame) into a canvas. */
export function drawHand(canvas, video, state, { mirror = true, dim = 0.25 } = {}) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.save();
  if (mirror) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  if (video && video.videoWidth) {
    // cover the canvas (crop, keep the aspect)
    const vr = video.videoWidth / video.videoHeight, cr = w / h;
    let sw = video.videoWidth, sh = video.videoHeight, sx = 0, sy = 0;
    if (vr > cr) { sw = sh * cr; sx = (video.videoWidth - sw) / 2; } else { sh = sw / cr; sy = (video.videoHeight - sh) / 2; }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, w, h);
    ctx.fillStyle = `rgba(10,12,18,${dim})`;
    ctx.fillRect(0, 0, w, h);
    const lm = state?.lm;
    if (lm) {
      const map = (p) => [((p.x * video.videoWidth) - sx) / sw * w, ((p.y * video.videoHeight) - sy) / sh * h];
      const col = state.driving === false && state.gesture === 'open' ? '#7fb4ff' : COLORS[state.gesture] || COLORS.none;
      ctx.lineCap = 'round';
      ctx.lineWidth = Math.max(2, w / 160);
      ctx.strokeStyle = col;
      ctx.shadowColor = col; ctx.shadowBlur = 10;
      ctx.beginPath();
      for (const [a, b] of BONES) { const [x1, y1] = map(lm[a]), [x2, y2] = map(lm[b]); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); }
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#fff';
      for (const p of lm) { const [x, y] = map(p); ctx.beginPath(); ctx.arc(x, y, Math.max(2, w / 220), 0, 7); ctx.fill(); }
      if (state.axes) {                                    // the hand's 3D orientation: its three axes
        const [ox, oy] = map(state.axes.o);
        ['#ff6b6b', '#5ee08f', '#6fa8ff'].forEach((c, j) => {
          const [x, y] = map(state.axes.axes[j]);
          ctx.strokeStyle = c; ctx.lineWidth = Math.max(2, w / 200);
          ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(x, y); ctx.stroke();
        });
      }
      if (state.gesture === 'pinch') {                      // a ring at the pinch
        const [x1, y1] = map(lm[4]), [x2, y2] = map(lm[8]);
        ctx.strokeStyle = COLORS.pinch; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc((x1 + x2) / 2, (y1 + y2) / 2, w / 28, 0, 7); ctx.stroke();
      }
    }
  } else {
    ctx.fillStyle = '#0d1016'; ctx.fillRect(0, 0, w, h);
  }
  ctx.restore();
}

/** The start checklist: each step ticks off when done, in order; then onDone(). Steps:
 *  camera on, hand seen, a fist (that pauses), a pinch (that grips), open the hand (go). */
export class Checklist {
  constructor(el, onDone, { gripText } = {}) {
    this.el = el;
    this.onDone = onDone;
    this.steps = [
      ['camera', 'Camera on'],
      ['hand', 'Show your hand: palm toward the camera, the whole hand in view'],
      ['fist', 'Make a fist ✊: that <b>pauses</b> the robot'],
      ['pinch', gripText || 'Open it, then pinch 🤏 thumb and index finger: that <b>grips</b>'],
      ['open', 'Open your hand ✋: now it <b>drives</b> the robot'],
    ];
    this.i = 0;
    this.held = 0;
    el.innerHTML = this.steps.map(([k, t]) => `<li data-k="${k}"><span class="tick"></span><span>${t}</span></li>`).join('');
    this._render();
  }
  _render() {
    [...this.el.children].forEach((li, j) => {
      li.classList.toggle('done', j < this.i);
      li.classList.toggle('now', j === this.i);
    });
  }
  /** camera: is it on; state: HandControl's frame state */
  update(camera, state) {
    if (this.i >= this.steps.length) return;
    const k = this.steps[this.i][0];
    const g = state?.gesture;
    const ok = k === 'camera' ? camera
      : k === 'hand' ? !!state?.tracked
      : k === 'fist' ? g === 'fist'
      : k === 'pinch' ? g === 'pinch'
      : g === 'open' && !state?.gripping;
    this.held = ok ? this.held + 1 : 0;
    if (this.held >= (k === 'camera' ? 1 : 6)) {             // (held for a few frames: not a flicker)
      this.i++;
      this.held = 0;
      this._render();
      if (this.i >= this.steps.length) this.onDone?.();
    }
  }
  get done() { return this.i >= this.steps.length; }
}
