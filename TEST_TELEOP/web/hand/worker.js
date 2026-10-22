// The hand tracker in a background thread (a classic worker: MediaPipe's loader needs
// importScripts): it gets camera frames (ImageBitmaps) and answers with the hand's 21 points, so
// the tracking never waits for the 3D view to draw, and the other way round.
// With a pose model, MediaPipe Pose runs too (on the frames that ask for it): the elbows and
// wrists, for the forearm's direction.
// Messages in:  {type: 'init', mp, model, poseModel}, {type: 'frame', bitmap, t, pose}
// Messages out: {type: 'ready', delegate}, {type: 'result', t, lm, wl, pose: {lm, wl} | null},
//               {type: 'error', message}
let hands = null, poser = null, lastT = 0;

function stamp(t) {                     // (MediaPipe wants strictly increasing timestamps [ms])
  lastT = Math.max(lastT + 1, Math.round(t));
  return lastT;
}

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      const { HandLandmarker, PoseLandmarker, FilesetResolver } = await import(`${m.mp}/vision_bundle.mjs`);
      const vision = await FilesetResolver.forVisionTasks(`${m.mp}/wasm`);
      let err = null;
      for (const delegate of ['GPU', 'CPU']) {
        try {
          const opts = {
            baseOptions: { modelAssetPath: m.model, delegate }, runningMode: 'VIDEO', numHands: 1,
            // (presence and tracking a little lower than the defaults: a tilted hand stays tracked)
            minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4 };
          if (delegate === 'GPU' && typeof OffscreenCanvas !== 'undefined') opts.canvas = new OffscreenCanvas(1, 1);
          hands = await HandLandmarker.createFromOptions(vision, opts);
          // warm up (the first detection compiles the GPU programs: ~0.2 s, better now than on the first frame)
          const c = new OffscreenCanvas(320, 240);
          c.getContext('2d').fillRect(0, 0, 320, 240);
          const bmp = c.transferToImageBitmap();
          hands.detectForVideo(bmp, stamp(performance.now()));
          if (m.poseModel) {
            try {
              const po = { baseOptions: { modelAssetPath: m.poseModel, delegate }, runningMode: 'VIDEO', numPoses: 1 };
              if (delegate === 'GPU' && typeof OffscreenCanvas !== 'undefined') po.canvas = new OffscreenCanvas(1, 1);
              poser = await PoseLandmarker.createFromOptions(vision, po);
              poser.detectForVideo(bmp, stamp(performance.now()));
            } catch (e) { poser = null; }               // (no forearm then: the hand alone still works)
          }
          bmp.close();
          self.postMessage({ type: 'ready', delegate });
          return;
        } catch (e) { err = e; hands = null; }
      }
      self.postMessage({ type: 'error', message: String(err?.message || err) });
    } else if (m.type === 'frame') {
      if (!hands) { m.bitmap.close(); self.postMessage({ type: 'result', t: m.t, lm: null, wl: null }); return; }
      const ts = stamp(m.t);
      const r = hands.detectForVideo(m.bitmap, ts);
      let pose = null;
      if (m.pose && poser) {
        const p = poser.detectForVideo(m.bitmap, ts);
        if (p.landmarks?.[0]) pose = { lm: p.landmarks[0], wl: p.worldLandmarks[0] };
      }
      m.bitmap.close();
      self.postMessage({ type: 'result', t: m.t, lm: r.landmarks?.[0] || null, wl: r.worldLandmarks?.[0] || null, pose });
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e?.message || e) });
  }
};
