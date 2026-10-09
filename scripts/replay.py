"""Inspect / replay a recorded episode.

    python scripts/replay.py data/episode_0000.hdf5 --info
    python scripts/replay.py data/episode_0000.hdf5 --video out.mp4 --camera front
    python scripts/replay.py data/episode_0000.hdf5 --video out.mp4 --camera operator   # what the VR user saw
    mjpython scripts/replay.py data/episode_0000.hdf5          # interactive viewer (macOS needs mjpython)
    python scripts/replay.py data/episode_0000.hdf5 --mode actions   # re-simulate from joint commands

--mode states  (default) sets the recorded full MuJoCo state every frame (exact replay)
--mode actions starts from frame 0 and re-simulates by feeding the recorded joint-space
               actions, which is a quick check that the action labels reproduce the demo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vrteleop import cloth as C  # noqa: E402
from vrteleop import cloth_config as CC  # noqa: E402
from vrteleop import scene as S  # noqa: E402


def info(h):
    print(f"file: {h.filename}")
    for k, v in h.attrs.items():
        if k != "model_xml":
            print(f"  attr {k}: {v}")
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:32s} {str(obj.shape):20s} {obj.dtype}")
    h.visititems(visit)
    T = h["time"].shape[0]
    print(f"  duration: {h['time'][-1]:.2f}s, {T} frames")
    if "observations/cloth_verts" in h:
        faces = h["cloth/faces"][:]
        tz = S.TABLE_Z
        for name, i in (("first", 0), ("last", T - 1)):
            m = C.fold_metrics_from(h["cloth/rest_uv"][:], faces, h["observations/cloth_verts"][i], tz)
            print(f"  cloth {name} frame: coverage {m['coverage']:.2f}, height {m['height'] * 100:.1f} cm")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode")
    p.add_argument("--info", action="store_true")
    p.add_argument("--mode", choices=["states", "actions"], default="states")
    p.add_argument("--video", default=None, help="write an mp4 instead of opening the viewer")
    p.add_argument("--camera", default="front",
                   help="a scene camera, or 'operator': follows the recorded headset pose (first-person)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    a = p.parse_args()

    h = h5py.File(a.episode, "r")
    if a.info:
        info(h)
        return
    task = h.attrs.get("task", "blocks")          # episodes recorded before tasks existed: blocks
    if isinstance(task, bytes):
        task = task.decode()
    if task == "screen":
        import TEST_TELEOP  # noqa: F401  (the screening test's scene: TEST_TELEOP/)
    cloth = None
    if task == "jeans":
        if "cloth_config" in h.attrs:                 # the exact garment and settings recorded
            cloth = CC.ClothConfig.from_json(h.attrs["cloth_config"])
        else:                                         # older episodes: default garment, recorded spacing
            sp = h.attrs.get("cloth_spacing")
            cloth = CC.load(overrides={"spacing": float(sp)} if sp else None)
    engine = h.attrs.get("cloth_engine", "mujoco")
    engine = engine.decode() if isinstance(engine, bytes) else engine
    # a cloth simulated outside MuJoCo (the GPU engine) replays from its recorded vertices, drawn
    # through the scene's render-only mesh; the robots replay from the recorded states
    sc = S.build_scene(task, cloth, engine if task == "jeans" else "mujoco")
    m, d = sc.model, mujoco.MjData(sc.model)
    render_mesh = None
    if task == "jeans" and engine != "mujoco":
        render_mesh = C.ClothRenderMesh(m, h["cloth/faces"][:])
        cloth_verts = h["observations/cloth_verts"]
        if a.mode == "actions":
            sys.exit(f"--mode actions needs the MuJoCo cloth; this episode used the '{engine}' engine")
    grasp = {}
    if sc.cloth is not None:
        grasp = {s: C.PinchGrasp(m, sc.cloth, s, sc.grasp_eq[s], slip_force=cloth.slip_force,
                                 release_gap=cloth.release_gap) for s in S.SIDES}
    if h["observations/full_qpos"].shape[1] != m.nq:
        sys.exit(f"episode state size {h['observations/full_qpos'].shape[1]} != model nq {m.nq}: "
                 f"the scene changed since this episode was recorded")
    qpos = h["observations/full_qpos"][:]
    qvel = h["observations/full_qvel"][:]
    act = h["action"][:]
    fps = float(h.attrs["fps"])
    T = len(qpos)

    d.qpos[:], d.qvel[:] = qpos[0], qvel[0]
    mujoco.mj_forward(m, d)
    substeps = max(1, int(round(1.0 / fps / m.opt.timestep)))
    ctrl_every = max(1, int(round(0.01 / m.opt.timestep)))     # pinch logic runs at 100 Hz like the server

    def advance(i):
        if a.mode == "states":
            d.qpos[:], d.qvel[:] = qpos[i], qvel[i]
            mujoco.mj_forward(m, d)
        else:
            for k, s in enumerate(S.SIDES):
                d.ctrl[sc.arm_act[s]] = act[i, 7 * k:7 * k + 6]
                d.ctrl[sc.grip_act[s]] = 255.0 * act[i, 7 * k + 6]
            for n in range(substeps):
                if grasp and n % ctrl_every == 0:
                    for k, s in enumerate(S.SIDES):
                        grasp[s].update(m, d, float(act[i, 7 * k + 6]), ctrl_every * m.opt.timestep)
                mujoco.mj_step(m, d)

    camera = a.camera
    operator = None
    if a.camera == "operator":
        # Re-use the head_cam slot and move it along the recorded headset pose. The WebXR
        # camera, like a MuJoCo camera, looks along its -z axis with +y up.
        camera = "head_cam"
        cid = m.camera(camera).id
        m.cam_fovy[cid] = 85.0                    # close to the Quest's vertical field of view
        operator = (cid, h["teleop/head_pose"][:].astype(float), m.cam_pos[cid].copy(), m.cam_quat[cid].copy())

    def place_operator(i, state={}):
        cid, poses, pos0, quat0 = operator
        p = poses[i]
        if not np.all(np.isfinite(p)):            # headset not tracking in this frame
            p = state.get("last", np.r_[pos0, quat0])
        q = p[3:] / np.linalg.norm(p[3:])
        if "pos" in state:                        # light smoothing of head jitter
            if np.dot(q, state["quat"]) < 0:
                q = -q
            state["pos"] = 0.6 * state["pos"] + 0.4 * p[:3]
            state["quat"] = 0.6 * state["quat"] + 0.4 * q
            state["quat"] /= np.linalg.norm(state["quat"])
        else:
            state["pos"], state["quat"] = p[:3].copy(), q.copy()
        state["last"] = p
        m.cam_pos[cid], m.cam_quat[cid] = state["pos"], state["quat"]

    if a.video:
        import imageio
        r = mujoco.Renderer(m, a.height, a.width)
        with imageio.get_writer(a.video, fps=fps, quality=8) as w:
            for i in range(T):
                advance(i)
                if operator:
                    place_operator(i)
                if render_mesh is not None:
                    render_mesh.update(m, d, cloth_verts[i], r)
                r.update_scene(d, camera=camera)
                w.append_data(r.render())
        print(f"wrote {a.video} ({T} frames)")
        if a.mode == "actions":
            err = np.linalg.norm(d.qpos[:] - qpos[-1])
            print(f"final full-qpos deviation vs recording: {err:.4f}")
        return

    from mujoco import viewer as mjviewer
    if render_mesh is not None:
        print("note: the interactive viewer shows the GPU-simulated jeans in their start pose; "
              "use --video to see them move")
    with mjviewer.launch_passive(m, d) as v:
        i = 0
        while v.is_running():
            t0 = time.time()
            advance(i)
            v.sync()
            i = (i + 1) % T
            if i == 0:
                d.qpos[:], d.qvel[:] = qpos[0], qvel[0]
                mujoco.mj_forward(m, d)
            time.sleep(max(0, 1 / fps - (time.time() - t0)))


if __name__ == "__main__":
    main()
