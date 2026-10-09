# Robot screening test

An online test that screens candidates for robot teleoperation **before** they come onsite.
A candidate controls a simulated robot arm (Universal Robots UR5e with a Robotiq 2F-85 gripper)
from home, using their **Android phone as a motion controller**, and watches it on their
**computer screen**. Three short manipulation tasks, after a practice round, are scored on
time, steps completed and efficiency. Nothing to install for candidates; up to about 15 at a time,
run from one Windows PC.

The full guide (tasks, controls, scoring, running a round, results): **[TEST_TELEOP/README.md](TEST_TELEOP/README.md)**.

## Quick start (Windows)

1. Install [Python 3.12](https://www.python.org/downloads/) (tick **"Add python.exe to PATH"**)
   and [Git](https://git-scm.com/downloads).
2. Get the code and set it up:

   ```bash
   git clone https://github.com/Thanuish/robot-screening-test.git
   ```

   Then double-click **`setup_windows.bat`** in the new folder (creates `.venv` and installs
   the packages; a few minutes).
3. Install Cloudflare's tunnel program once (gives the PC a public `https://` address for
   candidates, without opening your router):

   ```bash
   winget install Cloudflare.cloudflared
   ```

4. Double-click **`TEST_TELEOP\start_screening.bat`**. It opens the results page
   (`http://localhost:8000/admin`), which shows the public address. Create invite codes there and
   send each candidate a link.

Try it yourself first: create a code, open the link on your computer and scan the QR code with an
Android phone (Chrome). Without a phone, `python -m TEST_TELEOP.screen_bot` plays the whole test
automatically and prints the scores.

## What's in here

```
TEST_TELEOP/     the screening test: lobby, tasks and scoring, phone and monitor pages (see its README)
vrteleop/        the robot simulator it runs on (MuJoCo scene, arm control, recording, 3D viewer)
assets/          robot models from MuJoCo Menagerie: UR5e and Robotiq 2F-85 (see their LICENSE files)
config/          garment settings used by the simulator's scene code
scripts/         replay.py: replay a recorded attempt (data/screening/.../episode_XXXX.hdf5)
```

Recorded attempts and results go to `data/screening/` (not part of the repository).

This repository is the screening test taken out of the larger VR teleoperation project
([Thanuish/Sim](https://github.com/Thanuish/Sim)), with only the parts the test needs.

Credits: [MuJoCo](https://mujoco.org) (physics), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(robot models), [three.js](https://threejs.org) (3D in the browser, included in `vrteleop/web/vendor`).
