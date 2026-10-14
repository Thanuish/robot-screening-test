# Robot games: Buzz Wire, Cube Tasks, Tower

Teleoperation games for screening robot operators, all with the same realistic 3D lab, the same
robot (a UR5e with a Robotiq 2F-85 gripper, real physics) and the same controller: an Android
phone held like a VR controller. A menu on the computer screen chooses the game:

| Game | What you do | Score per level |
|---|---|---|
| **Buzz Wire** | slide a ring along a curved wire without touching it (3 levels) | progress, touches, touch time, time, smoothness |
| **Cube Tasks** | the screening test's tasks: a cube into a square, shapes into outlines, objects into bins | steps done, time left, grasps that worked (as in the screening test) |
| **Tower** | blocks appear one at a time: stack them as high as you can (cubes; then bricks and planks) | best height, falls, grasps that worked |

**Status:** milestones 2-4 done (physics, robot, phone, graphics); game menus in a first version.
Level editor and hosting come next.

## Try it

```
python -m buzzwire.server            # then open http://localhost:8000 and choose a game
python -m buzzwire.server --game tower   # (start straight in one game)
```

The screen shows two QR codes, one per kind of phone (they need the public tunnel:
`python -m TEST_TELEOP.lobby tunnel`):

| Link | Phone | Page | Controls |
|---|---|---|---|
| `/phone` | Android (Chrome) | `web/phone.html`, `phone.js` | **touch joysticks** (default), or motion control (the phone's camera tracks it, WebXR AR) |
| `/iphone` | iPhone / iPad (Safari) | `web/apple/index.html`, `iphone.js` | **touch joysticks** (default), or **tilt steering** (hold the left pad and tilt the phone; the gyroscope, Safari asks to allow motion) |

Joysticks: the **left stick** moves the gripper (forward = into the screen), the **right stick**
moves it up / down and turns it; **TILT** (levels with full rotation) makes the right stick tilt
and twist. Full deflection = 12 cm/s, gentle pushes move slowly. **GRAB / RELEASE** close and
open the gripper. Feedback: Android vibrates, the iPhone beeps (Safari can't vibrate); both flash
red on a touch of the wire. If motion tracking is lost (Android), both screens say so.

**Graphics:** physically based materials lit by an HDRI studio (reflections), a soft key light,
MSAA. Presets: **standard** (default, light enough for laptop graphics), **low** (chosen by
itself below 45 fps), **high** (`?q=high`: adds ambient occlusion and bloom). The robot's
positions (~50 a second) are interpolated, so motion is smooth at any frame rate. Each screen's
frame rate goes to the server's log. Assets: `buzzwire/web/assets/CREDITS.md` (CC0), three.js r170
in `buzzwire/web/vendor/` (MIT).

**Responsiveness:** the robot follows the phone about 90 ms behind (simulation alone, plus the
network): stiffer joint motors (`servo_stiffness = 3`), light smoothing (0.03 s), top speed
0.6 m/s. All in `config/buzzwire.toml` `[robot]` and `[controls]`.

## How a level goes

1. The robot already holds the wand (fixed in its hand: it can't slip out); its ring is around
   the wire at the start (the red sleeve).
2. Hold the pad on the phone and move: slide the ring along the wire without touching it.
   Touching buzzes (red flash, sound, the phone vibrates) and is counted.
3. When the ring reaches the red sleeve at the far end, the level is done.

**The guide** (a "virtual fixture", common in teleoperation): the arm's target is limited so the
ring can touch the wire (the touch counts) but never press into it, and the ring stays within 30
degrees of square to the wire. Pushing harder just keeps it touching; it can't be torn off.

Level 1 is straight with a big ring that keeps its angle; level 2 bends left and right (turn the
phone); level 3 also goes up and down (tilt the phone). Per level in `config/buzzwire.toml`: the
ring's size (`loop_radius`), `rotation` (`none`, `yaw`, `full`), `start_held` (false: pick the
wand up from its fork first) and `finish` (`hang`: set it on the end hook and let go).

**Score per level (0-100):** progress along the wire (40, full when completed), no touches (25),
little touch time (15), time left (10, if completed), smoothness (10, from the jerk of the
motion). Weights and thresholds in `config/buzzwire.toml` `[scoring]`.

## Checks

```
python -m buzzwire.bot --replay                 # Buzz Wire: every level, 0 touches; the replay scores the same
python -m buzzwire.bot --clumsy                 # wobbles past the ring's clearance: touches are counted
python -m buzzwire.bot --game cubes --replay    # Cube Tasks: every task done
python -m buzzwire.bot --game tower --replay    # Tower: stacks every block
```

## Files

```
config/buzzwire.toml   robot, controls, physics, view; Buzz Wire (board, wire, wand, scoring, levels);
                       [cubes] (which tasks of config/screening.toml); [tower] (platform, blocks, levels)
buzzwire/
  config.py    reads and checks the configuration
  spline.py    the wire's curve through the control points (arc length, closest point)
  scene.py     the MuJoCo model: robot, board, wire (capsule chain), forks, wand
  ik.py        inverse kinematics on Pinocchio (joint limits and speeds, table keep-out)
  base.py      what every game shares: controller input, teleop, IK, physics, clock, results, replay
  game.py      Buzz Wire: touches, the guide at the wire, progress
  cubes.py     Cube Tasks (the screening test's pick / shapes / sort tasks)
  tower.py     Tower: blocks one at a time, height, falls
  bot.py       plays every game through the controller messages (Buzz Wire: clean and clumsy)
  server.py    development server: the game in real time, the web pages
  web/         index.html + app.js: the game screen (three.js); phone.html + phone.js: the Android controller
  web/apple/   index.html + iphone.js: the iPhone controller
```
