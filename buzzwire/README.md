# Robot games: the teleoperation assessment

A screening test for robot operators. The candidate controls a robot arm (a UR5e with a Robotiq
2F-85 gripper, real physics on the server) from their **phone** and watches it on their
**computer**, seen from behind the robot. Three games: **Buzz Wire** (slide a ring along a wire
without touching it), **Cube Tasks** (shapes into outlines, objects into bins) and **Tower** (stack
blocks as high as you can).

## Sending the test to a candidate

```
python -m buzzwire.candidates new "Jane Doe" jane@example.com      # prints her personal link
python -m buzzwire.candidates list
```

Put the link in the invitation e-mail. The candidate opens it **on their computer**:

1. **Welcome**: name, consent.
2. **Connect the phone**: one QR code; the phone's camera opens it, and the page picks the iPhone
   or Android controller by itself (`/p/<code>`). Controls: two touch joysticks (the same for
   everyone; left = move, right = up / down and turn) and GRAB / RELEASE.
3. **Tutorial**: the controls one at a time, each checked before the next (move, up / down, turn,
   above the cube, lower, grab, put it down).
4. **Practice**: a short buzz wire and a small tower, with hints.
5. **Ready?**: "Start the assessment" or "Practise again" (at most 3 rounds; the number is recorded).
6. **Assessment**: shapes, sorting, buzz-wire curves, a cube tower. A card before each task says
   what to do; while playing, the screen shows only the robot and a thin timer (a red edge glow on
   a wire touch). No scores are shown.
7. **Thank you**.

Results: `data/buzzwire/candidates/<code>/<date>/results.json` (each stage's time, the practice
rounds, the phone, each task's metrics and score, the overall score) and an input log per game
(an exact replay). The flow is set in `config/buzzwire.toml` `[flow]` (which tasks, how many
practice rounds, how long the task cards show). One candidate at a time for now; many at once,
a permanent address and the admin page come with the hosting step.

## Running it

```
python -m buzzwire.server               # http://localhost:8000: a menu of the games (for you)
python -m TEST_TELEOP.lobby tunnel      # the public https address the phones and candidates need
```

| Link | What |
|---|---|
| `/c/<code>` | a candidate's test (computer) |
| `/p/<code>` | the candidate's phone: sends it to `/phone` (Android) or `/iphone` (iPhone) |
| `/` | the game menu (development) |
| `/phone` | Android controller: touch joysticks, or motion control (the camera tracks the phone) |
| `/iphone` | iPhone controller (`web/apple/`): touch joysticks, or tilt steering (the gyroscope) |

**View:** from behind and above the robot (`[view]`); the shoulder, upper arm and forearm are drawn
see-through (`arm_opacity`) so they don't hide the work area, and a ring on the table below the
gripper shows where it would come down (green: low enough to grab).

**Graphics:** physically based materials and HDRI lighting; presets **lite** (the default: no
shadows), **low**, **standard**, **high** (`?q=...`); it steps down by itself below 45 fps. The
robot's positions are interpolated, so motion is smooth at any frame rate. Assets:
`web/assets/CREDITS.md` (CC0), three.js r170 in `web/vendor/` (MIT).

**Responsiveness:** the robot follows about 90 ms behind the controls in the simulation (plus the
network): stiff joint motors, light smoothing, top speed 0.6 m/s (`[robot]`, `[controls]`).

## Buzz Wire: how a level goes

1. The robot already holds the wand (fixed in its hand: it can't slip out); its ring is around
   the wire at the start (the red sleeve).
2. Steer with the phone: slide the ring along the wire without touching it.
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
                       [cubes] (which tasks of config/screening.toml); [tower]; [flow] (the candidate's test)
buzzwire/
  config.py      reads and checks the configuration
  candidates.py  candidates and their personal links (data/buzzwire/candidates.json)
  session.py     a candidate's test: welcome, tutorial, practice, ready, assessment, thank you
  tutorial.py    the interactive tutorial (steps checked one by one)
  base.py        what every game shares: controller input, teleop, IK, physics, clock, results, replay
  game.py        Buzz Wire: touches, the guide at the wire, progress
  cubes.py       Cube Tasks (the screening test's pick / shapes / sort tasks)
  tower.py       Tower: blocks one at a time, height, falls
  scene.py       the MuJoCo scenes: robot and room; the buzz-wire board, wire and wand
  spline.py      the wire's curve (arc length, closest point)
  ik.py          inverse kinematics on Pinocchio (joint limits and speeds, table keep-out)
  bot.py         plays every game through the controller messages (checks)
  server.py      the server: the game in real time, the web pages, the candidate's test
  web/           index.html + app.js: the screen; phone.html + phone.js: the Android controller
  web/apple/     index.html + iphone.js: the iPhone controller
```
