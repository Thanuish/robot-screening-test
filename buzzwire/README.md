# Robot games: the teleoperation assessment

A screening test for robot operators. The candidate controls a robot arm (a UR5e with a Robotiq
2F-85 gripper, real physics on the server) from their **phone** and watches it on their
**computer**, seen from behind the robot. Three games: **Buzz Wire** (slide a ring along a wire
without touching it), **Cube Tasks** (shapes into outlines, objects into bins) and **Tower** (stack
blocks as high as you can).

## Sending the test to a candidate

**The admin page**, `http://localhost:8000/admin` (it opens only on the server PC, e.g. through the
SSH forward; a login comes with hosting): create a candidate (name, e-mail, **controls**), copy
their link, change their controls, see their status, stage, scores, practice rounds, phone and the
controls they actually used. The same from the command line:

```
python -m buzzwire.candidates new "Jane Doe" jane@example.com --controls touch   # prints her link
python -m buzzwire.candidates controls <code> motion
python -m buzzwire.candidates list
```

**Controls per candidate** (the default: `[flow] default_controls`): **motion** (move the phone:
Android phones are tracked by their camera, with a short warm-up; iPhones steer by tilting),
**touch** (two joysticks, the same on every phone) or **choice** (the candidate picks). An
Android phone that can't track motion falls back to touch; what was used is recorded.

Put the link in the invitation e-mail. The candidate opens it **on their computer**:

1. **Welcome**: name, consent.
2. **Connect the phone**: one QR code; the phone's camera opens it, and the page picks the iPhone
   or Android controller by itself (`/p/<code>`), with the controls set for this candidate.
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

**View:** from just behind the robot, over its right shoulder, at head height, close and wide
(`[view]`). Drag to turn the view; double-click to reset it. The shoulder, upper arm and forearm
are drawn see-through (`arm_opacity`) so they don't hide the work area.

**Depth cues** make height and distance readable on a flat screen:
- **Shadows** fall straight down from a light above the table, on every quality setting.
- **A faint 5 cm grid** on the table: its lines shrink with distance.
- **A ring and a line** from the fingertips down to the table: green when low enough to grab.

**Graphics:** physically based materials and HDRI lighting. Presets: **lite** (the default:
small shadows, no effects), **low**, **standard**, **high** (`?q=...`). It steps down by itself
below 45 fps. The
robot's positions are interpolated, so motion is smooth at any frame rate. Assets:
`web/assets/CREDITS.md` (CC0), three.js r170 in `web/vendor/` (MIT).

**Smooth motion** (`[controls]`):
- **Even playback.** Through the internet the phone's messages arrive unevenly (bursts, then
  gaps). The server keeps them with the phone's timestamps and plays them back evenly, a little
  later. The delay adapts to the connection: about 30 ms on a good one, at most 150 ms
  (`jitter_buffer_s`, `jitter_buffer_max_s`; `buzzwire/jitter.py`).
- **Adaptive smoothing.** A One Euro filter is strong while the hand is still (no shake) and
  weak while it moves fast (little lag).
- **Speed limits** on the robot: top speed 0.6 m/s (`[robot]`).
- **Tracking loss.** If the phone's camera tracking drops out (a very fast move, a blank view),
  the robot glides to a stop and waits. When tracking returns it carries on from where it is,
  without a jump.
- **Slow networks.** The phone skips messages while the network is behind, instead of piling up
  delay.

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
  candidates.py  candidates, their personal links and controls (data/buzzwire/candidates.json)
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
  web/admin.html the admin page (candidates, controls, results)
```
