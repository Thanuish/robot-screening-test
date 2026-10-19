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

**View** (as in the screening test, `[view]`): from the robot's "head", just behind its base,
looking down at the work area. Drag to turn the view; double-click to reset it. "Forward" on the
phone is always into the screen, also after the view is turned. The shoulder, upper arm and
forearm are drawn see-through (`arm_opacity` 0.25) so they don't hide the work area.

**Side view** (`[side_view]`): a small second view in the top-right corner, under the top bar.
It shows the gripper from the robot's left at table height and follows it smoothly. Whatever is
between that camera and the gripper (the arm, a post) is cut away.

**Depth cues** make height and distance readable on a flat screen:
- **A ring, a dot and a line** from the point between the finger pads down to the table, green
  when they are low enough to grab (`[depth_cues]`, as in the screening test).
- **Shadows** fall straight down from a light above the table, on every quality setting.
- **A faint 10 cm grid** on the table: its lines shrink with distance.

**Graphics:** physically based materials and HDRI lighting. Presets: **lite** (the default:
small shadows, no effects), **low**, **standard**, **high** (`?q=...`). It steps down by itself
below 45 fps. The
robot's positions are interpolated, so motion is smooth at any frame rate. Assets:
`web/assets/CREDITS.md` (CC0), three.js r170 in `web/vendor/` (MIT).

**From the phone to the robot**, the same chain and numbers as the screening test
(`vrteleop/`), so it feels as predictable. Each control tick (100 a second):
1. **Clutch.** While MOVE is held, the gripper's target moves and turns like the phone, relative
   to where both were when MOVE was pressed.
2. **Smoothing.** The target moves a fixed share of the way toward the phone each tick: a
   low-pass with a 0.12 s time constant (`[controls] filter = "simple"`, `smoothing_s`). The
   delay is the same at every hand speed. The adaptive One Euro filter is still there to try
   (`filter = "one_euro"`).
3. **Speed caps** after the smoothing: at most 0.4 m/s and 90 degrees a second (`[robot]`).
4. **Inverse kinematics** (`ik.py`): rotation counts as much as position. Three iterations per
   tick, sharing that tick's joint-speed budget. The forearm and wrist keep 2 cm above the
   table, every part of the gripper 3 mm (`arm_clearance`, `finger_clearance`). These distances
   are measured on the real collision shapes, so the fingers can reach down to grab.
5. **The motors** follow at the UR5e's own stiffness (`servo_stiffness` 1).

Measured with phone-like input (30 poses a second, 1.5 mm noise): the gripper is about 0.22 s
behind the hand at both 2 cm/s and 20 cm/s, wobbles 0.6 mm when the hand is still, and is within
0.6 degrees of a 90-degree tip half a second after it.

**Losing the phone:**
- **No messages.** No controller message for 0.35 s (`input_timeout_s`) counts as MOVE
  released, and the arm holds still. When messages come back the clutch takes a new anchor, so
  the arm doesn't jump to wherever the phone is now.
- **Tracking loss.** If the phone's camera tracking drops out (a very fast move, a blank view),
  MOVE is released as well. If the tracking relocates itself (its position jumps), the clutch
  re-anchors too.
- **Slow networks.** The phone skips messages while the network is behind, instead of piling up
  delay.
- **Jitter buffer.** An optional buffer evens out messages that arrive in bursts
  (`jitter_buffer_s`, `jitter_buffer_max_s`; `jitter.py`). It's off by default, as in the
  screening test.

**Vibration:** one short pulse (25 ms) on GRAB / RELEASE and when a wire touch starts, nothing
else. Vibration shakes the sensors the phone's tracking uses. Everything else is shown on the
phone's screen, which has no animations while tracking.

## Buzz Wire: how a level goes

1. The robot already holds the wand (fixed in its hand: it can't slip out); its ring is around
   the wire at the start (the red sleeve).
2. Steer with the phone: slide the ring along the wire without touching it.
   A touch is counted and shown: a red flash, a sound, one short pulse on the phone.
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
config/buzzwire.toml   robot, controls, physics, view, side view, depth cues; Buzz Wire (board, wire, wand, scoring, levels);
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
  ik.py          inverse kinematics on Pinocchio (joint limits and speeds, table keep-out on the real shapes)
  jitter.py      the optional jitter buffer for the phone's messages
  bot.py         plays every game through the controller messages (checks)
  server.py      the server: the game in real time, the web pages, the candidate's test
  web/           index.html + app.js: the screen; phone.html + phone.js: the Android controller
  web/apple/     index.html + iphone.js: the iPhone controller
  web/admin.html the admin page (candidates, controls, results)
```
