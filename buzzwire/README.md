# Robot games: the teleoperation assessment

A screening test for robot operators. The candidate controls a robot arm (a UR5e with a Robotiq
2F-85 gripper, real physics on the server) from their **phone** and watches it on their
**computer**, seen from behind the robot. Three games: **Cube Tasks** (shapes into outlines,
objects into bins), **Tower** (stack blocks as high as you can) and **Ring on pegs** (tilt each
ring onto a leaning peg, round after round). The older Buzz Wire game is no longer in the menu or
the test (its code is still in `game.py`).

## Sending the test to a candidate

**The admin page**, `http://localhost:8000/admin` (it opens only on the server PC, e.g. through the
SSH forward; a login comes with hosting): create a candidate (name, e-mail, **controls**), copy
their link, change their controls, see their status, stage, scores, practice rounds, phone and the
controls they actually used.

**Details** (per candidate) shows everything recorded, for every attempt:
- the time of each stage, the phone and controls used, when each tutorial step was learned, and
  the practice scores;
- a card per assessment task with its numbers and a timeline in words:
  - **Tower:** each block grabbed, placed (with the tower's height), taken off, or fallen (with
    the height before), and each new best height that stood;
  - **Cube Tasks:** each grasp, drop and step done;
  - **Ring on pegs:** each grasp, each ring on or off its peg, each drop and each round.

While a test is running, the details stay current. The same from the command line:

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
3. **Tutorial**: the controls one at a time, each checked before the next: first the mouse (drag
   to look around, scroll to zoom), then the phone (move, up / down, turn, above the cube, lower,
   grab, put it down).
4. **Practice**: a cube into a square, a small tower, one round of rings, with hints. A hint on
   the screen reminds them of the mouse (drag, scroll, double-click to reset).
5. **Ready?**: "Start the assessment" or "Practise again" (at most 3 rounds; the number is recorded).
6. **Assessment**: shapes, sorting, a cube tower, ring on pegs (5 minutes, as many rounds as they
   can). A card before each task says what to do, with a tip about the mouse. While playing, the
   screen shows only the robot and a thin timer. No scores are shown.
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
phone is always into the screen, also after the view is turned.

**The arm is solid.** Shading and occlusion are strong depth cues. Its big links (base,
shoulder, upper arm, forearm) fade to `arm_opacity` (0.25) only while they hide something. Every
frame, rays from the camera to the gripper, to the spot below it and to the table 8 cm around
that spot are tested against each link's box. A link becomes solid again, smoothly, once it no
longer hides anything.

**Side view** (`[side_view]`): a small second view in the top-right corner, under the top bar.
It shows the gripper from the robot's left at table height and follows it smoothly. Whatever is
between that camera and the gripper (the arm, a post) is cut away. A centimetre scale on its
left edge marks the finger pads' height ("14.1 cm", green when low enough to grab). It's exact at
the gripper's distance, because the side camera looks level from a fixed distance.

**Depth cues** make height and distance readable on a flat screen:
- **A ring, a dot and a line** from the point between the finger pads down to the table, green
  when they are low enough to grab (`[depth_cues]`, as in the screening test).
- **Shadows** fall straight down from a light above the table, on every quality setting. They're
  sharp-edged and dark, with little flat fill light.
- **A faint 10 cm grid** on the table: its lines shrink with distance.

**Graphics:** physically based materials and HDRI lighting. Presets: **lite** (the default:
small shadows, no effects), **low**, **standard**, **high** (`?q=...`). It steps down by itself
below 45 fps. The robot is shown slightly in the past, blended between the server's ~50 updates
a second, so it moves smoothly at any frame rate. That delay adapts to the connection: it covers
how late 95 % of the updates arrive (30-150 ms). The corner shows frame rate, quality and this
delay. Assets:
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
5. **The motors** follow at the UR5e's own stiffness (`servo_stiffness` 1). They also get the
   commanded velocity, as a real robot's servo controller does (`servo_feedforward`). Without
   it, a joint motor trails its command by kv/kp (0.1 s).

Measured with phone-like input (30 poses a second, 1.5 mm noise): the gripper is about 0.12 s
behind the hand at both 2 cm/s and 20 cm/s, and wobbles 0.8 mm when the hand is still (about a
pixel). It's within 0.15 degrees of a 90-degree tip half a second after it. Without the
feed-forward (the screening test's chain): 0.22 s, 0.6 mm, 0.6 degrees.

**No stalls.** Each open page has its own sender that always sends the newest frame. A slow
page (a background tab, a bad network) only misses frames, and can't hold up the simulation or
anyone else. The loop reads the network every tick. Phones and hidden pages get no robot poses
(phones don't use them). A level change takes about 50-80 ms, and the games are built once when
the server starts. The log notes slow ticks (`[loop]`), the longest gap between the phone's
messages, and each page's frame rate and display delay.

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

## Ring on pegs

Three rings (red, blue, green) stand on their edge in small wooden holders in a row in front of
the robot. Straight ahead of each, further out, is a peg of the same colour on a grey post. The
pegs lean 45, 60 and 80 degrees, so each ring needs a different tilt of the hand
(`rotation = "full"`).

**How to play:** take a ring by the top of its tube, lift it, tilt it so its hole lines up with
the peg, slide it on from the tip, and let go. When all three hang on their pegs and nothing is
held, the round counts. The rings jump back into their holders and the next round starts, until
the time is up. Screens show "Ring on pegs · round N", the rings hung this round as three ticks,
and "Round N!" on each new round.

**Rules**, checked 20 times a second (`[rings]`):
- **On its peg:** the ring's centre lies along the peg's axis, between its base and its tip, and
  close enough to the axis for the peg to be through the hole.
- **Done:** on its peg, not held, and nearly still. It counts while it hangs there; a ring that
  falls off un-counts.
- **Held:** lifted 2 cm above where it would rest, and within 8 cm of the gripper. Where a ring
  rests depends on how it lies: on its edge, or flat.
- **Grasps and drops:** a grasp is a GRAB that lifted a ring (at most one per GRAB). A drop is a
  ring that left the hand while GRAB was still closed, and not onto its peg.

**Score (0-100):** 80 × rings hung / (3 × `target_rounds`, i.e. 9 rings) + 20 × grasp efficiency
(grasps / GRAB presses, minus 1/3 per drop). Results: rounds, rings hung, each round's time,
grasps, drops, and the timeline.

**Its own camera** (`view_eye`, `view_look_at`): the rings stand to the robot's left, so the view
is from a little left of the robot. All three columns stand in a row, none behind the arm.

The config checks that each ring and peg is on the table and 0.35-0.85 m from the arm's base,
that a peg tilted more than 30 degrees has a post, and that the ring's hole is clearly bigger than
the peg.

## Buzz Wire (not in the menu or the test any more): how a level goes

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
python -m buzzwire.bot --game rings --replay    # Ring on pegs: 2 rounds (~41 s each), score 73.3; practice 1 round
python -m buzzwire.stress                       # every game shoved around like an impatient person:
                                                # no control tick may take over 50 ms
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
  rings.py       Ring on pegs: rings, holders, leaning pegs, rounds
  scene.py       the MuJoCo scenes: robot and room; the buzz-wire board, wire and wand
  spline.py      the wire's curve (arc length, closest point)
  ik.py          inverse kinematics on Pinocchio (joint limits and speeds, table keep-out on the real shapes)
  stress.py      the stress check: fast swings, shoving into the table; times every tick
  jitter.py      the optional jitter buffer for the phone's messages
  bot.py         plays every game through the controller messages (checks)
  server.py      the server: the game in real time, the web pages, the candidate's test
  web/           index.html + app.js: the screen; phone.html + phone.js: the Android controller
  web/apple/     index.html + iphone.js: the iPhone controller
  web/admin.html the admin page (candidates, controls, results)
```
