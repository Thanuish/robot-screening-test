# TEST_TELEOP: remote screening test with a phone

A short online test to screen candidates **before** they come onsite: does someone understand
how their hand motion maps to a robot? They control one robot arm from home with their **phone**
as a motion controller and watch it on their **computer screen**. No VR headset, nothing to
install. Up to about 15 people at a time, run from your laptop.

```
 Candidate's computer                 Your laptop                            Candidate's phone
 ┌────────────────────┐   https   ┌──────────────────────────┐   https   ┌───────────────────┐
 │ the robot (3D),    │ ◄───────► │ lobby: invite codes,     │ ◄───────► │ motion tracking   │
 │ instructions,      │  tunnel   │ one simulator per        │  tunnel   │ (Chrome, Android) │
 │ timer, QR code     │           │ candidate, results page  │           │ MOVE / GRAB       │
 └────────────────────┘           └──────────────────────────┘           └───────────────────┘
```

**What the candidate does** (about 20 minutes):

| Phase | What | Time limit | Scored |
|---|---|---|---|
| Connect | scan the QR code on the screen with the phone, tap START CONTROLLER | | |
| Practice 1 | touch three glowing targets with the gripper | 2:30 | no (but reach efficiency is shown to you) |
| Practice 2 | put a yellow cube into a green square | 2:30 | no |
| Task 1: shapes | put a big square, a small square and a bar into their outlines (the bar must be turned 90°) | 4:00 | yes |
| Task 2: sorting | red ball, blue cube, green cylinder into the bin of their colour | 3:00 | yes |
| Task 3: tilt and pour | take a book out of its holder and lay it flat on a mat (tilt 90°); pour 5 balls from a cup into a bowl (tip the cup) and set the cup down upright | 5:00 | yes |

Practice and tasks 1–2 use the phone's position and its turn left / right; the gripper keeps
pointing down. **Task 3 uses all six degrees of freedom**: tilting and twisting the phone also
tilts and twists the gripper (the phone shows "TILT ON"). It's closest to real manipulation
such as handling clothes: lifting things off a surface, turning them over, tipping and pouring.
Which phases use full rotation is the `rotation` setting of each phase in
`config/screening.toml`.

**Physics:** the same as the VR blocks task: rigid MuJoCo bodies, light blocks (80 g), the UR5e's
own joint motors (`servo_stiffness` in the configuration). The book weighs 200 g; the mug and
the bowl are hollow (rings of wall pieces), so the balls really pour, and the mug has torsional
friction so it doesn't spin in the fingers when tipped. Only quick flicks are capped: the gripper
follows the phone at up to 1 m/s and 180°/s, about a real UR5e's top speed (`max_speed`,
`max_turn_deg`). The arm collides with the table like a real one: to lay the book
flat, turn it and tilt the gripper sideways; tilted straight forward, the wrist hits the table
before the book is low enough.

The candidate controls **one arm**: the right arm of the pair (the other is hidden and plays no
part). The monitor starts first-person, like in VR: your eyes at the robot's head, the arm
reaching out like your own right arm.
Depth cues make height and position readable on a flat screen: a ring on the table straight
below the gripper (where it would come down) and a line down to it, both turning green when the
fingertips are low enough to grab (`[depth_cues]` in the configuration).

**Phone controls:** hold **MOVE** and move the phone: the gripper moves the same way. Turning
the phone turns the gripper. **GRAB** / **RELEASE** close and open it. **RE-CENTER** resets
"forward" to wherever the phone points.

**View** (on the computer): the arrow keys turn and tilt the camera, **+** / **−** (or the mouse
wheel) zoom, dragging with the mouse also turns it, and **R** resets it. "Forward" on the phone
always means **into the screen**, whichever way the view is turned: the monitor page tells the
simulator which way the camera faces, and the phone's motion is turned to match.

**The phone's camera:** the phone's position is tracked by Chrome's AR feature (WebXR / Google
ARCore), which works by watching the room through the back camera. That's why the camera turns
on; it can't be switched off for motion tracking. The candidate doesn't see the camera image
(the controller page covers it), and no image leaves the phone: only the phone's position and
rotation are sent. This needs an **Android phone with Google Play Services for AR** (most
Android phones from the last few years). iPhones can't do this in a browser.

**Score per task (0–100):** 60 × steps completed + 20 × time left (only if completed) + 20 ×
grasp efficiency (grabs that lifted something ÷ grab attempts, minus a third per object dropped).
**Overall** = mean of the three tasks. **Pass** = overall ≥ the pass mark (default 50, set on the
admin page or in the configuration), and practice 1 completed. **Reach efficiency** (straight-line distance ÷ distance moved in practice
1; 1.0 = perfectly direct) is the clearest single sign of whether someone gets the mapping.
Times are in simulation time, so a slow connection doesn't cost points. The weights are in the
configuration's `[scoring]`; the rules in `screen_game.py`. Changing them changes how results
compare with earlier candidates (each result records which configuration it ran with).

## Running a screening round

1. **Once:** install Cloudflare's tunnel program, which gives your laptop a public `https://`
   address without opening your router:

   ```bash
   winget install Cloudflare.cloudflared
   ```

2. **Start:** double-click **`TEST_TELEOP\start_screening.bat`**. It starts the lobby and the tunnel, and opens
   the results page (`http://localhost:8000/admin`) in your browser. The page shows the public
   address, which is new each time you start.
3. **Invite:** on the results page, create invite codes (for example 15 for a batch). Send each
   candidate one link (`https://<address>/?code=XXXXXXXX`). Each code works once.
4. **Watch:** the results page updates live: who is playing, in which phase, the scores, and
   PASS / FAIL. **Download CSV** exports everything. **Allow retake** gives someone a second
   attempt (the first is kept).

**Testing one part:** on the results page, **Test one part** makes a link that runs only the
part you pick (for example Task 3): phone pairing, then straight into that part, then done. Such
codes go into the batch "Testing" and get no pass / fail. From the command line:
`python -m TEST_TELEOP.lobby batch Testing --n 1 --only tilt` (parts: `reach`, `pick`, `shapes`,
`sort`, `tilt`).

**The public address** comes from Cloudflare's free quick tunnel, which gets a new random address
each time it starts. `start_screening.bat` runs it in its own window ("Screening tunnel"): keep
that window open and the address stays the same, also when you restart the lobby. It changes
after a reboot or if that window is closed. For an address that never changes, set up a named
Cloudflare tunnel on a domain you own (free Cloudflare account).

Keep the laptop on, plugged in and online while candidates play. Closing the window stops the
test for everyone. Each candidate needs a computer with a modern browser and an Android phone;
the first load is about 5 MB.

**Where results go:** `data/screening/` holds `candidates.json` (codes, names, status),
`settings.json` (how many at a time) and per candidate `<batch>/<code>/screen/attempt/` with
`results.json` (all metrics and scores), `screening.toml` (the configuration the attempt ran
with), one recording per phase (`episode_XXXX.hdf5`, replay with `scripts/replay.py`) and the
simulator's log.

**Privacy:** candidates agree to the recording on the start page. Decide how long you keep the
recordings, and delete a candidate's folder when you no longer need it.

**Test it yourself first:** run `start_screening.bat`, create a code, open the public address
on your computer and scan the QR code with your Android phone. Without a phone,
`python -m TEST_TELEOP.screen_bot` plays the whole test automatically and prints the scores.

## How it works

The test is built on the simulator in `vrteleop/` (the robots, physics, recording and the 3D
viewer) and adds only what is specific to the test. Importing `TEST_TELEOP` registers the task
`screen` and its game with vrteleop, which doesn't otherwise know about the test.

```
TEST_TELEOP/
  start_screening.bat  start the lobby and the public tunnel (double-click)
  lobby.py         the web server candidates connect to: invite codes, consent, one simulator
                   per candidate (a separate process), relays the monitor and phone to it,
                   results page and CSV (/admin, this PC only), Cloudflare quick tunnel
  worker.py        one candidate's simulator: vrteleop's server with the task registered
  config.py        reads and checks config/screening.toml (check_config.py: check it by hand)
  screen_scene.py  builds the objects, outlines, bins and targets from it (a vrteleop scene plugin)
  screen_game.py   phases, goal checks per step type, metrics, scores, results.json
  screen_bot.py    plays the whole test through the same messages a phone sends (self-check)
  web/
    screen.html/.js  the candidate's monitor page (the 3D view is vrteleop/web/main.js)
    phone.html/.js   the phone controller (WebXR AR session for motion tracking)
    admin.html       your results page
```

**Phone → robot:** the phone page sends its pose in every frame (about 60 times a second), in
the same message format as a VR controller. MOVE is the grip button (clutch), GRAB/RELEASE the
thumbstick, and the server's relative mapping does the rest. Practice and tasks 1–2 use only the
turn about the vertical axis (the gripper points down); task 3 the phone's whole orientation
(the phase's `rotation`, which the phone reads from the status). The monitor page sends the camera's heading
(`view` message); the server passes it on in its status (`view_yaw`), and the phone turns its
motion by it (and re-anchors the clutch when it changes, so the arm doesn't jump).

**Lag:** through the tunnel, a message takes about 50–70 ms each way, so the robot on the
monitor follows the phone about 0.15–0.25 s later (phone → your laptop → monitor, plus the
arm's own response). Candidates at home see the same.

**Settings:** `data/screening/settings.json`: `max_concurrent` (15), `idle_timeout_min` (an
abandoned test's simulator is stopped after this). Everything about the test itself is in the
configuration (next section).

## Changing the test: config/screening.toml

One file defines the whole test, in plain units (metres, degrees, seconds) with a comment on
every setting:

| Section | What |
|---|---|
| `[arm]` | which arm, where its base stands, start pose, top speed and turning speed, motor stiffness |
| `[controls]` | `motion_scale`: robot motion per phone motion (1.5) |
| `[view]`, `[depth_cues]` | the monitor's starting camera; the ring and line below the gripper |
| `[timing]`, `[tolerances]` | countdown; how close counts as "in", "touched", "at rest", "flat"... |
| `[scoring]` | the weights (60 / 20 / 20), the pass mark, which practice must be completed |
| `[materials.*]` | shiny or matte surfaces (ceramic, wood) |
| `[objects.*]` | every object: shape, size, colour, mass, friction, where it starts |
| `[markers.*]` | zones, outlines, bins, the bowl, the book holder |
| `[[phases]]` | the phases in order: text, time limit, scored or practice, rotation, objects, markers, and the **steps** to do |

Steps are building blocks: `reach` a target, `place_in_zone`, `place_in_slot` (turned to match),
`place_in_bin`, `remove_from` a holder, `lay_flat`, `pour_into` a bowl, `set_upright`. A new task
is a new `[[phases]]` entry made of these, with its objects and markers; no programming.

After editing, check the file: `python -m TEST_TELEOP.check_config`. It lists every problem
(a misspelt setting, a size below zero, an object off the table or out of the arm's reach, a
step naming a marker the phase doesn't show...). Then run the bot,
`python -m TEST_TELEOP.screen_bot`, to see that every task can still be done. New simulators
(each new candidate) use the changed file; the lobby keeps running. If the file is broken, the
admin page says so and keeps using the last good version. To try another file without touching
the real one, set `SCREEN_CONFIG=path/to/file.toml`.
