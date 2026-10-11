# Robot Buzz Wire

A teleoperation screening game, like the classic buzz-wire game: a robot arm holds a wand with
a metal loop around a curved wire. Guide the loop from one end to the other without touching
the wire, then hang the wand on the end hook. Played with an Android phone as a motion
controller (like a VR controller).

**Status:** milestone 2 of 7 done (physics, robot, phone control), graphics (milestone 4) in a
first realistic version. Game menus, the level editor and hosting come next.

## Try it

```
python -m buzzwire.server            # then open http://localhost:8000
```

The screen shows a QR code: scan it with an **Android phone** (Chrome) and tap START CONTROLLER.
The phone needs the public tunnel (`python -m TEST_TELEOP.lobby tunnel`). The phone is the only
controller:

- **Hold the blue pad** and move the phone: the robot follows (like a VR controller).
- **GRAB / RELEASE** close and open the gripper. **RE-CENTER**: "forward" = where you point now.
- The phone buzzes when the wand touches the wire, and on grab, release and level done.

**Graphics:** physically based materials lit by an HDRI studio, a soft key light, ambient
occlusion, bloom (the wire glows red when touched), anti-aliasing. Quality steps down by itself
when the frame rate drops below 45 fps (high, medium, low); `?q=low` forces a preset. The frame
rate of each screen is reported to the server's log. Assets: `buzzwire/web/assets/CREDITS.md`
(CC0), three.js r170 in `buzzwire/web/vendor/` (MIT).

## How a level goes

1. The wand rests on the start fork, its loop around the rubber sleeve at the start of the wire.
2. Pick it up by the handle, lift until the loop is centred on the wire.
3. Guide the loop along the wire. Touching the wire buzzes (red flash, sound) and is counted.
4. Set the wand on the end fork (the hook) and let go: the level is done.

**Score per level (0-100):** progress along the wire (40, full when hung), no touches (25),
little touch time (15), time left (10, if completed), smoothness (10, from the jerk of the
motion). Weights and thresholds in `config/buzzwire.toml` `[scoring]`.

## Checks

```
python -m buzzwire.bot --replay          # clean run: every level, 0 touches; replay gives identical scores
python -m buzzwire.bot --clumsy          # wobbles past the loop's clearance: touches are counted
```

## Files

```
config/buzzwire.toml   robot, controls, physics, view, board, wire, wand, scoring, levels
buzzwire/
  config.py    reads and checks the configuration
  spline.py    the wire's curve through the control points (arc length, closest point)
  scene.py     the MuJoCo model: robot, board, wire (capsule chain), forks, wand
  ik.py        inverse kinematics on Pinocchio (joint limits and speeds, table keep-out)
  game.py      one player's game: controller input, physics, touches, levels, scores, replay
  bot.py       plays through the controller messages (clean and clumsy)
  server.py    development server: the game in real time, the web pages
  web/         index.html + app.js: the game screen (three.js); phone.html + phone.js: the controller
```
