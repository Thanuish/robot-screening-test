# Robot Buzz Wire

A teleoperation screening game, like the classic buzz-wire game: a robot arm holds a wand with
a metal loop around a curved wire. Guide the loop from one end to the other without touching
the wire, then hang the wand on the end hook. Played with an Android phone as a motion
controller (like a VR controller), or the keyboard.

**Status:** milestone 2 of 7 (physics, robot, keyboard and phone control, a first 3D view).
Graphics, game menus, the level editor and hosting come next.

## Try it

```
python -m buzzwire.server            # then open http://localhost:8000
```

- **Phone:** scan the QR code on the page (needs the public tunnel:
  `python -m TEST_TELEOP.lobby tunnel`). Hold **MOVE** and move the phone; **GRAB** / **RELEASE**
  close and open the gripper; **RE-CENTER** makes "forward" the direction you point.
- **Keyboard:** click the view, then **W A S D** move, **R / F** up / down, **Q / E** turn,
  **I / K** tilt and **J / L** twist (level 3), **C** grab / release, **Shift** slow.
- The green ghost shows where you point; the robot follows it at a real robot's speed.

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
  web/         the player's screen (three.js)
```
