"""TEST_TELEOP: the remote screening test (phone as the controller, the robot on the
candidate's monitor). Built on vrteleop: importing this package registers the task "screen"
(screen_scene.py) and its game (screen_game.py) with the vrteleop scene and server.

    python -m TEST_TELEOP.lobby serve --tunnel     # the lobby (or double-click start_screening.bat)
    python -m TEST_TELEOP.screen_bot               # a bot plays the whole test (self-check)

See TEST_TELEOP/README.md.
"""
from . import screen_scene  # noqa: F401  (registers the task)
from . import screen_game   # noqa: F401  (registers the game)
