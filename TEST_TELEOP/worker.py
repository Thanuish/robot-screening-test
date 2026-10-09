"""One candidate's simulator: vrteleop's server with the screening task registered. The lobby
starts one per candidate:  python -m TEST_TELEOP.worker --task screen --http --port N ..."""
import TEST_TELEOP  # noqa: F401  (registers the task "screen" and its game)
from vrteleop.server import main

if __name__ == "__main__":
    main()
