"""Download the UR5e and Robotiq 2F-85 models from MuJoCo Menagerie into ./assets.

    python setup_assets.py

Uses a sparse git checkout if git is available, otherwise the GitHub API.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = "google-deepmind/mujoco_menagerie"
DIRS = ["universal_robots_ur5e", "robotiq_2f85"]
DEST = Path(__file__).resolve().parent / "assets"


def via_git() -> bool:
    if not shutil.which("git"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            f"https://github.com/{REPO}.git", tmp], check=True)
            subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", *DIRS], check=True)
        except subprocess.CalledProcessError:
            return False
        for d in DIRS:
            shutil.rmtree(DEST / d, ignore_errors=True)
            shutil.copytree(Path(tmp) / d, DEST / d)
    return True


def via_api():
    def get(url):
        with urllib.request.urlopen(url) as r:
            return r.read()
    for d in DIRS:
        stack = [d]
        while stack:
            path = stack.pop()
            for e in json.loads(get(f"https://api.github.com/repos/{REPO}/contents/{path}")):
                if e["type"] == "dir":
                    stack.append(e["path"])
                else:
                    out = DEST / e["path"]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(get(e["download_url"]))
                    print("  ", e["path"])


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    if all((DEST / d).exists() for d in DIRS) and "--force" not in sys.argv:
        print("Assets already present (use --force to re-download).")
        sys.exit(0)
    if not via_git():
        print("git unavailable/failed, using GitHub API...")
        via_api()
    print("Done:", ", ".join(DIRS))
