"""Check the test's configuration (config/screening.toml, or another file) and print a summary:

    python -m TEST_TELEOP.check_config [file]
"""
import sys

from . import config as C


def main():
    try:
        cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    except C.ConfigError as e:
        print(e)
        sys.exit(1)
    print(f"{cfg.path}: OK")
    print(f"  arm: {cfg.arm.side} {cfg.arm.model} at {cfg.arm.base}")
    print(f"  {len(cfg.objects)} objects, {len(cfg.markers)} markers, {len(cfg.materials)} materials")
    for p in cfg.phases:
        extra = f"  {p.limit:.0f} s, {len(p.steps)} steps{', scored' if p.scored else ''}" if p.kind == "task" else ""
        print(f"  phase {p.key:8s} {p.title}{extra}")


if __name__ == "__main__":
    main()
