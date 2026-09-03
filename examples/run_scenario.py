#!/usr/bin/env python3
"""Compatibility wrapper for `drone-swarm run`."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _main() -> int:
    """Import the installed/source-tree CLI only after `src` is on `sys.path`."""

    from drone_swarm.cli import main

    return main(("run", *sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(_main())
