"""Compatibility entry point; --method explicitly selects one pruning strategy."""

import sys

from ds_qnn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prune", *sys.argv[1:]]))
