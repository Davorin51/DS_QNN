"""Compatibility entry point; install the package, then prefer ds-qnn train."""

import sys

from ds_qnn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
