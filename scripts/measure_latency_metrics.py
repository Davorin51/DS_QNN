"""Compatibility entry point; install the package, then prefer ds-qnn benchmark."""

import sys

from ds_qnn.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
