"""``python -m hermes_report``: the same as the ``hermes-report`` command."""

import sys

from hermes_report.cli import main

if __name__ == "__main__":
  sys.exit(main())
