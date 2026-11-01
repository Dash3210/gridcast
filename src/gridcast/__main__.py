"""Enable ``python -m gridcast`` to run the CLI.

A GitHub Action and a reviewer on a machine without the console script on PATH both need this.
"""

from gridcast.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
