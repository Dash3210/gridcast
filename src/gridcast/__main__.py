"""Enable execution via ``python -m gridcast``.

Entry point for running the Gridcast CLI as a module without requiring
the console script to be installed on PATH.
"""

from gridcast.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
