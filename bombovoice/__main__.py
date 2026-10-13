import sys

from .cli import main

if __name__ == "__main__":  # guard: the speech worker process re-imports this module on Windows
    sys.exit(main())
