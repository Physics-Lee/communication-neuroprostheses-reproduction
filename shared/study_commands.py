"""Launch a repository study command through its public entry point."""

import subprocess
import sys
from pathlib import Path


def run_command(entry, implementation, usage, *, cwd=None):
    """Run the implementation with the public command's arguments.

    Parameters
    ----------
    entry : str
        Path to the public entry point.
    implementation : str
        Implementation path relative to the entry point's code directory.
    usage : str
        Arguments and a short description for the public help message.
    cwd : str or pathlib.Path, optional
        Working directory required by an upstream implementation.

    Returns
    -------
    int
        Child command exit code, or zero after printing help.
    """
    entry_path = Path(entry).resolve()
    args = sys.argv[1:]
    if args == ["--help"] or args == ["-h"]:
        print(f"Usage: python {entry_path.name} {usage}")
        return 0
    target = entry_path.parent / implementation
    command = [sys.executable, str(target), *args]
    return subprocess.run(command, cwd=cwd, check=False).returncode
