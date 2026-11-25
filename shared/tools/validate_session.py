"""Validate common session files and trace their trials to the publisher's files.

Usage (from the repository root):

    python shared/tools/validate_session.py PATH [PATH ...] [--source {files,trials,none}] [--json FILE]

Each PATH is a session file written by a study's ``export_sessions.py``. With
no PATH, every session file under ``studies/*/data/processed/`` is checked.
The exit status is zero when every file passes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from session_format import ROOT, format_report, validate_session  # noqa: E402
from study_paths import ARCHIVE_NAMES, study_checkout  # noqa: E402


def exported_sessions():
    """List every common session file present in the local study directories.

    Returns
    -------
    list of pathlib.Path
        Session files in study and file-name order.
    """
    found = []
    for study in ARCHIVE_NAMES:
        directory = study_checkout(study) / "data" / "processed"
        if directory.is_dir():
            found.extend(sorted(directory.glob("*/*.h5")))
    return found


def main():
    """Validate the requested session files and print a report for each.

    Returns
    -------
    int
        Zero when every file passes, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="session .h5 files (default: every exported session)")
    parser.add_argument("--source", choices=("files", "trials", "none"), default="files",
                        help="source check depth: existence of the source files (default), "
                             "re-read every trial, or none")
    parser.add_argument("--json", type=Path, help="also write the reports to this JSON file")
    args = parser.parse_args()
    paths = args.paths or exported_sessions()
    if not paths:
        print("no session files found")
        return 1
    reports = []
    for path in paths:
        report = validate_session(path, args.source)
        reports.append(report)
        print("\n".join(format_report(report)), flush=True)
    failed = [report for report in reports if not report.ok]
    print(f"\n{len(reports) - len(failed)} passed, {len(failed)} failed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([{
            "path": str(Path(report.path).resolve()),
            "ok": report.ok, "errors": report.errors, "warnings": report.warnings,
            "summary": report.summary} for report in reports], indent=1) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
