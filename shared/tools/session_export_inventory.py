"""List every session of every study and the state of its common session file.

Usage (from the repository root):

    python shared/tools/session_export_inventory.py [--source {files,none,trials}] [--write] [--study STUDY_ID]

For each study the sessions its exporter offers are compared with the files
under ``data/processed/``. A session is one of

- ``exported``: its file exists and passes ``validate_session``,
- ``invalid``: its file exists but fails the validation,
- ``pending``: the exporter offers it and no file exists,
- ``excluded``: the exporter lists it in ``EXCLUDED_SESSIONS`` with a reason.

``--source trials`` also re-reads every trial from the publisher's files.
``--write`` stores the result as ``session_export_inventory.json`` and
``session_export_inventory.md`` in ``docs/20261124_repository_migration/``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from session_format import ROOT, load_exporter, validate_session  # noqa: E402
from study_paths import ARCHIVE_NAMES, session_export_path  # noqa: E402

DOCS = ROOT / "docs" / "20261124_repository_migration"
STATES = ("exported", "invalid", "pending", "excluded")


def study_inventory(study, source):
    """Describe the sessions of one study.

    Parameters
    ----------
    study : str
        Study directory name.
    source : str
        Source-check depth passed to ``validate_session``.

    Returns
    -------
    dict
        Study identifier, exporter path, whether an exporter exists, the
        session rows, and per-state counts.
    """
    entry = ROOT / "studies" / study / "code" / "export_sessions.py"
    result = {"study": study, "exporter": entry.relative_to(ROOT).as_posix(),
              "has_exporter": entry.is_file(), "sessions": []}
    if not entry.is_file():
        result["state"] = "no exporter"
        return result
    exporter = load_exporter(result["exporter"])
    excluded = dict(getattr(exporter, "EXCLUDED_SESSIONS", {}))
    for item in exporter.list_sessions():
        name = item["session"]
        row = {"session": name, "sources": len(item.get("sources", []))}
        target = session_export_path(study, name)
        if name in excluded:
            row.update(state="excluded", note=excluded[name])
        elif target.exists():
            report = validate_session(target, source)
            row["file"] = target.relative_to(ROOT).as_posix()
            row["size_bytes"] = target.stat().st_size
            if report.ok:
                row.update(state="exported", trials=report.summary.get("trials"),
                           traced=report.summary.get("traced"),
                           neural=sorted(report.summary.get("neural", {})),
                           targets=sorted(report.summary.get("targets", {})))
            else:
                row.update(state="invalid", errors=report.errors[:5])
            if report.warnings:
                row["warnings"] = report.warnings[:5]
        else:
            row.update(state="pending", file=target.name)
        result["sessions"].append(row)
    for name, reason in excluded.items():
        if name not in {row["session"] for row in result["sessions"]}:
            result["sessions"].append({"session": name, "state": "excluded", "note": reason})
    result["state"] = "ok"
    return result


def counts(result):
    """Count the session rows of one study by state.

    Parameters
    ----------
    result : dict
        Result of ``study_inventory``.

    Returns
    -------
    dict
        Number of sessions in each state, trial total, and size total.
    """
    summary = {state: 0 for state in STATES}
    trials = size = 0
    for row in result["sessions"]:
        summary[row["state"]] += 1
        trials += row.get("trials") or 0
        size += row.get("size_bytes") or 0
    summary.update(trials=trials, size_bytes=size)
    return summary


def markdown(inventory):
    """Render the inventory as a Markdown document.

    Parameters
    ----------
    inventory : dict
        Output of ``build``.

    Returns
    -------
    str
        The document text.
    """
    lines = ["# Session export inventory", "", inventory["generated_utc"], "",
             "State of every session file in the common format. `exported` means the file exists and "
             "passed `shared/tools/validate_session.py`" + (" with every trial re-read from its source"
             if inventory["source"] == "trials" else " with its source files present"
             if inventory["source"] == "files" else " (structure only)") + ". "
             "Rebuild with `python shared/tools/session_export_inventory.py --write`.", "",
             "| Study | Sessions | Exported | Invalid | Pending | Excluded | Trials | Size |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    total = {key: 0 for key in STATES + ("trials", "size_bytes")}
    for result in inventory["studies"]:
        summary = counts(result)
        for key in total:
            total[key] += summary[key]
        lines.append(f"| `{result['study']}` | {len(result['sessions'])} | {summary['exported']} | "
                     f"{summary['invalid']} | {summary['pending']} | {summary['excluded']} | "
                     f"{summary['trials']} | {summary['size_bytes'] / 2**20:.1f} MiB |")
    lines.append(f"| **All** | **{sum(total[s] for s in STATES)}** | **{total['exported']}** | "
                 f"**{total['invalid']}** | **{total['pending']}** | **{total['excluded']}** | "
                 f"**{total['trials']}** | **{total['size_bytes'] / 2**30:.2f} GiB** |")
    for result in inventory["studies"]:
        lines.extend(["", f"## {result['study']}", ""])
        if result["state"] != "ok":
            lines.append(f"No `code/export_sessions.py` in this study.")
            continue
        lines.extend(["| Session | State | Trials | Size | Note |", "| --- | --- | --- | --- | --- |"])
        for row in result["sessions"]:
            size = f"{row['size_bytes'] / 2**20:.1f} MiB" if "size_bytes" in row else ""
            note = row.get("note") or (row.get("file") if row["state"] == "pending" else "")
            if row["state"] == "invalid":
                note = "; ".join(row.get("errors", []))
            note = note.replace("|", "\\|")
            lines.append(f"| `{row['session']}` | {row['state']} | {row.get('trials', '')} | {size} | {note} |")
    return "\n".join(lines) + "\n"


def build(source, studies):
    """Build the inventory of the requested studies.

    Parameters
    ----------
    source : str
        Source-check depth passed to ``validate_session``.
    studies : list of str
        Study directory names.

    Returns
    -------
    dict
        Timestamp, source depth, and one result per study.
    """
    from datetime import datetime, timezone
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source,
            "studies": [study_inventory(study, source) for study in studies]}


def main():
    """Print the inventory and optionally write it to the migration documents.

    Returns
    -------
    int
        Zero when no study has an invalid file or no exporter, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("files", "none", "trials"), default="files",
                        help="source-check depth for every exported file (default: files)")
    parser.add_argument("--write", action="store_true", help="write the JSON and Markdown documents")
    parser.add_argument("--study", action="append", default=[], help="limit to this study (repeatable)")
    args = parser.parse_args()
    studies = args.study or list(ARCHIVE_NAMES)
    start = time.time()
    inventory = build(args.source, studies)
    for result in inventory["studies"]:
        summary = counts(result)
        label = "FAIL" if summary["invalid"] else "ok"
        print(f"[{label}] {result['study']}: {result['state']}; "
              + ", ".join(f"{state} {summary[state]}" for state in STATES)
              + f", trials {summary['trials']}, {summary['size_bytes'] / 2**20:.1f} MiB", flush=True)
        for row in result["sessions"]:
            if row["state"] == "invalid":
                print(f"       invalid {row['session']}: {row.get('errors')}")
    print(f"{time.time() - start:.0f}s")
    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "session_export_inventory.json").write_text(
            json.dumps(inventory, indent=1) + "\n", encoding="utf-8")
        (DOCS / "session_export_inventory.md").write_text(markdown(inventory), encoding="utf-8")
        print(f"written: {DOCS / 'session_export_inventory.md'}")
    return 1 if any(counts(result)["invalid"] for result in inventory["studies"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
