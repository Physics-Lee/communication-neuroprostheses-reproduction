"""Check that each study tutorial points to real inputs and runnable commands.

Walks ``studies/*/tutorial.ipynb`` and verifies, for every study, that the
data-setup section comes first, that each ``python <script>`` command names an
existing repository file and only uses flags its implementation declares, that
a command without flags passes the number of positional arguments the public
usage string requires, and that each ``studies/...`` path either exists on
disk, is a download target under ``data/raw/``, or is an output produced by an
earlier step of the same notebook. Relative Markdown links are resolved from
the study directory.

Usage (from the repository root):

    python shared/tools/check_study_tutorials.py [--run] [--study STUDY_ID]

``--run`` also executes each public command with ``--help``. The check needs
no external data: on a fresh clone, inputs that a notebook step produces are
reported as ``produced`` instead of failing.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_COMMANDS = ("prepare_data.py", "export_sessions.py", "train.py", "infer.py", "decode.py",
                   "evaluate.py")
PATH_RE = re.compile(r"studies/[A-Za-z0-9_./\-]+")
COMMAND_RE = re.compile(r"^python (\S+\.py)([^\n]*)", re.M)
FLAG_RE = re.compile(r"(--?[A-Za-z][\w-]*)")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")
WRAPPER_RE = re.compile(r'run_command\(\s*__file__,\s*"([^"]+)",\s*"([^"]+)"')
OUTPUT_LINE_RE = re.compile(r"^\s*\*\*[^*]*output[^*]*\*\*[^\n]*", re.I | re.M)
PRODUCED_LINE_RE = re.compile(
    r"^\s*(?:After (?:extraction|preparing|conversion)[^\n]*"
    r"|The (?:directory|checkpoint directory) (?:contains|is))[^\n]*", re.I | re.M)


def cell_text(notebook):
    """Return the concatenated source text of every cell in a notebook.

    Parameters
    ----------
    notebook : dict
        Parsed ``.ipynb`` document.

    Returns
    -------
    str
        Markdown and code cell sources joined with blank lines.
    """
    return "\n\n".join("".join(cell["source"]) for cell in notebook["cells"])


def mentioned_paths(text):
    """Extract repository-relative study paths from notebook text.

    Parameters
    ----------
    text : str
        Notebook source text.

    Returns
    -------
    list of str
        Unique paths in order of appearance with trailing sentence
        punctuation removed.
    """
    seen = []
    for match in PATH_RE.findall(text):
        path = match.rstrip(".,;:)") or match
        if path not in seen:
            seen.append(path)
    return seen


def classify(path, produced):
    """Decide whether a path mentioned by a tutorial is accounted for.

    Parameters
    ----------
    path : str
        Repository-relative path.
    produced : list of str
        Paths that the notebook declares as outputs or tells the reader to
        check for after running a step.

    Returns
    -------
    str
        ``ok`` when the path exists, ``download target`` for an archive the
        reader must save under ``data/raw/``, ``produced`` when a notebook
        step writes it, otherwise ``missing``.
    """
    if (ROOT / path).exists():
        return "ok"
    parts = Path(path).parts
    if "data" in parts and parts[parts.index("data") + 1:parts.index("data") + 2] == ("raw",):
        return "download target"
    if any(path == item or path.startswith(item.rstrip("/") + "/") or item.startswith(path)
           for item in produced):
        return "produced"
    return "missing"


def implementation(script):
    """Follow a public command wrapper to the script it launches.

    Parameters
    ----------
    script : pathlib.Path
        Repository script named by a tutorial command.

    Returns
    -------
    tuple
        ``(path, usage)``: the script whose arguments should be checked and
        the public usage string. A ``run_command`` wrapper resolves to its
        implementation, a session exporter resolves to the shared
        ``export_cli`` that defines its arguments, and any other script is
        its own implementation with no usage string.
    """
    text = script.read_text(encoding="utf-8")
    match = WRAPPER_RE.search(text)
    if match is None:
        if "export_cli(" in text:
            return ROOT / "shared" / "session_format.py", None
        return script, None
    return (script.parent / match.group(1)).resolve(), match.group(2)


def positional_arguments(arguments):
    """Count the positional arguments on a tutorial command line.

    Parameters
    ----------
    arguments : str
        Text following the script name.

    Returns
    -------
    int
        Number of tokens that are neither flags nor quoted strings. Only
        meaningful for commands that use no flags.
    """
    stripped = re.sub(r"\"[^\"]*\"", "", arguments)
    return len([token for token in stripped.split() if not token.startswith("-")])


def usage_bounds(usage):
    """Return the required and maximum positional counts of a usage string.

    Parameters
    ----------
    usage : str
        Public usage string such as ``SESSION BLOCK_CSV [LIMIT]``.

    Returns
    -------
    tuple of int
        ``(required, maximum)`` positional argument counts.
    """
    required = len(re.findall(r"(?<![\w\]])[A-Z][A-Z_]*(?![\w\]])", re.sub(r"\[[^\]]*\]", "", usage)))
    maximum = len(re.findall(r"\b[A-Z][A-Z_]*\b", usage))
    return required, maximum


def check_study(study, run):
    """Audit one study tutorial and return its findings.

    Parameters
    ----------
    study : str
        Study directory name.
    run : bool
        Whether to execute the public commands with ``--help``.

    Returns
    -------
    dict
        ``paths`` maps each mentioned path to its classification,
        ``problems`` lists command, link, and layout failures, and ``help``
        maps public commands to the ``--help`` exit code.
    """
    notebook = json.loads((ROOT / "studies" / study / "tutorial.ipynb").read_text(encoding="utf-8"))
    text = cell_text(notebook)
    declared = OUTPUT_LINE_RE.findall(text) + PRODUCED_LINE_RE.findall(text)
    produced = [path for line in declared for path in mentioned_paths(line)]
    report = {"paths": {}, "problems": [], "help": {}}
    lines = [line.strip() for line in "".join(notebook["cells"][0]["source"]).splitlines()]
    headings = [line for line in lines if line]
    report["setup_first"] = len(headings) > 1 and headings[1] == "## Prepare the data"
    if not report["setup_first"]:
        report["problems"].append("data setup is not the first section")
    for path in mentioned_paths(text):
        report["paths"][path] = classify(path, produced)
        if report["paths"][path] == "missing":
            report["problems"].append(f"path is missing: {path}")
    for name, arguments in COMMAND_RE.findall(text):
        script = ROOT / name
        if not script.is_file():
            report["problems"].append(f"script not found: {name}")
            continue
        target, usage = implementation(script)
        source = target.read_text(encoding="utf-8")
        for flag in set(FLAG_RE.findall(arguments)):
            if f'"{flag}"' not in source and f"'{flag}'" not in source:
                report["problems"].append(f"flag {flag} is not accepted by {name}")
        flags = FLAG_RE.findall(arguments)
        if usage is not None and not flags:
            required, maximum = usage_bounds(usage)
            given = positional_arguments(arguments)
            if not required <= given <= maximum:
                report["problems"].append(
                    f"{name} takes {required} to {maximum} positional arguments, notebook shows {given}")
    for link in LINK_RE.findall(text):
        if not link.startswith(("http://", "https://", "mailto:")) and not (ROOT / "studies" / study / link).exists():
            report["problems"].append(f"unresolved link: {link}")
    if run:
        for command in PUBLIC_COMMANDS:
            entry = ROOT / "studies" / study / "code" / command
            if entry.is_file():
                result = subprocess.run([sys.executable, str(entry), "--help"],
                                        capture_output=True, text=True, cwd=str(ROOT))
                report["help"][command] = result.returncode
                if result.returncode != 0 or "usage:" not in result.stdout.lower():
                    report["problems"].append(f"{command} --help failed (exit {result.returncode})")
    return report


def main():
    """Audit every study tutorial and print a pass/fail summary.

    Returns
    -------
    int
        Zero when every tutorial passes, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="execute public commands with --help")
    parser.add_argument("--study", help="audit only this study directory")
    args = parser.parse_args()
    studies = sorted(p.name for p in (ROOT / "studies").iterdir()
                     if p.is_dir() and (p / "tutorial.ipynb").is_file())
    if args.study:
        studies = [s for s in studies if s == args.study]
    failed = 0
    for study in studies:
        report = check_study(study, args.run)
        counts = {}
        for status in report["paths"].values():
            counts[status] = counts.get(status, 0) + 1
        failed += bool(report["problems"])
        print(f"[{'FAIL' if report['problems'] else 'PASS'}] {study}  setup-first={report['setup_first']}"
              f"  paths={counts}  help-checked={len(report['help'])}")
        for path, status in report["paths"].items():
            if status != "ok":
                print(f"        {status:16s} {path}")
        for problem in report["problems"]:
            print(f"        ! {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
