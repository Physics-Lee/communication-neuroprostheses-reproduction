"""Execute notebooks top to bottom in the bci-repro kernel and store the outputs.

Usage (from the repository root, with bci-repro active):

    python shared/tools/run_notebook.py NOTEBOOK [NOTEBOOK ...] [--cwd DIRECTORY] [--timeout SECONDS]

The code cells of each notebook run in order in one fresh kernel that uses the
same Python interpreter as this script, started in the notebook's directory.
Each code cell's outputs and execution count are replaced. Markdown cells and
all other parts of the document are left as they are, and the file keeps its
JSON formatting. A cell that raises stops that notebook with a non-zero exit
status, after its error is written into the notebook.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager

STYLES = (
    {"indent": 1, "ensure_ascii": False},
    {"indent": 1, "sort_keys": True, "ensure_ascii": False},
    {"indent": 1, "sort_keys": True},
    {"indent": 1},
)


def detect_style(raw):
    """Return the JSON dump options that reproduce a notebook file.

    Parameters
    ----------
    raw : str
        Text of the notebook file.

    Returns
    -------
    dict
        Keyword arguments for ``json.dumps`` that write the file back in its
        own formatting, or the default style when none matches.
    """
    notebook = json.loads(raw)
    for style in STYLES:
        if json.dumps(notebook, **style) + "\n" == raw or json.dumps(notebook, **style) == raw:
            return style
    return dict(STYLES[0])


def message_output(message):
    """Convert one IOPub message to a notebook output entry.

    Parameters
    ----------
    message : dict
        A message from the kernel's IOPub channel.

    Returns
    -------
    dict or None
        An output entry for the notebook, or ``None`` for messages that carry
        nothing to store.
    """
    kind = message["msg_type"]
    content = message["content"]
    if kind == "stream":
        return {"output_type": "stream", "name": content["name"], "text": content["text"]}
    if kind in ("display_data", "execute_result", "update_display_data"):
        entry = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
        if kind == "execute_result":
            entry["execution_count"] = content.get("execution_id")
        return entry
    if kind == "error":
        return {"output_type": "error", "ename": content["ename"], "evalue": content["evalue"],
                "traceback": content["traceback"]}
    return None


class InterpreterKernelManager(KernelManager):
    """Kernel manager that starts this interpreter instead of a registered kernel.

    Notes
    -----
    ``KernelManager.kernel_spec`` has no setter, so the property is replaced.
    The kernel therefore runs under the Python that runs this script, which is
    ``bci-repro`` when the tool is started from that environment.
    """

    spec = None

    @property
    def kernel_spec(self):
        """jupyter_client.kernelspec.KernelSpec: the specification of this interpreter."""
        if self.spec is None:
            type(self).spec = KernelSpec(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                language="python", display_name="Python 3 (bci-repro)")
        return self.spec


def run_cell(client, code, timeout):
    """Execute one cell and collect its outputs.

    Parameters
    ----------
    client : jupyter_client.blocking.client.BlockingKernelClient
        Client of a started kernel.
    code : str
        Source of the cell.
    timeout : int or float
        Seconds to wait for each kernel message.

    Returns
    -------
    tuple
        ``(outputs, failed)``: the output entries and whether the cell raised.
    """
    outputs = []
    failed = False
    client.execute(code, allow_stdin=False, store_history=True)
    while True:
        message = client.get_iopub_msg(timeout=timeout)
        if message["msg_type"] == "status" and message["content"]["execution_state"] == "idle":
            break
        entry = message_output(message)
        if entry is not None:
            if entry["output_type"] == "error":
                failed = True
            if entry["output_type"] == "stream" and outputs and outputs[-1] == entry \
                    and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == entry["name"]:
                outputs[-1]["text"] += entry["text"]
            else:
                outputs.append(entry)
    return outputs, failed


def run_notebook(path, timeout, cwd=None):
    """Execute one notebook file in place.

    Parameters
    ----------
    path : pathlib.Path
        Notebook to execute.
    timeout : int or float
        Seconds to wait for each kernel message.
    cwd : pathlib.Path, optional
        Working directory for the kernel. Defaults to the notebook's directory.

    Returns
    -------
    dict
        Counts of code cells run and the index of the cell that failed, or
        ``None`` when every cell ran.
    """
    raw = path.read_text(encoding="utf-8")
    style = detect_style(raw)
    notebook = json.loads(raw)
    manager = InterpreterKernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(cwd or path.parent), env=dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline"))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=timeout)
        failed_at, count = None, 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            outputs, failed = run_cell(client, "".join(cell["source"]), timeout)
            cell["outputs"] = outputs
            cell["execution_count"] = count
            if failed:
                failed_at = index
                break
        result = {"code_cells": count, "failed_cell": failed_at}
        if failed_at is not None:
            result["error"] = notebook["cells"][failed_at]["outputs"][-1]
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    path.write_text(json.dumps(notebook, **style) + "\n", encoding="utf-8", newline="\n")
    return result


def main():
    """Execute the requested notebooks and report the outcome of each.

    Returns
    -------
    int
        Zero when every notebook ran without an error, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--cwd", type=Path, help="working directory for every kernel (default: the notebook's folder)")
    parser.add_argument("--timeout", type=float, default=600, help="seconds to wait for each kernel message")
    args = parser.parse_args()
    failed = 0
    for path in args.notebooks:
        start = time.time()
        result = run_notebook(path, args.timeout, args.cwd)
        state = "FAIL" if result["failed_cell"] is not None else "PASS"
        failed += state == "FAIL"
        line = f"[{state}] {path}  {result['code_cells']} code cells in {time.time() - start:.0f}s"
        if state == "FAIL":
            error = result["error"]
            line += f"\n       cell {result['failed_cell']}: {error['ename']}: {error['evalue']}"
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
