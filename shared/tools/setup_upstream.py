"""Install or verify the upstream source repositories for the study pipelines."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "shared" / "upstream_sources.json"
NATIVE_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".java")


def git(path, *args):
    """Run Git in a source checkout and return its standard output.

    Parameters
    ----------
    path : pathlib.Path
        Working directory of the Git command.
    *args : str
        Git command and arguments.

    Returns
    -------
    bytes
        Git standard output, retained as bytes for patch verification.
    """
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=60",
         "-c", "core.autocrlf=false", "-C", str(path), *args],
        env=env, capture_output=True, timeout=240, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def verify(path, source):
    """Verify the checkout revision, origin, and recorded source patch.

    Parameters
    ----------
    path : pathlib.Path
        Upstream checkout directory.
    source : dict
        Source entry from the repository manifest.
    """
    if not (path / ".git").is_dir():
        raise RuntimeError(f"Missing Git checkout at {path}")
    revision = git(path, "rev-parse", "HEAD").decode().strip()
    origin = git(path, "remote", "get-url", "origin").decode().strip()
    if revision != source["commit"] or origin.removesuffix(".git") != source["url"].removesuffix(".git"):
        raise RuntimeError(f"Checkout origin or commit differs from the manifest: {path}")
    diff = git(path, "diff", "--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-color", "HEAD", "--")
    patch = source.get("patch")
    expected = (ROOT / patch).read_bytes().replace(b"\r\n", b"\n") if patch else b""
    if diff.replace(b"\r\n", b"\n") != expected:
        raise RuntimeError(f"Checkout modifications differ from the recorded patch: {path}")
    ls_files = git(path, "ls-files", "-t", "-z").decode("utf-8").split("\0")
    prefix = source.get("windows_sparse_prefix")
    for entry in filter(None, ls_files):
        flag, filename = entry.split(" ", 1)
        if flag == "S":
            if not prefix or not filename.startswith(prefix + "/") or not filename.endswith(NATIVE_SUFFIXES):
                raise RuntimeError(f"Unexpected sparse checkout path: {filename}")
        elif not (path / filename).exists():
            raise RuntimeError(f"Missing source file: {path / filename}")


def install(path, source):
    """Fetch the pinned commit and apply its patch before installing it.

    Parameters
    ----------
    path : pathlib.Path
        Final location of the checkout.
    source : dict
        Source entry from the repository manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".setup-", dir=path.parent) as temporary:
        stage = Path(temporary) / "checkout"
        stage.mkdir()
        git(stage, "init", "--quiet")
        git(stage, "config", "core.autocrlf", "false")
        git(stage, "remote", "add", "origin", source["url"])
        git(stage, "fetch", "--depth=1", "--no-tags", "origin", source["commit"])
        prefix = source.get("windows_sparse_prefix")
        if os.name == "nt" and prefix:
            git(stage, "config", "core.sparseCheckout", "true")
            git(stage, "config", "core.sparseCheckoutCone", "false")
            patterns = ["/*", *(f"!/{prefix}/**/*{suffix}" for suffix in NATIVE_SUFFIXES)]
            (stage / ".git/info/sparse-checkout").write_text("\n".join(patterns) + "\n", encoding="utf-8")
        git(stage, "checkout", "--quiet", "--detach", source["commit"])
        if patch := source.get("patch"):
            git(stage, "apply", "--index", str(ROOT / patch))
        verify(stage, source)
        stage.rename(path)


def main():
    """Configure all studies or the studies requested on the command line.

    Returns
    -------
    int
        Zero if every selected checkout was installed or verified.
    """
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="append", choices=sources, help="Study and its runtime dependencies")
    parser.add_argument("--check", action="store_true", help="Verify existing checkouts without downloading")
    args = parser.parse_args()
    selected = set(args.study or sources)
    failures = 0
    for study, source in sources.items():
        if study not in selected and not selected.intersection(source["runtime_studies"]):
            continue
        path = ROOT / "studies" / study / "open_source_code_for_reference" / source["name"]
        try:
            if args.check or path.exists():
                verify(path, source)
            else:
                install(path, source)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"{study}: {error}")
            failures += 1
        else:
            print(f"{study}: {source['name']} {source['commit'][:12]} verified")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
