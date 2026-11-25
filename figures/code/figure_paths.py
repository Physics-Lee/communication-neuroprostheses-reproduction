"""Resolve the repository, benchmark configuration, and published figure bundle."""

from pathlib import Path

import yaml

FIGURES = Path(__file__).resolve().parents[1]
ROOT = FIGURES.parent
BENCHMARK = FIGURES / "reproduction_benchmark.yaml"
PUBLISHED = FIGURES / "results" / "20261124_repository_migration"


def load_benchmark():
    """Read the benchmark and its evaluation sources.

    Returns
    -------
    dict
        Figure configuration and repository-relative input paths.
    """
    with BENCHMARK.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)
