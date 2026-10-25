# Adapted for this project; original logic reviewed and re-validated.
"""Path resolution utilities.

All paths in the project are relative to the repository root.
No absolute paths are hard-coded anywhere in the codebase.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file's location until it finds a directory
    containing 'README.md' and 'src/'.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "README.md").exists() and (current / "src").is_dir():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root directory")


def get_results_dir() -> Path:
    """Return the path to the results directory."""
    return get_project_root() / "results"


def get_cache_dir() -> Path:
    """Return the path to the kernel cache directory."""
    d = get_results_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_tables_dir() -> Path:
    """Return the path to the results tables directory."""
    d = get_results_dir() / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_figures_dir() -> Path:
    """Return the path to the results figures directory."""
    d = get_results_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    """Return the path to the raw data directory."""
    return get_project_root() / "data" / "raw"


def get_configs_dir() -> Path:
    """Return the path to the configs directory."""
    return get_project_root() / "configs"
