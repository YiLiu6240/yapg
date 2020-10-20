from pathlib import Path


def find_project_root(anchor_file: str = "pyproject.toml") -> Path:
    cwd = Path.cwd()
    test_dir = cwd
    prev_dir = None
    while prev_dir != test_dir:
        if (test_dir / anchor_file).exists():
            return test_dir
        prev_dir = test_dir
        test_dir = test_dir.parent
    return cwd
