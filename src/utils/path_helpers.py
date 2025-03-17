from pathlib import Path


def find_project_root(marker: str = "README.md") -> Path:
    """
    Walks up the directory tree to find the project root by searching for a marker file.

    Parameters:
        marker (str): The filename to look for that indicates the project root.

    Returns:
        Path: The project root directory.

    Raises:
        FileNotFoundError: If the marker file is not found in any parent directory.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root marker: {marker}")
