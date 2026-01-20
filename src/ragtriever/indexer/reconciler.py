from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path


def matches_ignore_pattern(rel_path: str, patterns: list[str]) -> bool:
    """Check if a relative path matches any of the ignore patterns.

    Supports glob patterns like:
    - "**/.DS_Store" - match .DS_Store in any directory
    - ".git/**" - match everything under .git
    - "**/~$*" - match Office temp files in any directory
    - "00-Input/**" - match everything under 00-Input folder

    Args:
        rel_path: Relative path from vault root (forward slashes)
        patterns: List of glob patterns to match against

    Returns:
        True if the path matches any pattern and should be ignored
    """
    # Normalize path separators
    rel_path = rel_path.replace("\\", "/")

    for pattern in patterns:
        pattern = pattern.replace("\\", "/")

        # Handle ** prefix patterns (match in any directory)
        if pattern.startswith("**/"):
            # Match the pattern suffix against any path segment
            suffix = pattern[3:]  # Remove **/
            # Check if path ends with the pattern or contains it as a segment
            if fnmatch(rel_path, pattern) or fnmatch(rel_path, f"*/{suffix}"):
                return True
            # Also check each path component
            parts = rel_path.split("/")
            for i, part in enumerate(parts):
                # Check if this part matches the suffix
                if fnmatch(part, suffix):
                    return True
                # Check if remaining path matches
                remaining = "/".join(parts[i:])
                if fnmatch(remaining, suffix):
                    return True

        # Handle ** suffix patterns (match everything under a directory)
        elif pattern.endswith("/**"):
            prefix = pattern[:-3]  # Remove /**
            if rel_path.startswith(prefix + "/") or rel_path == prefix:
                return True

        # Handle patterns with ** in the middle
        elif "**" in pattern:
            if fnmatch(rel_path, pattern):
                return True

        # Simple pattern matching
        else:
            if fnmatch(rel_path, pattern):
                return True

    return False


@dataclass
class Reconciler:
    root: Path
    ignore: list[str]

    def scan_files(self) -> list[Path]:
        """Scan vault for files, respecting ignore patterns."""
        paths: list[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file():
                rel_path = str(p.relative_to(self.root)).replace("\\", "/")
                if not matches_ignore_pattern(rel_path, self.ignore):
                    paths.append(p)
        return paths
