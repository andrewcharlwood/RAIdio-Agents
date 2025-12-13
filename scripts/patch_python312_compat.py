#!/usr/bin/env python3
"""
Python 3.12 Compatibility Patcher for M3D-LaMed

Fixes known Python 3.12 incompatibilities in HuggingFace cached model files.
Run this after downloading models if using Python 3.12+.

Usage:
    python scripts/patch_python312_compat.py
    python scripts/patch_python312_compat.py --cache-dir ~/.cache/huggingface

Known Issues Fixed:
    1. collections.Sequence -> collections.abc.Sequence (removed in Python 3.10)
    2. collections.Mapping -> collections.abc.Mapping
    3. collections.MutableMapping -> collections.abc.MutableMapping
    4. distutils.version -> packaging.version (removed in Python 3.12)
"""

import argparse
import os
import re
import sys
from pathlib import Path


# Patterns to fix and their replacements
FIXES = [
    # collections ABC classes moved to collections.abc in Python 3.10
    (
        r'from collections import (Sequence|Mapping|MutableMapping|MutableSequence|Set|MutableSet|Callable|Iterable|Iterator)',
        r'from collections.abc import \1'
    ),
    (
        r'collections\.(Sequence|Mapping|MutableMapping|MutableSequence|Set|MutableSet|Callable|Iterable|Iterator)',
        r'collections.abc.\1'
    ),
    # distutils removed in Python 3.12
    (
        r'from distutils\.version import LooseVersion',
        'from packaging.version import Version as LooseVersion'
    ),
    (
        r'from distutils\.version import StrictVersion',
        'from packaging.version import Version as StrictVersion'
    ),
    (
        r'import distutils\.version',
        'import packaging.version'
    ),
]


def find_python_files(cache_dir: Path) -> list:
    """Find all Python files in the transformers_modules cache."""
    modules_dir = cache_dir / "modules" / "transformers_modules"

    if not modules_dir.exists():
        print(f"No transformers_modules found at {modules_dir}")
        return []

    python_files = []
    for py_file in modules_dir.rglob("*.py"):
        python_files.append(py_file)

    return python_files


def patch_file(filepath: Path, dry_run: bool = False) -> dict:
    """
    Apply compatibility patches to a Python file.

    Returns dict with patch results.
    """
    results = {
        "file": str(filepath),
        "patched": False,
        "changes": [],
    }

    try:
        content = filepath.read_text(encoding="utf-8")
        original_content = content

        for pattern, replacement in FIXES:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                results["changes"].append({
                    "pattern": pattern,
                    "matches": len(matches) if isinstance(matches[0], str) else len(matches),
                })

        if content != original_content:
            results["patched"] = True
            if not dry_run:
                filepath.write_text(content, encoding="utf-8")

    except Exception as e:
        results["error"] = str(e)

    return results


def get_default_cache_dir() -> Path:
    """Get the default HuggingFace cache directory."""
    # Check environment variable first
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home)

    # Check XDG cache
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "huggingface"

    # Default to ~/.cache/huggingface
    return Path.home() / ".cache" / "huggingface"


def main():
    parser = argparse.ArgumentParser(
        description="Patch HuggingFace cached files for Python 3.12 compatibility"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: auto-detect)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Get cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_default_cache_dir()

    print(f"Python 3.12 Compatibility Patcher")
    print(f"=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Cache directory: {cache_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Find Python files
    python_files = find_python_files(cache_dir)

    if not python_files:
        print("No Python files found to patch.")
        return 0

    print(f"Found {len(python_files)} Python files to check")
    print()

    # Patch files
    patched_count = 0
    error_count = 0

    for py_file in python_files:
        result = patch_file(py_file, dry_run=args.dry_run)

        if result.get("error"):
            error_count += 1
            print(f"ERROR: {py_file.name}: {result['error']}")

        elif result["patched"]:
            patched_count += 1
            action = "Would patch" if args.dry_run else "Patched"
            print(f"{action}: {py_file.relative_to(cache_dir)}")

            if args.verbose:
                for change in result["changes"]:
                    print(f"  - {change['pattern'][:50]}... ({change['matches']} matches)")

        elif args.verbose:
            print(f"OK: {py_file.name}")

    # Summary
    print()
    print(f"Summary:")
    print(f"  Files checked: {len(python_files)}")
    print(f"  Files {'needing patches' if args.dry_run else 'patched'}: {patched_count}")
    print(f"  Errors: {error_count}")

    if patched_count > 0 and args.dry_run:
        print()
        print("Run without --dry-run to apply patches.")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
