"""Provide stable helpers for the installed Streamlit application test suite.

Purpose
-------
The application-wide tests must work both from the strike package and after the
files are installed in the project. These helpers resolve the project root,
read source files, parse Python safely, and load modules without hiding errors.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def project_root() -> Path:
    """Return the audited project root from the environment or test location."""

    configured = os.environ.get("STREAMLIT_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[1]
    if not (root / "app").is_dir():
        raise FileNotFoundError(f"Streamlit project root is invalid: {root}")
    return root


def source_text(relative: str) -> str:
    """Read one required UTF-8 project file by its project-relative path."""

    path = project_root() / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required project file is missing: {relative}")
    return path.read_text(encoding="utf-8")


def source_tree(relative: str) -> ast.AST:
    """Parse one project Python file and return its syntax tree."""

    return ast.parse(source_text(relative), filename=relative)


def import_project_module(name: str) -> ModuleType:
    """Import one project module after placing the project root on sys.path."""

    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def python_files() -> tuple[Path, ...]:
    """Return every application Python file in stable order."""

    return tuple(sorted((project_root() / "app").rglob("*.py")))


def assert_module_documented(relative: str) -> None:
    """Require module, class, and function docstrings in one source file."""

    tree = source_tree(relative)
    if not ast.get_docstring(tree):
        raise AssertionError(f"Module docstring is missing: {relative}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                raise AssertionError(
                    f"Docstring is missing for {node.name} in {relative}"
                )
