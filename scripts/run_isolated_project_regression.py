#!/usr/bin/env python3
"""Run project tests in native-runtime-specific isolated processes.

Purpose
-------
Keep complete project regression coverage without loading the movement analytics
stack and the Transformer stack into the same Python interpreter. Native BERT,
DistilBERT, and LoRA tests use ``.venv-distilbert``. Ambiguous files whose names
contain ``transformer`` are collected in both safe candidate environments and
run only in the first environment that collects without an OpenMP conflict.

Inputs
------
- ``--project-root``: project directory containing tests and both environments.
- ``--phase``: a readable label such as ``pre-run`` or ``post-run``.
- The test filename, which determines the required native-runtime profile.

Logic
-----
1. Discover every regular ``test_*.py`` file below the tests directory.
2. Route native Transformer training tests to ``.venv-distilbert``.
3. Route analytics tests to the verified main ``.venv`` environment.
4. Probe ambiguous ``transformer`` files with collection-only subprocesses.
5. Select only an environment that collects without mixed OpenMP runtimes.
6. Build a temporary pure-Python pytest support bundle only when needed.
7. Run every test file in a fresh process with its selected environment.
8. Parse one temporary JUnit report per file and fail closed on any defect.

Outputs and downstream use
--------------------------
The script prints each file, selected runtime profile, pytest output, and one
aggregate result. The movement suite uses its exit code as a hard regression
gate before and after model execution.

Assumptions and limitations
---------------------------
The project already contains the dedicated ``.venv-distilbert`` environment
created for its verified Transformer benchmark. Scikit-learn and Pydantic are
not required there. A generic ``transformer`` filename is not enough to prove
that the test needs the native Transformer stack, so those files are probed
before execution. Mixed OpenMP warnings are never hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MIXED_OPENMP_MARKERS = (
    "Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp')",
    "Incompatible OpenMP runtime families loaded",
)
TRANSFORMER_TEST_NAME_MARKERS = (
    "bert",
    "distilbert",
    "lora",
)
AMBIGUOUS_RUNTIME_TEST_NAME_MARKERS = (
    "transformer",
)
TRANSFORMER_PYTHON_RELATIVE = Path('.venv-distilbert/bin/python')
PYTEST_SUPPORT_MODULES = (
    'pytest',
    '_pytest',
    'pluggy',
    'packaging',
    'iniconfig',
    'pygments',
    'py',
    'exceptiongroup',
    'tomli',
    'typing_extensions',
)
REQUIRED_PYTEST_SUPPORT_MODULES = {
    'pytest',
    '_pytest',
    'pluggy',
    'packaging',
    'iniconfig',
    'pygments',
    'py',
}
FORBIDDEN_PYTEST_SUPPORT_NAMES = {
    'numpy',
    'pandas',
    'sklearn',
    'torch',
    'transformers',
}


class IsolatedRegressionError(RuntimeError):
    """Raised when isolated regression execution or evidence is invalid."""


@dataclass(frozen=True)
class TestCounts:
    """Store aggregate JUnit counts for one test file or the full run."""

    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def passed(self) -> int:
        """Return tests that completed without failure, error, or skip."""

        return self.tests - self.failures - self.errors - self.skipped

    def add(self, other: 'TestCounts') -> 'TestCounts':
        """Return a new count object containing both count sets."""

        return TestCounts(
            tests=self.tests + other.tests,
            failures=self.failures + other.failures,
            errors=self.errors + other.errors,
            skipped=self.skipped + other.skipped,
        )


@dataclass(frozen=True)
class RuntimeSelection:
    """Describe the Python executable selected for one project test file."""

    python_path: Path
    profile_name: str


def discover_test_files(tests_root: Path) -> list[Path]:
    """Return sorted regular test files while rejecting unsafe paths."""

    if (
        not tests_root.exists()
        or tests_root.is_symlink()
        or not tests_root.is_dir()
    ):
        raise IsolatedRegressionError(
            f'Tests directory is missing or unsafe: {tests_root}'
        )

    # Discovery reads filenames only. Importing project tests in this controller
    # would defeat the native-runtime separation enforced by the subprocesses.
    files: list[Path] = []
    for path in sorted(tests_root.rglob('test_*.py')):
        if path.is_symlink() or not path.is_file():
            raise IsolatedRegressionError(
                f'Test path is not a regular file: {path}'
            )
        files.append(path)
    if not files:
        raise IsolatedRegressionError(
            f'No test files were found below: {tests_root}'
        )
    return files


def parse_junit_counts(report_path: Path) -> TestCounts:
    """Parse one pytest JUnit report and return validated integer counts."""

    if (
        not report_path.exists()
        or report_path.is_symlink()
        or not report_path.is_file()
    ):
        raise IsolatedRegressionError(
            f'JUnit report is missing or unsafe: {report_path}'
        )
    try:
        root = ET.parse(report_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise IsolatedRegressionError(
            f'JUnit report is invalid: {report_path}: {exc}'
        ) from exc

    # Pytest can emit either a testsuite root or a testsuites wrapper.
    suites = [root] if root.tag == 'testsuite' else list(
        root.findall('testsuite')
    )
    if not suites:
        raise IsolatedRegressionError(
            f'JUnit report has no testsuite element: {report_path}'
        )

    counts = TestCounts()
    for suite in suites:
        try:
            current = TestCounts(
                tests=int(suite.attrib.get('tests', '0')),
                failures=int(suite.attrib.get('failures', '0')),
                errors=int(suite.attrib.get('errors', '0')),
                skipped=int(suite.attrib.get('skipped', '0')),
            )
        except ValueError as exc:
            raise IsolatedRegressionError(
                f'JUnit report has non-integer counts: {report_path}'
            ) from exc
        if min(
            current.tests,
            current.failures,
            current.errors,
            current.skipped,
        ) < 0:
            raise IsolatedRegressionError(
                f'JUnit report has negative counts: {report_path}'
            )
        if current.passed < 0:
            raise IsolatedRegressionError(
                f'JUnit report counts are inconsistent: {report_path}'
            )
        counts = counts.add(current)
    return counts


def contains_mixed_openmp_warning(output: str) -> bool:
    """Return whether subprocess output reports incompatible OpenMP families."""

    return any(marker in output for marker in MIXED_OPENMP_MARKERS)


def requires_transformer_runtime(test_file: Path) -> bool:
    """Return whether a filename explicitly names a native Transformer job."""

    filename = test_file.name.lower()
    return any(
        marker in filename for marker in TRANSFORMER_TEST_NAME_MARKERS
    )


def has_ambiguous_runtime_name(test_file: Path) -> bool:
    """Return whether a filename needs collection-based runtime selection."""

    filename = test_file.name.lower()
    return any(
        marker in filename
        for marker in AMBIGUOUS_RUNTIME_TEST_NAME_MARKERS
    ) and not requires_transformer_runtime(test_file)


def require_executable_python(python_path: Path, description: str) -> Path:
    """Return a safe executable Python path or fail before running tests."""

    if not python_path.exists():
        raise IsolatedRegressionError(
            f'Missing {description}: {python_path}'
        )
    try:
        resolved = python_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise IsolatedRegressionError(
            f'Broken {description}: {python_path}'
        ) from exc
    if not resolved.is_file() or not os.access(python_path, os.X_OK):
        raise IsolatedRegressionError(
            f'Unsafe or non-executable {description}: {python_path}'
        )
    return python_path


def select_python_for_test(
    project_root: Path,
    main_python: Path,
    test_file: Path,
) -> RuntimeSelection:
    """Select the preferred environment for one unambiguous test file."""

    if not requires_transformer_runtime(test_file):
        return RuntimeSelection(
            python_path=require_executable_python(
                main_python,
                'main analytics Python',
            ),
            profile_name='analytics',
        )

    # Native Transformer tests use the environment that already completed the
    # BERT and LoRA benchmarks. This keeps torch out of movement-model tests.
    transformer_python = project_root / TRANSFORMER_PYTHON_RELATIVE
    return RuntimeSelection(
        python_path=require_executable_python(
            transformer_python,
            'dedicated Transformer Python',
        ),
        profile_name='transformer',
    )


def runtime_candidates_for_test(
    project_root: Path,
    main_python: Path,
    test_file: Path,
) -> tuple[RuntimeSelection, ...]:
    """Return ordered safe runtime candidates for one test file.

    Generic Transformer dataset tests can depend on project schemas such as
    Pydantic without importing torch. They therefore start with analytics and
    fall back to the dedicated Transformer environment only after a clean
    collection probe proves that environment is suitable.
    """

    preferred = select_python_for_test(
        project_root,
        main_python,
        test_file,
    )
    if not has_ambiguous_runtime_name(test_file):
        return (preferred,)

    transformer = RuntimeSelection(
        python_path=require_executable_python(
            project_root / TRANSFORMER_PYTHON_RELATIVE,
            'dedicated Transformer Python',
        ),
        profile_name='transformer',
    )
    return (preferred, transformer)


def regression_environment(
    project_root: Path,
    python_path: Path,
) -> dict[str, str]:
    """Build deterministic thread and import settings for one test process."""

    environment = dict(os.environ)
    virtual_environment = python_path.parent.parent
    current_path = environment.get('PATH', '')
    environment.update(
        {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONHASHSEED': '42',
            'PYTHONPATH': str(project_root / 'src'),
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
            'VIRTUAL_ENV': str(virtual_environment),
            'PATH': str(python_path.parent) + os.pathsep + current_path,
        }
    )
    # A parent PYTHONHOME can force the wrong standard library or site-packages
    # into the selected virtual environment, so it is removed deterministically.
    environment.pop('PYTHONHOME', None)
    return environment


def python_has_pytest(
    project_root: Path,
    python_path: Path,
) -> bool:
    """Return whether one selected interpreter can import pytest directly."""

    completed = subprocess.run(
        [str(python_path), '-c', 'import pytest'],
        cwd=project_root,
        env=regression_environment(project_root, python_path),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def pytest_support_sources(
    project_root: Path,
    main_python: Path,
) -> dict[str, Path]:
    """Locate only the pure-Python modules needed to start pytest.

    The selected Transformer interpreter must not receive the main environment's
    entire site-packages directory because that can expose compiled analytics
    libraries and recreate the Intel/LLVM OpenMP conflict.
    """

    code = "\n".join(
        (
            'import importlib, json',
            f'names = {PYTEST_SUPPORT_MODULES!r}',
            'result = {}',
            'for name in names:',
            '    try:',
            '        module = importlib.import_module(name)',
            '    except ModuleNotFoundError:',
            '        continue',
            '    result[name] = module.__file__',
            'print(json.dumps(result, sort_keys=True))',
        )
    )
    completed = subprocess.run(
        [str(main_python), '-c', code],
        cwd=project_root,
        env=regression_environment(project_root, main_python),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise IsolatedRegressionError(
            f'Could not locate pure pytest support modules: {detail}'
        )
    try:
        raw_sources = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IsolatedRegressionError(
            'Pytest support inventory was not valid JSON.'
        ) from exc
    if not isinstance(raw_sources, dict):
        raise IsolatedRegressionError(
            'Pytest support inventory must be a JSON object.'
        )

    sources: dict[str, Path] = {}
    for name, raw_path in raw_sources.items():
        if not isinstance(name, str) or not isinstance(raw_path, str):
            raise IsolatedRegressionError(
                'Pytest support inventory contains invalid values.'
            )
        source = Path(raw_path).resolve()
        if source.is_symlink() or not source.is_file():
            raise IsolatedRegressionError(
                f'Pytest support module is missing or unsafe: {name}'
            )
        sources[name] = source

    missing = sorted(REQUIRED_PYTEST_SUPPORT_MODULES - set(sources))
    if missing:
        raise IsolatedRegressionError(
            'Required pytest support modules are missing: ' + ', '.join(missing)
        )
    return sources


def build_pytest_support_directory(
    project_root: Path,
    main_python: Path,
    destination: Path,
) -> Path:
    """Copy a minimal pure-Python pytest runtime into a temporary directory.

    Inputs
    ------
    ``main_python`` supplies pytest and its pure dependencies. ``destination``
    is deleted automatically with the isolated regression temporary directory.

    Safety and downstream use
    -------------------------
    Only named pytest support modules are copied. Compiled files and analytics or
    Transformer packages are rejected. The dedicated Transformer interpreter can
    then collect and execute tests without importing the main environment.
    """

    if destination.exists():
        raise IsolatedRegressionError(
            f'Pytest support destination already exists: {destination}'
        )
    destination.mkdir(parents=True, mode=0o700)

    for name, source_file in pytest_support_sources(
        project_root,
        main_python,
    ).items():
        if source_file.name == '__init__.py':
            source = source_file.parent
            target = destination / source.name
            shutil.copytree(source, target)
        else:
            source = source_file
            target = destination / source.name
            shutil.copy2(source, target)

    copied_names = {path.name for path in destination.iterdir()}
    forbidden = sorted(FORBIDDEN_PYTEST_SUPPORT_NAMES & copied_names)
    if forbidden:
        raise IsolatedRegressionError(
            'Pytest support copied forbidden packages: ' + ', '.join(forbidden)
        )

    compiled_suffixes = {'.so', '.dylib', '.dll', '.pyd'}
    compiled_files = [
        path
        for path in destination.rglob('*')
        if path.is_file() and path.suffix.lower() in compiled_suffixes
    ]
    if compiled_files:
        raise IsolatedRegressionError(
            'Pytest support unexpectedly contains compiled libraries.'
        )
    return destination


def pytest_command(
    project_root: Path,
    selected_python: Path,
    relative_test: Path,
    report_path: Path | None,
    pytest_support_root: Path | None,
    *,
    collect_only: bool = False,
) -> list[str]:
    """Build a pytest command without exposing another environment's packages."""

    pytest_arguments = ['-q']
    if collect_only:
        pytest_arguments.append('--collect-only')
    pytest_arguments.append(str(relative_test))
    if report_path is not None:
        pytest_arguments.append(f'--junitxml={report_path}')
    if python_has_pytest(project_root, selected_python):
        return [str(selected_python), '-m', 'pytest', *pytest_arguments]
    if pytest_support_root is None:
        raise IsolatedRegressionError(
            'Selected runtime has no pytest and no safe support bundle.'
        )

    bootstrap = (
        "import sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import pytest; "
        "raise SystemExit(pytest.main(sys.argv[2:]))"
    )
    return [
        str(selected_python),
        '-c',
        bootstrap,
        str(pytest_support_root),
        *pytest_arguments,
    ]


def probe_test_collection(
    selection: RuntimeSelection,
    project_root: Path,
    test_file: Path,
    pytest_support_root: Path | None,
) -> tuple[bool, str]:
    """Collect one test in a fresh process and report runtime suitability.

    A successful probe means imports completed, tests were discovered, and no
    Intel/LLVM OpenMP conflict was reported. The probe never executes tests and
    therefore cannot convert a real assertion failure into a fallback.
    """

    relative = test_file.relative_to(project_root)
    command = pytest_command(
        project_root,
        selection.python_path,
        relative,
        None,
        pytest_support_root,
        collect_only=True,
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=regression_environment(project_root, selection.python_path),
        check=False,
        capture_output=True,
        text=True,
    )
    output = '\n'.join(
        value for value in (completed.stdout, completed.stderr) if value
    ).strip()
    if contains_mixed_openmp_warning(output):
        return False, 'mixed OpenMP runtimes were reported during collection'
    if completed.returncode != 0:
        final_line = output.splitlines()[-1] if output else 'collection failed'
        return False, final_line
    return True, output


def choose_runtime_for_test(
    project_root: Path,
    main_python: Path,
    test_file: Path,
    pytest_support_root: Path | None,
) -> RuntimeSelection:
    """Choose a runtime without masking collection or native-library defects."""

    candidates = runtime_candidates_for_test(
        project_root,
        main_python,
        test_file,
    )
    if len(candidates) == 1:
        return candidates[0]

    failures: list[str] = []
    for selection in candidates:
        usable, detail = probe_test_collection(
            selection,
            project_root,
            test_file,
            pytest_support_root,
        )
        if usable:
            return selection
        failures.append(f'{selection.profile_name}: {detail}')

    relative = test_file.relative_to(project_root)
    raise IsolatedRegressionError(
        f'No safe runtime could collect {relative}: ' + '; '.join(failures)
    )


def run_test_file(
    python_path: Path,
    project_root: Path,
    test_file: Path,
    report_path: Path,
    pytest_support_root: Path | None = None,
) -> tuple[TestCounts, str]:
    """Run one test file in a fresh interpreter and validate its evidence."""

    relative = test_file.relative_to(project_root)
    command = pytest_command(
        project_root,
        python_path,
        relative,
        report_path,
        pytest_support_root,
    )
    # Capture both streams so a successful pytest status cannot hide a native
    # runtime conflict emitted in the warning summary.
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=regression_environment(project_root, python_path),
        check=False,
        capture_output=True,
        text=True,
    )
    combined_output = '\n'.join(
        value for value in (completed.stdout, completed.stderr) if value
    ).strip()
    if combined_output:
        print(combined_output, flush=True)

    if contains_mixed_openmp_warning(combined_output):
        raise IsolatedRegressionError(
            f'Mixed OpenMP runtimes were reported by: {relative}'
        )
    if completed.returncode != 0:
        raise IsolatedRegressionError(
            f'Regression test file failed: {relative}'
        )

    counts = parse_junit_counts(report_path)
    if counts.tests <= 0:
        raise IsolatedRegressionError(
            f'No tests were collected from: {relative}'
        )
    if counts.failures or counts.errors:
        raise IsolatedRegressionError(
            f'JUnit evidence reports failures for: {relative}'
        )
    return counts, combined_output


def run_regression(
    project_root: Path,
    main_python: Path,
    phase: str,
) -> TestCounts:
    """Run every discovered test file and return the aggregate counts."""

    tests_root = project_root / 'tests'
    test_files = discover_test_files(tests_root)
    aggregate = TestCounts()

    print('Isolated regression phase:', phase)
    print('Discovered test files:', len(test_files))
    # JUnit files live in a controlled temporary directory and are removed
    # automatically after aggregate counts have been validated.
    with tempfile.TemporaryDirectory(
        prefix='movement_isolated_regression_'
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        transformer_tests = [
            path
            for path in test_files
            if requires_transformer_runtime(path)
            or has_ambiguous_runtime_name(path)
        ]
        pytest_support_root: Path | None = None
        if transformer_tests:
            transformer_python = require_executable_python(
                project_root / TRANSFORMER_PYTHON_RELATIVE,
                'dedicated Transformer Python',
            )
            if not python_has_pytest(project_root, transformer_python):
                pytest_support_root = build_pytest_support_directory(
                    project_root,
                    main_python,
                    temporary_root / 'pytest_support',
                )

        for index, test_file in enumerate(test_files, start=1):
            relative = test_file.relative_to(project_root)
            selection = choose_runtime_for_test(
                project_root,
                main_python,
                test_file,
                pytest_support_root,
            )
            print(
                f'[{index}/{len(test_files)}] {relative.as_posix()} '
                f'[{selection.profile_name}]',
                flush=True,
            )
            report_path = temporary_root / f'pytest_{index:04d}.xml'
            counts, _ = run_test_file(
                selection.python_path,
                project_root,
                test_file,
                report_path,
                pytest_support_root,
            )
            aggregate = aggregate.add(counts)

    if aggregate.tests <= 0 or aggregate.failures or aggregate.errors:
        raise IsolatedRegressionError(
            'Aggregate isolated regression evidence is invalid.'
        )
    print(
        'ISOLATED PROJECT REGRESSION: PASSED '
        f'({aggregate.passed} passed, '
        f'{aggregate.skipped} skipped, '
        f'{aggregate.tests} total)'
    )
    return aggregate


def parse_arguments(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for one isolated regression run."""

    parser = argparse.ArgumentParser(
        description='Run each pytest file in a native-runtime-specific process.'
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        required=True,
        help='Project root containing environments and tests.',
    )
    parser.add_argument(
        '--phase',
        default='project-regression',
        help='Human-readable phase included in output.',
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    """Validate paths, run isolated regression, and return a shell status."""

    args = parse_arguments(argv)
    project_root = args.project_root.expanduser().resolve()
    main_python = project_root / '.venv/bin/python'
    if not project_root.is_dir() or project_root.is_symlink():
        raise IsolatedRegressionError(
            f'Project root is missing or unsafe: {project_root}'
        )
    require_executable_python(main_python, 'main analytics Python')
    run_regression(project_root, main_python, str(args.phase))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except IsolatedRegressionError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
