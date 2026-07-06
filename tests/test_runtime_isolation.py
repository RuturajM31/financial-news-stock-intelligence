"""Focused tests for native-runtime-specific project regression isolation."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def load_runtime_runner() -> object:
    """Load the installed runtime-isolation script as one test module."""

    script_path = (
        Path(__file__).resolve().parents[1]
        / 'scripts'
        / 'run_isolated_project_regression.py'
    )
    spec = importlib.util.spec_from_file_location(
        'movement_runtime_isolation_for_tests',
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load runtime-isolation script.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_executable(path: Path) -> None:
    """Create a tiny executable file for environment-selection tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
    os.chmod(path, 0o755)


def test_discover_test_files_returns_sorted_regular_files(
    tmp_path: Path,
) -> None:
    """Prepare test files, run discovery, and check deterministic ordering."""

    # Prepare data.
    runner = load_runtime_runner()
    tests_root = tmp_path / 'tests'
    nested = tests_root / 'nested'
    nested.mkdir(parents=True)
    (tests_root / 'test_z.py').write_text('def test_z(): pass\n')
    (nested / 'test_a.py').write_text('def test_a(): pass\n')
    # Run function.
    result = runner.discover_test_files(tests_root)
    # Check result.
    assert [path.name for path in result] == ['test_a.py', 'test_z.py']


def test_discover_test_files_rejects_empty_directory(
    tmp_path: Path,
) -> None:
    """Prepare an empty directory, run discovery, and check rejection."""

    # Prepare data.
    runner = load_runtime_runner()
    tests_root = tmp_path / 'tests'
    tests_root.mkdir()
    # Run function and check result.
    with pytest.raises(runner.IsolatedRegressionError, match='No test files'):
        runner.discover_test_files(tests_root)


def test_parse_junit_counts_calculates_passed_tests(
    tmp_path: Path,
) -> None:
    """Prepare JUnit evidence, run parsing, and check aggregate counts."""

    # Prepare data.
    runner = load_runtime_runner()
    report = tmp_path / 'report.xml'
    report.write_text(
        '<testsuite tests="5" failures="1" errors="0" skipped="1" />',
        encoding='utf-8',
    )
    # Run function.
    counts = runner.parse_junit_counts(report)
    # Check result.
    assert counts.tests == 5
    assert counts.passed == 3
    assert counts.failures == 1
    assert counts.skipped == 1


def test_mixed_openmp_warning_is_detected() -> None:
    """Prepare warning output, run detection, and check fail-closed marker."""

    # Prepare data.
    runner = load_runtime_runner()
    output = (
        "Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') "
        'loaded at the same time.'
    )
    # Run function.
    detected = runner.contains_mixed_openmp_warning(output)
    # Check result.
    assert detected is True


def test_bert_test_uses_dedicated_transformer_environment(
    tmp_path: Path,
) -> None:
    """Prepare both environments, select BERT runtime, and check routing."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    main_python = project_root / '.venv/bin/python'
    transformer_python = project_root / '.venv-distilbert/bin/python'
    make_executable(main_python)
    make_executable(transformer_python)
    bert_test = project_root / 'tests/test_bert_smoke_runner.py'
    # Run function.
    selected = runner.select_python_for_test(
        project_root,
        main_python,
        bert_test,
    )
    # Check result.
    assert selected.python_path == transformer_python
    assert selected.profile_name == 'transformer'


def test_regular_test_uses_main_analytics_environment(
    tmp_path: Path,
) -> None:
    """Prepare both environments, select regular runtime, and check routing."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    main_python = project_root / '.venv/bin/python'
    transformer_python = project_root / '.venv-distilbert/bin/python'
    make_executable(main_python)
    make_executable(transformer_python)
    regular_test = project_root / 'tests/test_movement_pipeline.py'
    # Run function.
    selected = runner.select_python_for_test(
        project_root,
        main_python,
        regular_test,
    )
    # Check result.
    assert selected.python_path == main_python
    assert selected.profile_name == 'analytics'


def test_run_test_file_executes_in_fresh_process(
    tmp_path: Path,
) -> None:
    """Prepare one passing test, run isolation, and check JUnit evidence."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    tests_root = project_root / 'tests'
    tests_root.mkdir(parents=True)
    test_file = tests_root / 'test_one.py'
    test_file.write_text(
        'def test_one():\n    assert 2 + 2 == 4\n',
        encoding='utf-8',
    )
    report = tmp_path / 'result.xml'
    # Run function.
    counts, output = runner.run_test_file(
        Path(sys.executable),
        project_root,
        test_file,
        report,
    )
    # Check result.
    assert counts.tests == 1
    assert counts.passed == 1
    assert 'passed' in output


def test_pytest_support_bundle_excludes_native_model_packages(
    tmp_path: Path,
) -> None:
    """Prepare module sources, Run the copy, and Check the safety boundary."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    project_root.mkdir()
    destination = tmp_path / 'pytest_support'
    # Run function.
    support_root = runner.build_pytest_support_directory(
        project_root,
        Path(sys.executable),
        destination,
    )
    # Check result.
    copied_names = {path.name for path in support_root.iterdir()}
    assert 'pytest' in copied_names
    assert '_pytest' in copied_names
    assert copied_names.isdisjoint(
        {'numpy', 'pandas', 'sklearn', 'torch', 'transformers'}
    )
    assert not any(
        path.suffix.lower() in {'.so', '.dylib', '.dll', '.pyd'}
        for path in support_root.rglob('*')
        if path.is_file()
    )


def test_transformer_test_uses_safe_pytest_bridge_without_sklearn(
    tmp_path: Path,
) -> None:
    """Prepare a no-site Python, Run the bridge, and Check a clean pass."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    tests_root = project_root / 'tests'
    tests_root.mkdir(parents=True)
    test_file = tests_root / 'test_bert_bridge.py'
    test_file.write_text(
        'def test_bridge():\n    assert 3 * 7 == 21\n',
        encoding='utf-8',
    )
    transformer_python = project_root / '.venv-distilbert/bin/python'
    transformer_python.parent.mkdir(parents=True)
    transformer_python.write_text(
        '#!/bin/sh\n'
        f'exec "{sys.executable}" -S "$@"\n',
        encoding='utf-8',
    )
    os.chmod(transformer_python, 0o755)
    support_root = runner.build_pytest_support_directory(
        project_root,
        Path(sys.executable),
        tmp_path / 'pytest_support',
    )
    report = tmp_path / 'bridge.xml'
    # Run function.
    counts, output = runner.run_test_file(
        transformer_python,
        project_root,
        test_file,
        report,
        support_root,
    )
    # Check result.
    assert counts.tests == 1
    assert counts.passed == 1
    assert 'passed' in output


def test_generic_transformer_name_is_ambiguous_not_native() -> None:
    """Prepare a dataset test, Run classification, and Check ambiguity."""

    # Prepare data.
    runner = load_runtime_runner()
    test_file = Path('tests/test_transformer_dataset.py')
    # Run function.
    native = runner.requires_transformer_runtime(test_file)
    ambiguous = runner.has_ambiguous_runtime_name(test_file)
    # Check result.
    assert native is False
    assert ambiguous is True


def test_ambiguous_test_candidates_start_with_analytics(
    tmp_path: Path,
) -> None:
    """Prepare both runtimes, Run candidate selection, and Check ordering."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    main_python = project_root / '.venv/bin/python'
    transformer_python = project_root / '.venv-distilbert/bin/python'
    make_executable(main_python)
    make_executable(transformer_python)
    test_file = project_root / 'tests/test_transformer_dataset.py'
    # Run function.
    candidates = runner.runtime_candidates_for_test(
        project_root,
        main_python,
        test_file,
    )
    # Check result.
    assert [item.profile_name for item in candidates] == [
        'analytics',
        'transformer',
    ]


def test_runtime_choice_falls_back_after_collection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare two probes, Run selection, and Check safe fallback."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    main_python = project_root / '.venv/bin/python'
    transformer_python = project_root / '.venv-distilbert/bin/python'
    make_executable(main_python)
    make_executable(transformer_python)
    test_file = project_root / 'tests/test_transformer_dataset.py'

    def fake_probe(selection, *_args):
        """Return one failed analytics probe and one passing fallback."""

        if selection.profile_name == 'analytics':
            return False, 'missing optional analytics dependency'
        return True, 'collection passed'

    monkeypatch.setattr(runner, 'probe_test_collection', fake_probe)
    # Run function.
    selected = runner.choose_runtime_for_test(
        project_root,
        main_python,
        test_file,
        None,
    )
    # Check result.
    assert selected.profile_name == 'transformer'
    assert selected.python_path == transformer_python


def test_runtime_choice_rejects_when_all_collection_probes_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare failed probes, Run selection, and Check fail-closed result."""

    # Prepare data.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    main_python = project_root / '.venv/bin/python'
    transformer_python = project_root / '.venv-distilbert/bin/python'
    make_executable(main_python)
    make_executable(transformer_python)
    test_file = project_root / 'tests/test_transformer_dataset.py'

    def failed_probe(selection, *_args):
        """Return a deterministic failed probe for every candidate."""

        return False, f'{selection.profile_name} collection failed'

    monkeypatch.setattr(runner, 'probe_test_collection', failed_probe)
    # Run function and check result.
    with pytest.raises(
        runner.IsolatedRegressionError,
        match='No safe runtime could collect',
    ):
        runner.choose_runtime_for_test(
            project_root,
            main_python,
            test_file,
            None,
        )


def test_ambiguous_transformer_dataset_uses_pydantic_capable_runtime(
    tmp_path: Path,
) -> None:
    """Prepare the reported failure, Run probing, and Check analytics wins."""

    # Prepare data: the analytics interpreter has Pydantic, while the dedicated
    # Transformer wrapper starts without site-packages and receives pytest only.
    runner = load_runtime_runner()
    project_root = tmp_path / 'project'
    tests_root = project_root / 'tests'
    tests_root.mkdir(parents=True)
    test_file = tests_root / 'test_transformer_dataset.py'
    test_file.write_text(
        'from pydantic import BaseModel\n\n'
        'class Row(BaseModel):\n'
        '    value: int\n\n'
        'def test_row():\n'
        '    assert Row(value=7).value == 7\n',
        encoding='utf-8',
    )

    main_python = project_root / '.venv/bin/python'
    main_python.parent.mkdir(parents=True)
    main_python.write_text(
        '#!/bin/sh\n'
        f'exec "{sys.executable}" "$@"\n',
        encoding='utf-8',
    )
    os.chmod(main_python, 0o755)

    transformer_python = project_root / '.venv-distilbert/bin/python'
    transformer_python.parent.mkdir(parents=True)
    transformer_python.write_text(
        '#!/bin/sh\n'
        f'exec "{sys.executable}" -S "$@"\n',
        encoding='utf-8',
    )
    os.chmod(transformer_python, 0o755)
    support_root = runner.build_pytest_support_directory(
        project_root,
        main_python,
        tmp_path / 'pytest_support',
    )
    # Run function.
    selected = runner.choose_runtime_for_test(
        project_root,
        main_python,
        test_file,
        support_root,
    )
    # Check result.
    assert selected.profile_name == 'analytics'
    assert selected.python_path == main_python
