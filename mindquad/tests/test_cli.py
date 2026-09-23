"""Tests for the Mindquad CLI wrapper and auto-unlock behavior."""

import argparse
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch

import yaml

from mindquad.cli import (
    build_snakemake_command,
    build_unlock_command,
    check_if_lock_error,
    get_snakemake_working_directory,
    is_directory_locked,
    load_makefile_arguments,
    main,
    unlock_working_directory,
)


def test_load_makefile_arguments_dict(tmp_path: Path) -> None:
    """Test loading arguments from a dictionary YAML makefile."""
    makefile_file = tmp_path / "args.yaml"
    makefile_content = {
        "dry-run": True,
        "rerun-incomplete": True,
        "latency-wait": 60,
        "verbose": False,
    }
    makefile_file.write_text(yaml.dump(makefile_content), encoding="utf-8")

    parsed_args = load_makefile_arguments(str(makefile_file))
    assert "--dry-run" in parsed_args
    assert "--rerun-incomplete" in parsed_args
    assert "--latency-wait" in parsed_args
    assert "60" in parsed_args
    assert "--verbose" not in parsed_args


def test_load_makefile_arguments_list(tmp_path: Path) -> None:
    """Test loading arguments from a list YAML makefile."""
    makefile_file = tmp_path / "args_list.yaml"
    makefile_content = ["--dry-run", "--printshellcmds"]
    makefile_file.write_text(yaml.dump(makefile_content), encoding="utf-8")

    parsed_args = load_makefile_arguments(str(makefile_file))
    assert parsed_args == ["--dry-run", "--printshellcmds"]


def test_get_snakemake_working_directory(tmp_path: Path) -> None:
    """Test resolving working directory from extra arguments."""
    custom_dir = tmp_path / "custom_work"
    custom_dir.mkdir()

    # Default to current working directory when not provided
    default_dir = get_snakemake_working_directory([])
    assert default_dir == Path.cwd().resolve()

    # Parsed from --directory
    resolved_dir_flag = get_snakemake_working_directory(
        ["--directory", str(custom_dir)]
    )
    assert resolved_dir_flag == custom_dir.resolve()

    # Parsed from -d
    resolved_short_flag = get_snakemake_working_directory(
        ["-d", str(custom_dir)]
    )
    assert resolved_short_flag == custom_dir.resolve()

    # Parsed from --directory=...
    resolved_equals_flag = get_snakemake_working_directory(
        [f"--directory={custom_dir}"]
    )
    assert resolved_equals_flag == custom_dir.resolve()


def test_is_directory_locked(tmp_path: Path) -> None:
    """Test detection of active Snakemake directory locks."""
    # When .snakemake does not exist
    assert not is_directory_locked(tmp_path)

    snakemake_dir = tmp_path / ".snakemake"
    locks_dir = snakemake_dir / "locks"
    locks_dir.mkdir(parents=True)

    # When locks dir is empty
    assert not is_directory_locked(tmp_path)

    # When lock file exists
    lock_file = locks_dir / "0.output.lock"
    lock_file.write_text("/some/output/file", encoding="utf-8")
    assert is_directory_locked(tmp_path)

    # When lock file is removed
    lock_file.unlink()
    assert not is_directory_locked(tmp_path)


def test_check_if_lock_error(tmp_path: Path) -> None:
    """Test checking logs for lock error signatures."""
    # If lock directory contains lock files, immediately returns True
    locks_dir = tmp_path / ".snakemake" / "locks"
    locks_dir.mkdir(parents=True)
    lock_file = locks_dir / "0.output.lock"
    lock_file.write_text("/test/target", encoding="utf-8")
    assert check_if_lock_error(tmp_path)

    # Remove lock file, check via log file inspection
    lock_file.unlink()
    log_dir = tmp_path / ".snakemake" / "log"
    log_dir.mkdir(parents=True)

    log_file = log_dir / "2026-09-23T100000.snakemake.log"
    log_file.write_text(
        "LockException:\nError: Directory cannot be locked.\n"
        "It can be removed with the --unlock argument.\n",
        encoding="utf-8",
    )
    assert check_if_lock_error(tmp_path)

    # Unrelated log file does not trigger lock detection
    log_file.unlink()
    unrelated_log = log_dir / "2026-09-23T110000.snakemake.log"
    unrelated_log.write_text(
        "RuleException: Command failed with exit status 1\n",
        encoding="utf-8",
    )
    assert not check_if_lock_error(tmp_path)


def test_build_unlock_command() -> None:
    """Test building the Snakemake unlock command."""
    mock_args = argparse.Namespace(
        config="/path/to/config.yaml",
        cores=4,
    )
    snakefile_path = "/path/to/Snakefile"

    command = build_unlock_command(
        args=mock_args,
        snakefile_path=snakefile_path,
        extra_arguments=["--directory", "/custom/workdir"],
    )
    assert command == [
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        "/path/to/config.yaml",
        "--unlock",
        "--directory",
        "/custom/workdir",
    ]


def test_build_snakemake_command() -> None:
    """Test building the main Snakemake execution command."""
    mock_args = argparse.Namespace(
        config="/path/to/config.yaml",
        cores=8,
        submit="slurm",
        go=True,
        unlock=False,
        makefile=None,
    )
    snakefile_path = "/path/to/Snakefile"

    command = build_snakemake_command(
        args=mock_args,
        snakefile_path=snakefile_path,
        extra_arguments=["--rerun-incomplete"],
    )
    expected_command = [
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        "/path/to/config.yaml",
        "--cores",
        "8",
        "--profile",
        "slurm",
        "--rerun-triggers",
        "mtime",
        "--rerun-incomplete",
    ]
    assert command == expected_command


def test_unlock_working_directory_success() -> None:
    """Test unlock_working_directory when subprocess succeeds."""
    mock_args = argparse.Namespace(
        config="/path/to/config.yaml",
    )
    with patch("subprocess.run") as mock_subprocess_run:
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        success = unlock_working_directory(
            args=mock_args,
            snakefile_path="/path/to/Snakefile",
        )
        assert success is True
        mock_subprocess_run.assert_called_once()


def test_unlock_working_directory_failure() -> None:
    """Test unlock_working_directory when subprocess fails."""
    mock_args = argparse.Namespace(
        config="/path/to/config.yaml",
    )
    with patch("subprocess.run") as mock_subprocess_run:
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["snakemake", "--unlock"],
        )
        success = unlock_working_directory(
            args=mock_args,
            snakefile_path="/path/to/Snakefile",
        )
        assert success is False


def test_main_preemptive_auto_unlock(tmp_path: Path) -> None:
    """Test main() auto-unlocking preemptively when locks are detected."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy: true", encoding="utf-8")

    mock_args = argparse.Namespace(
        config=str(config_file),
        cores=2,
        submit=None,
        makefile=None,
        go=False,
        unlock=False,
    )

    with patch("mindquad.cli.parse_arguments", return_value=mock_args), \
         patch("mindquad.cli.is_directory_locked", return_value=True), \
         patch("mindquad.cli.unlock_working_directory") as mock_unlock, \
         patch("subprocess.run") as mock_subprocess_run:

        mock_subprocess_run.return_value = MagicMock(returncode=0)
        main()

        # Both unlock and execution must have been called
        mock_unlock.assert_called_once()
        mock_subprocess_run.assert_called_once()


def test_main_reactive_auto_unlock(tmp_path: Path) -> None:
    """Test main() auto-unlocking when execution hits a lock error."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dummy: true", encoding="utf-8")

    mock_args = argparse.Namespace(
        config=str(config_file),
        cores=2,
        submit=None,
        makefile=None,
        go=False,
        unlock=False,
    )

    first_failure = subprocess.CalledProcessError(
        returncode=1,
        cmd=["snakemake"],
    )
    retry_success = MagicMock(returncode=0)

    with patch("mindquad.cli.parse_arguments", return_value=mock_args), \
         patch("mindquad.cli.is_directory_locked", return_value=False), \
         patch("mindquad.cli.check_if_lock_error", return_value=True), \
         patch("mindquad.cli.unlock_working_directory") as mock_unlock, \
         patch("subprocess.run") as mock_run:

        mock_unlock.return_value = True
        mock_run.side_effect = [first_failure, retry_success]

        main()

        mock_unlock.assert_called_once()
        assert mock_run.call_count == 2
