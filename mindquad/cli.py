"""Command line interface for the mindquad pipeline."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import List, Optional
import yaml


def ascii_art() -> str:
    """Return mindquad ascii banner."""
    pink_code = '\033[95m'
    reset_code = '\033[0m'
    return f"{pink_code}" + r"""
    MINDQUAD: Neuroimaging Analysis Pipeline
    for fMRI, MRS, Structural and Diffusion data
    ===========================================
          _---~~(~~-_.
        _{        )   )
      ,   ) -~~- ( ,-' )_
     (  `-,_..`., )-- '_,)
    ( ` _)  (  -~( -_ `,  }
    (_-  _  ~_-~~~~`,  ,' )
      `~ -^(    __;-,((()))
            ~~~~ {_ -_(())
                   `\  }
                     { }

""" + f"{reset_code}"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=ascii_art(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="Path to the pipeline configuration YAML file.",
    )
    parser.add_argument(
        "-n",
        "--cores",
        type=int,
        default=1,
        help="Number of cores to use.",
    )
    parser.add_argument(
        "-s",
        "--submit",
        type=str,
        help="HPC config file (Snakemake profile).",
    )
    parser.add_argument(
        "-m",
        "--makefile",
        type=str,
        help="YAML file containing additional snakemake arguments.",
    )
    parser.add_argument(
        "-g",
        "--go",
        action="store_true",
        default=False,
        help=(
            "Ignore code/param changes and only trigger reruns "
            "based on file modification times."
        ),
    )
    parser.add_argument(
        "-u",
        "--unlock",
        action="store_true",
        default=False,
        help="Unlock the working directory.",
    )

    return parser.parse_args()


def load_makefile_arguments(makefile_path: str) -> List[str]:
    """Load additional Snakemake arguments from a YAML file.

    Args:
        makefile_path: Path to the YAML makefile.

    Returns:
        List[str]: Parsed arguments for Snakemake.
    """
    if not os.path.isfile(makefile_path):
        print(
            f"Error: Additional arguments file not found: {makefile_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(makefile_path, "r", encoding="utf-8") as file_handle:
        try:
            parsed_data = yaml.safe_load(file_handle)
        except yaml.YAMLError as yaml_parse_error:
            print(
                f"Error parsing YAML makefile: {yaml_parse_error}",
                file=sys.stderr,
            )
            sys.exit(1)

    extra_arguments: List[str] = []
    if isinstance(parsed_data, list):
        extra_arguments = [
            str(argument_item) for argument_item in parsed_data
        ]
    elif isinstance(parsed_data, dict):
        for argument_key, argument_value in parsed_data.items():
            if isinstance(argument_value, bool):
                if argument_value:
                    extra_arguments.append(f"--{argument_key}")
            else:
                extra_arguments.extend(
                    [f"--{argument_key}", str(argument_value)]
                )

    return extra_arguments


def get_snakemake_working_directory(
    extra_arguments: Optional[List[str]] = None,
) -> Path:
    """Determine the Snakemake working directory from CLI arguments.

    Args:
        extra_arguments: Additional CLI flags passed to Snakemake.

    Returns:
        Path: Resolved directory path for the Snakemake execution.
    """
    if extra_arguments:
        for argument_index, argument_item in enumerate(extra_arguments):
            if argument_item in ("--directory", "-d"):
                if argument_index + 1 < len(extra_arguments):
                    target_directory = extra_arguments[argument_index + 1]
                    return Path(target_directory).resolve()
            elif argument_item.startswith("--directory="):
                target_directory = argument_item.split("=", 1)[1]
                return Path(target_directory).resolve()

    return Path.cwd().resolve()


def is_directory_locked(working_directory: Path) -> bool:
    """Check if the Snakemake working directory contains active lock files.

    Args:
        working_directory: Directory containing the .snakemake folder.

    Returns:
        bool: True if lock files are detected, False otherwise.
    """
    locks_directory = working_directory / ".snakemake" / "locks"
    if not locks_directory.is_dir():
        return False

    lock_files = [
        file_path
        for file_path in locks_directory.iterdir()
        if file_path.is_file() and file_path.name.endswith(".lock")
    ]
    return len(lock_files) > 0


def check_if_lock_error(working_directory: Path) -> bool:
    """Check if the latest execution failed due to directory locks.

    Args:
        working_directory: Snakemake execution directory.

    Returns:
        bool: True if failure was caused by directory locks.
    """
    if is_directory_locked(working_directory):
        return True

    log_directory = working_directory / ".snakemake" / "log"
    if not log_directory.is_dir():
        return False

    log_files = sorted(
        [
            file_entry
            for file_entry in log_directory.iterdir()
            if file_entry.name.endswith(".snakemake.log")
        ],
        key=lambda file_entry: file_entry.stat().st_mtime,
        reverse=True,
    )
    if not log_files:
        return False

    latest_log_path = log_files[0]
    try:
        log_content = latest_log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        lock_indicators = [
            "LockException",
            "Directory cannot be locked",
            "unlock the directory with the following command",
        ]
        return any(
            indicator_text in log_content
            for indicator_text in lock_indicators
        )
    except OSError:
        return False


def build_unlock_command(
    args: argparse.Namespace,
    snakefile_path: str,
    extra_arguments: Optional[List[str]] = None,
) -> List[str]:
    """Build the Snakemake command to unlock the working directory.

    Args:
        args: Parsed command line arguments.
        snakefile_path: Absolute path to the main Snakefile.
        extra_arguments: Optional additional Snakemake arguments.

    Returns:
        List[str]: Snakemake unlock command components.
    """
    unlock_command = [
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        args.config,
        "--unlock",
    ]

    if extra_arguments:
        for argument_index, argument_item in enumerate(extra_arguments):
            if argument_item in ("--directory", "-d"):
                if argument_index + 1 < len(extra_arguments):
                    directory_value = extra_arguments[argument_index + 1]
                    unlock_command.extend([argument_item, directory_value])
            elif argument_item.startswith("--directory="):
                unlock_command.append(argument_item)

    return unlock_command


def unlock_working_directory(
    args: argparse.Namespace,
    snakefile_path: str,
    extra_arguments: Optional[List[str]] = None,
) -> bool:
    """Automatically unlock the Snakemake working directory.

    Args:
        args: Parsed command line arguments.
        snakefile_path: Absolute path to the main Snakefile.
        extra_arguments: Optional additional Snakemake arguments.

    Returns:
        bool: True if directory was successfully unlocked, False otherwise.
    """
    unlock_command = build_unlock_command(
        args=args,
        snakefile_path=snakefile_path,
        extra_arguments=extra_arguments,
    )
    print(
        "Working directory is locked. Automatically unlocking...",
        flush=True,
    )
    try:
        subprocess.run(unlock_command, check=True)
        print("Successfully unlocked working directory.", flush=True)
        return True
    except subprocess.CalledProcessError as unlock_error:
        print(
            "Failed to automatically unlock working directory "
            f"(return code {unlock_error.returncode}).",
            file=sys.stderr,
        )
        return False


def build_snakemake_command(
    args: argparse.Namespace,
    snakefile_path: str,
    extra_arguments: Optional[List[str]] = None,
) -> List[str]:
    """Build the Snakemake command list from the given arguments.

    Args:
        args: Parsed command line arguments.
        snakefile_path: Absolute path to the main Snakefile.
        extra_arguments: Optional pre-loaded additional arguments.

    Returns:
        List[str]: A list of strings representing the Snakemake command.
    """
    command_components = [
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        args.config,
        "--cores",
        str(args.cores),
    ]

    if args.submit:
        command_components.extend(["--profile", args.submit])

    if args.go:
        command_components.extend(["--rerun-triggers", "mtime"])

    if args.unlock:
        command_components.extend(["--unlock"])

    if extra_arguments is None and args.makefile:
        extra_arguments = load_makefile_arguments(args.makefile)

    if extra_arguments:
        command_components.extend(extra_arguments)

    return command_components


def main() -> None:
    """Main entrypoint for the CLI wrapper."""
    args = parse_arguments()

    if not os.path.isfile(args.config):
        print(
            f"Error: Config file not found: {args.config}",
            file=sys.stderr,
        )
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    snakefile_path = os.path.join(current_dir, "workflow", "Snakefile")

    if not os.path.isfile(snakefile_path):
        print(
            "Error: Could not locate Snakefile at expected path: "
            f"{snakefile_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    extra_arguments: List[str] = []
    if args.makefile:
        extra_arguments = load_makefile_arguments(args.makefile)

    working_directory = get_snakemake_working_directory(extra_arguments)

    if not args.unlock and is_directory_locked(working_directory):
        unlock_working_directory(
            args=args,
            snakefile_path=snakefile_path,
            extra_arguments=extra_arguments,
        )

    command_to_run = build_snakemake_command(
        args=args,
        snakefile_path=snakefile_path,
        extra_arguments=extra_arguments,
    )

    try:
        subprocess.run(command_to_run, check=True)
    except subprocess.CalledProcessError as execution_error:
        if not args.unlock and check_if_lock_error(working_directory):
            print(
                "Execution failed due to a locked working directory. "
                "Attempting to unlock and rerun...",
                flush=True,
            )
            unlock_succeeded = unlock_working_directory(
                args=args,
                snakefile_path=snakefile_path,
                extra_arguments=extra_arguments,
            )
            if unlock_succeeded:
                try:
                    subprocess.run(command_to_run, check=True)
                    return
                except subprocess.CalledProcessError as retry_error:
                    print(
                        "Snakemake execution failed on retry with return "
                        f"code {retry_error.returncode}.",
                        file=sys.stderr,
                    )
                    sys.exit(retry_error.returncode)

        print(
            "Snakemake execution failed with return code "
            f"{execution_error.returncode}.",
            file=sys.stderr,
        )
        sys.exit(execution_error.returncode)
    except FileNotFoundError:
        print(
            "Error: 'snakemake' command not found. Please ensure it is "
            "installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
