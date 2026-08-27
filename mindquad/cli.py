"""Command line interface for the mindquad pipeline."""

import argparse
import os
import shlex
import subprocess
import sys
from typing import List

def ascii_art() -> str:
    """
    Returns ascii art
    """
    return r"""\033[95mL
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
\033[0m
    """

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

    return parser.parse_args()


def build_snakemake_command(args: argparse.Namespace, snakefile_path: str) -> List[str]:
    """Build the Snakemake command list from the given arguments.

    Args:
        args: Parsed command line arguments.
        snakefile_path: Absolute path to the main Snakefile.

    Returns:
        List[str]: A list of strings representing the Snakemake command.
    """
    cmd = [
        "snakemake",
        "--snakefile",
        snakefile_path,
        "--configfile",
        args.config,
        "--cores",
        str(args.cores),
    ]

    if args.submit:
        cmd.extend(["--profile", args.submit])

    if args.makefile:
        if not os.path.isfile(args.makefile):
            print(
                f"Error: Additional arguments file not found: {args.makefile}",
                file=sys.stderr,
            )
            sys.exit(1)

        import yaml
        
        with open(args.makefile, "r", encoding="utf-8") as file_handle:
            try:
                data = yaml.safe_load(file_handle)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML makefile: {exc}", file=sys.stderr)
                sys.exit(1)

        extra_args = []
        if isinstance(data, list):
            extra_args = [str(x) for x in data]
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, bool):
                    if v:
                        extra_args.append(f"--{k}")
                else:
                    extra_args.extend([f"--{k}", str(v)])
                    
        if extra_args:
            cmd.extend(extra_args)

    return cmd


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
            f"Error: Could not locate Snakefile at expected path: {snakefile_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = build_snakemake_command(args, snakefile_path)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as error:
        print(
            f"Snakemake execution failed with return code {error.returncode}.",
            file=sys.stderr,
        )
        sys.exit(error.returncode)
    except FileNotFoundError:
        print(
            "Error: 'snakemake' command not found. Please ensure it is installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
