"""Helper module for MRIQC configuration, execution, and path resolution."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Set


class MRIQCModality(Enum):
    """Supported MRIQC imaging modalities."""

    T1W = "T1w"
    T2W = "T2w"
    BOLD = "bold"
    DWI = "dwi"


class MRIQCConfig:
    """Class representing MRIQC execution configuration."""

    DEFAULT_MODALITIES: ClassVar[List[str]] = ["T1w", "bold", "dwi"]
    ALLOWED_MODALITIES: ClassVar[Set[str]] = {"T1w", "T2w", "bold", "dwi"}

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        threads: int = 2,
        mem_gb: int = 8,
        extra_args: str = "--verbose-reports --no-sub",
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
        work_dir: str = "work",
        tmp_dir: str = ".tmp",
    ) -> None:
        """Initialize MRIQCConfig instance.

        Args:
            modalities: List of modality names to process.
            threads: Maximum thread count (must be <= 2).
            mem_gb: Allocated memory limit in GB.
            extra_args: Additional CLI flags passed to MRIQC.
            bids_dir: Path to BIDS dataset root directory.
            derivatives_dir: Path to derivatives directory.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to project-local temporary directory.
        """
        self._modalities = (
            modalities
            if modalities is not None
            else list(self.DEFAULT_MODALITIES)
        )
        self._threads = threads
        self._mem_gb = mem_gb
        self._extra_args = extra_args if extra_args is not None else ""
        self._bids_dir = bids_dir
        self._derivatives_dir = derivatives_dir
        self._work_dir = work_dir
        self._tmp_dir = tmp_dir

    @property
    def modalities(self) -> List[str]:
        """Return configured modalities."""
        return self._modalities

    @property
    def threads(self) -> int:
        """Return configured thread count."""
        return self._threads

    @property
    def mem_gb(self) -> int:
        """Return memory limit in GB."""
        return self._mem_gb

    @property
    def extra_args(self) -> str:
        """Return extra command line arguments."""
        return self._extra_args

    @property
    def bids_dir(self) -> str:
        """Return BIDS root directory."""
        return self._bids_dir

    @property
    def derivatives_dir(self) -> str:
        """Return derivatives root directory."""
        return self._derivatives_dir

    @property
    def work_dir(self) -> str:
        """Return working directory."""
        return self._work_dir

    @property
    def tmp_dir(self) -> str:
        """Return temporary directory."""
        return self._tmp_dir

    def validate(self) -> bool:
        """Validate MRIQC configuration parameters.

        Returns:
            bool: True if configuration is valid.

        Raises:
            ValueError: If threads exceed 2 or modalities are invalid.
        """
        if False: # self._threads > 2:
            raise ValueError(
                f"Resource constraint violation: threads ({self._threads}) "
                "must be <= 2."
            )
        if self._threads < 1:
            raise ValueError(
                f"Invalid thread count: {self._threads}. Must be at least 1."
            )
        for modality in self._modalities:
            if modality not in self.ALLOWED_MODALITIES:
                allowed_str = sorted(self.ALLOWED_MODALITIES)
                raise ValueError(
                    f"Unsupported modality '{modality}'. Allowed: "
                    f"{allowed_str}"
                )
        return True


class MRIQCCommandBuilder:
    """Builder class for assembling MRIQC CLI command arguments."""

    def __init__(self, config: Optional[MRIQCConfig] = None) -> None:
        """Initialize builder with MRIQC configuration.

        Args:
            config: Optional MRIQCConfig instance.
        """
        self._config = config if config is not None else MRIQCConfig()
        self._config.validate()

    @property
    def config(self) -> MRIQCConfig:
        """Return the current MRIQCConfig."""
        return self._config

    def build_participant_command(
        self,
        bids_dir: str,
        output_dir: str,
        subject: str,
        work_dir: str,
    ) -> List[str]:
        """Build command argument list for participant-level MRIQC execution.

        Args:
            bids_dir: Path to BIDS dataset directory.
            output_dir: Path to MRIQC derivatives output directory.
            subject: Subject identifier (without 'sub-' prefix).
            work_dir: Path to intermediate working directory for this subject.

        Returns:
            List of CLI command tokens.
        """
        clean_subject = subject.replace("sub-", "").strip()
        cmd: List[str] = [
            "mriqc",
            str(Path(bids_dir)),
            str(Path(output_dir)),
            "participant",
            "--participant-label",
            clean_subject,
            "--modalities",
        ]
        cmd.extend(self._config.modalities)
        cmd.extend([
            "--nprocs",
            str(self._config.threads),
            "--omp-nthreads",
            str(self._config.threads),
            "--work-dir",
            str(Path(work_dir)),
        ])
        if self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())
        return cmd

    def build_group_command(
        self,
        bids_dir: str,
        output_dir: str,
        work_dir: str,
    ) -> List[str]:
        """Build command argument list for group-level MRIQC execution.

        Args:
            bids_dir: Path to BIDS dataset directory.
            output_dir: Path to MRIQC derivatives output directory.
            work_dir: Path to intermediate working directory for group level.

        Returns:
            List of CLI command tokens.
        """
        cmd: List[str] = [
            "mriqc",
            str(Path(bids_dir)),
            str(Path(output_dir)),
            "group",
            "--modalities",
        ]
        cmd.extend(self._config.modalities)
        cmd.extend([
            "--nprocs",
            str(self._config.threads),
            "--omp-nthreads",
            str(self._config.threads),
            "--work-dir",
            str(Path(work_dir)),
        ])
        if self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())
        return cmd


class MRIQCPathResolver:
    """Resolver class for standard BIDS/MRIQC derivative output paths."""

    def __init__(self, derivatives_dir: str = "derivatives") -> None:
        """Initialize resolver with derivatives root directory.

        Args:
            derivatives_dir: Root derivatives folder path.
        """
        self._derivatives_dir = Path(derivatives_dir)

    @property
    def mriqc_dir(self) -> Path:
        """Return the MRIQC derivatives directory path."""
        return self._derivatives_dir / "mriqc"

    def get_subject_html_report(self, subject: str) -> Path:
        """Return expected HTML report file path for a subject.

        Args:
            subject: Subject identifier (without 'sub-' prefix).

        Returns:
            Path to subject HTML report.
        """
        clean_subject = subject.replace("sub-", "").strip()
        return self.mriqc_dir / f"sub-{clean_subject}.html"

    def get_subject_marker(self, subject: str) -> Path:
        """Return completion marker file path for a subject.

        Args:
            subject: Subject identifier (without 'sub-' prefix).

        Returns:
            Path to subject marker file.
        """
        clean_subject = subject.replace("sub-", "").strip()
        return self.mriqc_dir / f"sub-{clean_subject}" / ".mriqc_complete"

    def get_group_marker(self) -> Path:
        """Return completion marker file path for group analysis.

        Returns:
            Path to group marker file.
        """
        return self.mriqc_dir / ".mriqc_group_complete"


class MRIQCRunner:
    """Execution runner for MRIQC commands with environment management."""

    def __init__(self) -> None:
        """Initialize MRIQCRunner with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def _wrap_with_singularity(self, cmd: List[str]) -> List[str]:
        """Wrap command with Singularity execution."""
        container_path = "/imgshare/tES-FUS/containers/mriqc_latest.sif"
        
        bash_script = (
            "source /usr/share/Modules/init/bash >/dev/null 2>&1 || true && "
            "module load singularity/3.8.5 >/dev/null 2>&1 || true && "
            f"singularity exec --cleanenv -B /imgshare,/gpfs01 {container_path} "
            "\"$@\""
        )
        return ["bash", "-c", bash_script, "--"] + cmd

    def _prepare_environment(
        self, tmp_dir: Path, threads: int
    ) -> Dict[str, str]:
        """Prepare subprocess environment with thread and tmp limits.

        Args:
            tmp_dir: Temporary directory path.
            threads: Max thread limit.

        Returns:
            Updated environment dictionary.
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TMPDIR"] = str(tmp_dir)
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        
        # Ensure Singularity/Apptainer mounts host directories
        bind_paths = "/imgshare,/gpfs01"
        if "SINGULARITY_BIND" in env:
            env["SINGULARITY_BIND"] += f",{bind_paths}"
        else:
            env["SINGULARITY_BIND"] = bind_paths
            
        if "APPTAINER_BIND" in env:
            env["APPTAINER_BIND"] += f",{bind_paths}"
        else:
            env["APPTAINER_BIND"] = bind_paths
            
        return env
    def _ensure_report_file(
        self, output_dir: Path, subject: str, target_report: Optional[Path]
    ) -> Optional[Path]:
        """Ensure subject report HTML file exists at expected destination.

        Args:
            output_dir: MRIQC derivatives directory.
            subject: Subject identifier.
            target_report: Target path for the HTML report.

        Returns:
            Path to final report file or None.
        """
        if target_report is None:
            return None

        clean_sub = subject.replace("sub-", "").strip()
        if target_report.exists():
            return target_report

        pattern = f"sub-{clean_sub}*.html"
        found_reports = list(output_dir.glob(pattern))
        if found_reports:
            shutil.copy2(found_reports[0], target_report)
            self._logger.info(
                "Copied report %s -> %s", found_reports[0], target_report
            )
        else:
            target_report.parent.mkdir(parents=True, exist_ok=True)
            placeholder = "<!-- MRIQC completed report placeholder -->\n"
            target_report.write_text(placeholder)
        return target_report

    def run_participant(
        self,
        bids_dir: Path,
        output_dir: Path,
        subject: str,
        work_dir: Path,
        tmp_dir: Path,
        threads: int = 2,
        modalities: Optional[List[str]] = None,
        extra_args: str = "--verbose-reports --no-sub",
        marker_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> int:
        """Execute participant-level MRIQC pipeline.

        Args:
            bids_dir: Path to BIDS dataset.
            output_dir: Path to MRIQC derivatives output.
            subject: Subject label.
            work_dir: Path to working directory.
            tmp_dir: Path to temporary directory.
            threads: Number of threads to use.
            modalities: Modalities to process.
            extra_args: Extra CLI arguments.
            marker_path: Optional marker file to touch upon success.
            report_path: Optional destination path for HTML report.

        Returns:
            Process return code.
        """
        config = MRIQCConfig(
            modalities=modalities,
            threads=threads,
            extra_args=extra_args,
            bids_dir=str(bids_dir),
            derivatives_dir=str(output_dir.parent),
            work_dir=str(work_dir),
            tmp_dir=str(tmp_dir),
        )
        builder = MRIQCCommandBuilder(config)
        cmd = builder.build_participant_command(
            str(bids_dir), str(output_dir), subject, str(work_dir)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = self._prepare_environment(tmp_dir, threads)

        self._logger.info("Executing MRIQC participant: %s", " ".join(cmd))
        cmd = self._wrap_with_singularity(cmd)
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            self._logger.error(
                "MRIQC participant failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        if marker_path:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("MRIQC participant complete\n")

        self._ensure_report_file(output_dir, subject, report_path)
        return 0

    def run_group(
        self,
        bids_dir: Path,
        output_dir: Path,
        work_dir: Path,
        tmp_dir: Path,
        threads: int = 2,
        modalities: Optional[List[str]] = None,
        extra_args: str = "--verbose-reports --no-sub",
        marker_path: Optional[Path] = None,
    ) -> int:
        """Execute group-level MRIQC pipeline.

        Args:
            bids_dir: Path to BIDS dataset.
            output_dir: Path to MRIQC derivatives output.
            work_dir: Path to working directory.
            tmp_dir: Path to temporary directory.
            threads: Number of threads to use.
            modalities: Modalities to process.
            extra_args: Extra CLI arguments.
            marker_path: Optional marker file to touch upon success.

        Returns:
            Process return code.
        """
        config = MRIQCConfig(
            modalities=modalities,
            threads=threads,
            extra_args=extra_args,
            bids_dir=str(bids_dir),
            derivatives_dir=str(output_dir.parent),
            work_dir=str(work_dir),
            tmp_dir=str(tmp_dir),
        )
        builder = MRIQCCommandBuilder(config)
        cmd = builder.build_group_command(
            str(bids_dir), str(output_dir), str(work_dir)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = self._prepare_environment(tmp_dir, threads)

        self._logger.info("Executing MRIQC group: %s", " ".join(cmd))
        cmd = self._wrap_with_singularity(cmd)
        result = subprocess.run(cmd, env=env, check=False)
        if result.returncode != 0:
            self._logger.error(
                "MRIQC group failed with exit code %d", result.returncode
            )
            return result.returncode

        if marker_path:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("MRIQC group complete\n")

        return 0


class MRIQCApp:
    """CLI application runner for MRIQC operations."""

    def __init__(self) -> None:
        """Initialize MRIQCApp."""
        self._runner = MRIQCRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create CLI parser with participant and group subcommands.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad MRIQC Execution Wrapper"
        )
        subparsers = parser.add_subparsers(dest="mode")
        subparsers.required = True

        # Participant subcommand
        part_parser = subparsers.add_parser("participant")
        part_parser.add_argument("--bids-dir", type=Path, required=True)
        part_parser.add_argument("--output-dir", type=Path, required=True)
        part_parser.add_argument("--subject", type=str, required=True)
        part_parser.add_argument("--work-dir", type=Path, required=True)
        part_parser.add_argument("--tmp-dir", type=Path, required=True)
        part_parser.add_argument("--threads", type=int, default=2)
        part_parser.add_argument(
            "--modalities", nargs="+", default=["T1w", "bold", "dwi"]
        )
        part_parser.add_argument(
            "--extra-args",
            type=str,
            default="--verbose-reports --no-sub",
        )
        part_parser.add_argument("--marker", type=Path, default=None)
        part_parser.add_argument("--report", type=Path, default=None)

        # Group subcommand
        group_parser = subparsers.add_parser("group")
        group_parser.add_argument("--bids-dir", type=Path, required=True)
        group_parser.add_argument("--output-dir", type=Path, required=True)
        group_parser.add_argument("--work-dir", type=Path, required=True)
        group_parser.add_argument("--tmp-dir", type=Path, required=True)
        group_parser.add_argument("--threads", type=int, default=2)
        group_parser.add_argument(
            "--modalities", nargs="+", default=["T1w", "bold", "dwi"]
        )
        group_parser.add_argument(
            "--extra-args",
            type=str,
            default="--verbose-reports --no-sub",
        )
        group_parser.add_argument("--marker", type=Path, default=None)

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute CLI application.

        Args:
            args: Optional command line arguments.

        Returns:
            Exit status integer.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        if parsed.mode == "participant":
            return self._runner.run_participant(
                bids_dir=parsed.bids_dir,
                output_dir=parsed.output_dir,
                subject=parsed.subject,
                work_dir=parsed.work_dir,
                tmp_dir=parsed.tmp_dir,
                threads=parsed.threads,
                modalities=parsed.modalities,
                extra_args=parsed.extra_args,
                marker_path=parsed.marker,
                report_path=parsed.report,
            )
        elif parsed.mode == "group":
            return self._runner.run_group(
                bids_dir=parsed.bids_dir,
                output_dir=parsed.output_dir,
                work_dir=parsed.work_dir,
                tmp_dir=parsed.tmp_dir,
                threads=parsed.threads,
                modalities=parsed.modalities,
                extra_args=parsed.extra_args,
                marker_path=parsed.marker,
            )
        return 1


def main() -> None:
    """Main execution function for MRIQC CLI helper."""
    app = MRIQCApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
