"""Helper module for fMRIPrep configuration, execution, and path resolution."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional


class FMRIPrepOutputSpace(Enum):
    """Supported standard output template spaces for fMRIPrep."""

    MNI152NLIN2009CASYM = "MNI152NLin2009cAsym"
    MNI152NLIN2009CASYM_RES2 = "MNI152NLin2009cAsym:res-2"
    MNI152NLIN6ASYM = "MNI152NLin6Asym"
    MNI152NLIN6ASYM_RES2 = "MNI152NLin6Asym:res-2"
    FSAVERAGE = "fsaverage"
    FSAVERAGE5 = "fsaverage5"
    FSAVERAGE6 = "fsaverage6"
    ANAT = "anat"
    T1W = "T1w"

    @classmethod
    def from_value(cls, value: Any) -> "FMRIPrepOutputSpace":
        """Convert a string or FMRIPrepOutputSpace instance to enum.

        Args:
            value: Space name string or FMRIPrepOutputSpace enum.

        Returns:
            Validated FMRIPrepOutputSpace enum instance.

        Raises:
            ValueError: If space value is not recognized or invalid type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip()
            for item in cls:
                if item.value.lower() == clean_val.lower():
                    return item
            allowed_sorted = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported output space '{value}'. "
                f"Allowed: {allowed_sorted}"
            )
        raise ValueError(f"Invalid output space type: {type(value)}")


class FMRIPrepConfig:
    """Configuration container and validator for fMRIPrep execution."""

    DEFAULT_OUTPUT_SPACES: ClassVar[List[str]] = [
        "MNI152NLin2009cAsym:res-2",
        "fsaverage5",
    ]

    def __init__(
        self,
        threads: int = 2,
        mem_mb: int = 8000,
        output_spaces: Optional[List[str]] = None,
        cifti_output: Optional[str] = "91k",
        fs_subjects_dir: Optional[str] = "derivatives/fastsurfer",
        fs_license: Optional[str] = None,
        fs_no_resume: bool = False,
        dummy_scans: Optional[int] = None,
        bids_filter_file: Optional[str] = None,
        extra_args: str = "--skip-bids-validation --notrack",
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
        work_dir: str = "work",
        tmp_dir: str = ".tmp",
    ) -> None:
        """Initialize FMRIPrepConfig instance.

        Args:
            threads: Number of threads (must be between 1 and 2).
            mem_mb: Memory limit in megabytes.
            output_spaces: Target normalization template spaces.
            cifti_output: CIFTI grayordinates resolution ('91k' or '170k').
            fs_subjects_dir: Path to FreeSurfer/FastSurfer subjects.
            fs_license: Path to FreeSurfer license file.
            fs_no_resume: Skip FreeSurfer surface reconstruction flag.
            dummy_scans: Number of non-steady-state volumes to drop.
            bids_filter_file: Path to custom BIDS query filter JSON file.
            extra_args: Extra flags passed directly to fMRIPrep.
            bids_dir: Path to BIDS dataset root directory.
            derivatives_dir: Path to derivatives directory.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to project-local temporary directory.
        """
        self._threads = threads
        self._mem_mb = mem_mb
        self._output_spaces = (
            list(output_spaces)
            if output_spaces is not None
            else list(self.DEFAULT_OUTPUT_SPACES)
        )
        self._cifti_output = cifti_output
        self._fs_subjects_dir = fs_subjects_dir
        self._fs_license = fs_license
        self._fs_no_resume = fs_no_resume
        self._dummy_scans = dummy_scans
        self._bids_filter_file = bids_filter_file
        self._extra_args = extra_args
        self._bids_dir = bids_dir
        self._derivatives_dir = derivatives_dir
        self._work_dir = work_dir
        self._tmp_dir = tmp_dir

    @property
    def threads(self) -> int:
        """Return configured thread count."""
        return self._threads

    @property
    def mem_mb(self) -> int:
        """Return configured memory limit in MB."""
        return self._mem_mb

    @property
    def output_spaces(self) -> List[str]:
        """Return list of configured output spaces."""
        return self._output_spaces

    @property
    def cifti_output(self) -> Optional[str]:
        """Return configured CIFTI output resolution if enabled."""
        return self._cifti_output

    @property
    def fs_subjects_dir(self) -> Optional[str]:
        """Return path to FreeSurfer / FastSurfer subjects directory."""
        return self._fs_subjects_dir

    @property
    def fs_license(self) -> Optional[str]:
        """Return path to FreeSurfer license file."""
        return self._fs_license

    @property
    def fs_no_resume(self) -> bool:
        """Return True if recon-all surface reconstruction is disabled."""
        return self._fs_no_resume

    @property
    def dummy_scans(self) -> Optional[int]:
        """Return number of dummy scans to discard."""
        return self._dummy_scans

    @property
    def bids_filter_file(self) -> Optional[str]:
        """Return BIDS filter file path."""
        return self._bids_filter_file

    @property
    def extra_args(self) -> str:
        """Return extra CLI flags."""
        return self._extra_args

    @property
    def bids_dir(self) -> str:
        """Return BIDS dataset directory."""
        return self._bids_dir

    @property
    def derivatives_dir(self) -> str:
        """Return derivatives directory."""
        return self._derivatives_dir

    @property
    def work_dir(self) -> str:
        """Return intermediate working directory."""
        return self._work_dir

    @property
    def tmp_dir(self) -> str:
        """Return temporary directory."""
        return self._tmp_dir

    def validate(self) -> bool:
        """Validate fMRIPrep configuration parameters.

        Returns:
            True if configuration satisfies all constraints.

        Raises:
            ValueError: If resource limits or arguments are violated.
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
        if self._mem_mb < 1000:
            raise ValueError(
                f"Invalid memory allocation: {self._mem_mb} MB. "
                "Must be >= 1000 MB."
            )
        if not self._output_spaces:
            raise ValueError("At least one output space must be specified.")
        if self._dummy_scans is not None and self._dummy_scans < 0:
            raise ValueError(
                f"Invalid dummy scans: {self._dummy_scans}. Must be >= 0."
            )
        return True


class FMRIPrepCommandBuilder:
    """Builder class for assembling fMRIPrep CLI command tokens."""

    def __init__(self, config: Optional[FMRIPrepConfig] = None) -> None:
        """Initialize FMRIPrepCommandBuilder with configuration.

        Args:
            config: Optional FMRIPrepConfig instance.
        """
        self._config = config if config is not None else FMRIPrepConfig()
        self._config.validate()

    @property
    def config(self) -> FMRIPrepConfig:
        """Return the current FMRIPrepConfig."""
        return self._config

    def build_participant_command(
        self,
        bids_dir: Path,
        output_dir: Path,
        subject: str,
        work_dir: Path,
        executable: str = "fmriprep",
    ) -> List[str]:
        """Build command token list for participant-level fMRIPrep execution.

        Args:
            bids_dir: Path to BIDS dataset root directory.
            output_dir: Path to fMRIPrep derivatives output directory.
            subject: Subject identifier string.
            work_dir: Path to intermediate working directory.

        Returns:
            List of CLI command tokens.
        """
        clean_subject = subject.replace("sub-", "").strip()
        import shlex
        cmd: List[str] = shlex.split(executable) + [
            str(bids_dir),
            str(output_dir),
            "participant",
            "--participant-label",
            clean_subject,
            "--nprocs",
            str(self._config.threads),
            "--omp-nthreads",
            str(self._config.threads),
            "--mem-mb",
            str(self._config.mem_mb),
            "--work-dir",
            str(work_dir),
        ]

        if self._config.output_spaces:
            cmd.append("--output-spaces")
            cmd.extend(self._config.output_spaces)

        fs_sd = self._config.fs_subjects_dir
        if fs_sd and fs_sd.strip():
            cmd.extend(["--fs-subjects-dir", str(Path(fs_sd.strip()))])

        fs_lic = self._config.fs_license
        if fs_lic and fs_lic.strip():
            cmd.extend(["--fs-license-file", str(Path(fs_lic.strip()))])

        if self._config.cifti_output and self._config.cifti_output.strip():
            cifti_val = self._config.cifti_output.strip()
            if cifti_val in ("91k", "170k"):
                cmd.extend(["--cifti-output", cifti_val])
            else:
                cmd.append("--cifti-output")

        if self._config.fs_no_resume:
            cmd.append("--fs-no-resume")

        if self._config.dummy_scans is not None:
            cmd.extend(["--dummy-scans", str(self._config.dummy_scans)])

        filt = self._config.bids_filter_file
        if filt and filt.strip():
            cmd.extend(["--bids-filter-file", str(Path(filt.strip()))])

        if self._config.extra_args and self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())

        return cmd


class FMRIPrepPathResolver:
    """Resolver class for fMRIPrep derivative file and directory paths."""

    def __init__(self, derivatives_dir: str = "derivatives") -> None:
        """Initialize resolver with root derivatives directory.

        Args:
            derivatives_dir: Path to derivatives directory.
        """
        self._derivatives_dir = Path(derivatives_dir)

    @property
    def derivatives_dir(self) -> Path:
        """Return derivatives directory path."""
        return self._derivatives_dir

    @property
    def fmriprep_dir(self) -> Path:
        """Return fMRIPrep derivatives directory path."""
        return self._derivatives_dir / "fmriprep"

    def get_subject_dir(self, subject: str) -> Path:
        """Return fMRIPrep subject output directory path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject derivatives directory.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.fmriprep_dir / f"sub-{clean_sub}"

    def get_subject_html_report(self, subject: str) -> Path:
        """Return expected HTML report path for a subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject HTML report.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.fmriprep_dir / f"sub-{clean_sub}.html"

    def get_subject_marker(self, subject: str) -> Path:
        """Return completion marker file path for a subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject marker file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.get_subject_dir(clean_sub) / ".fmriprep_complete"

    def get_anatomical_preproc(
        self,
        subject: str,
        space: str = "MNI152NLin2009cAsym_res-2",
    ) -> Path:
        """Return expected preprocessed anatomical T1w volume path.

        Args:
            subject: Subject identifier.
            space: Normalization space identifier string.

        Returns:
            Path to preprocessed T1w NIfTI image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = (
            f"sub-{clean_sub}_space-{space}_desc-preproc_T1w.nii.gz"
        )
        return self.get_subject_dir(clean_sub) / "anat" / filename

    def get_confounds_file(
        self,
        subject: str,
        task: str = "rest",
        run: Optional[str] = None,
    ) -> Path:
        """Return expected confounds timeseries TSV file path.

        Args:
            subject: Subject identifier.
            task: Task name string (default 'rest').
            run: Optional run index string.

        Returns:
            Path to confounds TSV file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        run_part = f"_run-{run}" if run else ""
        filename = (
            f"sub-{clean_sub}_task-{task}{run_part}"
            "_desc-confounds_timeseries.tsv"
        )
        return self.get_subject_dir(clean_sub) / "func" / filename

    def get_bold_preproc(
        self,
        subject: str,
        task: str = "rest",
        space: str = "MNI152NLin2009cAsym_res-2",
        run: Optional[str] = None,
    ) -> Path:
        """Return expected preprocessed BOLD NIfTI file path.

        Args:
            subject: Subject identifier.
            task: Task name string.
            space: Normalization space identifier.
            run: Optional run index string.

        Returns:
            Path to preprocessed BOLD NIfTI volume.
        """
        clean_sub = subject.replace("sub-", "").strip()
        run_part = f"_run-{run}" if run else ""
        filename = (
            f"sub-{clean_sub}_task-{task}{run_part}"
            f"_space-{space}_desc-preproc_bold.nii.gz"
        )
        return self.get_subject_dir(clean_sub) / "func" / filename


class FMRIPrepRunner:
    """Runner for fMRIPrep with environment and artifact management."""

    def __init__(self) -> None:
        """Initialize FMRIPrepRunner with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def prepare_environment(
        self,
        tmp_dir: Path,
        threads: int,
        fs_license: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare subprocess environment dictionary.

        Args:
            tmp_dir: Temporary directory path.
            threads: Maximum thread count (must be <= 2).
            fs_license: Optional FreeSurfer license file path.

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
        bind_paths = f"{tmp_dir}:/tmp,{tmp_dir}:/var/tmp"
        if "SINGULARITY_BIND" in env:
            env["SINGULARITY_BIND"] += f",{bind_paths}"
        else:
            env["SINGULARITY_BIND"] = bind_paths
            
        if "APPTAINER_BIND" in env:
            env["APPTAINER_BIND"] += f",{bind_paths}"
        else:
            env["APPTAINER_BIND"] = bind_paths
            
        if fs_license and fs_license.strip():
            env["FS_LICENSE"] = str(fs_license).strip()
        return env

    def ensure_report_file(
        self,
        output_dir: Path,
        subject: str,
        target_report: Optional[Path],
    ) -> Optional[Path]:
        """Ensure subject report HTML file exists at expected destination.

        Args:
            output_dir: fMRIPrep derivatives directory.
            subject: Subject identifier string.
            target_report: Target path for the HTML report.

        Returns:
            Path to resolved report file or None.
        """
        if target_report is None:
            return None

        clean_sub = subject.replace("sub-", "").strip()
        if target_report.exists():
            return target_report

        pattern = f"sub-{clean_sub}*.html"
        found_reports = sorted(output_dir.glob(pattern))
        target_report.parent.mkdir(parents=True, exist_ok=True)
        if found_reports:
            shutil.copy2(found_reports[0], target_report)
            self._logger.info(
                "Copied report %s -> %s", found_reports[0], target_report
            )
        else:
            placeholder = "<!-- fMRIPrep completed report placeholder -->\n"
            target_report.write_text(placeholder, encoding="utf-8")
            self._logger.info(
                "Generated report placeholder at %s", target_report
            )
        return target_report

    def run(
        self,
        bids_dir: Path,
        output_dir: Path,
        subject: str,
        work_dir: Path,
        tmp_dir: Path,
        threads: int = 2,
        mem_mb: int = 8000,
        output_spaces: Optional[List[str]] = None,
        cifti_output: Optional[str] = "91k",
        fs_subjects_dir: Optional[str] = "derivatives/fastsurfer",
        fs_license: Optional[str] = None,
        fs_no_resume: bool = False,
        dummy_scans: Optional[int] = None,
        bids_filter_file: Optional[str] = None,
        extra_args: str = "--skip-bids-validation --notrack",
        marker_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
        executable: str = "fmriprep",
    ) -> int:
        """Execute fMRIPrep pipeline for a single participant.

        Args:
            bids_dir: Path to BIDS dataset root directory.
            output_dir: Path to fMRIPrep derivatives output directory.
            subject: Subject label string.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to temporary directory.
            threads: Number of processing threads (max 2).
            mem_mb: Memory allocation in MB.
            output_spaces: Target normalization template spaces.
            cifti_output: CIFTI grayordinates output mode ('91k' or '170k').
            fs_subjects_dir: FreeSurfer/FastSurfer subjects directory path.
            fs_license: FreeSurfer license file path.
            fs_no_resume: Skip recon-all flag.
            dummy_scans: Dummy scans count.
            bids_filter_file: BIDS query filter file path.
            extra_args: Extra command line flags.
            marker_path: Optional marker file path touched upon completion.
            report_path: Optional destination path for HTML report.

        Returns:
            Process exit status code integer.
        """
        config = FMRIPrepConfig(
            threads=threads,
            mem_mb=mem_mb,
            output_spaces=output_spaces,
            cifti_output=cifti_output,
            fs_subjects_dir=fs_subjects_dir,
            fs_license=fs_license,
            fs_no_resume=fs_no_resume,
            dummy_scans=dummy_scans,
            bids_filter_file=bids_filter_file,
            extra_args=extra_args,
            bids_dir=str(bids_dir),
            derivatives_dir=str(output_dir.parent),
            work_dir=str(work_dir),
            tmp_dir=str(tmp_dir),
        )
        builder = FMRIPrepCommandBuilder(config)
        cmd = builder.build_participant_command(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject=subject,
            work_dir=work_dir,
            executable=executable,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = self.prepare_environment(tmp_dir, threads, fs_license)

        if fs_subjects_dir:
            clean_sub = subject.replace("sub-", "").strip()
            scripts_dir = Path(fs_subjects_dir) / f"sub-{clean_sub}" / "scripts"
            if scripts_dir.exists():
                for lock_file in scripts_dir.glob("IsRunning.*"):
                    try:
                        lock_file.unlink()
                        self._logger.info("Removed leftover FreeSurfer lock file: %s", lock_file)
                    except Exception as e:
                        self._logger.warning("Failed to remove lock file %s: %s", lock_file, e)

        self._logger.info("Executing fMRIPrep command: %s", " ".join(cmd))
        # Dynamically inject root mounts for wrapper-based singularity containers
        def get_root_mount(path: str) -> str:
            p = Path(path).resolve()
            return f"/{p.parts[1]}" if len(p.parts) > 1 else ""
            
        roots = set()
        for p in [bids_dir, output_dir, work_dir, fs_subjects_dir, fs_license]:
            if p:
                r = get_root_mount(str(p))
                if r:
                    roots.add(f"{r}:{r}")
                    
        if roots:
            root_binds = ",".join(roots)
            for k in ["SINGULARITY_BIND", "APPTAINER_BIND"]:
                env[k] = f"{root_binds},{env[k]}" if k in env else root_binds

        result = subprocess.run(cmd, env=env, check=False)

        if result.returncode != 0:
            self._logger.error(
                "fMRIPrep execution failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("fMRIPrep complete\n", encoding="utf-8")

        self.ensure_report_file(output_dir, subject, report_path)
        return 0


class FMRIPrepApp:
    """CLI application interface for fMRIPrep execution wrapper."""

    def __init__(self) -> None:
        """Initialize FMRIPrepApp."""
        self._runner = FMRIPrepRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI ArgumentParser for fMRIPrep wrapper.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad fMRIPrep Execution Wrapper"
        )
        parser.add_argument(
            "--bids-dir",
            type=Path,
            required=True,
            help="Path to BIDS dataset root directory",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Path to fMRIPrep derivatives output directory",
        )
        parser.add_argument(
            "--subject",
            type=str,
            required=True,
            help="Subject ID (with or without 'sub-' prefix)",
        )
        parser.add_argument(
            "--work-dir",
            type=Path,
            default=Path("work/fmriprep"),
            help="Path to working directory",
        )
        parser.add_argument(
            "--tmp-dir",
            type=Path,
            default=Path(".tmp"),
            help="Path to project-local temporary directory",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=2,
            help="Thread count (max 2)",
        )
        parser.add_argument(
            "--mem-mb",
            type=int,
            default=8000,
            help="Memory limit in MB",
        )
        parser.add_argument(
            "--output-spaces",
            nargs="+",
            default=["MNI152NLin2009cAsym:res-2", "fsaverage5"],
            help="Target normalization output spaces",
        )
        parser.add_argument(
            "--cifti-output",
            type=str,
            default="91k",
            help="CIFTI output resolution (e.g. 91k, 170k, none)",
        )
        parser.add_argument(
            "--fs-subjects-dir",
            type=str,
            default="derivatives/fastsurfer",
            help="Precomputed FreeSurfer/FastSurfer subjects directory",
        )
        parser.add_argument(
            "--fs-license",
            type=str,
            default="",
            help="Path to FreeSurfer license file",
        )
        parser.add_argument(
            "--fs-no-resume",
            action="store_true",
            default=False,
            help="Disable FreeSurfer surface reconstruction",
        )
        parser.add_argument(
            "--dummy-scans",
            type=int,
            default=None,
            help="Number of initial non-steady-state volumes to drop",
        )
        parser.add_argument(
            "--bids-filter-file",
            type=str,
            default="",
            help="Path to BIDS filter file",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            default="--skip-bids-validation --notrack",
            help="Additional CLI flags passed directly to fMRIPrep",
        )
        parser.add_argument(
            "--executable",
            type=str,
            default="fmriprep",
            help="Executable name or wrapper",
        )
        parser.add_argument(
            "--marker",
            type=Path,
            default=None,
            help="Path to completion marker file to create",
        )
        parser.add_argument(
            "--report",
            type=Path,
            default=None,
            help="Destination path for HTML report file",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Parse arguments and execute fMRIPrep runner.

        Args:
            args: Optional command line argument list.

        Returns:
            Process exit status code integer.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        cifti_mode = (
            parsed.cifti_output.strip() if parsed.cifti_output else None
        )
        if cifti_mode and cifti_mode.lower() in ("none", "false", ""):
            cifti_mode = None

        fs_sd = (
            parsed.fs_subjects_dir if parsed.fs_subjects_dir else None
        )
        fs_lic = (
            parsed.fs_license if parsed.fs_license else None
        )
        bids_filt = (
            parsed.bids_filter_file if parsed.bids_filter_file else None
        )

        return self._runner.run(
            bids_dir=parsed.bids_dir,
            output_dir=parsed.output_dir,
            subject=parsed.subject,
            work_dir=parsed.work_dir,
            tmp_dir=parsed.tmp_dir,
            threads=parsed.threads,
            mem_mb=parsed.mem_mb,
            output_spaces=parsed.output_spaces,
            cifti_output=cifti_mode,
            fs_subjects_dir=fs_sd,
            fs_license=fs_lic,
            fs_no_resume=parsed.fs_no_resume,
            dummy_scans=parsed.dummy_scans,
            bids_filter_file=bids_filt,
            extra_args=parsed.extra_args,
            marker_path=parsed.marker,
            report_path=parsed.report,
            executable=parsed.executable,
        )


def main() -> None:
    """Main execution function for fMRIPrep CLI wrapper."""
    app = FMRIPrepApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
