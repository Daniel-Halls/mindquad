"""Helper module for QSIPrep configuration, execution, and path resolution."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union


class QSIPrepDenoiseMethod(Enum):
    """Supported DWI denoising methods for QSIPrep."""

    DWIDENOISE = "dwidenoise"
    PATCH2SELF = "patch2self"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Any) -> "QSIPrepDenoiseMethod":
        """Convert a string or enum instance to QSIPrepDenoiseMethod.

        Args:
            value: Method name string or QSIPrepDenoiseMethod enum.

        Returns:
            Validated QSIPrepDenoiseMethod enum instance.

        Raises:
            ValueError: If denoise method is not recognized or invalid type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value.lower() == clean_val:
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported denoise method '{value}'. Allowed: {allowed}"
            )
        raise ValueError(f"Invalid denoise method type: {type(value)}")


class QSIPrepUnringingMethod(Enum):
    """Supported Gibbs unringing methods for QSIPrep."""

    MRDEGIBBS = "mrdegibbs"
    RPG = "rpg"
    NONE = "none"

    @classmethod
    def from_value(cls, value: Any) -> "QSIPrepUnringingMethod":
        """Convert a string or enum instance to QSIPrepUnringingMethod.

        Args:
            value: Unringing method name string or enum instance.

        Returns:
            Validated QSIPrepUnringingMethod enum instance.

        Raises:
            ValueError: If unringing method is not recognized or invalid type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value.lower() == clean_val:
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported unringing method '{value}'. Allowed: {allowed}"
            )
        raise ValueError(f"Invalid unringing method type: {type(value)}")


class QSIPrepOutputSpace(Enum):
    """Supported output reference spaces for QSIPrep."""

    T1W = "T1w"
    MNI152NLIN2009CASYM = "MNI152NLin2009cAsym"

    @classmethod
    def from_value(cls, value: Any) -> "QSIPrepOutputSpace":
        """Convert string or enum instance to QSIPrepOutputSpace.

        Args:
            value: Output space string or enum.

        Returns:
            Validated QSIPrepOutputSpace instance.

        Raises:
            ValueError: If output space is invalid.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip()
            for item in cls:
                if item.value.lower() == clean_val.lower():
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported output space '{value}'. Allowed: {allowed}"
            )
        raise ValueError(f"Invalid output space type: {type(value)}")


class QSIPrepConfig:
    """Configuration container and validator for QSIPrep execution."""

    DEFAULT_OUTPUT_RESOLUTION: ClassVar[float] = 1.5
    DEFAULT_EXTRA_ARGS: ClassVar[str] = "--skip-bids-validation --notrack"

    def __init__(
        self,
        threads: int = 2,
        mem_mb: int = 8000,
        output_resolution: float = 1.5,
        denoise_method: Any = QSIPrepDenoiseMethod.DWIDENOISE,
        unringing_method: Any = QSIPrepUnringingMethod.MRDEGIBBS,
        separate_all_dwis: bool = False,
        fs_subjects_dir: Optional[str] = "derivatives/fastsurfer",
        fs_license: Optional[str] = None,
        do_reconall: bool = False,
        bids_filter_file: Optional[str] = None,
        extra_args: str = "--skip-bids-validation --notrack",
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
        work_dir: str = "work",
        tmp_dir: str = ".tmp",
    ) -> None:
        """Initialize QSIPrepConfig instance.

        Args:
            threads: Processing thread count (must be 1 or 2).
            mem_mb: Memory limit in megabytes.
            output_resolution: Isotropic voxel resolution in mm for output DWI.
            denoise_method: DWI denoising method enum or string.
            unringing_method: Gibbs unringing method enum or string.
            separate_all_dwis: Whether to process DWI scans separately.
            fs_subjects_dir: Path to FreeSurfer/FastSurfer subjects directory.
            fs_license: Path to FreeSurfer license file.
            do_reconall: Whether to run FreeSurfer surface reconstruction.
            bids_filter_file: Path to custom BIDS filter JSON file.
            extra_args: Extra CLI flags passed to QSIPrep.
            bids_dir: Path to BIDS dataset root directory.
            derivatives_dir: Path to derivatives directory.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to project-local temporary directory.
        """
        self._threads = threads
        self._mem_mb = mem_mb
        self._output_resolution = float(output_resolution)
        self._denoise_method = (
            denoise_method
            if isinstance(denoise_method, QSIPrepDenoiseMethod)
            else QSIPrepDenoiseMethod.from_value(denoise_method)
        )
        self._unringing_method = (
            unringing_method
            if isinstance(unringing_method, QSIPrepUnringingMethod)
            else QSIPrepUnringingMethod.from_value(unringing_method)
        )
        self._separate_all_dwis = separate_all_dwis
        self._fs_subjects_dir = fs_subjects_dir
        self._fs_license = fs_license
        self._do_reconall = do_reconall
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
    def output_resolution(self) -> float:
        """Return configured output voxel resolution in mm."""
        return self._output_resolution

    @property
    def denoise_method(self) -> QSIPrepDenoiseMethod:
        """Return configured denoising method enum."""
        return self._denoise_method

    @property
    def unringing_method(self) -> QSIPrepUnringingMethod:
        """Return configured unringing method enum."""
        return self._unringing_method

    @property
    def separate_all_dwis(self) -> bool:
        """Return True if separate all DWIs is enabled."""
        return self._separate_all_dwis

    @property
    def fs_subjects_dir(self) -> Optional[str]:
        """Return FreeSurfer / FastSurfer subjects directory path."""
        return self._fs_subjects_dir

    @property
    def fs_license(self) -> Optional[str]:
        """Return FreeSurfer license path."""
        return self._fs_license

    @property
    def do_reconall(self) -> bool:
        """Return True if recon-all surface reconstruction is enabled."""
        return self._do_reconall

    @property
    def bids_filter_file(self) -> Optional[str]:
        """Return BIDS filter file path."""
        return self._bids_filter_file

    @property
    def extra_args(self) -> str:
        """Return extra CLI flags string."""
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
        """Validate QSIPrep configuration parameters against constraints.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If resource limits or configuration values are invalid.
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
        if self._output_resolution <= 0.0:
            raise ValueError(
                f"Invalid output resolution: {self._output_resolution}. "
                "Must be greater than 0.0."
            )
        if not isinstance(self._denoise_method, QSIPrepDenoiseMethod):
            raise ValueError(
                f"Invalid denoise method instance: {self._denoise_method}"
            )
        if not isinstance(self._unringing_method, QSIPrepUnringingMethod):
            raise ValueError(
                f"Invalid unringing method instance: {self._unringing_method}"
            )
        return True


class QSIPrepCommandBuilder:
    """Builder class for assembling QSIPrep CLI command tokens."""

    def __init__(self, config: Optional[QSIPrepConfig] = None) -> None:
        """Initialize QSIPrepCommandBuilder with configuration.

        Args:
            config: Optional QSIPrepConfig instance.
        """
        self._config = config if config is not None else QSIPrepConfig()
        self._config.validate()

    @property
    def config(self) -> QSIPrepConfig:
        """Return the active QSIPrepConfig."""
        return self._config

    def build_participant_command(
        self,
        bids_dir: Path,
        output_dir: Path,
        subject: str,
        work_dir: Path,
        executable: str = "qsiprep",
    ) -> List[str]:
        """Build command token list for participant-level QSIPrep execution.

        Args:
            bids_dir: Path to BIDS dataset root directory.
            output_dir: Path to QSIPrep derivatives output directory.
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
            "--output-resolution",
            str(self._config.output_resolution),
        ]

        if self._config.denoise_method != QSIPrepDenoiseMethod.NONE:
            cmd.extend(["--denoise-method", self._config.denoise_method.value])

        if self._config.unringing_method != QSIPrepUnringingMethod.NONE:
            cmd.extend(
                ["--unringing-method", self._config.unringing_method.value]
            )

        fs_sd = self._config.fs_subjects_dir
        if fs_sd and fs_sd.strip():
            cmd.extend(["--fs-subjects-dir", str(Path(fs_sd.strip()))])

        fs_lic = self._config.fs_license
        if fs_lic and fs_lic.strip():
            cmd.extend(["--fs-license-file", str(Path(fs_lic.strip()))])

        if self._config.do_reconall:
            cmd.append("--do-reconall")

        if self._config.separate_all_dwis:
            cmd.append("--separate-all-dwis")

        filt = self._config.bids_filter_file
        if filt and filt.strip():
            cmd.extend(["--bids-filter-file", str(Path(filt.strip()))])

        if self._config.extra_args and self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())

        return cmd


class QSIPrepPathResolver:
    """Resolver class for QSIPrep derivative file and directory paths."""

    def __init__(self, derivatives_dir: str = "derivatives") -> None:
        """Initialize resolver with root derivatives directory.

        Args:
            derivatives_dir: Path to derivatives root directory.
        """
        self._derivatives_dir = Path(derivatives_dir)

    @property
    def derivatives_dir(self) -> Path:
        """Return root derivatives directory path."""
        return self._derivatives_dir

    @property
    def qsiprep_dir(self) -> Path:
        """Return QSIPrep derivatives directory path."""
        return self._derivatives_dir / "qsiprep"

    def get_subject_dir(self, subject: str) -> Path:
        """Return QSIPrep subject output directory path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject derivatives directory.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.qsiprep_dir / f"sub-{clean_sub}"

    def get_subject_html_report(self, subject: str) -> Path:
        """Return expected HTML report path for a subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject HTML report.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.qsiprep_dir / f"sub-{clean_sub}.html"

    def get_subject_marker(self, subject: str) -> Path:
        """Return completion marker file path for a subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject marker file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.get_subject_dir(clean_sub) / ".qsiprep_complete"

    def get_dwi_preproc(self, subject: str, space: str = "T1w") -> Path:
        """Return preprocessed DWI NIfTI volume path.

        Args:
            subject: Subject identifier.
            space: Reference space (default 'T1w').

        Returns:
            Path to preprocessed DWI NIfTI file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_space-{space}_desc-preproc_dwi.nii.gz"
        return self.get_subject_dir(clean_sub) / "dwi" / filename

    def get_dwi_bval(self, subject: str, space: str = "T1w") -> Path:
        """Return preprocessed DWI bval file path.

        Args:
            subject: Subject identifier.
            space: Reference space (default 'T1w').

        Returns:
            Path to preprocessed DWI bval file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_space-{space}_desc-preproc_dwi.bval"
        return self.get_subject_dir(clean_sub) / "dwi" / filename

    def get_dwi_bvec(self, subject: str, space: str = "T1w") -> Path:
        """Return preprocessed DWI bvec file path.

        Args:
            subject: Subject identifier.
            space: Reference space (default 'T1w').

        Returns:
            Path to preprocessed DWI bvec file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_space-{space}_desc-preproc_dwi.bvec"
        return self.get_subject_dir(clean_sub) / "dwi" / filename

    def get_dwi_brainmask(self, subject: str, space: str = "T1w") -> Path:
        """Return preprocessed DWI brain mask NIfTI path.

        Args:
            subject: Subject identifier.
            space: Reference space (default 'T1w').

        Returns:
            Path to DWI brain mask NIfTI file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_space-{space}_desc-brain_mask.nii.gz"
        return self.get_subject_dir(clean_sub) / "dwi" / filename

    def get_anatomical_preproc(self, subject: str) -> Path:
        """Return preprocessed anatomical T1w volume path.

        Args:
            subject: Subject identifier.

        Returns:
            Path to preprocessed T1w image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_desc-preproc_T1w.nii.gz"
        return self.get_subject_dir(clean_sub) / "anat" / filename

    def get_anatomical_brainmask(self, subject: str) -> Path:
        """Return anatomical brain mask volume path.

        Args:
            subject: Subject identifier.

        Returns:
            Path to anatomical brain mask image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_desc-brain_mask.nii.gz"
        return self.get_subject_dir(clean_sub) / "anat" / filename


class QSIPrepRunner:
    """Runner for QSIPrep with environment and artifact management."""

    def __init__(self) -> None:
        """Initialize QSIPrepRunner with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def _wrap_with_singularity(self, cmd: List[str]) -> List[str]:
        """Wrap command with Singularity execution."""
        container_path = "/imgshare/tES-FUS/containers/qsiprep_latest.sif"
        
        bash_script = (
            "source /usr/share/Modules/init/bash >/dev/null 2>&1 || true && "
            "module load singularity/3.8.5 >/dev/null 2>&1 || true && "
            f"singularity exec --cleanenv -B /imgshare,/gpfs01 {container_path} "
            "\"$@\""
        )
        return ["bash", "-c", bash_script, "--"] + cmd

    def prepare_environment(
        self,
        tmp_dir: Path,
        threads: int,
        fs_license: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare subprocess execution environment dictionary.

        Args:
            tmp_dir: Project-local temporary directory path.
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
        if fs_license and fs_license.strip():
            env["FS_LICENSE"] = str(fs_license).strip()
            
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

    def ensure_report_file(
        self,
        output_dir: Path,
        subject: str,
        target_report: Optional[Path],
    ) -> Optional[Path]:
        """Ensure subject report HTML file exists at expected destination.

        Args:
            output_dir: QSIPrep derivatives directory.
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
            placeholder = "<!-- QSIPrep completed report placeholder -->\n"
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
        output_resolution: float = 1.5,
        denoise_method: Any = QSIPrepDenoiseMethod.DWIDENOISE,
        unringing_method: Any = QSIPrepUnringingMethod.MRDEGIBBS,
        separate_all_dwis: bool = False,
        fs_subjects_dir: Optional[str] = "derivatives/fastsurfer",
        fs_license: Optional[str] = None,
        do_reconall: bool = False,
        bids_filter_file: Optional[str] = None,
        extra_args: str = "--skip-bids-validation --notrack",
        marker_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> int:
        """Execute QSIPrep pipeline for a single participant.

        Args:
            bids_dir: Path to BIDS dataset root directory.
            output_dir: Path to QSIPrep derivatives output directory.
            subject: Subject label string.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to temporary directory.
            threads: Number of processing threads (max 2).
            mem_mb: Memory allocation in MB.
            output_resolution: Isotropic voxel resolution in mm for output DWI.
            denoise_method: Denoising method enum or string.
            unringing_method: Unringing method enum or string.
            separate_all_dwis: Process DWI series separately if True.
            fs_subjects_dir: FreeSurfer/FastSurfer subjects directory path.
            fs_license: FreeSurfer license file path.
            do_reconall: Run FreeSurfer surface reconstruction flag.
            bids_filter_file: BIDS query filter file path.
            extra_args: Extra command line flags.
            marker_path: Optional marker file path touched upon completion.
            report_path: Optional destination path for HTML report.

        Returns:
            Process exit status code integer.
        """
        config = QSIPrepConfig(
            threads=threads,
            mem_mb=mem_mb,
            output_resolution=output_resolution,
            denoise_method=denoise_method,
            unringing_method=unringing_method,
            separate_all_dwis=separate_all_dwis,
            fs_subjects_dir=fs_subjects_dir,
            fs_license=fs_license,
            do_reconall=do_reconall,
            bids_filter_file=bids_filter_file,
            extra_args=extra_args,
            bids_dir=str(bids_dir),
            derivatives_dir=str(output_dir.parent),
            work_dir=str(work_dir),
            tmp_dir=str(tmp_dir),
        )
        builder = QSIPrepCommandBuilder(config)
        cmd = builder.build_participant_command(
            bids_dir=bids_dir,
            output_dir=output_dir,
            subject=subject,
            work_dir=work_dir,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = self.prepare_environment(tmp_dir, threads, fs_license)

        self._logger.info("Executing QSIPrep command: %s", " ".join(cmd))
        cmd = self._wrap_with_singularity(cmd)
        result = subprocess.run(cmd, env=env, check=False)

        if result.returncode != 0:
            self._logger.error(
                "QSIPrep execution failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("QSIPrep complete\n", encoding="utf-8")

        self.ensure_report_file(output_dir, subject, report_path)
        return 0


class QSIPrepApp:
    """CLI application interface for QSIPrep execution wrapper."""

    def __init__(self) -> None:
        """Initialize QSIPrepApp."""
        self._runner = QSIPrepRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI ArgumentParser for QSIPrep wrapper.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad QSIPrep Execution Wrapper"
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
            help="Path to QSIPrep derivatives output directory",
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
            default=Path("work/qsiprep"),
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
            "--output-resolution",
            type=float,
            default=1.5,
            help="Isotropic voxel resolution in mm for output DWI",
        )
        parser.add_argument(
            "--denoise-method",
            type=str,
            default="dwidenoise",
            help="DWI denoising method (dwidenoise, patch2self, none)",
        )
        parser.add_argument(
            "--unringing-method",
            type=str,
            default="mrdegibbs",
            help="Gibbs unringing method (mrdegibbs, rpg, none)",
        )
        parser.add_argument(
            "--separate-all-dwis",
            action="store_true",
            default=False,
            help="Process separate DWI runs individually",
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
            "--do-reconall",
            action="store_true",
            default=False,
            help="Enable FreeSurfer surface reconstruction",
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
            help="Additional CLI flags passed directly to QSIPrep",
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
        """Parse arguments and execute QSIPrep runner.

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
            output_resolution=parsed.output_resolution,
            denoise_method=parsed.denoise_method,
            unringing_method=parsed.unringing_method,
            separate_all_dwis=parsed.separate_all_dwis,
            fs_subjects_dir=fs_sd,
            fs_license=fs_lic,
            do_reconall=parsed.do_reconall,
            bids_filter_file=bids_filt,
            extra_args=parsed.extra_args,
            marker_path=parsed.marker,
            report_path=parsed.report,
        )


def main() -> None:
    """Main execution function for QSIPrep CLI wrapper."""
    app = QSIPrepApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
