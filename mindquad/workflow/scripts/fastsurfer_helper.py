"""Helper module for FastSurfer execution, configuration, and resolution."""

import argparse
from enum import Enum
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


class FastSurferDevice(Enum):
    """Supported computing devices for FastSurfer."""

    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
    MPS = "mps"

    @classmethod
    def from_value(cls, value: Any) -> "FastSurferDevice":
        """Convert a string or enum to FastSurferDevice.

        Args:
            value: Device name string or FastSurferDevice enum.

        Returns:
            Validated FastSurferDevice enum instance.

        Raises:
            ValueError: If device value is not recognized or invalid.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value == clean_val:
                    return item
            allowed_sorted = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported device '{value}'. Allowed: {allowed_sorted}"
            )
        raise ValueError(f"Invalid device type: {type(value)}")


class FastSurferConfig:
    """Configuration container and validator for FastSurfer execution."""

    def __init__(
        self,
        threads: int = 2,
        device: Any = FastSurferDevice.CPU,
        fs_license: Optional[str] = None,
        extra_args: str = "",
        batch_size: int = 1,
        seg_only: bool = False,
        surf_only: bool = False,
        parallel: bool = False,
    ) -> None:
        """Initialize FastSurfer configuration parameters.

        Args:
            threads: Number of processing threads.
            device: Computing device enum or string.
            fs_license: Path to FreeSurfer license file.
            extra_args: Additional command line arguments string.
            batch_size: Inference batch size (default 1).
            seg_only: Run only whole-brain segmentation.
            surf_only: Run only surface reconstruction.
            parallel: Run surface reconstruction in parallel.
        """
        self._threads = threads
        self._device = (
            device
            if isinstance(device, FastSurferDevice)
            else FastSurferDevice.from_value(device)
        )
        self._fs_license = fs_license
        self._extra_args = extra_args
        self._batch_size = batch_size
        self._seg_only = seg_only
        self._surf_only = surf_only
        self._parallel = parallel

    @property
    def threads(self) -> int:
        """Return configured thread count."""
        return self._threads

    @property
    def device(self) -> FastSurferDevice:
        """Return target compute device enum."""
        return self._device

    @property
    def fs_license(self) -> Optional[str]:
        """Return FreeSurfer license path if set."""
        return self._fs_license

    @property
    def extra_args(self) -> str:
        """Return extra CLI flags."""
        return self._extra_args

    @property
    def batch_size(self) -> int:
        """Return inference batch size."""
        return self._batch_size

    @property
    def seg_only(self) -> bool:
        """Return True if running segmentation only."""
        return self._seg_only

    @property
    def surf_only(self) -> bool:
        """Return True if running surface reconstruction only."""
        return self._surf_only

    @property
    def parallel(self) -> bool:
        """Return True if running parallel hemisphere processing."""
        return self._parallel

    def validate(self) -> bool:
        """Validate FastSurfer configuration parameters.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If thread limits or device settings are violated.
        """
        if self._threads < 1:
            raise ValueError(
                f"Invalid thread count: {self._threads}. Must be at least 1."
            )
        if not isinstance(self._device, FastSurferDevice):
            raise ValueError(f"Invalid device instance: {self._device}")
        if self._batch_size < 1:
            raise ValueError(
                f"Invalid batch size: {self._batch_size}. Must be >= 1."
            )
        if self._seg_only and self._surf_only:
            raise ValueError(
                "Cannot specify both seg_only and surf_only simultaneously."
            )
        return True


class FastSurferCommandBuilder:
    """Builder class for assembling FastSurfer CLI command arguments."""

    def __init__(self, config: Optional[FastSurferConfig] = None) -> None:
        """Initialize command builder with FastSurfer configuration.

        Args:
            config: Optional FastSurferConfig instance.
        """
        self._config = config if config is not None else FastSurferConfig()
        self._config.validate()

    @property
    def config(self) -> FastSurferConfig:
        """Return current FastSurferConfig."""
        return self._config

    def build_command(
        self,
        t1_path: Path,
        subjects_dir: Path,
        subject_id: str,
        executable: str = "run_fastsurfer.sh",
    ) -> List[str]:
        """Build command argument list for FastSurfer execution.

        Args:
            t1_path: Path to input structural T1w image.
            subjects_dir: Path to output subjects directory.
            subject_id: Subject identifier string.
            executable: FastSurfer script or singularity command.

        Returns:
            List of CLI command tokens.
        """
        clean_sid = subject_id.strip()

        command_tokens: List[str] = []
        if executable.startswith("singularity run"):
            command_tokens.extend(shlex.split(executable))
        else:
            resolved_exe = shutil.which(executable) or executable
            command_tokens.extend(["bash", resolved_exe])

        command_tokens.extend([
            "--t1",
            str(t1_path),
            "--sd",
            str(subjects_dir),
            "--sid",
            clean_sid,
            "--threads",
            str(self._config.threads),
        ])

        if self._config.device:
            command_tokens.extend(["--device", self._config.device.value])

        if self._config.batch_size > 1:
            command_tokens.extend(["--batch", str(self._config.batch_size)])

        if self._config.fs_license and self._config.fs_license.strip():
            command_tokens.extend(
                ["--fs_license", self._config.fs_license.strip()]
            )

        if self._config.seg_only:
            command_tokens.append("--seg_only")
        elif self._config.surf_only:
            command_tokens.append("--surf_only")

        # Always generate FreeSurfer-compatible parcellations for fMRIPrep
        if "--fsaparc" not in command_tokens and not self._config.seg_only:
            command_tokens.append("--fsaparc")

        if self._config.parallel:
            command_tokens.append("--parallel")

        if self._config.extra_args and self._config.extra_args.strip():
            command_tokens.extend(self._config.extra_args.strip().split())

        return command_tokens


class FastSurferPathResolver:
    """Resolver class for FastSurfer input and output file paths."""

    def __init__(
        self,
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
    ) -> None:
        """Initialize resolver with BIDS and derivatives directories.

        Args:
            bids_dir: Path to BIDS dataset root folder.
            derivatives_dir: Path to derivatives root folder.
        """
        self._bids_dir = Path(bids_dir)
        self._derivatives_dir = Path(derivatives_dir)

    @property
    def bids_dir(self) -> Path:
        """Return BIDS dataset directory."""
        return self._bids_dir

    @property
    def derivatives_dir(self) -> Path:
        """Return root derivatives directory."""
        return self._derivatives_dir

    @property
    def fastsurfer_dir(self) -> Path:
        """Return FastSurfer derivatives output directory."""
        return self._derivatives_dir / "fastsurfer"

    def get_subject_dir(self, subject: str) -> Path:
        """Return FastSurfer subject directory path.

        Args:
            subject: Subject identifier (with or without 'sub-' prefix).

        Returns:
            Path to subject output directory.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.fastsurfer_dir / f"sub-{clean_sub}"

    def get_scripts_dir(self, subject: str) -> Path:
        """Return path to FastSurfer subject scripts directory.

        Args:
            subject: Subject identifier.

        Returns:
            Path to scripts directory.
        """
        return self.get_subject_dir(subject) / "scripts"

    def get_is_running_file(self, subject: str) -> Path:
        """Return path to FastSurfer IsRunning.lh+rh lock file.

        Args:
            subject: Subject identifier.

        Returns:
            Path to IsRunning.lh+rh file.
        """
        return self.get_scripts_dir(subject) / "IsRunning.lh+rh"

    def resolve_t1w_path(self, subject: str) -> Path:
        """Resolve T1w anatomical image path for a subject in BIDS directory.

        Args:
            subject: Subject identifier (with or without 'sub-' prefix).

        Returns:
            Path to resolved T1w NIfTI image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        anat_dir = self._bids_dir / f"sub-{clean_sub}" / "anat"
        standard_path = anat_dir / f"sub-{clean_sub}_T1w.nii.gz"

        if standard_path.exists():
            return standard_path

        if anat_dir.exists():
            candidates: List[Path] = []
            for pattern in ["*T1w*.nii.gz", "*T1w*.nii"]:
                candidates.extend(sorted(anat_dir.glob(pattern)))
            if candidates:
                return candidates[0]

        return standard_path

    def get_segmentation_file(self, subject: str) -> Path:
        """Return path to deep-learning whole-brain segmentation file.

        Args:
            subject: Subject identifier.

        Returns:
            Path to aparc.DKTatlas+aseg.deep.mgz file.
        """
        subject_dir = self.get_subject_dir(subject)
        return subject_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz"

    def get_orig_mgz_file(self, subject: str) -> Path:
        """Return path to conformed orig.mgz volume.

        Args:
            subject: Subject identifier.

        Returns:
            Path to orig.mgz file.
        """
        return self.get_subject_dir(subject) / "mri" / "orig.mgz"

    def get_brainmask_file(self, subject: str) -> Path:
        """Return path to brainmask.mgz file.

        Args:
            subject: Subject identifier.

        Returns:
            Path to brainmask.mgz file.
        """
        return self.get_subject_dir(subject) / "mri" / "brainmask.mgz"

    def get_aseg_file(self, subject: str) -> Path:
        """Return path to aseg.mgz volume.

        Args:
            subject: Subject identifier.

        Returns:
            Path to aseg.mgz file.
        """
        return self.get_subject_dir(subject) / "mri" / "aseg.mgz"

    def get_surface_file(self, subject: str, hemi: str, surface: str) -> Path:
        """Return path to cortical surface file.

        Args:
            subject: Subject identifier.
            hemi: Hemisphere ('lh' or 'rh').
            surface: Surface type ('white', 'pial', 'sphere', etc.).

        Returns:
            Path to surface file.
        """
        return self.get_subject_dir(subject) / "surf" / f"{hemi}.{surface}"

    def get_stats_file(self, subject: str, stats_name: str) -> Path:
        """Return path to morphometric statistics table.

        Args:
            subject: Subject identifier.
            stats_name: Statistics file name (e.g. 'aseg.stats').

        Returns:
            Path to stats file.
        """
        return self.get_subject_dir(subject) / "stats" / stats_name

    def get_completion_marker(self, subject: str) -> Path:
        """Return path to FastSurfer completion marker file.

        Args:
            subject: Subject identifier.

        Returns:
            Path to .fastsurfer_complete marker file.
        """
        return self.get_subject_dir(subject) / ".fastsurfer_complete"


class FastSurferRunner:
    """Execution runner for FastSurfer commands with environment management."""

    def __init__(self) -> None:
        """Initialize FastSurferRunner with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def cleanup_is_running_lock(
        self,
        subjects_dir: Path,
        subject_id: str,
    ) -> bool:
        """Check for and delete FastSurfer IsRunning lock files before launch.

        Args:
            subjects_dir: FastSurfer subjects output directory.
            subject_id: Subject identifier string.

        Returns:
            bool: True if an IsRunning lock file was removed, False otherwise.
        """
        clean_subject_id = subject_id.strip()
        if clean_subject_id.startswith("sub-"):
            subject_directory = Path(subjects_dir) / clean_subject_id
        else:
            prefixed_directory = Path(subjects_dir) / f"sub-{clean_subject_id}"
            if prefixed_directory.is_dir():
                subject_directory = prefixed_directory
            else:
                subject_directory = Path(subjects_dir) / clean_subject_id

        scripts_directory = subject_directory / "scripts"
        if not scripts_directory.is_dir():
            return False

        is_running_file = scripts_directory / "IsRunning.lh+rh"
        lock_removed = False

        if is_running_file.exists():
            self._logger.info(
                "Found FastSurfer lock file '%s'. "
                "Deleting before launching FastSurfer.",
                is_running_file,
            )
            try:
                is_running_file.unlink()
                lock_removed = True
            except OSError as removal_error:
                self._logger.warning(
                    "Failed to delete FastSurfer lock file '%s': %s",
                    is_running_file,
                    removal_error,
                )

        # Also check for any extra IsRunning lock variants (e.g. lh or rh)
        for lock_file_path in scripts_directory.glob("IsRunning*"):
            if lock_file_path.is_file():
                self._logger.info(
                    "Found additional FastSurfer lock file '%s'. "
                    "Deleting before launching FastSurfer.",
                    lock_file_path,
                )
                try:
                    lock_file_path.unlink()
                    lock_removed = True
                except OSError as removal_error:
                    self._logger.warning(
                        "Failed to delete lock file '%s': %s",
                        lock_file_path,
                        removal_error,
                    )

        return lock_removed

    def prepare_environment(
        self,
        tmp_dir: Path,
        threads: int,
        fs_license: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare environment dictionary with thread and license settings.

        Args:
            tmp_dir: Project-local temporary directory path.
            threads: Maximum thread count.
            fs_license: Optional path to FreeSurfer license file.

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

        # Force Singularity to mount NVIDIA drivers
        env["SINGULARITY_NV"] = "1"
        env["APPTAINER_NV"] = "1"

        # Force PyTorch to use exactly one GPU
        env["CUDA_VISIBLE_DEVICES"] = "0"

        if fs_license and fs_license.strip():
            env["FS_LICENSE"] = str(fs_license).strip()
        return env

    def run(
        self,
        t1_path: Path,
        subjects_dir: Path,
        subject_id: str,
        tmp_dir: Path,
        threads: int = 2,
        device: Any = FastSurferDevice.CPU,
        fs_license: Optional[str] = None,
        extra_args: str = "",
        marker_path: Optional[Path] = None,
        executable: str = "run_fastsurfer.sh",
    ) -> int:
        """Execute FastSurfer pipeline for a structural T1w volume.

        Args:
            t1_path: Path to input T1w NIfTI volume.
            subjects_dir: Output FreeSurfer/FastSurfer subjects directory.
            subject_id: Subject identifier string.
            tmp_dir: Project-local temporary directory.
            threads: Processing thread count.
            device: Compute device string or FastSurferDevice enum.
            fs_license: Optional path to FreeSurfer license file.
            extra_args: Additional CLI flags string.
            marker_path: Optional marker file path touched upon completion.
            executable: FastSurfer execution command or script.

        Returns:
            Process exit status code integer.
        """
        config = FastSurferConfig(
            threads=threads,
            device=device,
            fs_license=fs_license,
            extra_args=extra_args,
        )
        builder = FastSurferCommandBuilder(config)
        cmd = builder.build_command(
            t1_path=t1_path,
            subjects_dir=subjects_dir,
            subject_id=subject_id,
            executable=executable,
        )

        subjects_dir.mkdir(parents=True, exist_ok=True)
        env = self.prepare_environment(tmp_dir, threads, fs_license)

        # Check for and delete FastSurfer IsRunning lock files before launch
        self.cleanup_is_running_lock(
            subjects_dir=subjects_dir,
            subject_id=subject_id,
        )

        self._logger.info("Executing FastSurfer command: %s", " ".join(cmd))

        # Dynamically inject root mounts for singularity containers
        def get_root_mount(target_path_str: str) -> str:
            resolved_path = Path(target_path_str).resolve()
            return (
                f"/{resolved_path.parts[1]}"
                if len(resolved_path.parts) > 1
                else ""
            )

        mount_roots = set()
        for path_argument in [t1_path, subjects_dir, fs_license]:
            if path_argument:
                root_prefix = get_root_mount(str(path_argument))
                if root_prefix:
                    mount_roots.add(f"{root_prefix}:{root_prefix}")

        if mount_roots:
            root_binds = ",".join(mount_roots)
            for env_var_name in ["SINGULARITY_BIND", "APPTAINER_BIND"]:
                if env_var_name in env:
                    env[env_var_name] = f"{root_binds},{env[env_var_name]}"
                else:
                    env[env_var_name] = root_binds

        result = subprocess.run(cmd, env=env, check=False)

        if result.returncode != 0:
            self._logger.error(
                "FastSurfer execution failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("FastSurfer complete\n", encoding="utf-8")

        return 0


class FastSurferApp:
    """CLI application interface for FastSurfer execution helper."""

    def __init__(self) -> None:
        """Initialize FastSurferApp."""
        self._runner = FastSurferRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI ArgumentParser for FastSurfer.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad FastSurfer Execution Wrapper"
        )
        parser.add_argument(
            "--t1",
            type=Path,
            required=True,
            help="Path to structural T1w NIfTI image",
        )
        parser.add_argument(
            "--sd",
            type=Path,
            required=True,
            help="Output subjects directory (derivatives/fastsurfer)",
        )
        parser.add_argument(
            "--sid",
            type=str,
            required=True,
            help="Subject ID (e.g. sub-19081001)",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=2,
            help="Thread count",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Computing device (cpu, cuda, auto, mps)",
        )
        parser.add_argument(
            "--fs-license",
            type=str,
            default="",
            help="Path to FreeSurfer license file",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            default="",
            help="Additional arguments passed to run_fastsurfer.sh",
        )
        parser.add_argument(
            "--marker",
            type=Path,
            default=None,
            help="Path to completion marker file to create",
        )
        parser.add_argument(
            "--executable",
            type=str,
            default="run_fastsurfer.sh",
            help="Executable or singularity wrapper for fastsurfer",
        )
        parser.add_argument(
            "--tmp-dir",
            type=Path,
            default=Path(".tmp"),
            help="Path to project-local temporary directory",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Parse arguments and execute FastSurfer runner.

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

        return self._runner.run(
            t1_path=parsed.t1,
            subjects_dir=parsed.sd,
            subject_id=parsed.sid,
            tmp_dir=parsed.tmp_dir,
            threads=parsed.threads,
            device=parsed.device,
            fs_license=parsed.fs_license if parsed.fs_license else None,
            extra_args=parsed.extra_args,
            marker_path=parsed.marker,
            executable=parsed.executable,
        )


def main() -> None:
    """Main execution function for FastSurfer CLI wrapper."""
    app = FastSurferApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
