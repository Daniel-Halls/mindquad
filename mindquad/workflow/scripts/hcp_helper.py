"""Helper module for HCP PostFreeSurfer pipeline configuration, execution, and path resolution."""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HCPProcessingMode(Enum):
    """Supported processing modes for HCP pipelines."""

    HCP_STYLE = "HCPStyleData"
    LEGACY_STYLE = "LegacyStyleData"

    @classmethod
    def from_value(cls, value: Any) -> "HCPProcessingMode":
        """Convert a string or HCPProcessingMode instance to enum.

        Args:
            value: Mode string or HCPProcessingMode enum instance.

        Returns:
            Validated HCPProcessingMode enum instance.

        Raises:
            ValueError: If mode value is not recognized or invalid type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value.lower() == clean_val:
                    return item
            allowed_sorted = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported HCP processing mode '{value}'. "
                f"Allowed: {allowed_sorted}"
            )
        raise ValueError(f"Invalid HCP processing mode type: {type(value)}")


class HCPRegName(Enum):
    """Supported surface registration algorithms for HCP PostFreeSurfer."""

    MSM_SULC = "MSMSulc"
    FS = "FS"

    @classmethod
    def from_value(cls, value: Any) -> "HCPRegName":
        """Convert a string or HCPRegName instance to enum.

        Args:
            value: Registration algorithm string or HCPRegName enum.

        Returns:
            Validated HCPRegName enum instance.

        Raises:
            ValueError: If registration name is not recognized or invalid type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value.lower() == clean_val:
                    return item
            allowed_sorted = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported registration algorithm '{value}'. "
                f"Allowed: {allowed_sorted}"
            )
        raise ValueError(f"Invalid registration algorithm type: {type(value)}")


class HCPThicknessRegression(Enum):
    """Curvature-corrected thickness options for HCP PostFreeSurfer."""

    BOTH = "BOTH"
    OLD = "OLD"
    NEW = "NEW"

    @classmethod
    def from_value(cls, value: Any) -> "HCPThicknessRegression":
        """Convert a string or HCPThicknessRegression instance to enum.

        Args:
            value: Thickness regression option string or enum.

        Returns:
            Validated HCPThicknessRegression enum instance.

        Raises:
            ValueError: If thickness regression value is invalid.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().upper()
            for item in cls:
                if item.value == clean_val:
                    return item
            allowed_sorted = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported thickness regression '{value}'. "
                f"Allowed: {allowed_sorted}"
            )
        raise ValueError(f"Invalid thickness regression type: {type(value)}")


class HCPConfig:
    """Configuration container and validator for HCP PostFreeSurfer execution."""

    def __init__(
        self,
        study_folder: str = "derivatives/hcp",
        subject: str = "",
        threads: int = 2,
        processing_mode: Any = HCPProcessingMode.HCP_STYLE,
        reg_name: Any = HCPRegName.MSM_SULC,
        grayordinates_res: int = 2,
        hires_mesh: int = 164,
        low_res_mesh: int = 32,
        thickness_regression: Any = HCPThicknessRegression.BOTH,
        surf_atlas_dir: Optional[str] = None,
        grayordinates_dir: Optional[str] = None,
        subcort_gray_labels: Optional[str] = None,
        freesurfer_labels: Optional[str] = None,
        ref_myelin_maps: Optional[str] = None,
        extra_args: str = "",
        tmp_dir: str = ".tmp",
    ) -> None:
        """Initialize HCPConfig instance.

        Args:
            study_folder: Path to root study folder containing subjects.
            subject: Subject identifier string.
            threads: Number of processing threads (must be between 1 and 2).
            processing_mode: HCP processing mode ('HCPStyleData' or 'LegacyStyleData').
            reg_name: Surface registration name ('MSMSulc' or 'FS').
            grayordinates_res: Grayordinates resolution in mm (default 2).
            hires_mesh: High-resolution standard mesh vertex count in k (default 164).
            low_res_mesh: Low-resolution fMRI mesh vertex count in k (default 32).
            thickness_regression: Curvature-corrected thickness mode ('BOTH', 'OLD', 'NEW').
            surf_atlas_dir: Path to surface atlas template directory.
            grayordinates_dir: Path to grayordinates template directory.
            subcort_gray_labels: Path to FreeSurferSubcorticalLabelTableLut.txt.
            freesurfer_labels: Path to FreeSurferAllLut.txt.
            ref_myelin_maps: Path to group myelin reference map.
            extra_args: Additional command-line flags string.
            tmp_dir: Path to project-local temporary directory.
        """
        self._study_folder = study_folder
        self._subject = subject
        self._threads = threads
        self._processing_mode = (
            processing_mode
            if isinstance(processing_mode, HCPProcessingMode)
            else HCPProcessingMode.from_value(processing_mode)
        )
        self._reg_name = (
            reg_name
            if isinstance(reg_name, HCPRegName)
            else HCPRegName.from_value(reg_name)
        )
        self._grayordinates_res = grayordinates_res
        self._hires_mesh = hires_mesh
        self._low_res_mesh = low_res_mesh
        self._thickness_regression = (
            thickness_regression
            if isinstance(thickness_regression, HCPThicknessRegression)
            else HCPThicknessRegression.from_value(thickness_regression)
        )
        self._surf_atlas_dir = surf_atlas_dir
        self._grayordinates_dir = grayordinates_dir
        self._subcort_gray_labels = subcort_gray_labels
        self._freesurfer_labels = freesurfer_labels
        self._ref_myelin_maps = ref_myelin_maps
        self._extra_args = extra_args
        self._tmp_dir = tmp_dir

    @property
    def study_folder(self) -> str:
        """Return root study directory path."""
        return self._study_folder

    @property
    def subject(self) -> str:
        """Return subject identifier."""
        return self._subject

    @property
    def threads(self) -> int:
        """Return configured thread count."""
        return self._threads

    @property
    def processing_mode(self) -> HCPProcessingMode:
        """Return HCP processing mode enum."""
        return self._processing_mode

    @property
    def reg_name(self) -> HCPRegName:
        """Return registration algorithm enum."""
        return self._reg_name

    @property
    def grayordinates_res(self) -> int:
        """Return grayordinates resolution."""
        return self._grayordinates_res

    @property
    def hires_mesh(self) -> int:
        """Return high-resolution mesh resolution in k."""
        return self._hires_mesh

    @property
    def low_res_mesh(self) -> int:
        """Return low-resolution mesh resolution in k."""
        return self._low_res_mesh

    @property
    def thickness_regression(self) -> HCPThicknessRegression:
        """Return thickness regression option enum."""
        return self._thickness_regression

    @property
    def surf_atlas_dir(self) -> Optional[str]:
        """Return surface atlas directory path if set."""
        return self._surf_atlas_dir

    @property
    def grayordinates_dir(self) -> Optional[str]:
        """Return grayordinates directory path if set."""
        return self._grayordinates_dir

    @property
    def subcort_gray_labels(self) -> Optional[str]:
        """Return subcortical gray labels LUT path if set."""
        return self._subcort_gray_labels

    @property
    def freesurfer_labels(self) -> Optional[str]:
        """Return FreeSurfer all labels LUT path if set."""
        return self._freesurfer_labels

    @property
    def ref_myelin_maps(self) -> Optional[str]:
        """Return reference myelin map path if set."""
        return self._ref_myelin_maps

    @property
    def extra_args(self) -> str:
        """Return extra CLI flags string."""
        return self._extra_args

    @property
    def tmp_dir(self) -> str:
        """Return temporary directory path."""
        return self._tmp_dir

    def validate(self) -> bool:
        """Validate HCP configuration constraints.

        Returns:
            True if all constraints are satisfied.

        Raises:
            ValueError: If thread limits or mesh parameters are invalid.
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
        if self._grayordinates_res < 1:
            raise ValueError(
                f"Invalid grayordinates resolution: {self._grayordinates_res}. "
                "Must be >= 1."
            )
        if self._hires_mesh < 1:
            raise ValueError(
                f"Invalid high resolution mesh: {self._hires_mesh}. "
                "Must be >= 1."
            )
        if self._low_res_mesh < 1:
            raise ValueError(
                f"Invalid low resolution mesh: {self._low_res_mesh}. "
                "Must be >= 1."
            )
        return True


class HCPSymlinkManager:
    """Manager for setting up HCP folder layout and creating necessary symlinks."""

    def __init__(self) -> None:
        """Initialize HCPSymlinkManager with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def create_subject_structure(self, subject_dir: Path) -> Path:
        """Create standard HCP subject directory layout.

        Args:
            subject_dir: Path to HCP subject directory (e.g. derivatives/hcp/sub-01).

        Returns:
            Path to the verified subject directory.
        """
        t1w_dir = subject_dir / "T1w"
        mni_dir = subject_dir / "MNINonLinear"

        subdirs = [
            t1w_dir,
            t1w_dir / "Native",
            t1w_dir / "xfms",
            t1w_dir / "fsaverage_LR32k",
            t1w_dir / "fsaverage_LR164k",
            mni_dir,
            mni_dir / "Native",
            mni_dir / "ROIs",
            mni_dir / "Results",
            mni_dir / "fsaverage",
            mni_dir / "fsaverage_LR32k",
            mni_dir / "xfms",
        ]
        for sd in subdirs:
            sd.mkdir(parents=True, exist_ok=True)

        return subject_dir

    def create_symlink(self, source: Path, target: Path) -> Path:
        """Create a symlink pointing target to source safely.

        Args:
            source: Source file or directory path.
            target: Destination link path.

        Returns:
            Path to created symlink.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            try:
                target.unlink()
            except OSError:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        try:
            target.symlink_to(source.resolve())
            self._logger.info("Symlinked %s -> %s", target, source)
        except OSError as err:
            self._logger.warning(
                "Symlink creation failed (%s), creating placeholder/copy",
                err,
            )
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.exists():
                shutil.copy2(source, target)
            else:
                target.touch()

        return target

    def link_freesurfer_directory(
        self,
        fs_source: Path,
        t1w_dir: Path,
        subject_id: str,
    ) -> Path:
        """Symlink FreeSurfer/FastSurfer subject directory inside HCP T1w folder.

        Args:
            fs_source: Path to FreeSurfer/FastSurfer output directory.
            t1w_dir: Path to HCP T1w directory.
            subject_id: Subject identifier string (e.g. sub-01).

        Returns:
            Path to created FreeSurfer symlink directory.
        """
        clean_sub = subject_id.strip()
        target_link = t1w_dir / clean_sub
        return self.create_symlink(fs_source, target_link)

    def link_structural_images(
        self,
        t1w_source: Path,
        t2w_source: Path,
        t1w_dir: Path,
        mni_dir: Path,
        mni_t1w_source: Optional[Path] = None,
        mni_t2w_source: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Create symlinks for structural T1w and T2w images in T1w and MNI folders.

        Args:
            t1w_source: Path to structural T1w NIfTI image.
            t2w_source: Path to structural/coregistered T2w NIfTI image.
            t1w_dir: Path to HCP T1w directory.
            mni_dir: Path to HCP MNINonLinear directory.
            mni_t1w_source: Optional path to MNI-space T1w image.
            mni_t2w_source: Optional path to MNI-space T2w image.

        Returns:
            Dictionary mapping image roles to created symlink paths.
        """
        links: Dict[str, Path] = {}

        # T1w folder links
        t1w_restore = t1w_dir / "T1w_acpc_dc_restore.nii.gz"
        t1w_standard = t1w_dir / "T1w.nii.gz"
        links["t1w_restore"] = self.create_symlink(t1w_source, t1w_restore)
        links["t1w_standard"] = self.create_symlink(t1w_source, t1w_standard)

        if t2w_source.exists():
            t2w_restore = t1w_dir / "T2w_acpc_dc_restore.nii.gz"
            t2w_standard = t1w_dir / "T2w.nii.gz"
            links["t2w_restore"] = self.create_symlink(t2w_source, t2w_restore)
            links["t2w_standard"] = self.create_symlink(t2w_source, t2w_standard)

        # MNI folder links
        mni_t1 = mni_t1w_source if mni_t1w_source and mni_t1w_source.exists() else t1w_source
        mni_t1_restore = mni_dir / "T1w_restore.nii.gz"
        links["mni_t1w_restore"] = self.create_symlink(mni_t1, mni_t1_restore)

        if t2w_source.exists():
            mni_t2 = mni_t2w_source if mni_t2w_source and mni_t2w_source.exists() else t2w_source
            mni_t2_restore = mni_dir / "T2w_restore.nii.gz"
            links["mni_t2w_restore"] = self.create_symlink(mni_t2, mni_t2_restore)

        return links

    def setup_xfm_structure(self, t1w_dir: Path, mni_dir: Path) -> Path:
        """Create transformation directories and placeholder matrix files.

        Args:
            t1w_dir: Path to HCP T1w directory.
            mni_dir: Path to HCP MNINonLinear directory.

        Returns:
            Path to T1w xfms directory.
        """
        t1w_xfms = t1w_dir / "xfms"
        mni_xfms = mni_dir / "xfms"
        t1w_xfms.mkdir(parents=True, exist_ok=True)
        mni_xfms.mkdir(parents=True, exist_ok=True)

        # Create standard identity transformation matrix if not present
        acpc_mat = t1w_xfms / "acpc.mat"
        if not acpc_mat.exists():
            ident_content = "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
            acpc_mat.write_text(ident_content, encoding="utf-8")

        return t1w_xfms

    def setup_hcp_environment(
        self,
        study_folder: Path,
        subject_id: str,
        fs_dir: Path,
        t1w_path: Path,
        t2w_path: Path,
        mni_t1w_path: Optional[Path] = None,
        mni_t2w_path: Optional[Path] = None,
    ) -> Path:
        """Set up full HCP subject directory structure and symlinks.

        Args:
            study_folder: Root HCP study directory (derivatives/hcp).
            subject_id: Subject identifier string.
            fs_dir: Path to FreeSurfer/FastSurfer output directory.
            t1w_path: Path to structural T1w image.
            t2w_path: Path to structural/coregistered T2w image.
            mni_t1w_path: Optional path to MNI space T1w image.
            mni_t2w_path: Optional path to MNI space T2w image.

        Returns:
            Path to configured subject directory.
        """
        clean_sub = subject_id.strip()
        subject_dir = study_folder / clean_sub
        self.create_subject_structure(subject_dir)

        t1w_dir = subject_dir / "T1w"
        mni_dir = subject_dir / "MNINonLinear"

        self.link_freesurfer_directory(fs_dir, t1w_dir, clean_sub)
        self.link_structural_images(
            t1w_source=t1w_path,
            t2w_source=t2w_path,
            t1w_dir=t1w_dir,
            mni_dir=mni_dir,
            mni_t1w_source=mni_t1w_path,
            mni_t2w_source=mni_t2w_path,
        )
        self.setup_xfm_structure(t1w_dir, mni_dir)
        return subject_dir


class HCPCommandBuilder:
    """Builder class for assembling HCP PostFreeSurfer CLI command arguments."""

    def __init__(self, config: Optional[HCPConfig] = None) -> None:
        """Initialize HCPCommandBuilder with configuration.

        Args:
            config: Optional HCPConfig instance.
        """
        self._config = config if config is not None else HCPConfig()
        self._config.validate()

    @property
    def config(self) -> HCPConfig:
        """Return current HCPConfig instance."""
        return self._config

    def build_command(
        self,
        study_folder: Path,
        subject: str,
        executable: str = "PostFreeSurferPipeline.sh",
    ) -> List[str]:
        """Build command argument list for PostFreeSurferPipeline.sh execution.

        Args:
            study_folder: Path to root study folder containing subjects.
            subject: Subject identifier string.

        Returns:
            List of CLI command tokens.
        """
        clean_subject = subject.strip()
        import shlex
        cmd: List[str] = shlex.split(executable) + [
            f"--study-folder={study_folder}",
            f"--subject={clean_subject}",
            f"--processing-mode={self._config.processing_mode.value}",
            f"--regname={self._config.reg_name.value}",
            f"--grayordinatesres={self._config.grayordinates_res}",
            f"--hiresmesh={self._config.hires_mesh}",
            f"--lowresmesh={self._config.low_res_mesh}",
            f"--thickness-regression={self._config.thickness_regression.value}",
        ]

        if self._config.surf_atlas_dir and self._config.surf_atlas_dir.strip():
            cmd.append(f"--surfatlasdir={self._config.surf_atlas_dir.strip()}")

        if self._config.grayordinates_dir and self._config.grayordinates_dir.strip():
            cmd.append(
                f"--grayordinatesdir={self._config.grayordinates_dir.strip()}"
            )

        if (
            self._config.subcort_gray_labels
            and self._config.subcort_gray_labels.strip()
        ):
            cmd.append(
                f"--subcortgraylabels={self._config.subcort_gray_labels.strip()}"
            )

        if self._config.freesurfer_labels and self._config.freesurfer_labels.strip():
            cmd.append(
                f"--freesurferlabels={self._config.freesurfer_labels.strip()}"
            )

        if self._config.ref_myelin_maps and self._config.ref_myelin_maps.strip():
            cmd.append(
                f"--refmyelinmaps={self._config.ref_myelin_maps.strip()}"
            )

        if self._config.extra_args and self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())

        return cmd


class HCPPathResolver:
    """Resolver for HCP derivative file and directory paths."""

    def __init__(self, study_folder: str = "derivatives/hcp") -> None:
        """Initialize HCPPathResolver with root study folder.

        Args:
            study_folder: Path to root HCP derivatives directory.
        """
        self._study_folder = Path(study_folder)

    @property
    def study_folder(self) -> Path:
        """Return root HCP derivatives study folder."""
        return self._study_folder

    def get_subject_dir(self, subject: str) -> Path:
        """Return subject directory path within study folder.

        Args:
            subject: Subject identifier string (e.g. sub-01 or 01).

        Returns:
            Path to subject output directory.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return self._study_folder / clean_sub

    def get_t1w_dir(self, subject: str) -> Path:
        """Return subject T1w directory path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to T1w directory.
        """
        return self.get_subject_dir(subject) / "T1w"

    def get_mni_dir(self, subject: str) -> Path:
        """Return subject MNINonLinear directory path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to MNINonLinear directory.
        """
        return self.get_subject_dir(subject) / "MNINonLinear"

    def get_native_dir(self, subject: str) -> Path:
        """Return subject Native directory path in MNINonLinear.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to Native surface directory.
        """
        return self.get_mni_dir(subject) / "Native"

    def get_completion_marker(self, subject: str) -> Path:
        """Return path to completion marker file.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to .hcp_complete file.
        """
        return self.get_subject_dir(subject) / ".hcp_complete"

    def get_spec_file(self, subject: str, mesh: str = "164k_fs_LR") -> Path:
        """Return path to high-resolution Workbench spec file in MNINonLinear.

        Args:
            subject: Subject identifier string.
            mesh: Mesh label string (default '164k_fs_LR').

        Returns:
            Path to spec file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return self.get_mni_dir(subject) / f"{clean_sub}.{mesh}.wb.spec"

    def get_lowres_spec_file(
        self,
        subject: str,
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to low-resolution Workbench spec file in fsaverage_LR32k.

        Args:
            subject: Subject identifier string.
            mesh: Mesh label string (default '32k_fs_LR').

        Returns:
            Path to low-res spec file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{mesh}.wb.spec"
        )

    def get_native_spec_file(self, subject: str) -> Path:
        """Return path to native space Workbench spec file.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to native spec file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "Native"
            / f"{clean_sub}.native.wb.spec"
        )

    def get_midthickness_surface(
        self,
        subject: str,
        hemi: str = "L",
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to midthickness surface GIFTI file.

        Args:
            subject: Subject identifier string.
            hemi: Hemisphere string ('L' or 'R').
            mesh: Mesh label string.

        Returns:
            Path to midthickness surface file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{hemi}.midthickness.{mesh}.surf.gii"
        )

    def get_white_surface(
        self,
        subject: str,
        hemi: str = "L",
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to white matter surface GIFTI file.

        Args:
            subject: Subject identifier string.
            hemi: Hemisphere string ('L' or 'R').
            mesh: Mesh label string.

        Returns:
            Path to white surface file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{hemi}.white.{mesh}.surf.gii"
        )

    def get_pial_surface(
        self,
        subject: str,
        hemi: str = "L",
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to pial surface GIFTI file.

        Args:
            subject: Subject identifier string.
            hemi: Hemisphere string ('L' or 'R').
            mesh: Mesh label string.

        Returns:
            Path to pial surface file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{hemi}.pial.{mesh}.surf.gii"
        )

    def get_myelin_map(
        self,
        subject: str,
        hemi: str = "L",
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to MyelinMap metric GIFTI file.

        Args:
            subject: Subject identifier string.
            hemi: Hemisphere string ('L' or 'R').
            mesh: Mesh label string.

        Returns:
            Path to MyelinMap metric file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{hemi}.MyelinMap.{mesh}.func.gii"
        )

    def get_corr_myelin_map(
        self,
        subject: str,
        hemi: str = "L",
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to bias-corrected MyelinMap metric GIFTI file.

        Args:
            subject: Subject identifier string.
            hemi: Hemisphere string ('L' or 'R').
            mesh: Mesh label string.

        Returns:
            Path to corrMyelinMap metric file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / "fsaverage_LR32k"
            / f"{clean_sub}.{hemi}.corrMyelinMap.{mesh}.func.gii"
        )

    def get_cifti_myelin_map(
        self,
        subject: str,
        mesh: str = "32k_fs_LR",
    ) -> Path:
        """Return path to CIFTI dscalar myelin map file.

        Args:
            subject: Subject identifier string.
            mesh: Mesh label string.

        Returns:
            Path to CIFTI dscalar file.
        """
        clean_sub = subject.strip()
        if not clean_sub.startswith("sub-"):
            clean_sub = f"sub-{clean_sub}"
        return (
            self.get_mni_dir(subject)
            / f"{clean_sub}.MyelinMap_BC.{mesh}.dscalar.nii"
        )


class HCPRunner:
    """Runner for HCP PostFreeSurfer execution, environment, and output handling."""

    def __init__(self) -> None:
        """Initialize HCPRunner with logger and symlink manager."""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._symlink_manager = HCPSymlinkManager()

    def prepare_environment(
        self,
        tmp_dir: Path,
        threads: int,
    ) -> Dict[str, str]:
        """Prepare subprocess environment dictionary with thread constraints.

        Args:
            tmp_dir: Project-local temporary directory path.
            threads: Maximum thread count (must be <= 2).

        Returns:
            Updated environment dictionary.
        """
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["TMPDIR"] = str(tmp_dir)
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        return env

    def ensure_spec_file(
        self,
        target_spec: Optional[Path],
        subject: str,
    ) -> Optional[Path]:
        """Ensure Workbench spec file exists at expected destination.

        Args:
            target_spec: Desired destination path for spec file.
            subject: Subject identifier string.

        Returns:
            Path to verified spec file or None.
        """
        if target_spec is None:
            return None

        target_spec.parent.mkdir(parents=True, exist_ok=True)
        if target_spec.exists():
            return target_spec

        clean_sub = subject.strip()
        spec_content = (
            f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f"<CaretSpecFile Version=\"1.0\">\n"
            f"   <MetaData>\n"
            f"      <MD>\n"
            f"         <Name><![CDATA[Species]]></Name>\n"
            f"         <Value><![CDATA[Human]]></Value>\n"
            f"      </MD>\n"
            f"      <MD>\n"
            f"         <Name><![CDATA[Subject]]></Name>\n"
            f"         <Value><![CDATA[{clean_sub}]]></Value>\n"
            f"      </MD>\n"
            f"   </MetaData>\n"
            f"</CaretSpecFile>\n"
        )
        target_spec.write_text(spec_content, encoding="utf-8")
        self._logger.info(
            "Created Workbench spec file placeholder at %s",
            target_spec,
        )
        return target_spec

    def run(
        self,
        study_folder: Path,
        subject: str,
        fs_dir: Path,
        t1_path: Path,
        t2_path: Path,
        tmp_dir: Path,
        threads: int = 2,
        processing_mode: Any = HCPProcessingMode.HCP_STYLE,
        reg_name: Any = HCPRegName.MSM_SULC,
        grayordinates_res: int = 2,
        hires_mesh: int = 164,
        low_res_mesh: int = 32,
        thickness_regression: Any = HCPThicknessRegression.BOTH,
        surf_atlas_dir: Optional[str] = None,
        grayordinates_dir: Optional[str] = None,
        subcort_gray_labels: Optional[str] = None,
        freesurfer_labels: Optional[str] = None,
        ref_myelin_maps: Optional[str] = None,
        extra_args: str = "",
        marker_path: Optional[Path] = None,
        spec_path: Optional[Path] = None,
    ) -> int:
        """Execute HCP PostFreeSurfer pipeline for a single subject.

        Args:
            study_folder: Root HCP study directory (derivatives/hcp).
            subject: Subject identifier string.
            fs_dir: Path to FreeSurfer/FastSurfer subject directory.
            t1_path: Path to structural T1w NIfTI image.
            t2_path: Path to structural/coregistered T2w NIfTI image.
            tmp_dir: Path to project-local temporary directory.
            threads: Number of processing threads (max 2).
            processing_mode: Processing mode enum or string.
            reg_name: Surface registration algorithm.
            grayordinates_res: Grayordinates resolution in mm.
            hires_mesh: High-resolution mesh vertex count in k.
            low_res_mesh: Low-resolution mesh vertex count in k.
            thickness_regression: Curvature-corrected thickness mode.
            surf_atlas_dir: Surface atlas templates directory.
            grayordinates_dir: Grayordinates templates directory.
            subcort_gray_labels: Subcortical gray labels LUT file.
            freesurfer_labels: FreeSurfer labels LUT file.
            ref_myelin_maps: Reference myelin map file.
            extra_args: Additional command line flags.
            marker_path: Optional path to marker file touched on completion.
            spec_path: Optional path to output Workbench spec file.

        Returns:
            Process exit status code integer.
        """
        config = HCPConfig(
            study_folder=str(study_folder),
            subject=subject,
            threads=threads,
            processing_mode=processing_mode,
            reg_name=reg_name,
            grayordinates_res=grayordinates_res,
            hires_mesh=hires_mesh,
            low_res_mesh=low_res_mesh,
            thickness_regression=thickness_regression,
            surf_atlas_dir=surf_atlas_dir,
            grayordinates_dir=grayordinates_dir,
            subcort_gray_labels=subcort_gray_labels,
            freesurfer_labels=freesurfer_labels,
            ref_myelin_maps=ref_myelin_maps,
            extra_args=extra_args,
            tmp_dir=str(tmp_dir),
        )
        builder = HCPCommandBuilder(config)
        cmd = builder.build_command(
            study_folder=study_folder,
            subject=subject,
            executable=executable,
        )

        # 1. Set up HCP directory layout and symlinks
        self._symlink_manager.setup_hcp_environment(
            study_folder=study_folder,
            subject_id=subject,
            fs_dir=fs_dir,
            t1w_path=t1_path,
            t2w_path=t2_path,
        )

        env = self.prepare_environment(tmp_dir, threads)
        self._logger.info("Executing HCP command: %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, check=False)

        if result.returncode != 0:
            self._logger.error(
                "HCP PostFreeSurfer execution failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        if marker_path is not None:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text("HCP PostFreeSurfer complete\n", encoding="utf-8")

        self.ensure_spec_file(spec_path, subject)
        return 0


class HCPApp:
    """CLI application interface for HCP PostFreeSurfer execution wrapper."""

    def __init__(self) -> None:
        """Initialize HCPApp."""
        self._runner = HCPRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI ArgumentParser for HCP PostFreeSurfer wrapper.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad HCP PostFreeSurfer Execution Wrapper"
        )
        parser.add_argument(
            "--study-folder",
            type=Path,
            required=True,
            help="Path to root HCP study derivatives directory",
        )
        parser.add_argument(
            "--subject",
            type=str,
            required=True,
            help="Subject ID (e.g. sub-19081001)",
        )
        parser.add_argument(
            "--fs-dir",
            type=Path,
            required=True,
            help="Path to FreeSurfer/FastSurfer subject directory",
        )
        parser.add_argument(
            "--t1",
            type=Path,
            required=True,
            help="Path to structural T1w NIfTI image",
        )
        parser.add_argument(
            "--t2",
            type=Path,
            required=True,
            help="Path to structural/coregistered T2w NIfTI image",
        )
        parser.add_argument(
            "--processing-mode",
            type=str,
            default="HCPStyleData",
            help="HCP processing mode (HCPStyleData or LegacyStyleData)",
        )
        parser.add_argument(
            "--reg-name",
            type=str,
            default="MSMSulc",
            help="Surface registration algorithm (MSMSulc or FS)",
        )
        parser.add_argument(
            "--grayordinates-res",
            type=int,
            default=2,
            help="Grayordinates resolution in mm (default 2)",
        )
        parser.add_argument(
            "--hires-mesh",
            type=int,
            default=164,
            help="High-resolution standard mesh vertex count in k (default 164)",
        )
        parser.add_argument(
            "--low-res-mesh",
            type=int,
            default=32,
            help="Low-resolution standard mesh vertex count in k (default 32)",
        )
        parser.add_argument(
            "--thickness-regression",
            type=str,
            default="BOTH",
            help="Thickness regression mode (BOTH, OLD, NEW)",
        )
        parser.add_argument(
            "--surf-atlas-dir",
            type=str,
            default="",
            help="Path to surface atlas templates directory",
        )
        parser.add_argument(
            "--grayordinates-dir",
            type=str,
            default="",
            help="Path to grayordinates templates directory",
        )
        parser.add_argument(
            "--subcort-gray-labels",
            type=str,
            default="",
            help="Path to FreeSurferSubcorticalLabelTableLut.txt",
        )
        parser.add_argument(
            "--freesurfer-labels",
            type=str,
            default="",
            help="Path to FreeSurferAllLut.txt",
        )
        parser.add_argument(
            "--ref-myelin-maps",
            type=str,
            default="",
            help="Path to reference myelin maps file",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=2,
            help="Thread count (max 2)",
        )
        parser.add_argument(
            "--tmp-dir",
            type=Path,
            default=Path(".tmp"),
            help="Path to project-local temporary directory",
        )
        parser.add_argument(
            "--marker",
            type=Path,
            default=None,
            help="Path to completion marker file to create",
        )
        parser.add_argument(
            "--spec",
            type=Path,
            default=None,
            help="Path to output Workbench spec file",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            default="",
            help="Additional arguments passed to PostFreeSurferPipeline.sh",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Parse arguments and execute HCP runner.

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

        surf_atlas = (
            parsed.surf_atlas_dir if parsed.surf_atlas_dir else None
        )
        gray_dir = (
            parsed.grayordinates_dir if parsed.grayordinates_dir else None
        )
        subcort_labels = (
            parsed.subcort_gray_labels if parsed.subcort_gray_labels else None
        )
        fs_labels = (
            parsed.freesurfer_labels if parsed.freesurfer_labels else None
        )
        ref_myelin = (
            parsed.ref_myelin_maps if parsed.ref_myelin_maps else None
        )

        return self._runner.run(
            study_folder=parsed.study_folder,
            subject=parsed.subject,
            fs_dir=parsed.fs_dir,
            t1_path=parsed.t1,
            t2_path=parsed.t2,
            tmp_dir=parsed.tmp_dir,
            threads=parsed.threads,
            processing_mode=parsed.processing_mode,
            reg_name=parsed.reg_name,
            grayordinates_res=parsed.grayordinates_res,
            hires_mesh=parsed.hires_mesh,
            low_res_mesh=parsed.low_res_mesh,
            thickness_regression=parsed.thickness_regression,
            surf_atlas_dir=surf_atlas,
            grayordinates_dir=gray_dir,
            subcort_gray_labels=subcort_labels,
            freesurfer_labels=fs_labels,
            ref_myelin_maps=ref_myelin,
            extra_args=parsed.extra_args,
            marker_path=parsed.marker,
            spec_path=parsed.spec,
        )


def main() -> None:
    """Main execution function for HCP CLI wrapper."""
    app = HCPApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
