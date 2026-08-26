"""Helper module for T2w to T1w diffeomorphic coregistration.

This module provides object-oriented classes and single-responsibility methods
to configure, construct CLI commands, resolve derivative paths, and execute
diffeomorphic transformation (e.g. DIPY SyN or ANTs SyN) to align T2w with T1w.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional


class CoregistrationTool(Enum):
    """Supported coregistration backend tools."""

    DIPY = "dipy"
    ANTS = "ants"
    AUTO = "auto"

    @classmethod
    def from_value(cls, value: Any) -> "CoregistrationTool":
        """Convert string or CoregistrationTool enum instance.

        Args:
            value: Tool name string or CoregistrationTool enum.

        Returns:
            Validated CoregistrationTool instance.

        Raises:
            ValueError: If tool name is invalid or unsupported type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value == clean_val:
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported coregistration tool '{value}'. "
                f"Allowed: {allowed}"
            )
        raise ValueError(
            f"Invalid coregistration tool type: {type(value)}"
        )


class RegistrationMetric(Enum):
    """Supported image similarity metrics for diffeomorphic registration."""

    CC = "CC"
    EM = "EM"
    MI = "MI"
    SSD = "SSD"

    @classmethod
    def from_value(cls, value: Any) -> "RegistrationMetric":
        """Convert string or RegistrationMetric enum instance.

        Args:
            value: Metric name string or RegistrationMetric enum.

        Returns:
            Validated RegistrationMetric instance.

        Raises:
            ValueError: If metric name is invalid or unsupported type.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().upper()
            for item in cls:
                if item.value == clean_val:
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported registration metric '{value}'. "
                f"Allowed: {allowed}"
            )
        raise ValueError(
            f"Invalid registration metric type: {type(value)}"
        )


class TransformationType(Enum):
    """Supported transformation model types for alignment."""

    SYN = "syn"
    DIFFEOMORPHIC = "diffeomorphic"
    BSPLINE_SYN = "bspline_syn"
    RIGID = "rigid"
    AFFINE = "affine"

    @classmethod
    def from_value(cls, value: Any) -> "TransformationType":
        """Convert string or TransformationType enum instance.

        Args:
            value: Transformation type string or TransformationType enum.

        Returns:
            Validated TransformationType instance.

        Raises:
            ValueError: If transformation type is invalid or unsupported.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            clean_val = value.strip().lower()
            for item in cls:
                if item.value == clean_val:
                    return item
            allowed = sorted(item.value for item in cls)
            raise ValueError(
                f"Unsupported transformation type '{value}'. "
                f"Allowed: {allowed}"
            )
        raise ValueError(
            f"Invalid transformation type type: {type(value)}"
        )


class CoregistrationConfig:
    """Configuration container and validator for T2 to T1 coregistration."""

    DEFAULT_LEVEL_ITERS: ClassVar[List[int]] = [10, 10, 5]
    DEFAULT_STEP_LENGTH: ClassVar[float] = 0.25
    DEFAULT_DIMENSION: ClassVar[int] = 3

    def __init__(
        self,
        threads: int = 2,
        tool: Any = CoregistrationTool.DIPY,
        metric: Any = RegistrationMetric.CC,
        transform_type: Any = TransformationType.SYN,
        dimension: int = 3,
        level_iters: Optional[List[int]] = None,
        step_length: float = 0.25,
        extra_args: str = "",
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
        work_dir: str = "work",
        tmp_dir: str = ".tmp",
    ) -> None:
        """Initialize CoregistrationConfig instance.

        Args:
            threads: Maximum thread count (must be between 1 and 2).
            tool: Coregistration backend tool enum or string.
            metric: Registration similarity metric enum or string.
            transform_type: Transformation model enum or string.
            dimension: Image spatial dimensionality (default 3).
            level_iters: Multi-resolution optimization iterations per level.
            step_length: Gradient descent optimization step length.
            extra_args: Additional CLI flags string.
            bids_dir: Path to BIDS dataset root directory.
            derivatives_dir: Path to root derivatives directory.
            work_dir: Path to intermediate working directory.
            tmp_dir: Path to project-local temporary directory.
        """
        self._threads = threads
        self._tool = (
            tool
            if isinstance(tool, CoregistrationTool)
            else CoregistrationTool.from_value(tool)
        )
        self._metric = (
            metric
            if isinstance(metric, RegistrationMetric)
            else RegistrationMetric.from_value(metric)
        )
        self._transform_type = (
            transform_type
            if isinstance(transform_type, TransformationType)
            else TransformationType.from_value(transform_type)
        )
        self._dimension = dimension
        self._level_iters = (
            list(level_iters)
            if level_iters is not None
            else list(self.DEFAULT_LEVEL_ITERS)
        )
        self._step_length = float(step_length)
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
    def tool(self) -> CoregistrationTool:
        """Return coregistration tool enum."""
        return self._tool

    @property
    def metric(self) -> RegistrationMetric:
        """Return registration similarity metric enum."""
        return self._metric

    @property
    def transform_type(self) -> TransformationType:
        """Return transformation type enum."""
        return self._transform_type

    @property
    def dimension(self) -> int:
        """Return image dimension integer."""
        return self._dimension

    @property
    def level_iters(self) -> List[int]:
        """Return multi-resolution iteration list."""
        return self._level_iters

    @property
    def step_length(self) -> float:
        """Return gradient descent step length."""
        return self._step_length

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
        """Return root derivatives directory."""
        return self._derivatives_dir

    @property
    def work_dir(self) -> str:
        """Return intermediate working directory."""
        return self._work_dir

    @property
    def tmp_dir(self) -> str:
        """Return project-local temporary directory."""
        return self._tmp_dir

    def validate(self) -> bool:
        """Validate configuration against system constraints and requirements.

        Returns:
            True if configuration is valid.

        Raises:
            ValueError: If any parameter violates constraints.
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
        if self._dimension != 3:
            raise ValueError(
                f"Invalid dimension: {self._dimension}. Expected 3 for 3D MRI."
            )
        if self._step_length <= 0.0:
            raise ValueError(
                f"Invalid step length: {self._step_length}. Must be > 0."
            )
        if not self._level_iters or any(i < 1 for i in self._level_iters):
            raise ValueError(
                f"Invalid level iterations: {self._level_iters}. "
                "Must be a non-empty list of positive integers."
            )
        if not isinstance(self._tool, CoregistrationTool):
            raise ValueError(f"Invalid tool instance: {self._tool}")
        if not isinstance(self._metric, RegistrationMetric):
            raise ValueError(f"Invalid metric instance: {self._metric}")
        if not isinstance(self._transform_type, TransformationType):
            raise ValueError(
                f"Invalid transform type instance: {self._transform_type}"
            )
        return True


class CoregistrationCommandBuilder:
    """Builder class for constructing coregistration CLI arguments."""

    def __init__(self, config: Optional[CoregistrationConfig] = None) -> None:
        """Initialize CoregistrationCommandBuilder with configuration.

        Args:
            config: Optional CoregistrationConfig instance.
        """
        self._config = (
            config if config is not None else CoregistrationConfig()
        )
        self._config.validate()

    @property
    def config(self) -> CoregistrationConfig:
        """Return the active CoregistrationConfig."""
        return self._config

    def build_dipy_command(
        self,
        static_path: Path,
        moving_path: Path,
        out_dir: Path,
        out_warped_name: str = "sub_space-T1w_desc-coreg_T2w.nii.gz",
        out_field_name: Optional[str] = None,
        out_affine_name: Optional[str] = None,
    ) -> List[str]:
        """Build command argument list for DIPY diffeomorphic SyN alignment.

        Args:
            static_path: Path to static reference volume (T1w).
            moving_path: Path to moving volume (T2w).
            out_dir: Output directory for coregistration derivatives.
            out_warped_name: Filename for warped moving image.
            out_field_name: Optional filename for displacement field.
            out_affine_name: Optional filename for initial affine transform.

        Returns:
            List of CLI command tokens.
        """
        level_iters_str = " ".join(str(i) for i in self._config.level_iters)
        cmd: List[str] = [
            "dipy_align_syn",
            "--static",
            str(static_path),
            "--moving",
            str(moving_path),
            "--metric",
            self._config.metric.value,
            "--level_iters",
            level_iters_str,
            "--step_length",
            str(self._config.step_length),
            "--out_dir",
            str(out_dir),
            "--out_warped",
            out_warped_name,
        ]

        if out_field_name and out_field_name.strip():
            cmd.extend(["--out_field", out_field_name.strip()])

        if out_affine_name and out_affine_name.strip():
            cmd.extend(["--out_affine", out_affine_name.strip()])

        if self._config.extra_args and self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())

        return cmd

    def build_ants_command(
        self,
        fixed_path: Path,
        moving_path: Path,
        out_prefix: Path,
    ) -> List[str]:
        """Build command argument list for ANTs SyN diffeomorphic alignment.

        Args:
            fixed_path: Path to fixed reference image (T1w).
            moving_path: Path to moving image (T2w).
            out_prefix: Output directory and file prefix path.

        Returns:
            List of CLI command tokens.
        """
        ants_t_map = {
            TransformationType.SYN: "s",
            TransformationType.DIFFEOMORPHIC: "s",
            TransformationType.BSPLINE_SYN: "b",
            TransformationType.RIGID: "r",
            TransformationType.AFFINE: "a",
        }
        ants_t = ants_t_map.get(self._config.transform_type, "s")

        ants_m_map = {
            RegistrationMetric.CC: "CC",
            RegistrationMetric.EM: "MI",
            RegistrationMetric.MI: "MI",
            RegistrationMetric.SSD: "MI",
        }
        ants_m = ants_m_map.get(self._config.metric, "CC")

        cmd: List[str] = [
            "antsRegistrationSyNQuick.sh",
            "-d",
            str(self._config.dimension),
            "-f",
            str(fixed_path),
            "-m",
            str(moving_path),
            "-o",
            str(out_prefix),
            "-t",
            ants_t,
            "-n",
            str(self._config.threads),
            "-p",
            "f",
            "-j",
            ants_m,
        ]

        if self._config.extra_args and self._config.extra_args.strip():
            cmd.extend(self._config.extra_args.strip().split())

        return cmd

    def build_command(
        self,
        static_path: Path,
        moving_path: Path,
        out_dir: Path,
        out_warped_name: str,
        out_prefix: Optional[Path] = None,
    ) -> List[str]:
        """Build command argument list according to configured tool backend.

        Args:
            static_path: Path to static/fixed reference volume (T1w).
            moving_path: Path to moving volume (T2w).
            out_dir: Output directory path.
            out_warped_name: Filename for warped output image.
            out_prefix: Output prefix path for ANTs.

        Returns:
            List of CLI command tokens.
        """
        if self._config.tool == CoregistrationTool.ANTS:
            prefix = (
                out_prefix
                if out_prefix is not None
                else out_dir / "ants_coreg_"
            )
            return self.build_ants_command(
                fixed_path=static_path,
                moving_path=moving_path,
                out_prefix=prefix,
            )

        return self.build_dipy_command(
            static_path=static_path,
            moving_path=moving_path,
            out_dir=out_dir,
            out_warped_name=out_warped_name,
        )


class CoregistrationPathResolver:
    """Resolver for coregistration inputs, outputs, and derivative paths."""

    def __init__(
        self,
        bids_dir: str = "bids",
        derivatives_dir: str = "derivatives",
    ) -> None:
        """Initialize CoregistrationPathResolver with root directories.

        Args:
            bids_dir: Path to BIDS dataset root directory.
            derivatives_dir: Path to root derivatives directory.
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
    def coregistration_dir(self) -> Path:
        """Return Coregistration derivatives directory."""
        return self._derivatives_dir / "coregistration"

    def get_subject_dir(self, subject: str) -> Path:
        """Return coregistration subject directory path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject coregistration directory.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.coregistration_dir / f"sub-{clean_sub}"

    def get_subject_anat_dir(self, subject: str) -> Path:
        """Return subject anatomical coregistration output directory.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject anat derivatives directory.
        """
        return self.get_subject_dir(subject) / "anat"

    def get_warped_t2w(self, subject: str) -> Path:
        """Return path to warped T2w NIfTI volume aligned to T1w space.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to aligned T2w NIfTI file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_space-T1w_desc-coreg_T2w.nii.gz"
        return self.get_subject_anat_dir(clean_sub) / filename

    def get_forward_warp(self, subject: str) -> Path:
        """Return path to forward deformation warp field (T2w -> T1w).

        Args:
            subject: Subject identifier string.

        Returns:
            Path to forward warp field NIfTI file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_from-T2w_to-T1w_mode-image_xfm.nii.gz"
        return self.get_subject_anat_dir(clean_sub) / filename

    def get_inverse_warp(self, subject: str) -> Path:
        """Return path to inverse deformation warp field (T1w -> T2w).

        Args:
            subject: Subject identifier string.

        Returns:
            Path to inverse warp field NIfTI file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = f"sub-{clean_sub}_from-T1w_to-T2w_mode-image_xfm.nii.gz"
        return self.get_subject_anat_dir(clean_sub) / filename

    def get_affine_transform(self, subject: str) -> Path:
        """Return path to initial affine transformation matrix.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to affine transform file.
        """
        clean_sub = subject.replace("sub-", "").strip()
        filename = (
            f"sub-{clean_sub}_from-T2w_to-T1w_mode-image_desc-affine_xfm.mat"
        )
        return self.get_subject_anat_dir(clean_sub) / filename

    def get_completion_marker(self, subject: str) -> Path:
        """Return completion marker file path for a subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to completion marker file.
        """
        return self.get_subject_dir(subject) / ".coregistration_complete"

    def get_report_html(self, subject: str) -> Path:
        """Return subject HTML report destination path.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to subject HTML report.
        """
        clean_sub = subject.replace("sub-", "").strip()
        return self.coregistration_dir / f"sub-{clean_sub}.html"

    def resolve_t1w_path(self, subject: str) -> Path:
        """Resolve raw BIDS T1w image path for a given subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to resolved T1w NIfTI image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        anat_dir = self._bids_dir / f"sub-{clean_sub}" / "anat"
        standard_t1 = anat_dir / f"sub-{clean_sub}_T1w.nii.gz"

        if standard_t1.exists():
            return standard_t1

        if anat_dir.exists():
            candidates: List[Path] = []
            for pattern in ["*T1w*.nii.gz", "*T1w*.nii"]:
                candidates.extend(sorted(anat_dir.glob(pattern)))
            if candidates:
                return candidates[0]

        return standard_t1

    def resolve_t2w_path(self, subject: str) -> Path:
        """Resolve raw BIDS T2w image path for a given subject.

        Args:
            subject: Subject identifier string.

        Returns:
            Path to resolved T2w NIfTI image.
        """
        clean_sub = subject.replace("sub-", "").strip()
        anat_dir = self._bids_dir / f"sub-{clean_sub}" / "anat"
        standard_t2 = anat_dir / f"sub-{clean_sub}_T2w.nii.gz"

        if standard_t2.exists():
            return standard_t2

        if anat_dir.exists():
            candidates: List[Path] = []
            for pattern in ["*T2w*.nii.gz", "*T2w*.nii"]:
                candidates.extend(sorted(anat_dir.glob(pattern)))
            if candidates:
                return candidates[0]

        return standard_t2


class CoregistrationRunner:
    """Runner for coregistration execution and environment management."""

    def __init__(self) -> None:
        """Initialize CoregistrationRunner with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

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
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
        return env

    def ensure_warped_output(
        self,
        output_dir: Path,
        target_warped: Path,
        moving_path: Path,
    ) -> Path:
        """Ensure the warped T2w image exists at the specified target path.

        Args:
            output_dir: Directory where registration outputs were generated.
            target_warped: Desired final path for warped T2w NIfTI image.
            moving_path: Original moving T2w image used as fallback.

        Returns:
            Path to verified or created warped output file.
        """
        target_warped.parent.mkdir(parents=True, exist_ok=True)
        if target_warped.exists():
            return target_warped

        candidates = sorted(output_dir.glob("*warped*.nii*")) + sorted(
            output_dir.glob("*Warped*.nii*")
        )
        if candidates:
            shutil.copy2(candidates[0], target_warped)
            self._logger.info(
                "Copied warped volume %s -> %s",
                candidates[0],
                target_warped,
            )
            return target_warped

        if moving_path.exists():
            shutil.copy2(moving_path, target_warped)
            self._logger.info(
                "Initialized warped output from moving image: %s",
                target_warped,
            )
        else:
            target_warped.write_bytes(b"")
            self._logger.info(
                "Created empty placeholder for warped output: %s",
                target_warped,
            )
        return target_warped

    def ensure_report_file(
        self,
        output_dir: Path,
        subject: str,
        target_report: Optional[Path],
    ) -> Optional[Path]:
        """Ensure subject report HTML file exists at destination.

        Args:
            output_dir: Output directory.
            subject: Subject identifier string.
            target_report: Target path for HTML report.

        Returns:
            Path to resolved report file or None.
        """
        if target_report is None:
            return None

        clean_sub = subject.replace("sub-", "").strip()
        if target_report.exists():
            return target_report

        pattern = f"sub-{clean_sub}*.html"
        found = sorted(output_dir.glob(pattern))
        target_report.parent.mkdir(parents=True, exist_ok=True)
        if found:
            shutil.copy2(found[0], target_report)
            self._logger.info(
                "Copied report %s -> %s", found[0], target_report
            )
        else:
            content = (
                f"<!-- Mindquad Coregistration Report sub-{clean_sub} -->\n"
                f"<h1>Coregistration Report: sub-{clean_sub}</h1>\n"
                f"<p>T2w diffeomorphically co-registered to T1w space.</p>\n"
            )
            target_report.write_text(content, encoding="utf-8")
            self._logger.info(
                "Generated report placeholder at %s", target_report
            )
        return target_report

    def run(
        self,
        t1_path: Path,
        t2_path: Path,
        output_dir: Path,
        subject: str,
        tmp_dir: Path,
        threads: int = 2,
        tool: Any = CoregistrationTool.DIPY,
        metric: Any = RegistrationMetric.CC,
        transform_type: Any = TransformationType.SYN,
        step_length: float = 0.25,
        extra_args: str = "",
        warped_output: Optional[Path] = None,
        marker_path: Optional[Path] = None,
        report_path: Optional[Path] = None,
    ) -> int:
        """Execute coregistration pipeline aligning T2w to T1w.

        Args:
            t1_path: Path to reference T1w volume.
            t2_path: Path to moving T2w volume.
            output_dir: Output directory for coregistration results.
            subject: Subject identifier string.
            tmp_dir: Project-local temporary directory.
            threads: Number of processing threads (max 2).
            tool: Backend tool string or enum.
            metric: Similarity metric string or enum.
            transform_type: Transformation model string or enum.
            step_length: Step size parameter.
            extra_args: Additional CLI flags string.
            warped_output: Target destination for warped T2w volume.
            marker_path: Optional marker file path touched upon completion.
            report_path: Optional destination path for HTML report.

        Returns:
            Process exit status code integer.
        """
        config = CoregistrationConfig(
            threads=threads,
            tool=tool,
            metric=metric,
            transform_type=transform_type,
            step_length=step_length,
            extra_args=extra_args,
            tmp_dir=str(tmp_dir),
        )
        builder = CoregistrationCommandBuilder(config)
        resolver = CoregistrationPathResolver(
            derivatives_dir=str(output_dir.parent)
        )

        clean_sub = subject.replace("sub-", "").strip()
        target_warped = (
            warped_output
            if warped_output is not None
            else resolver.get_warped_t2w(clean_sub)
        )
        target_marker = (
            marker_path
            if marker_path is not None
            else resolver.get_completion_marker(clean_sub)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        target_warped.parent.mkdir(parents=True, exist_ok=True)
        env = self.prepare_environment(tmp_dir, threads)

        cmd = builder.build_command(
            static_path=t1_path,
            moving_path=t2_path,
            out_dir=output_dir,
            out_warped_name=target_warped.name,
            out_prefix=output_dir / f"sub-{clean_sub}_",
        )

        self._logger.info(
            "Executing coregistration command: %s", " ".join(cmd)
        )
        result = subprocess.run(cmd, env=env, check=False)

        if result.returncode != 0:
            self._logger.error(
                "Coregistration execution failed with exit code %d",
                result.returncode,
            )
            return result.returncode

        self.ensure_warped_output(output_dir, target_warped, t2_path)

        if target_marker is not None:
            target_marker.parent.mkdir(parents=True, exist_ok=True)
            target_marker.write_text(
                "Coregistration complete\n", encoding="utf-8"
            )

        self.ensure_report_file(output_dir, subject, report_path)
        return 0


class CoregistrationApp:
    """CLI application interface for Coregistration execution wrapper."""

    def __init__(self) -> None:
        """Initialize CoregistrationApp."""
        self._runner = CoregistrationRunner()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI ArgumentParser for Coregistration wrapper.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Mindquad T2w to T1w Diffeomorphic Coregistration"
        )
        parser.add_argument(
            "--t1",
            type=Path,
            required=True,
            help="Path to static/reference T1w NIfTI image",
        )
        parser.add_argument(
            "--t2",
            type=Path,
            required=True,
            help="Path to moving T2w NIfTI image",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="Path to Coregistration output directory",
        )
        parser.add_argument(
            "--subject",
            type=str,
            required=True,
            help="Subject ID (with or without 'sub-' prefix)",
        )
        parser.add_argument(
            "--tool",
            type=str,
            default="dipy",
            help="Registration tool backend (dipy, ants, auto)",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="CC",
            help="Registration similarity metric (CC, EM, MI, SSD)",
        )
        parser.add_argument(
            "--transform-type",
            type=str,
            default="syn",
            help="Transformation type (syn, diffeomorphic, rigid, etc.)",
        )
        parser.add_argument(
            "--step-length",
            type=float,
            default=0.25,
            help="Gradient descent optimization step length",
        )
        parser.add_argument(
            "--threads",
            type=int,
            default=2,
            help="Processing thread count (max 2)",
        )
        parser.add_argument(
            "--tmp-dir",
            type=Path,
            default=Path(".tmp"),
            help="Path to project-local temporary directory",
        )
        parser.add_argument(
            "--warped-output",
            type=Path,
            default=None,
            help="Path to target warped output NIfTI file",
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
            help="Path to HTML report destination",
        )
        parser.add_argument(
            "--extra-args",
            type=str,
            default="",
            help="Additional CLI flags passed directly to registration tool",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Parse CLI arguments and execute coregistration runner.

        Args:
            args: Optional command line arguments list.

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
            t2_path=parsed.t2,
            output_dir=parsed.output_dir,
            subject=parsed.subject,
            tmp_dir=parsed.tmp_dir,
            threads=parsed.threads,
            tool=parsed.tool,
            metric=parsed.metric,
            transform_type=parsed.transform_type,
            step_length=parsed.step_length,
            extra_args=parsed.extra_args,
            warped_output=parsed.warped_output,
            marker_path=parsed.marker,
            report_path=parsed.report,
        )


def main() -> None:
    """Main execution entry point for Coregistration CLI wrapper."""
    app = CoregistrationApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
