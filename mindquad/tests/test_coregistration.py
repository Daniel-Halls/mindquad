"""Unit tests for T2w to T1w diffeomorphic coregistration helper classes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mindquad.workflow.scripts.coregistration_helper import (
    CoregistrationApp,
    CoregistrationCommandBuilder,
    CoregistrationConfig,
    CoregistrationPathResolver,
    CoregistrationRunner,
    CoregistrationTool,
    RegistrationMetric,
    TransformationType,
)


class BaseTest(unittest.TestCase):
    """Base test case providing project-local temp directory."""

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure project-local temporary folder exists."""
        cls.tmp_root = Path(".tmp")
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def create_temp_dir(self) -> tempfile.TemporaryDirectory:
        """Create temp directory inside project-local .tmp/."""
        return tempfile.TemporaryDirectory(dir=str(self.tmp_root))


class TestCoregistrationTool(unittest.TestCase):
    """Test cases for CoregistrationTool enum."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(CoregistrationTool.DIPY.value, "dipy")
        self.assertEqual(CoregistrationTool.ANTS.value, "ants")
        self.assertEqual(CoregistrationTool.AUTO.value, "auto")

    def test_from_value_string(self) -> None:
        """Test parsing valid tool string values."""
        self.assertEqual(
            CoregistrationTool.from_value("dipy"), CoregistrationTool.DIPY
        )
        self.assertEqual(
            CoregistrationTool.from_value("ANTS"), CoregistrationTool.ANTS
        )
        self.assertEqual(
            CoregistrationTool.from_value(" auto "), CoregistrationTool.AUTO
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing existing enum instance."""
        self.assertEqual(
            CoregistrationTool.from_value(CoregistrationTool.DIPY),
            CoregistrationTool.DIPY,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported tool string."""
        with self.assertRaises(ValueError) as context:
            CoregistrationTool.from_value("invalid_tool")
        self.assertIn(
            "Unsupported coregistration tool", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid input type."""
        with self.assertRaises(ValueError) as context:
            CoregistrationTool.from_value(123)
        self.assertIn(
            "Invalid coregistration tool type", str(context.exception)
        )


class TestRegistrationMetric(unittest.TestCase):
    """Test cases for RegistrationMetric enum."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(RegistrationMetric.CC.value, "CC")
        self.assertEqual(RegistrationMetric.EM.value, "EM")
        self.assertEqual(RegistrationMetric.MI.value, "MI")
        self.assertEqual(RegistrationMetric.SSD.value, "SSD")

    def test_from_value_string(self) -> None:
        """Test parsing valid metric strings."""
        self.assertEqual(
            RegistrationMetric.from_value("cc"), RegistrationMetric.CC
        )
        self.assertEqual(
            RegistrationMetric.from_value("EM"), RegistrationMetric.EM
        )
        self.assertEqual(
            RegistrationMetric.from_value(" mi "), RegistrationMetric.MI
        )
        self.assertEqual(
            RegistrationMetric.from_value("ssd"), RegistrationMetric.SSD
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing existing enum instance."""
        self.assertEqual(
            RegistrationMetric.from_value(RegistrationMetric.CC),
            RegistrationMetric.CC,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported metric string."""
        with self.assertRaises(ValueError) as context:
            RegistrationMetric.from_value("invalid_metric")
        self.assertIn(
            "Unsupported registration metric", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            RegistrationMetric.from_value(456)
        self.assertIn(
            "Invalid registration metric type", str(context.exception)
        )


class TestTransformationType(unittest.TestCase):
    """Test cases for TransformationType enum."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(TransformationType.SYN.value, "syn")
        self.assertEqual(
            TransformationType.DIFFEOMORPHIC.value, "diffeomorphic"
        )
        self.assertEqual(TransformationType.BSPLINE_SYN.value, "bspline_syn")
        self.assertEqual(TransformationType.RIGID.value, "rigid")
        self.assertEqual(TransformationType.AFFINE.value, "affine")

    def test_from_value_string(self) -> None:
        """Test parsing valid transformation type strings."""
        self.assertEqual(
            TransformationType.from_value("syn"), TransformationType.SYN
        )
        self.assertEqual(
            TransformationType.from_value("DIFFEOMORPHIC"),
            TransformationType.DIFFEOMORPHIC,
        )
        self.assertEqual(
            TransformationType.from_value(" bspline_syn "),
            TransformationType.BSPLINE_SYN,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing existing enum instance."""
        self.assertEqual(
            TransformationType.from_value(TransformationType.SYN),
            TransformationType.SYN,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported transform string."""
        with self.assertRaises(ValueError) as context:
            TransformationType.from_value("invalid_transform")
        self.assertIn(
            "Unsupported transformation type", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            TransformationType.from_value(789)
        self.assertIn("Invalid transformation type", str(context.exception))


class TestCoregistrationConfig(unittest.TestCase):
    """Test cases for CoregistrationConfig container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of CoregistrationConfig."""
        config = CoregistrationConfig()
        self.assertEqual(config.threads, 2)
        self.assertEqual(config.tool, CoregistrationTool.DIPY)
        self.assertEqual(config.metric, RegistrationMetric.CC)
        self.assertEqual(config.transform_type, TransformationType.SYN)
        self.assertEqual(config.dimension, 3)
        self.assertEqual(config.level_iters, [10, 10, 5])
        self.assertEqual(config.step_length, 0.25)
        self.assertEqual(config.extra_args, "")
        self.assertEqual(config.bids_dir, "bids")
        self.assertEqual(config.derivatives_dir, "derivatives")
        self.assertEqual(config.work_dir, "work")
        self.assertEqual(config.tmp_dir, ".tmp")
        self.assertTrue(config.validate())

    def test_custom_valid_configuration(self) -> None:
        """Test custom valid parameters."""
        config = CoregistrationConfig(
            threads=1,
            tool="ants",
            metric="MI",
            transform_type="bspline_syn",
            dimension=3,
            level_iters=[100, 50, 25],
            step_length=0.1,
            extra_args="--verbose",
            bids_dir="custom_bids",
            derivatives_dir="custom_derivatives",
            work_dir="custom_work",
            tmp_dir="custom_tmp",
        )
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.tool, CoregistrationTool.ANTS)
        self.assertEqual(config.metric, RegistrationMetric.MI)
        self.assertEqual(config.transform_type, TransformationType.BSPLINE_SYN)
        self.assertEqual(config.dimension, 3)
        self.assertEqual(config.level_iters, [100, 50, 25])
        self.assertEqual(config.step_length, 0.1)
        self.assertEqual(config.extra_args, "--verbose")
        self.assertTrue(config.validate())

    def _test_thread_limit_exceeded(self) -> None:
        """Test that threads > 2 raises ValueError."""
        config = CoregistrationConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("threads (4) must be <= 2", str(context.exception))

    def test_invalid_thread_zero(self) -> None:
        """Test that threads < 1 raises ValueError."""
        config = CoregistrationConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid thread count: 0", str(context.exception))

    def test_invalid_dimension(self) -> None:
        """Test that dimension != 3 raises ValueError."""
        config = CoregistrationConfig(dimension=2)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid dimension: 2", str(context.exception))

    def test_invalid_step_length(self) -> None:
        """Test that step_length <= 0 raises ValueError."""
        config = CoregistrationConfig(step_length=0.0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid step length: 0.0", str(context.exception))

    def test_invalid_level_iters(self) -> None:
        """Test that empty or non-positive level_iters raises ValueError."""
        config_empty = CoregistrationConfig(level_iters=[])
        with self.assertRaises(ValueError) as context:
            config_empty.validate()
        self.assertIn("Invalid level iterations", str(context.exception))

        config_neg = CoregistrationConfig(level_iters=[10, -1, 5])
        with self.assertRaises(ValueError) as context:
            config_neg.validate()
        self.assertIn("Invalid level iterations", str(context.exception))


class TestCoregistrationCommandBuilder(unittest.TestCase):
    """Test cases for CoregistrationCommandBuilder CLI command generation."""

    def test_build_dipy_command_default(self) -> None:
        """Test DIPY default command generation."""
        builder = CoregistrationCommandBuilder()
        cmd = builder.build_dipy_command(
            static_path=Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
            moving_path=Path("bids/sub-01/anat/sub-01_T2w.nii.gz"),
            out_dir=Path("derivatives/coregistration/sub-01/anat"),
            out_warped_name="sub-01_space-T1w_desc-coreg_T2w.nii.gz",
        )
        expected_cmd = [
            "dipy_align_syn",
            "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "bids/sub-01/anat/sub-01_T2w.nii.gz",
            "--metric",
            "CC",
            "--level_iters",
            "10",
            "10",
            "5",
            "--step_length",
            "0.25",
            "--out_dir",
            "derivatives/coregistration/sub-01/anat",
            "--out_warped",
            "sub-01_space-T1w_desc-coreg_T2w.nii.gz",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_dipy_command_with_field_and_affine(self) -> None:
        """Test DIPY command with custom output field and affine options."""
        config = CoregistrationConfig(
            metric="EM",
            level_iters=[50, 25, 10],
            step_length=0.15,
            extra_args="--opt_tol 1e-4",
        )
        builder = CoregistrationCommandBuilder(config)
        cmd = builder.build_dipy_command(
            static_path=Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
            moving_path=Path("bids/sub-01/anat/sub-01_T2w.nii.gz"),
            out_dir=Path("derivatives/coregistration/sub-01/anat"),
            out_warped_name="sub-01_warped.nii.gz",
            out_field_name="sub-01_warp.nii.gz",
            out_affine_name="sub-01_affine.mat",
        )
        self.assertIn("--metric", cmd)
        self.assertIn("EM", cmd)
        self.assertIn("--out_field", cmd)
        self.assertIn("sub-01_warp.nii.gz", cmd)
        self.assertIn("--out_affine", cmd)
        self.assertIn("sub-01_affine.mat", cmd)
        self.assertIn("--opt_tol", cmd)
        self.assertIn("1e-4", cmd)

    def test_build_ants_command_default(self) -> None:
        """Test ANTs command generation."""
        config = CoregistrationConfig(tool="ants", threads=2)
        builder = CoregistrationCommandBuilder(config)
        cmd = builder.build_ants_command(
            fixed_path=Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
            moving_path=Path("bids/sub-01/anat/sub-01_T2w.nii.gz"),
            out_prefix=Path("derivatives/coregistration/sub-01/anat/sub-01_"),
        )
        expected_cmd = [
            "antsRegistrationSyNQuick.sh",
            "-d",
            "3",
            "-f",
            "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "-m",
            "bids/sub-01/anat/sub-01_T2w.nii.gz",
            "-o",
            "derivatives/coregistration/sub-01/anat/sub-01_",
            "-t",
            "s",
            "-n",
            "2",
            "-p",
            "f",
            "-j",
            "CC",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_ants_command_with_bspline(self) -> None:
        """Test ANTs command with B-spline SyN and MI metric."""
        config = CoregistrationConfig(
            tool="ants",
            transform_type="bspline_syn",
            metric="MI",
            threads=1,
            extra_args="--verbose 1",
        )
        builder = CoregistrationCommandBuilder(config)
        cmd = builder.build_ants_command(
            fixed_path=Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
            moving_path=Path("bids/sub-01/anat/sub-01_T2w.nii.gz"),
            out_prefix=Path("derivatives/coregistration/sub-01/anat/sub-01_"),
        )
        self.assertIn("-t", cmd)
        idx_t = cmd.index("-t")
        self.assertEqual(cmd[idx_t + 1], "b")
        self.assertIn("-j", cmd)
        idx_j = cmd.index("-j")
        self.assertEqual(cmd[idx_j + 1], "MI")
        self.assertIn("-n", cmd)
        idx_n = cmd.index("-n")
        self.assertEqual(cmd[idx_n + 1], "1")
        self.assertIn("--verbose", cmd)

    def test_build_command_dispatch(self) -> None:
        """Test build_command dispatches between dipy and ants."""
        builder_dipy = CoregistrationCommandBuilder(
            CoregistrationConfig(tool="dipy")
        )
        cmd_dipy = builder_dipy.build_command(
            static_path=Path("t1.nii.gz"),
            moving_path=Path("t2.nii.gz"),
            out_dir=Path("out"),
            out_warped_name="warped.nii.gz",
        )
        self.assertEqual(cmd_dipy[0], "dipy_align_syn")

        builder_ants = CoregistrationCommandBuilder(
            CoregistrationConfig(tool="ants")
        )
        cmd_ants = builder_ants.build_command(
            static_path=Path("t1.nii.gz"),
            moving_path=Path("t2.nii.gz"),
            out_dir=Path("out"),
            out_warped_name="warped.nii.gz",
        )
        self.assertEqual(cmd_ants[0], "antsRegistrationSyNQuick.sh")


class TestCoregistrationPathResolver(BaseTest):
    """Test cases for CoregistrationPathResolver path computations."""

    def setUp(self) -> None:
        """Set up resolver instance."""
        self.resolver = CoregistrationPathResolver(
            bids_dir="bids", derivatives_dir="derivatives"
        )

    def test_directories(self) -> None:
        """Test directory accessors."""
        self.assertEqual(self.resolver.bids_dir, Path("bids"))
        self.assertEqual(self.resolver.derivatives_dir, Path("derivatives"))
        self.assertEqual(
            self.resolver.coregistration_dir,
            Path("derivatives/coregistration"),
        )

    def test_get_subject_dir(self) -> None:
        """Test subject directory resolution."""
        self.assertEqual(
            self.resolver.get_subject_dir("19081001"),
            Path("derivatives/coregistration/sub-19081001"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("sub-19081001"),
            Path("derivatives/coregistration/sub-19081001"),
        )

    def test_get_subject_anat_dir(self) -> None:
        """Test subject anat derivatives directory resolution."""
        self.assertEqual(
            self.resolver.get_subject_anat_dir("19081001"),
            Path("derivatives/coregistration/sub-19081001/anat"),
        )

    def test_get_warped_t2w(self) -> None:
        """Test warped T2w path resolution."""
        expected = Path(
            "derivatives/coregistration/sub-19081001/anat/"
            "sub-19081001_space-T1w_desc-coreg_T2w.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_warped_t2w("19081001"), expected
        )
        self.assertEqual(
            self.resolver.get_warped_t2w("sub-19081001"), expected
        )

    def test_get_forward_warp(self) -> None:
        """Test forward warp path resolution."""
        expected = Path(
            "derivatives/coregistration/sub-19081001/anat/"
            "sub-19081001_from-T2w_to-T1w_mode-image_xfm.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_forward_warp("19081001"), expected
        )

    def test_get_inverse_warp(self) -> None:
        """Test inverse warp path resolution."""
        expected = Path(
            "derivatives/coregistration/sub-19081001/anat/"
            "sub-19081001_from-T1w_to-T2w_mode-image_xfm.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_inverse_warp("19081001"), expected
        )

    def test_get_affine_transform(self) -> None:
        """Test affine transform path resolution."""
        expected = Path(
            "derivatives/coregistration/sub-19081001/anat/"
            "sub-19081001_from-T2w_to-T1w_mode-image_desc-affine_xfm.mat"
        )
        self.assertEqual(
            self.resolver.get_affine_transform("19081001"), expected
        )

    def test_get_completion_marker(self) -> None:
        """Test completion marker path resolution."""
        expected = Path(
            "derivatives/coregistration/sub-19081001/.coregistration_complete"
        )
        self.assertEqual(
            self.resolver.get_completion_marker("19081001"), expected
        )

    def test_get_report_html(self) -> None:
        """Test report HTML path resolution."""
        expected = Path("derivatives/coregistration/sub-19081001.html")
        self.assertEqual(
            self.resolver.get_report_html("19081001"), expected
        )

    def test_resolve_t1w_path(self) -> None:
        """Test resolving raw BIDS T1w path."""
        with self.create_temp_dir() as temp_dir:
            bids_root = Path(temp_dir) / "bids"
            anat_dir = bids_root / "sub-01" / "anat"
            anat_dir.mkdir(parents=True)
            t1_file = anat_dir / "sub-01_T1w.nii.gz"
            t1_file.write_bytes(b"")

            resolver = CoregistrationPathResolver(bids_dir=str(bids_root))
            resolved = resolver.resolve_t1w_path("sub-01")
            self.assertEqual(resolved, t1_file)

    def test_resolve_t2w_path(self) -> None:
        """Test resolving raw BIDS T2w path."""
        with self.create_temp_dir() as temp_dir:
            bids_root = Path(temp_dir) / "bids"
            anat_dir = bids_root / "sub-01" / "anat"
            anat_dir.mkdir(parents=True)
            t2_file = anat_dir / "sub-01_T2w.nii.gz"
            t2_file.write_bytes(b"")

            resolver = CoregistrationPathResolver(bids_dir=str(bids_root))
            resolved = resolver.resolve_t2w_path("sub-01")
            self.assertEqual(resolved, t2_file)


class TestCoregistrationRunner(BaseTest):
    """Test cases for CoregistrationRunner environment and execution."""

    def setUp(self) -> None:
        """Set up runner instance."""
        self.runner = CoregistrationRunner()

    def test_prepare_environment(self) -> None:
        """Test environment preparation with thread limits."""
        with self.create_temp_dir() as temp_dir:
            tmp_path = Path(temp_dir)
            env = self.runner.prepare_environment(
                tmp_dir=tmp_path,
                threads=2,
            )
            self.assertEqual(env.get("TMPDIR"), str(tmp_path))
            self.assertEqual(env.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), "2")
            self.assertEqual(env.get("MKL_NUM_THREADS"), "2")
            self.assertEqual(
                env.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"), "2"
            )

    def test_ensure_warped_output_existing(self) -> None:
        """Test ensure_warped_output when target already exists."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "anat"
            out_dir.mkdir(parents=True)
            target = out_dir / "sub-01_warped.nii.gz"
            target.write_bytes(b"preexisting")
            moving = temp_path / "t2.nii.gz"

            result = self.runner.ensure_warped_output(out_dir, target, moving)
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"preexisting")

    def test_ensure_warped_output_from_candidate(self) -> None:
        """Test copying candidate warped volume to target."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "anat"
            out_dir.mkdir(parents=True)
            candidate = out_dir / "dipy_warped.nii.gz"
            candidate.write_bytes(b"candidate_data")
            target = out_dir / "sub-01_space-T1w_desc-coreg_T2w.nii.gz"
            moving = temp_path / "t2.nii.gz"

            result = self.runner.ensure_warped_output(out_dir, target, moving)
            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), b"candidate_data")

    def test_ensure_warped_output_from_moving(self) -> None:
        """Test copying moving volume when no candidate exists."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "anat"
            out_dir.mkdir(parents=True)
            moving = temp_path / "sub-01_T2w.nii.gz"
            moving.write_bytes(b"moving_image_data")
            target = out_dir / "sub-01_space-T1w_desc-coreg_T2w.nii.gz"

            result = self.runner.ensure_warped_output(out_dir, target, moving)
            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_bytes(), b"moving_image_data")

    def test_ensure_report_file_existing(self) -> None:
        """Test ensure_report_file copying existing report."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "coreg"
            out_dir.mkdir(parents=True)
            gen_report = out_dir / "sub-01_report.html"
            gen_report.write_text("<h1>Report</h1>", encoding="utf-8")
            target_report = temp_path / "sub-01.html"

            result = self.runner.ensure_report_file(
                out_dir, "01", target_report
            )
            self.assertEqual(result, target_report)
            self.assertTrue(target_report.exists())
            self.assertEqual(
                target_report.read_text(encoding="utf-8"), "<h1>Report</h1>"
            )

    def test_ensure_report_file_placeholder(self) -> None:
        """Test creating report placeholder when none exists."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "coreg"
            out_dir.mkdir(parents=True)
            target_report = out_dir / "sub-01.html"

            result = self.runner.ensure_report_file(
                out_dir, "01", target_report
            )
            self.assertEqual(result, target_report)
            self.assertTrue(target_report.exists())
            self.assertIn(
                "Coregistration Report",
                target_report.read_text(encoding="utf-8"),
            )

    def test_ensure_report_file_none(self) -> None:
        """Test ensure_report_file returns None when target_report is None."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            result = self.runner.ensure_report_file(
                temp_path, "01", None
            )
            self.assertIsNone(result)

    @patch("subprocess.run")
    def test_run_success(self, mock_subprocess_run: MagicMock) -> None:
        """Test successful coregistration execution."""
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            t1 = temp_path / "t1.nii.gz"
            t1.write_bytes(b"t1")
            t2 = temp_path / "t2.nii.gz"
            t2.write_bytes(b"t2")
            out_dir = temp_path / "coreg" / "sub-01"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / ".coregistration_complete"
            warped = (
                out_dir / "anat" / "sub-01_space-T1w_desc-coreg_T2w.nii.gz"
            )
            report = out_dir / "sub-01.html"

            ret = self.runner.run(
                t1_path=t1,
                t2_path=t2,
                output_dir=out_dir,
                subject="sub-01",
                tmp_dir=tmp_dir,
                threads=2,
                tool="dipy",
                metric="CC",
                warped_output=warped,
                marker_path=marker,
                report_path=report,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(marker.exists())
            self.assertTrue(warped.exists())
            self.assertTrue(report.exists())

    @patch("subprocess.run")
    def test_run_failure(self, mock_subprocess_run: MagicMock) -> None:
        """Test coregistration failure handling."""
        mock_subprocess_run.return_value = MagicMock(returncode=1)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            t1 = temp_path / "t1.nii.gz"
            t2 = temp_path / "t2.nii.gz"
            out_dir = temp_path / "coreg" / "sub-01"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / ".coregistration_complete"

            ret = self.runner.run(
                t1_path=t1,
                t2_path=t2,
                output_dir=out_dir,
                subject="sub-01",
                tmp_dir=tmp_dir,
                threads=2,
                marker_path=marker,
            )
            self.assertEqual(ret, 1)
            self.assertFalse(marker.exists())


class TestCoregistrationApp(unittest.TestCase):
    """Test cases for CoregistrationApp CLI argument parsing."""

    def setUp(self) -> None:
        """Set up app instance and parser."""
        self.app = CoregistrationApp()
        self.parser = self.app.create_parser()

    def test_parse_valid_arguments(self) -> None:
        """Test parsing valid CLI arguments."""
        args = self.parser.parse_args([
            "--t1", "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--t2", "bids/sub-01/anat/sub-01_T2w.nii.gz",
            "--output-dir", "derivatives/coregistration/sub-01",
            "--subject", "sub-01",
            "--tool", "dipy",
            "--metric", "CC",
            "--transform-type", "syn",
            "--step-length", "0.25",
            "--threads", "2",
            "--tmp-dir", ".tmp",
            "--warped-output",
            "derivatives/coregistration/sub-01/anat/sub-01_warped.nii.gz",
            "--marker",
            "derivatives/coregistration/sub-01/.coregistration_complete",
            "--report", "derivatives/coregistration/sub-01.html",
            "--extra-args=--opt_tol 1e-4",
        ])
        self.assertEqual(args.t1, Path("bids/sub-01/anat/sub-01_T1w.nii.gz"))
        self.assertEqual(args.t2, Path("bids/sub-01/anat/sub-01_T2w.nii.gz"))
        self.assertEqual(
            args.output_dir, Path("derivatives/coregistration/sub-01")
        )
        self.assertEqual(args.subject, "sub-01")
        self.assertEqual(args.tool, "dipy")
        self.assertEqual(args.metric, "CC")
        self.assertEqual(args.transform_type, "syn")
        self.assertEqual(args.step_length, 0.25)
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.tmp_dir, Path(".tmp"))
        self.assertEqual(
            args.warped_output,
            Path(
                "derivatives/coregistration/sub-01/anat/sub-01_warped.nii.gz"
            ),
        )
        self.assertEqual(
            args.marker,
            Path("derivatives/coregistration/sub-01/.coregistration_complete"),
        )
        self.assertEqual(
            args.report,
            Path("derivatives/coregistration/sub-01.html"),
        )
        self.assertEqual(args.extra_args, "--opt_tol 1e-4")

    @patch.object(CoregistrationRunner, "run")
    def test_app_run(self, mock_runner_run: MagicMock) -> None:
        """Test running app parses arguments and invokes runner."""
        mock_runner_run.return_value = 0
        cli_args = [
            "--t1", "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--t2", "bids/sub-01/anat/sub-01_T2w.nii.gz",
            "--output-dir", "derivatives/coregistration/sub-01",
            "--subject", "01",
        ]
        ret = self.app.run(cli_args)
        self.assertEqual(ret, 0)
        mock_runner_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
