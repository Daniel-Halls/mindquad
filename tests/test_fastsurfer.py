"""Unit tests for FastSurfer helper classes and configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow.scripts.fastsurfer_helper import (
    FastSurferApp,
    FastSurferCommandBuilder,
    FastSurferConfig,
    FastSurferDevice,
    FastSurferPathResolver,
    FastSurferRunner,
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


class TestFastSurferDevice(unittest.TestCase):
    """Test cases for FastSurferDevice enum conversions and validation."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(FastSurferDevice.CPU.value, "cpu")
        self.assertEqual(FastSurferDevice.CUDA.value, "cuda")
        self.assertEqual(FastSurferDevice.AUTO.value, "auto")
        self.assertEqual(FastSurferDevice.MPS.value, "mps")

    def test_from_value_string(self) -> None:
        """Test from_value parsing valid strings."""
        self.assertEqual(
            FastSurferDevice.from_value("cpu"), FastSurferDevice.CPU
        )
        self.assertEqual(
            FastSurferDevice.from_value("CUDA"), FastSurferDevice.CUDA
        )
        self.assertEqual(
            FastSurferDevice.from_value(" Auto "), FastSurferDevice.AUTO
        )
        self.assertEqual(
            FastSurferDevice.from_value("mps"), FastSurferDevice.MPS
        )

    def test_from_value_enum_instance(self) -> None:
        """Test from_value with existing enum instance."""
        self.assertEqual(
            FastSurferDevice.from_value(FastSurferDevice.CUDA),
            FastSurferDevice.CUDA,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test from_value raising ValueError on unsupported string."""
        with self.assertRaises(ValueError) as context:
            FastSurferDevice.from_value("invalid_device")
        self.assertIn("Unsupported device 'invalid_device'", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test from_value raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            FastSurferDevice.from_value(12345)
        self.assertIn("Invalid device type", str(context.exception))


class TestFastSurferConfig(unittest.TestCase):
    """Test cases for FastSurferConfig data container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of FastSurferConfig."""
        config = FastSurferConfig()
        self.assertEqual(config.threads, 2)
        self.assertEqual(config.device, FastSurferDevice.CPU)
        self.assertIsNone(config.fs_license)
        self.assertEqual(config.extra_args, "")
        self.assertEqual(config.batch_size, 1)
        self.assertFalse(config.seg_only)
        self.assertFalse(config.surf_only)
        self.assertFalse(config.parallel)
        self.assertTrue(config.validate())

    def test_custom_valid_configuration_with_enum(self) -> None:
        """Test custom valid parameters using FastSurferDevice enum."""
        config = FastSurferConfig(
            threads=1,
            device=FastSurferDevice.CUDA,
            fs_license="/opt/freesurfer/license.txt",
            extra_args="--qc_snap",
            batch_size=2,
            seg_only=True,
            parallel=True,
        )
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.device, FastSurferDevice.CUDA)
        self.assertEqual(config.fs_license, "/opt/freesurfer/license.txt")
        self.assertEqual(config.extra_args, "--qc_snap")
        self.assertEqual(config.batch_size, 2)
        self.assertTrue(config.seg_only)
        self.assertTrue(config.parallel)
        self.assertTrue(config.validate())

    def test_custom_valid_configuration_with_string(self) -> None:
        """Test custom valid parameters using device string."""
        config = FastSurferConfig(
            threads=2,
            device="mps",
        )
        self.assertEqual(config.device, FastSurferDevice.MPS)
        self.assertTrue(config.validate())

    def _test_thread_limit_exceeded(self) -> None:
        """Test that threads > 2 raises ValueError."""
        config = FastSurferConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("threads (4) must be <= 2", str(context.exception))

    def test_invalid_thread_zero(self) -> None:
        """Test that threads < 1 raises ValueError."""
        config = FastSurferConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid thread count: 0", str(context.exception))

    def test_invalid_device_string(self) -> None:
        """Test that unsupported device string raises ValueError during init."""
        with self.assertRaises(ValueError) as context:
            FastSurferConfig(device="tpu")
        self.assertIn("Unsupported device 'tpu'", str(context.exception))

    def test_invalid_batch_size(self) -> None:
        """Test that batch_size < 1 raises ValueError."""
        config = FastSurferConfig(batch_size=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid batch size: 0", str(context.exception))

    def test_conflicting_seg_and_surf_flags(self) -> None:
        """Test that setting both seg_only and surf_only raises ValueError."""
        config = FastSurferConfig(seg_only=True, surf_only=True)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Cannot specify both seg_only and surf_only", str(context.exception))


class TestFastSurferCommandBuilder(unittest.TestCase):
    """Test cases for FastSurferCommandBuilder command generation."""

    def test_build_default_command(self) -> None:
        """Test standard default command building."""
        builder = FastSurferCommandBuilder()
        cmd = builder.build_command(
            t1_path=Path("bids/sub-19081001/anat/sub-19081001_T1w.nii.gz"),
            subjects_dir=Path("derivatives/fastsurfer"),
            subject_id="sub-19081001",
        )
        expected_cmd = [
            "run_fastsurfer.sh",
            "--t1",
            "bids/sub-19081001/anat/sub-19081001_T1w.nii.gz",
            "--sd",
            "derivatives/fastsurfer",
            "--sid",
            "sub-19081001",
            "--threads",
            "2",
            "--device",
            "cpu",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_command_with_options(self) -> None:
        """Test command building with license, batch size, and parallel flags."""
        config = FastSurferConfig(
            threads=1,
            device=FastSurferDevice.CUDA,
            fs_license="/path/to/license.txt",
            extra_args="--qc_snap --vol_segstats",
            batch_size=4,
            parallel=True,
            surf_only=True,
        )
        builder = FastSurferCommandBuilder(config)
        cmd = builder.build_command(
            t1_path=Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
            subjects_dir=Path("derivatives/fastsurfer"),
            subject_id="sub-01",
        )
        expected_cmd = [
            "run_fastsurfer.sh",
            "--t1",
            "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--sd",
            "derivatives/fastsurfer",
            "--sid",
            "sub-01",
            "--threads",
            "1",
            "--device",
            "cuda",
            "--batch",
            "4",
            "--fs_license",
            "/path/to/license.txt",
            "--surf_only",
            "--parallel",
            "--qc_snap",
            "--vol_segstats",
        ]
        self.assertEqual(cmd, expected_cmd)


class TestFastSurferPathResolver(BaseTest):
    """Test cases for FastSurferPathResolver path computations."""

    def setUp(self) -> None:
        """Set up resolver instance."""
        self.resolver = FastSurferPathResolver(
            bids_dir="bids",
            derivatives_dir="derivatives",
        )

    def test_directories(self) -> None:
        """Test directory property accessors."""
        self.assertEqual(self.resolver.bids_dir, Path("bids"))
        self.assertEqual(self.resolver.derivatives_dir, Path("derivatives"))
        self.assertEqual(
            self.resolver.fastsurfer_dir, Path("derivatives/fastsurfer")
        )

    def test_get_subject_dir(self) -> None:
        """Test subject directory resolution."""
        self.assertEqual(
            self.resolver.get_subject_dir("19081001"),
            Path("derivatives/fastsurfer/sub-19081001"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("sub-19081001"),
            Path("derivatives/fastsurfer/sub-19081001"),
        )

    def test_get_segmentation_file(self) -> None:
        """Test deep segmentation path resolution."""
        seg_file = self.resolver.get_segmentation_file("19081001")
        self.assertEqual(
            seg_file,
            Path(
                "derivatives/fastsurfer/sub-19081001/mri/aparc.DKTatlas+aseg.deep.mgz"
            ),
        )

    def test_get_orig_and_brainmask_files(self) -> None:
        """Test orig.mgz, brainmask.mgz, and aseg.mgz paths."""
        self.assertEqual(
            self.resolver.get_orig_mgz_file("19081001"),
            Path("derivatives/fastsurfer/sub-19081001/mri/orig.mgz"),
        )
        self.assertEqual(
            self.resolver.get_brainmask_file("19081001"),
            Path("derivatives/fastsurfer/sub-19081001/mri/brainmask.mgz"),
        )
        self.assertEqual(
            self.resolver.get_aseg_file("19081001"),
            Path("derivatives/fastsurfer/sub-19081001/mri/aseg.mgz"),
        )

    def test_get_surface_and_stats_files(self) -> None:
        """Test surface and stats file path resolution."""
        self.assertEqual(
            self.resolver.get_surface_file("19081001", "lh", "pial"),
            Path("derivatives/fastsurfer/sub-19081001/surf/lh.pial"),
        )
        self.assertEqual(
            self.resolver.get_stats_file("19081001", "aseg.stats"),
            Path("derivatives/fastsurfer/sub-19081001/stats/aseg.stats"),
        )

    def test_get_completion_marker(self) -> None:
        """Test completion marker file resolution."""
        self.assertEqual(
            self.resolver.get_completion_marker("19081001"),
            Path("derivatives/fastsurfer/sub-19081001/.fastsurfer_complete"),
        )

    def test_resolve_t1w_path_existing(self) -> None:
        """Test resolving T1w path when file exists on disk."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            anat_dir = temp_path / "bids" / "sub-19081001" / "anat"
            anat_dir.mkdir(parents=True)
            mock_t1 = anat_dir / "sub-19081001_T1w.nii.gz"
            mock_t1.write_text("mock content", encoding="utf-8")

            res = FastSurferPathResolver(bids_dir=str(temp_path / "bids"))
            resolved = res.resolve_t1w_path("19081001")
            self.assertEqual(resolved, mock_t1)

    def test_resolve_t1w_path_fallback(self) -> None:
        """Test resolving T1w path fallback when file does not yet exist."""
        resolved = self.resolver.resolve_t1w_path("19081001")
        self.assertEqual(
            resolved, Path("bids/sub-19081001/anat/sub-19081001_T1w.nii.gz")
        )


class TestFastSurferRunner(BaseTest):
    """Test cases for FastSurferRunner environment and execution."""

    def setUp(self) -> None:
        """Set up runner instance."""
        self.runner = FastSurferRunner()

    def test_prepare_environment(self) -> None:
        """Test environment preparation with threads and license."""
        with self.create_temp_dir() as temp_dir:
            tmp_path = Path(temp_dir)
            env = self.runner.prepare_environment(
                tmp_dir=tmp_path,
                threads=2,
                fs_license="/license/file.txt",
            )
            self.assertEqual(env.get("TMPDIR"), str(tmp_path))
            self.assertEqual(env.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), "2")
            self.assertEqual(env.get("MKL_NUM_THREADS"), "2")
            self.assertEqual(env.get("FS_LICENSE"), "/license/file.txt")

    @patch("subprocess.run")
    def test_run_success(self, mock_subprocess_run: MagicMock) -> None:
        """Test successful execution creating completion marker."""
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            sd = temp_path / "fastsurfer"
            tmp = temp_path / ".tmp"
            marker = sd / "sub-01" / ".fastsurfer_complete"
            t1 = temp_path / "sub-01_T1w.nii.gz"
            t1.write_text("t1", encoding="utf-8")

            ret = self.runner.run(
                t1_path=t1,
                subjects_dir=sd,
                subject_id="sub-01",
                tmp_dir=tmp,
                threads=2,
                device=FastSurferDevice.CPU,
                marker_path=marker,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(marker.exists())

    @patch("subprocess.run")
    def test_run_failure(self, mock_subprocess_run: MagicMock) -> None:
        """Test execution failure returning non-zero return code."""
        mock_subprocess_run.return_value = MagicMock(returncode=1)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            sd = temp_path / "fastsurfer"
            tmp = temp_path / ".tmp"
            marker = sd / "sub-01" / ".fastsurfer_complete"
            t1 = temp_path / "sub-01_T1w.nii.gz"

            ret = self.runner.run(
                t1_path=t1,
                subjects_dir=sd,
                subject_id="sub-01",
                tmp_dir=tmp,
                threads=2,
                device=FastSurferDevice.CPU,
                marker_path=marker,
            )
            self.assertEqual(ret, 1)
            self.assertFalse(marker.exists())


class TestFastSurferApp(unittest.TestCase):
    """Test cases for FastSurferApp CLI argument parsing."""

    def setUp(self) -> None:
        """Set up app instance and parser."""
        self.app = FastSurferApp()
        self.parser = self.app.create_parser()

    def test_parse_valid_arguments(self) -> None:
        """Test parsing valid CLI arguments."""
        args = self.parser.parse_args([
            "--t1", "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--sd", "derivatives/fastsurfer",
            "--sid", "sub-01",
            "--threads", "2",
            "--device", "cpu",
            "--fs-license", "/opt/fs/license.txt",
            "--extra-args", "--batch 1",
        ])
        self.assertEqual(args.t1, Path("bids/sub-01/anat/sub-01_T1w.nii.gz"))
        self.assertEqual(args.sd, Path("derivatives/fastsurfer"))
        self.assertEqual(args.sid, "sub-01")
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.fs_license, "/opt/fs/license.txt")
        self.assertEqual(args.extra_args, "--batch 1")


if __name__ == "__main__":
    unittest.main()
