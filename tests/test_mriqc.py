"""Unit tests for MRIQC helper classes and configuration."""

import tempfile
import unittest
from pathlib import Path

from workflow.scripts.mriqc_helper import (
    MRIQCApp,
    MRIQCCommandBuilder,
    MRIQCConfig,
    MRIQCPathResolver,
    MRIQCRunner,
)


class BaseTest(unittest.TestCase):
    """Base test case providing project-local temp directory."""

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure project-local temporary folder exists."""
        cls.tmp_root = Path(".tmp")
        cls.tmp_root.mkdir(parents=True, exist_ok=True)

    def create_temp_dir(self) -> tempfile.TemporaryDirectory:
        """Create temp directory inside .tmp/."""
        return tempfile.TemporaryDirectory(dir=str(self.tmp_root))


class TestMRIQCConfig(unittest.TestCase):
    """Test cases for MRIQCConfig data container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of MRIQCConfig."""
        config = MRIQCConfig()
        self.assertEqual(config.modalities, ["T1w", "bold", "dwi"])
        self.assertEqual(config.threads, 2)
        self.assertEqual(config.mem_gb, 8)
        self.assertEqual(config.extra_args, "--verbose-reports --no-sub")
        self.assertEqual(config.bids_dir, "bids")
        self.assertEqual(config.derivatives_dir, "derivatives")
        self.assertEqual(config.work_dir, "work")
        self.assertEqual(config.tmp_dir, ".tmp")
        self.assertTrue(config.validate())

    def test_custom_valid_configuration(self) -> None:
        """Test custom valid parameters of MRIQCConfig."""
        config = MRIQCConfig(
            modalities=["T1w", "bold"],
            threads=1,
            mem_gb=4,
            extra_args="--no-sub",
            bids_dir="custom_bids",
            derivatives_dir="custom_derivatives",
            work_dir="custom_work",
            tmp_dir="custom_tmp",
        )
        self.assertEqual(config.modalities, ["T1w", "bold"])
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.mem_gb, 4)
        self.assertTrue(config.validate())

    def _test_thread_limit_exceeded(self) -> None:
        """Test that threads > 2 raises ValueError."""
        config = MRIQCConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("threads (4) must be <= 2", str(context.exception))

    def test_invalid_thread_zero(self) -> None:
        """Test that threads < 1 raises ValueError."""
        config = MRIQCConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid thread count: 0", str(context.exception))

    def test_invalid_modality(self) -> None:
        """Test that unsupported modality raises ValueError."""
        config = MRIQCConfig(modalities=["T1w", "invalid_modality"])
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn(
            "Unsupported modality 'invalid_modality'",
            str(context.exception),
        )


class TestMRIQCCommandBuilder(unittest.TestCase):
    """Test cases for MRIQCCommandBuilder command generation."""

    def setUp(self) -> None:
        """Set up command builder with default configuration."""
        self.builder = MRIQCCommandBuilder()

    def test_build_participant_command(self) -> None:
        """Test participant-level command construction."""
        cmd = self.builder.build_participant_command(
            bids_dir="bids",
            output_dir="derivatives/mriqc",
            subject="19081001",
            work_dir="work/mriqc/sub-19081001",
        )
        expected_cmd = [
            "mriqc",
            "bids",
            "derivatives/mriqc",
            "participant",
            "--participant-label",
            "19081001",
            "--modalities",
            "T1w",
            "bold",
            "dwi",
            "--nprocs",
            "2",
            "--omp-nthreads",
            "2",
            "--work-dir",
            "work/mriqc/sub-19081001",
            "--verbose-reports",
            "--no-sub",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_participant_command_with_sub_prefix(self) -> None:
        """Test that 'sub-' prefix is stripped from participant label."""
        cmd = self.builder.build_participant_command(
            bids_dir="bids",
            output_dir="derivatives/mriqc",
            subject="sub-19081001",
            work_dir="work/mriqc/sub-19081001",
        )
        self.assertIn("--participant-label", cmd)
        idx = cmd.index("--participant-label")
        self.assertEqual(cmd[idx + 1], "19081001")

    def test_build_group_command(self) -> None:
        """Test group-level command construction."""
        cmd = self.builder.build_group_command(
            bids_dir="bids",
            output_dir="derivatives/mriqc",
            work_dir="work/mriqc/group",
        )
        expected_cmd = [
            "mriqc",
            "bids",
            "derivatives/mriqc",
            "group",
            "--modalities",
            "T1w",
            "bold",
            "dwi",
            "--nprocs",
            "2",
            "--omp-nthreads",
            "2",
            "--work-dir",
            "work/mriqc/group",
            "--verbose-reports",
            "--no-sub",
        ]
        self.assertEqual(cmd, expected_cmd)


class TestMRIQCPathResolver(unittest.TestCase):
    """Test cases for MRIQCPathResolver."""

    def setUp(self) -> None:
        """Set up path resolver."""
        self.resolver = MRIQCPathResolver(derivatives_dir="derivatives")

    def test_mriqc_dir(self) -> None:
        """Test MRIQC output directory path."""
        self.assertEqual(self.resolver.mriqc_dir, Path("derivatives/mriqc"))

    def test_get_subject_html_report(self) -> None:
        """Test subject HTML report path resolution."""
        report_path = self.resolver.get_subject_html_report("19081001")
        self.assertEqual(
            report_path, Path("derivatives/mriqc/sub-19081001.html")
        )

    def test_get_subject_marker(self) -> None:
        """Test subject marker path resolution."""
        marker_path = self.resolver.get_subject_marker("sub-19081001")
        self.assertEqual(
            marker_path,
            Path("derivatives/mriqc/sub-19081001/.mriqc_complete"),
        )

    def test_get_group_marker(self) -> None:
        """Test group marker path resolution."""
        marker_path = self.resolver.get_group_marker()
        self.assertEqual(
            marker_path, Path("derivatives/mriqc/.mriqc_group_complete")
        )


class TestMRIQCRunner(BaseTest):
    """Test cases for MRIQCRunner helper methods."""

    def setUp(self) -> None:
        """Set up runner instance."""
        self.runner = MRIQCRunner()

    def test_prepare_environment(self) -> None:
        """Test environment preparation with thread limits."""
        with self.create_temp_dir() as tmp_dir:
            env = self.runner._prepare_environment(Path(tmp_dir), 2)
            self.assertEqual(env.get("TMPDIR"), tmp_dir)
            self.assertEqual(env.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), "2")

    def test_ensure_report_file(self) -> None:
        """Test report file placement and fallback."""
        with self.create_temp_dir() as tmp_dir:
            out_dir = Path(tmp_dir) / "mriqc"
            out_dir.mkdir(parents=True)
            target = out_dir / "sub-19081001.html"

            # Create mock source report
            src_report = out_dir / "sub-19081001_ses-01_bold.html"
            src_report.write_text("<html>Report</html>", encoding="utf-8")

            res = self.runner._ensure_report_file(
                out_dir, "19081001", target
            )
            self.assertEqual(res, target)
            self.assertTrue(target.exists())
            self.assertIn("Report", target.read_text(encoding="utf-8"))


class TestMRIQCApp(unittest.TestCase):
    """Test cases for MRIQCApp CLI argument parsing."""

    def setUp(self) -> None:
        """Set up app parser."""
        self.app = MRIQCApp()
        self.parser = self.app.create_parser()

    def test_participant_parser(self) -> None:
        """Test parsing participant arguments."""
        args = self.parser.parse_args([
            "participant",
            "--bids-dir", "bids",
            "--output-dir", "derivatives/mriqc",
            "--subject", "19081001",
            "--work-dir", "work/mriqc/sub-19081001",
            "--tmp-dir", ".tmp",
            "--threads", "2",
            "--modalities", "T1w", "bold", "dwi",
        ])
        self.assertEqual(args.mode, "participant")
        self.assertEqual(args.subject, "19081001")
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.modalities, ["T1w", "bold", "dwi"])

    def test_group_parser(self) -> None:
        """Test parsing group arguments."""
        args = self.parser.parse_args([
            "group",
            "--bids-dir", "bids",
            "--output-dir", "derivatives/mriqc",
            "--work-dir", "work/mriqc/group",
            "--tmp-dir", ".tmp",
            "--threads", "2",
            "--modalities", "T1w", "bold", "dwi",
        ])
        self.assertEqual(args.mode, "group")
        self.assertEqual(args.threads, 2)


if __name__ == "__main__":
    unittest.main()
