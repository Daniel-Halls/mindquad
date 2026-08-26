"""Unit tests for fMRIPrep helper classes and configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow.scripts.fmriprep_helper import (
    FMRIPrepApp,
    FMRIPrepCommandBuilder,
    FMRIPrepConfig,
    FMRIPrepOutputSpace,
    FMRIPrepPathResolver,
    FMRIPrepRunner,
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


class TestFMRIPrepOutputSpace(unittest.TestCase):
    """Test cases for FMRIPrepOutputSpace enum conversions and validation."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(
            FMRIPrepOutputSpace.MNI152NLIN2009CASYM_RES2.value,
            "MNI152NLin2009cAsym:res-2",
        )
        self.assertEqual(
            FMRIPrepOutputSpace.FSAVERAGE5.value,
            "fsaverage5",
        )
        self.assertEqual(
            FMRIPrepOutputSpace.ANAT.value,
            "anat",
        )

    def test_from_value_string(self) -> None:
        """Test parsing valid output space strings."""
        self.assertEqual(
            FMRIPrepOutputSpace.from_value("MNI152NLin2009cAsym:res-2"),
            FMRIPrepOutputSpace.MNI152NLIN2009CASYM_RES2,
        )
        self.assertEqual(
            FMRIPrepOutputSpace.from_value("fsaverage5"),
            FMRIPrepOutputSpace.FSAVERAGE5,
        )
        self.assertEqual(
            FMRIPrepOutputSpace.from_value(" anat "),
            FMRIPrepOutputSpace.ANAT,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing existing enum instance."""
        self.assertEqual(
            FMRIPrepOutputSpace.from_value(FMRIPrepOutputSpace.FSAVERAGE),
            FMRIPrepOutputSpace.FSAVERAGE,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported space string."""
        with self.assertRaises(ValueError) as context:
            FMRIPrepOutputSpace.from_value("invalid_space")
        self.assertIn("Unsupported output space", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            FMRIPrepOutputSpace.from_value(999)
        self.assertIn("Invalid output space type", str(context.exception))


class TestFMRIPrepConfig(unittest.TestCase):
    """Test cases for FMRIPrepConfig container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of FMRIPrepConfig."""
        config = FMRIPrepConfig()
        self.assertEqual(config.threads, 2)
        self.assertEqual(config.mem_mb, 8000)
        self.assertEqual(
            config.output_spaces,
            ["MNI152NLin2009cAsym:res-2", "fsaverage5"],
        )
        self.assertEqual(config.cifti_output, "91k")
        self.assertEqual(config.fs_subjects_dir, "derivatives/fastsurfer")
        self.assertIsNone(config.fs_license)
        self.assertFalse(config.fs_no_reconall)
        self.assertIsNone(config.dummy_scans)
        self.assertIsNone(config.bids_filter_file)
        self.assertEqual(config.extra_args, "--skip-bids-validation --notrack")
        self.assertEqual(config.bids_dir, "bids")
        self.assertEqual(config.derivatives_dir, "derivatives")
        self.assertEqual(config.work_dir, "work")
        self.assertEqual(config.tmp_dir, ".tmp")
        self.assertTrue(config.validate())

    def test_custom_valid_configuration(self) -> None:
        """Test custom valid parameters."""
        config = FMRIPrepConfig(
            threads=1,
            mem_mb=4000,
            output_spaces=["MNI152NLin6Asym:res-2"],
            cifti_output="170k",
            fs_subjects_dir="/custom/fastsurfer",
            fs_license="/opt/freesurfer/license.txt",
            fs_no_reconall=True,
            dummy_scans=4,
            bids_filter_file="config/filter.json",
            extra_args="--verbose-reports",
            bids_dir="custom_bids",
            derivatives_dir="custom_derivatives",
            work_dir="custom_work",
            tmp_dir="custom_tmp",
        )
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.mem_mb, 4000)
        self.assertEqual(config.output_spaces, ["MNI152NLin6Asym:res-2"])
        self.assertEqual(config.cifti_output, "170k")
        self.assertEqual(config.fs_subjects_dir, "/custom/fastsurfer")
        self.assertEqual(config.fs_license, "/opt/freesurfer/license.txt")
        self.assertTrue(config.fs_no_reconall)
        self.assertEqual(config.dummy_scans, 4)
        self.assertEqual(config.bids_filter_file, "config/filter.json")
        self.assertEqual(config.extra_args, "--verbose-reports")
        self.assertTrue(config.validate())

    def _test_thread_limit_exceeded(self) -> None:
        """Test that threads > 2 raises ValueError."""
        config = FMRIPrepConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("threads (4) must be <= 2", str(context.exception))

    def test_invalid_thread_zero(self) -> None:
        """Test that threads < 1 raises ValueError."""
        config = FMRIPrepConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid thread count: 0", str(context.exception))

    def test_invalid_mem_mb(self) -> None:
        """Test that memory < 1000 MB raises ValueError."""
        config = FMRIPrepConfig(mem_mb=500)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid memory allocation", str(context.exception))

    def test_empty_output_spaces(self) -> None:
        """Test that empty output spaces list raises ValueError."""
        config = FMRIPrepConfig(output_spaces=[])
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn(
            "At least one output space must be specified",
            str(context.exception),
        )

    def test_invalid_dummy_scans(self) -> None:
        """Test that negative dummy scans raises ValueError."""
        config = FMRIPrepConfig(dummy_scans=-1)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid dummy scans: -1", str(context.exception))


class TestFMRIPrepCommandBuilder(unittest.TestCase):
    """Test cases for FMRIPrepCommandBuilder command generation."""

    def test_build_default_command(self) -> None:
        """Test standard default command generation."""
        builder = FMRIPrepCommandBuilder()
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/fmriprep"),
            subject="19081001",
            work_dir=Path("work/fmriprep/sub-19081001"),
        )
        expected_cmd = [
            "fmriprep",
            "bids",
            "derivatives/fmriprep",
            "participant",
            "--participant-label",
            "19081001",
            "--nprocs",
            "2",
            "--omp-nthreads",
            "2",
            "--mem-mb",
            "8000",
            "--work-dir",
            "work/fmriprep/sub-19081001",
            "--output-spaces",
            "MNI152NLin2009cAsym:res-2",
            "fsaverage5",
            "--fs-subjects-dir",
            "derivatives/fastsurfer",
            "--cifti-output",
            "91k",
            "--skip-bids-validation",
            "--notrack",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_command_with_subject_prefix(self) -> None:
        """Test command building strips sub- prefix from subject."""
        builder = FMRIPrepCommandBuilder()
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/fmriprep"),
            subject="sub-19081001",
            work_dir=Path("work/fmriprep/sub-19081001"),
        )
        self.assertIn("--participant-label", cmd)
        idx = cmd.index("--participant-label")
        self.assertEqual(cmd[idx + 1], "19081001")

    def test_build_command_with_all_options(self) -> None:
        """Test command building with license, dummy scans, and filter file."""
        config = FMRIPrepConfig(
            threads=1,
            mem_mb=4096,
            output_spaces=["MNI152NLin6Asym:res-2"],
            cifti_output=None,
            fs_subjects_dir="derivatives/fastsurfer",
            fs_license="/opt/freesurfer/license.txt",
            fs_no_reconall=True,
            dummy_scans=2,
            bids_filter_file="config/filter.json",
            extra_args="--verbose-reports --no-sub",
        )
        builder = FMRIPrepCommandBuilder(config)
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/fmriprep"),
            subject="sub-01",
            work_dir=Path("work/fmriprep/sub-01"),
        )
        expected_cmd = [
            "fmriprep",
            "bids",
            "derivatives/fmriprep",
            "participant",
            "--participant-label",
            "01",
            "--nprocs",
            "1",
            "--omp-nthreads",
            "1",
            "--mem-mb",
            "4096",
            "--work-dir",
            "work/fmriprep/sub-01",
            "--output-spaces",
            "MNI152NLin6Asym:res-2",
            "--fs-subjects-dir",
            "derivatives/fastsurfer",
            "--fs-license-file",
            "/opt/freesurfer/license.txt",
            "--fs-no-reconall",
            "--dummy-scans",
            "2",
            "--bids-filter-file",
            "config/filter.json",
            "--verbose-reports",
            "--no-sub",
        ]
        self.assertEqual(cmd, expected_cmd)


class TestFMRIPrepPathResolver(BaseTest):
    """Test cases for FMRIPrepPathResolver path computations."""

    def setUp(self) -> None:
        """Set up resolver instance."""
        self.resolver = FMRIPrepPathResolver(derivatives_dir="derivatives")

    def test_directories(self) -> None:
        """Test directory accessors."""
        self.assertEqual(self.resolver.derivatives_dir, Path("derivatives"))
        self.assertEqual(
            self.resolver.fmriprep_dir, Path("derivatives/fmriprep")
        )

    def test_get_subject_dir(self) -> None:
        """Test subject directory resolution."""
        self.assertEqual(
            self.resolver.get_subject_dir("19081001"),
            Path("derivatives/fmriprep/sub-19081001"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("sub-19081001"),
            Path("derivatives/fmriprep/sub-19081001"),
        )

    def test_get_subject_html_report(self) -> None:
        """Test subject HTML report resolution."""
        self.assertEqual(
            self.resolver.get_subject_html_report("19081001"),
            Path("derivatives/fmriprep/sub-19081001.html"),
        )

    def test_get_subject_marker(self) -> None:
        """Test subject completion marker resolution."""
        self.assertEqual(
            self.resolver.get_subject_marker("19081001"),
            Path("derivatives/fmriprep/sub-19081001/.fmriprep_complete"),
        )

    def test_get_anatomical_preproc(self) -> None:
        """Test preprocessed anatomical path resolution."""
        expected = Path(
            "derivatives/fmriprep/sub-19081001/anat/"
            "sub-19081001_space-MNI152NLin2009cAsym_res-2_"
            "desc-preproc_T1w.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_anatomical_preproc("19081001"),
            expected,
        )

    def test_get_confounds_file(self) -> None:
        """Test confounds timeseries TSV path resolution."""
        expected_plain = Path(
            "derivatives/fmriprep/sub-19081001/func/"
            "sub-19081001_task-rest_desc-confounds_timeseries.tsv"
        )
        self.assertEqual(
            self.resolver.get_confounds_file("19081001", task="rest"),
            expected_plain,
        )
        expected_run = Path(
            "derivatives/fmriprep/sub-19081001/func/"
            "sub-19081001_task-rest_run-01_desc-confounds_timeseries.tsv"
        )
        self.assertEqual(
            self.resolver.get_confounds_file(
                "19081001", task="rest", run="01"
            ),
            expected_run,
        )

    def test_get_bold_preproc(self) -> None:
        """Test preprocessed BOLD path resolution."""
        expected_plain = Path(
            "derivatives/fmriprep/sub-19081001/func/"
            "sub-19081001_task-rest_space-MNI152NLin2009cAsym_res-2_"
            "desc-preproc_bold.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_bold_preproc("19081001", task="rest"),
            expected_plain,
        )
        expected_run = Path(
            "derivatives/fmriprep/sub-19081001/func/"
            "sub-19081001_task-rest_run-02_"
            "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_bold_preproc(
                "19081001", task="rest", run="02"
            ),
            expected_run,
        )


class TestFMRIPrepRunner(BaseTest):
    """Test cases for FMRIPrepRunner environment and execution."""

    def setUp(self) -> None:
        """Set up runner instance."""
        self.runner = FMRIPrepRunner()

    def test_prepare_environment(self) -> None:
        """Test environment preparation with thread and license settings."""
        with self.create_temp_dir() as temp_dir:
            tmp_path = Path(temp_dir)
            env = self.runner.prepare_environment(
                tmp_dir=tmp_path,
                threads=2,
                fs_license="/license/fs.txt",
            )
            self.assertEqual(env.get("TMPDIR"), str(tmp_path))
            self.assertEqual(env.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), "2")
            self.assertEqual(env.get("MKL_NUM_THREADS"), "2")
            self.assertEqual(env.get("FS_LICENSE"), "/license/fs.txt")

    def test_ensure_report_file_existing_copy(self) -> None:
        """Test copying found report to target report path."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "fmriprep"
            out_dir.mkdir(parents=True)
            generated_report = out_dir / "sub-19081001.html"
            generated_report.write_text("<h1>Report</h1>", encoding="utf-8")

            target_report = temp_path / "reports" / "sub-19081001.html"
            result = self.runner.ensure_report_file(
                output_dir=out_dir,
                subject="19081001",
                target_report=target_report,
            )
            self.assertEqual(result, target_report)
            self.assertTrue(target_report.exists())
            self.assertEqual(
                target_report.read_text(encoding="utf-8"), "<h1>Report</h1>"
            )

    def test_ensure_report_file_placeholder(self) -> None:
        """Test creating placeholder when report file is not generated."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "fmriprep"
            out_dir.mkdir(parents=True)
            target_report = out_dir / "sub-19081001.html"

            result = self.runner.ensure_report_file(
                output_dir=out_dir,
                subject="19081001",
                target_report=target_report,
            )
            self.assertEqual(result, target_report)
            self.assertTrue(target_report.exists())
            report_text = target_report.read_text(encoding="utf-8")
            self.assertIn("placeholder", report_text)

    @patch("subprocess.run")
    def test_run_success(self, mock_subprocess_run: MagicMock) -> None:
        """Test successful execution creating marker and report."""
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            bids_dir = temp_path / "bids"
            out_dir = temp_path / "fmriprep"
            work_dir = temp_path / "work"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / "sub-01" / ".fmriprep_complete"
            report = out_dir / "sub-01.html"

            ret = self.runner.run(
                bids_dir=bids_dir,
                output_dir=out_dir,
                subject="sub-01",
                work_dir=work_dir,
                tmp_dir=tmp_dir,
                threads=2,
                mem_mb=8000,
                marker_path=marker,
                report_path=report,
            )
            self.assertEqual(ret, 0)
            self.assertTrue(marker.exists())
            self.assertTrue(report.exists())

    @patch("subprocess.run")
    def test_run_failure(self, mock_subprocess_run: MagicMock) -> None:
        """Test execution failure returning non-zero return code."""
        mock_subprocess_run.return_value = MagicMock(returncode=1)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            bids_dir = temp_path / "bids"
            out_dir = temp_path / "fmriprep"
            work_dir = temp_path / "work"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / "sub-01" / ".fmriprep_complete"
            report = out_dir / "sub-01.html"

            ret = self.runner.run(
                bids_dir=bids_dir,
                output_dir=out_dir,
                subject="sub-01",
                work_dir=work_dir,
                tmp_dir=tmp_dir,
                threads=2,
                mem_mb=8000,
                marker_path=marker,
                report_path=report,
            )
            self.assertEqual(ret, 1)
            self.assertFalse(marker.exists())


class TestFMRIPrepApp(unittest.TestCase):
    """Test cases for FMRIPrepApp CLI argument parsing."""

    def setUp(self) -> None:
        """Set up app instance and parser."""
        self.app = FMRIPrepApp()
        self.parser = self.app.create_parser()

    def test_parse_valid_arguments(self) -> None:
        """Test parsing valid CLI arguments."""
        args = self.parser.parse_args([
            "--bids-dir", "bids",
            "--output-dir", "derivatives/fmriprep",
            "--subject", "sub-19081001",
            "--work-dir", "work/fmriprep",
            "--threads", "2",
            "--mem-mb", "8000",
            "--output-spaces", "MNI152NLin2009cAsym:res-2", "fsaverage5",
            "--cifti-output", "91k",
            "--fs-subjects-dir", "derivatives/fastsurfer",
            "--fs-license", "/opt/fs/license.txt",
            "--extra-args=--notrack",
            "--marker", "derivatives/fmriprep/sub-19081001/.fmriprep_complete",
            "--report", "derivatives/fmriprep/sub-19081001.html",
        ])
        self.assertEqual(args.bids_dir, Path("bids"))
        self.assertEqual(args.output_dir, Path("derivatives/fmriprep"))
        self.assertEqual(args.subject, "sub-19081001")
        self.assertEqual(args.work_dir, Path("work/fmriprep"))
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.mem_mb, 8000)
        self.assertEqual(
            args.output_spaces, ["MNI152NLin2009cAsym:res-2", "fsaverage5"]
        )
        self.assertEqual(args.cifti_output, "91k")
        self.assertEqual(args.fs_subjects_dir, "derivatives/fastsurfer")
        self.assertEqual(args.fs_license, "/opt/fs/license.txt")
        self.assertEqual(args.extra_args, "--notrack")
        self.assertEqual(
            args.marker,
            Path("derivatives/fmriprep/sub-19081001/.fmriprep_complete"),
        )
        self.assertEqual(
            args.report,
            Path("derivatives/fmriprep/sub-19081001.html"),
        )


if __name__ == "__main__":
    unittest.main()
