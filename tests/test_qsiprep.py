"""Unit tests for QSIPrep helper classes and configuration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow.scripts.qsiprep_helper import (
    QSIPrepApp,
    QSIPrepCommandBuilder,
    QSIPrepConfig,
    QSIPrepDenoiseMethod,
    QSIPrepOutputSpace,
    QSIPrepPathResolver,
    QSIPrepRunner,
    QSIPrepUnringingMethod,
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


class TestQSIPrepDenoiseMethod(unittest.TestCase):
    """Test cases for QSIPrepDenoiseMethod enum."""

    def test_enum_members(self) -> None:
        """Test enum member values."""
        self.assertEqual(QSIPrepDenoiseMethod.DWIDENOISE.value, "dwidenoise")
        self.assertEqual(QSIPrepDenoiseMethod.PATCH2SELF.value, "patch2self")
        self.assertEqual(QSIPrepDenoiseMethod.NONE.value, "none")

    def test_from_value_string(self) -> None:
        """Test parsing valid denoise method strings."""
        self.assertEqual(
            QSIPrepDenoiseMethod.from_value("dwidenoise"),
            QSIPrepDenoiseMethod.DWIDENOISE,
        )
        self.assertEqual(
            QSIPrepDenoiseMethod.from_value("patch2self"),
            QSIPrepDenoiseMethod.PATCH2SELF,
        )
        self.assertEqual(
            QSIPrepDenoiseMethod.from_value(" NONE "),
            QSIPrepDenoiseMethod.NONE,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test parsing existing enum instance."""
        self.assertEqual(
            QSIPrepDenoiseMethod.from_value(QSIPrepDenoiseMethod.DWIDENOISE),
            QSIPrepDenoiseMethod.DWIDENOISE,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported method."""
        with self.assertRaises(ValueError) as context:
            QSIPrepDenoiseMethod.from_value("invalid_denoise")
        self.assertIn("Unsupported denoise method", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid input type."""
        with self.assertRaises(ValueError) as context:
            QSIPrepDenoiseMethod.from_value(123)
        self.assertIn("Invalid denoise method type", str(context.exception))


class TestQSIPrepUnringingMethod(unittest.TestCase):
    """Test cases for QSIPrepUnringingMethod enum."""

    def test_enum_members(self) -> None:
        """Test enum member values."""
        self.assertEqual(QSIPrepUnringingMethod.MRDEGIBBS.value, "mrdegibbs")
        self.assertEqual(QSIPrepUnringingMethod.RPG.value, "rpg")
        self.assertEqual(QSIPrepUnringingMethod.NONE.value, "none")

    def test_from_value_string(self) -> None:
        """Test parsing valid unringing method strings."""
        self.assertEqual(
            QSIPrepUnringingMethod.from_value("mrdegibbs"),
            QSIPrepUnringingMethod.MRDEGIBBS,
        )
        self.assertEqual(
            QSIPrepUnringingMethod.from_value("rpg"),
            QSIPrepUnringingMethod.RPG,
        )
        self.assertEqual(
            QSIPrepUnringingMethod.from_value(" none "),
            QSIPrepUnringingMethod.NONE,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test parsing existing enum instance."""
        self.assertEqual(
            QSIPrepUnringingMethod.from_value(
                QSIPrepUnringingMethod.MRDEGIBBS
            ),
            QSIPrepUnringingMethod.MRDEGIBBS,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported unringing method."""
        with self.assertRaises(ValueError) as context:
            QSIPrepUnringingMethod.from_value("invalid_unringing")
        self.assertIn("Unsupported unringing method", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid input type."""
        with self.assertRaises(ValueError) as context:
            QSIPrepUnringingMethod.from_value(456)
        self.assertIn("Invalid unringing method type", str(context.exception))


class TestQSIPrepOutputSpace(unittest.TestCase):
    """Test cases for QSIPrepOutputSpace enum."""

    def test_enum_members(self) -> None:
        """Test enum member values."""
        self.assertEqual(QSIPrepOutputSpace.T1W.value, "T1w")
        self.assertEqual(
            QSIPrepOutputSpace.MNI152NLIN2009CASYM.value,
            "MNI152NLin2009cAsym",
        )

    def test_from_value_string(self) -> None:
        """Test parsing valid output space strings."""
        self.assertEqual(
            QSIPrepOutputSpace.from_value("T1w"),
            QSIPrepOutputSpace.T1W,
        )
        self.assertEqual(
            QSIPrepOutputSpace.from_value("MNI152NLin2009cAsym"),
            QSIPrepOutputSpace.MNI152NLIN2009CASYM,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on invalid space string."""
        with self.assertRaises(ValueError) as context:
            QSIPrepOutputSpace.from_value("unsupported_space")
        self.assertIn("Unsupported output space", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid space type."""
        with self.assertRaises(ValueError) as context:
            QSIPrepOutputSpace.from_value(789)
        self.assertIn("Invalid output space type", str(context.exception))


class TestQSIPrepConfig(unittest.TestCase):
    """Test cases for QSIPrepConfig container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of QSIPrepConfig."""
        config = QSIPrepConfig()
        self.assertEqual(config.threads, 2)
        self.assertEqual(config.mem_mb, 8000)
        self.assertEqual(config.output_resolution, 1.5)
        self.assertEqual(
            config.denoise_method, QSIPrepDenoiseMethod.DWIDENOISE
        )
        self.assertEqual(
            config.unringing_method, QSIPrepUnringingMethod.MRDEGIBBS
        )
        self.assertFalse(config.separate_all_dwis)
        self.assertEqual(config.fs_subjects_dir, "derivatives/fastsurfer")
        self.assertIsNone(config.fs_license)
        self.assertFalse(config.do_reconall)
        self.assertIsNone(config.bids_filter_file)
        self.assertEqual(config.extra_args, "--skip-bids-validation --notrack")
        self.assertEqual(config.bids_dir, "bids")
        self.assertEqual(config.derivatives_dir, "derivatives")
        self.assertEqual(config.work_dir, "work")
        self.assertEqual(config.tmp_dir, ".tmp")
        self.assertTrue(config.validate())

    def test_custom_valid_configuration(self) -> None:
        """Test custom valid parameters."""
        config = QSIPrepConfig(
            threads=1,
            mem_mb=4000,
            output_resolution=1.2,
            denoise_method="patch2self",
            unringing_method="rpg",
            separate_all_dwis=True,
            fs_subjects_dir="/custom/fastsurfer",
            fs_license="/opt/freesurfer/license.txt",
            do_reconall=True,
            bids_filter_file="config/filter.json",
            extra_args="--verbose-reports",
            bids_dir="custom_bids",
            derivatives_dir="custom_derivatives",
            work_dir="custom_work",
            tmp_dir="custom_tmp",
        )
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.mem_mb, 4000)
        self.assertEqual(config.output_resolution, 1.2)
        self.assertEqual(
            config.denoise_method, QSIPrepDenoiseMethod.PATCH2SELF
        )
        self.assertEqual(
            config.unringing_method, QSIPrepUnringingMethod.RPG
        )
        self.assertTrue(config.separate_all_dwis)
        self.assertEqual(config.fs_subjects_dir, "/custom/fastsurfer")
        self.assertEqual(config.fs_license, "/opt/freesurfer/license.txt")
        self.assertTrue(config.do_reconall)
        self.assertEqual(config.bids_filter_file, "config/filter.json")
        self.assertEqual(config.extra_args, "--verbose-reports")
        self.assertTrue(config.validate())

    def _test_thread_limit_exceeded(self) -> None:
        """Test that threads > 2 raises ValueError."""
        config = QSIPrepConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("threads (4) must be <= 2", str(context.exception))

    def test_invalid_thread_zero(self) -> None:
        """Test that threads < 1 raises ValueError."""
        config = QSIPrepConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid thread count: 0", str(context.exception))

    def test_invalid_mem_mb(self) -> None:
        """Test that memory < 1000 MB raises ValueError."""
        config = QSIPrepConfig(mem_mb=500)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid memory allocation", str(context.exception))

    def test_invalid_output_resolution(self) -> None:
        """Test that non-positive output resolution raises ValueError."""
        config = QSIPrepConfig(output_resolution=0.0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("Invalid output resolution: 0.0", str(context.exception))


class TestQSIPrepCommandBuilder(unittest.TestCase):
    """Test cases for QSIPrepCommandBuilder command generation."""

    def test_build_default_command(self) -> None:
        """Test standard default command generation."""
        builder = QSIPrepCommandBuilder()
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/qsiprep"),
            subject="19081001",
            work_dir=Path("work/qsiprep/sub-19081001"),
        )
        expected_cmd = [
            "qsiprep",
            "bids",
            "derivatives/qsiprep",
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
            "work/qsiprep/sub-19081001",
            "--output-resolution",
            "1.5",
            "--denoise-method",
            "dwidenoise",
            "--unringing-method",
            "mrdegibbs",
            "--fs-subjects-dir",
            "derivatives/fastsurfer",
            "--skip-bids-validation",
            "--notrack",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_command_with_subject_prefix(self) -> None:
        """Test command building strips sub- prefix from subject."""
        builder = QSIPrepCommandBuilder()
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/qsiprep"),
            subject="sub-19081001",
            work_dir=Path("work/qsiprep/sub-19081001"),
        )
        self.assertIn("--participant-label", cmd)
        idx = cmd.index("--participant-label")
        self.assertEqual(cmd[idx + 1], "19081001")

    def test_build_command_with_all_options(self) -> None:
        """Test command building with license, recon-all, and separate DWIs."""
        config = QSIPrepConfig(
            threads=1,
            mem_mb=4096,
            output_resolution=1.2,
            denoise_method="patch2self",
            unringing_method="rpg",
            separate_all_dwis=True,
            fs_subjects_dir="derivatives/fastsurfer",
            fs_license="/opt/freesurfer/license.txt",
            do_reconall=True,
            bids_filter_file="config/filter.json",
            extra_args="--verbose-reports --no-sub",
        )
        builder = QSIPrepCommandBuilder(config)
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/qsiprep"),
            subject="sub-01",
            work_dir=Path("work/qsiprep/sub-01"),
        )
        expected_cmd = [
            "qsiprep",
            "bids",
            "derivatives/qsiprep",
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
            "work/qsiprep/sub-01",
            "--output-resolution",
            "1.2",
            "--denoise-method",
            "patch2self",
            "--unringing-method",
            "rpg",
            "--fs-subjects-dir",
            "derivatives/fastsurfer",
            "--fs-license-file",
            "/opt/freesurfer/license.txt",
            "--do-reconall",
            "--separate-all-dwis",
            "--bids-filter-file",
            "config/filter.json",
            "--verbose-reports",
            "--no-sub",
        ]
        self.assertEqual(cmd, expected_cmd)

    def test_build_command_with_none_denoise_and_unringing(self) -> None:
        """Test command building when denoise and unringing are none."""
        config = QSIPrepConfig(
            denoise_method=QSIPrepDenoiseMethod.NONE,
            unringing_method=QSIPrepUnringingMethod.NONE,
            fs_subjects_dir="",
            fs_license="",
        )
        builder = QSIPrepCommandBuilder(config)
        cmd = builder.build_participant_command(
            bids_dir=Path("bids"),
            output_dir=Path("derivatives/qsiprep"),
            subject="sub-01",
            work_dir=Path("work/qsiprep/sub-01"),
        )
        self.assertNotIn("--denoise-method", cmd)
        self.assertNotIn("--unringing-method", cmd)
        self.assertNotIn("--fs-subjects-dir", cmd)
        self.assertNotIn("--fs-license-file", cmd)


class TestQSIPrepPathResolver(BaseTest):
    """Test cases for QSIPrepPathResolver path computations."""

    def setUp(self) -> None:
        """Set up resolver instance."""
        self.resolver = QSIPrepPathResolver(derivatives_dir="derivatives")

    def test_directories(self) -> None:
        """Test directory accessors."""
        self.assertEqual(self.resolver.derivatives_dir, Path("derivatives"))
        self.assertEqual(
            self.resolver.qsiprep_dir, Path("derivatives/qsiprep")
        )

    def test_get_subject_dir(self) -> None:
        """Test subject directory resolution."""
        self.assertEqual(
            self.resolver.get_subject_dir("19081001"),
            Path("derivatives/qsiprep/sub-19081001"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("sub-19081001"),
            Path("derivatives/qsiprep/sub-19081001"),
        )

    def test_get_subject_html_report(self) -> None:
        """Test subject HTML report resolution."""
        self.assertEqual(
            self.resolver.get_subject_html_report("19081001"),
            Path("derivatives/qsiprep/sub-19081001.html"),
        )

    def test_get_subject_marker(self) -> None:
        """Test subject completion marker resolution."""
        self.assertEqual(
            self.resolver.get_subject_marker("19081001"),
            Path("derivatives/qsiprep/sub-19081001/.qsiprep_complete"),
        )

    def test_get_dwi_preproc(self) -> None:
        """Test preprocessed DWI path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/dwi/"
            "sub-19081001_space-T1w_desc-preproc_dwi.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_dwi_preproc("19081001", space="T1w"),
            expected,
        )

    def test_get_dwi_bval(self) -> None:
        """Test preprocessed DWI bval path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/dwi/"
            "sub-19081001_space-T1w_desc-preproc_dwi.bval"
        )
        self.assertEqual(
            self.resolver.get_dwi_bval("19081001", space="T1w"),
            expected,
        )

    def test_get_dwi_bvec(self) -> None:
        """Test preprocessed DWI bvec path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/dwi/"
            "sub-19081001_space-T1w_desc-preproc_dwi.bvec"
        )
        self.assertEqual(
            self.resolver.get_dwi_bvec("19081001", space="T1w"),
            expected,
        )

    def test_get_dwi_brainmask(self) -> None:
        """Test preprocessed DWI brain mask path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/dwi/"
            "sub-19081001_space-T1w_desc-brain_mask.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_dwi_brainmask("19081001", space="T1w"),
            expected,
        )

    def test_get_anatomical_preproc(self) -> None:
        """Test preprocessed anatomical path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/anat/"
            "sub-19081001_desc-preproc_T1w.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_anatomical_preproc("19081001"),
            expected,
        )

    def test_get_anatomical_brainmask(self) -> None:
        """Test anatomical brain mask path resolution."""
        expected = Path(
            "derivatives/qsiprep/sub-19081001/anat/"
            "sub-19081001_desc-brain_mask.nii.gz"
        )
        self.assertEqual(
            self.resolver.get_anatomical_brainmask("19081001"),
            expected,
        )


class TestQSIPrepRunner(BaseTest):
    """Test cases for QSIPrepRunner environment and execution."""

    def setUp(self) -> None:
        """Set up runner instance."""
        self.runner = QSIPrepRunner()

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
            out_dir = temp_path / "qsiprep"
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
            out_dir = temp_path / "qsiprep"
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

    def test_ensure_report_file_none(self) -> None:
        """Test ensure_report_file returns None when target_report is None."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            out_dir = temp_path / "qsiprep"
            out_dir.mkdir(parents=True)
            result = self.runner.ensure_report_file(
                output_dir=out_dir,
                subject="19081001",
                target_report=None,
            )
            self.assertIsNone(result)

    @patch("subprocess.run")
    def test_run_success(self, mock_subprocess_run: MagicMock) -> None:
        """Test successful execution creating marker and report."""
        mock_subprocess_run.return_value = MagicMock(returncode=0)
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            bids_dir = temp_path / "bids"
            out_dir = temp_path / "qsiprep"
            work_dir = temp_path / "work"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / "sub-01" / ".qsiprep_complete"
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
            out_dir = temp_path / "qsiprep"
            work_dir = temp_path / "work"
            tmp_dir = temp_path / ".tmp"
            marker = out_dir / "sub-01" / ".qsiprep_complete"
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


class TestQSIPrepApp(unittest.TestCase):
    """Test cases for QSIPrepApp CLI argument parsing."""

    def setUp(self) -> None:
        """Set up app instance and parser."""
        self.app = QSIPrepApp()
        self.parser = self.app.create_parser()

    def test_parse_valid_arguments(self) -> None:
        """Test parsing valid CLI arguments."""
        args = self.parser.parse_args([
            "--bids-dir", "bids",
            "--output-dir", "derivatives/qsiprep",
            "--subject", "sub-19081001",
            "--work-dir", "work/qsiprep",
            "--threads", "2",
            "--mem-mb", "8000",
            "--output-resolution", "1.5",
            "--denoise-method", "dwidenoise",
            "--unringing-method", "mrdegibbs",
            "--separate-all-dwis",
            "--fs-subjects-dir", "derivatives/fastsurfer",
            "--fs-license", "/opt/fs/license.txt",
            "--do-reconall",
            "--extra-args=--notrack",
            "--marker", "derivatives/qsiprep/sub-19081001/.qsiprep_complete",
            "--report", "derivatives/qsiprep/sub-19081001.html",
        ])
        self.assertEqual(args.bids_dir, Path("bids"))
        self.assertEqual(args.output_dir, Path("derivatives/qsiprep"))
        self.assertEqual(args.subject, "sub-19081001")
        self.assertEqual(args.work_dir, Path("work/qsiprep"))
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.mem_mb, 8000)
        self.assertEqual(args.output_resolution, 1.5)
        self.assertEqual(args.denoise_method, "dwidenoise")
        self.assertEqual(args.unringing_method, "mrdegibbs")
        self.assertTrue(args.separate_all_dwis)
        self.assertEqual(args.fs_subjects_dir, "derivatives/fastsurfer")
        self.assertEqual(args.fs_license, "/opt/fs/license.txt")
        self.assertTrue(args.do_reconall)
        self.assertEqual(args.extra_args, "--notrack")
        self.assertEqual(
            args.marker,
            Path("derivatives/qsiprep/sub-19081001/.qsiprep_complete"),
        )
        self.assertEqual(
            args.report,
            Path("derivatives/qsiprep/sub-19081001.html"),
        )

    @patch.object(QSIPrepRunner, "run")
    def test_app_run(self, mock_runner_run: MagicMock) -> None:
        """Test running app parses arguments and invokes runner."""
        mock_runner_run.return_value = 0
        cli_args = [
            "--bids-dir", "bids",
            "--output-dir", "derivatives/qsiprep",
            "--subject", "19081001",
        ]
        ret = self.app.run(cli_args)
        self.assertEqual(ret, 0)
        mock_runner_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
