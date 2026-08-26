"""Unit tests for MRS helper classes, path resolution, CLI command building, and execution."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from workflow.scripts.mrs_helper import (
    MRSApp,
    MRSConfig,
    MRSFitAlgorithm,
    MRSFitCommandBuilder,
    MRSPathResolver,
    MRSPreprocCommandBuilder,
    MRSQuantitiesManager,
    MRSReportGenerator,
    MRSRunner,
    MRSSegmentCommandBuilder,
    MRSSequenceType,
    MRSTissueSegmentationManager,
    MRSTissueType,
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


class TestMRSFitAlgorithm(unittest.TestCase):
    """Test cases for MRSFitAlgorithm enum conversions and validation."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(MRSFitAlgorithm.NEWTON.value, "Newton")
        self.assertEqual(MRSFitAlgorithm.MH.value, "MH")
        self.assertEqual(MRSFitAlgorithm.AUTO.value, "auto")

    def test_from_value_string(self) -> None:
        """Test from_value parsing valid strings."""
        self.assertEqual(
            MRSFitAlgorithm.from_value("newton"), MRSFitAlgorithm.NEWTON
        )
        self.assertEqual(
            MRSFitAlgorithm.from_value("Newton"), MRSFitAlgorithm.NEWTON
        )
        self.assertEqual(
            MRSFitAlgorithm.from_value("MH"), MRSFitAlgorithm.MH
        )
        self.assertEqual(
            MRSFitAlgorithm.from_value(" mh "), MRSFitAlgorithm.MH
        )
        self.assertEqual(
            MRSFitAlgorithm.from_value("auto"), MRSFitAlgorithm.AUTO
        )

    def test_from_value_enum_instance(self) -> None:
        """Test from_value with existing enum instance."""
        self.assertEqual(
            MRSFitAlgorithm.from_value(MRSFitAlgorithm.NEWTON),
            MRSFitAlgorithm.NEWTON,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test from_value raising ValueError on unsupported string."""
        with self.assertRaises(ValueError) as context:
            MRSFitAlgorithm.from_value("invalid_algo")
        self.assertIn("Unsupported MRS fit algorithm 'invalid_algo'", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test from_value raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            MRSFitAlgorithm.from_value(12345)
        self.assertIn("Invalid MRS fit algorithm type", str(context.exception))


class TestMRSSequenceType(unittest.TestCase):
    """Test cases for MRSSequenceType enum conversions."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(MRSSequenceType.PRESS.value, "press")
        self.assertEqual(MRSSequenceType.STEAM.value, "steam")
        self.assertEqual(MRSSequenceType.SLASER.value, "slaser")
        self.assertEqual(MRSSequenceType.MEGA_PRESS.value, "mega_press")
        self.assertEqual(MRSSequenceType.SPECIAL.value, "special")
        self.assertEqual(MRSSequenceType.SEMI_LASER.value, "semi_laser")
        self.assertEqual(MRSSequenceType.GENERIC.value, "generic")

    def test_from_value_string(self) -> None:
        """Test parsing valid sequence strings with dashes and spaces."""
        self.assertEqual(
            MRSSequenceType.from_value("press"), MRSSequenceType.PRESS
        )
        self.assertEqual(
            MRSSequenceType.from_value("MEGA-PRESS"), MRSSequenceType.MEGA_PRESS
        )
        self.assertEqual(
            MRSSequenceType.from_value(" sLASER "), MRSSequenceType.SLASER
        )

    def test_from_value_enum_instance(self) -> None:
        """Test from_value with existing enum instance."""
        self.assertEqual(
            MRSSequenceType.from_value(MRSSequenceType.STEAM),
            MRSSequenceType.STEAM,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test from_value raising ValueError on unsupported sequence."""
        with self.assertRaises(ValueError) as context:
            MRSSequenceType.from_value("unknown_seq")
        self.assertIn("Unsupported MRS sequence 'unknown_seq'", str(context.exception))

    def test_from_value_invalid_type(self) -> None:
        """Test from_value raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            MRSSequenceType.from_value(["press"])
        self.assertIn("Invalid MRS sequence type", str(context.exception))


class TestMRSTissueType(unittest.TestCase):
    """Test cases for MRSTissueType enum conversions."""

    def test_enum_members(self) -> None:
        """Test tissue type enum values."""
        self.assertEqual(MRSTissueType.GRAY_MATTER.value, "GM")
        self.assertEqual(MRSTissueType.WHITE_MATTER.value, "WM")
        self.assertEqual(MRSTissueType.CSF.value, "CSF")

    def test_from_value_string(self) -> None:
        """Test parsing tissue strings."""
        self.assertEqual(
            MRSTissueType.from_value("gm"), MRSTissueType.GRAY_MATTER
        )
        self.assertEqual(
            MRSTissueType.from_value("WM"), MRSTissueType.WHITE_MATTER
        )
        self.assertEqual(
            MRSTissueType.from_value(" csf "), MRSTissueType.CSF
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on invalid tissue string."""
        with self.assertRaises(ValueError) as context:
            MRSTissueType.from_value("bone")
        self.assertIn("Unsupported MRS tissue type 'bone'", str(context.exception))


class TestMRSConfig(unittest.TestCase):
    """Test cases for MRSConfig data container and validation."""

    def test_default_configuration(self) -> None:
        """Test default parameters of MRSConfig."""
        config = MRSConfig()
        self.assertEqual(config.threads, 2)
        self.assertIsNone(config.basis)
        self.assertIsNone(config.h2o_ref)
        self.assertEqual(config.fit_algorithm, MRSFitAlgorithm.NEWTON)
        self.assertEqual(config.ppm_min, 0.2)
        self.assertEqual(config.ppm_max, 4.2)
        self.assertEqual(config.ppm_range, (0.2, 4.2))
        self.assertEqual(config.baseline_order, 2)
        self.assertEqual(config.internal_reference, "Cr")
        self.assertEqual(config.extra_args, "")
        self.assertTrue(config.validate())

    def test_custom_valid_configuration(self) -> None:
        """Test custom valid parameters."""
        config = MRSConfig(
            threads=1,
            basis="/path/to/basis",
            h2o_ref="/path/to/ref.nii.gz",
            fit_algorithm=MRSFitAlgorithm.MH,
            ppm_min=0.5,
            ppm_max=4.0,
            baseline_order=1,
            internal_reference="tCr",
            extra_args="--ignore Gly",
            work_dir="work/mrs",
            tmp_dir=".tmp/mrs",
        )
        self.assertEqual(config.threads, 1)
        self.assertEqual(config.basis, "/path/to/basis")
        self.assertEqual(config.h2o_ref, "/path/to/ref.nii.gz")
        self.assertEqual(config.fit_algorithm, MRSFitAlgorithm.MH)
        self.assertEqual(config.ppm_min, 0.5)
        self.assertEqual(config.ppm_max, 4.0)
        self.assertEqual(config.baseline_order, 1)
        self.assertEqual(config.internal_reference, "tCr")
        self.assertEqual(config.extra_args, "--ignore Gly")
        self.assertEqual(config.work_dir, "work/mrs")
        self.assertEqual(config.tmp_dir, ".tmp/mrs")
        self.assertTrue(config.validate())

    def test_thread_capping(self) -> None:
        """Test thread count is capped at 2."""
        config = MRSConfig(threads=8)
        self.assertEqual(config.threads, 2)
        config_low = MRSConfig(threads=0)
        self.assertEqual(config_low.threads, 1)

    def test_invalid_ppm_range(self) -> None:
        """Test validation error when ppm_min >= ppm_max."""
        config = MRSConfig(ppm_min=4.5, ppm_max=4.0)
        with self.assertRaises(ValueError) as context:
            config.validate()
        self.assertIn("must be less than ppm_max", str(context.exception))


class TestMRSPathResolver(BaseTest):
    """Test cases for MRSPathResolver."""

    def setUp(self) -> None:
        """Initialize resolver for tests."""
        self.resolver = MRSPathResolver(
            bids_dir="test_bids", derivatives_dir="test_derivatives"
        )

    def test_directory_properties(self) -> None:
        """Test basic directory property getters."""
        self.assertEqual(self.resolver.bids_dir, Path("test_bids"))
        self.assertEqual(self.resolver.derivatives_dir, Path("test_derivatives"))
        self.assertEqual(
            self.resolver.mrs_dir, Path("test_derivatives") / "mrs"
        )

    def test_get_subject_dir(self) -> None:
        """Test subject directory resolution with and without prefix."""
        self.assertEqual(
            self.resolver.get_subject_dir("19081001"),
            Path("test_derivatives/mrs/sub-19081001"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("sub-19081001"),
            Path("test_derivatives/mrs/sub-19081001"),
        )

    def test_resolve_svs_path(self) -> None:
        """Test resolving SVS file with existing file and fallback."""
        with self.create_temp_dir() as temp_dir:
            bids_dir = Path(temp_dir) / "bids"
            subject_mrs = bids_dir / "sub-01" / "mrs"
            subject_mrs.mkdir(parents=True, exist_ok=True)
            svs_file = subject_mrs / "sub-01_svs.nii.gz"
            svs_file.write_text("svs_mock", encoding="utf-8")

            res = MRSPathResolver(bids_dir=bids_dir)
            resolved = res.resolve_svs_path("sub-01")
            self.assertEqual(resolved, svs_file)

    def test_resolve_water_ref_path(self) -> None:
        """Test resolving water reference file."""
        with self.create_temp_dir() as temp_dir:
            bids_dir = Path(temp_dir) / "bids"
            subject_mrs = bids_dir / "sub-01" / "mrs"
            subject_mrs.mkdir(parents=True, exist_ok=True)
            ref_file = subject_mrs / "sub-01_ref.nii.gz"
            ref_file.write_text("ref_mock", encoding="utf-8")

            res = MRSPathResolver(bids_dir=bids_dir)
            self.assertEqual(res.resolve_water_ref_path("sub-01"), ref_file)

            # Test when no ref file exists
            self.assertIsNone(res.resolve_water_ref_path("sub-02"))

    def test_resolve_t1w_path(self) -> None:
        """Test resolving anatomical T1w file."""
        with self.create_temp_dir() as temp_dir:
            bids_dir = Path(temp_dir) / "bids"
            subject_anat = bids_dir / "sub-01" / "anat"
            subject_anat.mkdir(parents=True, exist_ok=True)
            t1_file = subject_anat / "sub-01_T1w.nii.gz"
            t1_file.write_text("t1_mock", encoding="utf-8")

            res = MRSPathResolver(bids_dir=bids_dir)
            self.assertEqual(res.resolve_t1w_path("sub-01"), t1_file)
            self.assertIsNone(res.resolve_t1w_path("sub-02"))

    def test_derivative_file_getters(self) -> None:
        """Test derivative output filename getters."""
        self.assertEqual(
            self.resolver.get_preproc_file("01"),
            Path("test_derivatives/mrs/sub-01/svs_processed.nii.gz"),
        )
        self.assertEqual(
            self.resolver.get_tissue_fractions_file("01"),
            Path("test_derivatives/mrs/sub-01/tissue_fractions.json"),
        )
        self.assertEqual(
            self.resolver.get_quantities_csv("01"),
            Path("test_derivatives/mrs/sub-01/quantities.csv"),
        )
        self.assertEqual(
            self.resolver.get_report_html("01"),
            Path("test_derivatives/mrs/sub-01.html"),
        )
        self.assertEqual(
            self.resolver.get_marker_file("01"),
            Path("test_derivatives/mrs/sub-01/.mrs_complete"),
        )


class TestMRSCommandBuilders(unittest.TestCase):
    """Test cases for FSL-MRS command builders."""

    def test_preproc_command_builder(self) -> None:
        """Test building fsl_mrs_preproc command."""
        builder = MRSPreprocCommandBuilder()
        cmd = builder.build_preproc_command(
            data_path=Path("raw_svs.nii.gz"),
            output_dir=Path("preproc_out"),
            ref_path=Path("ref_svs.nii.gz"),
            extra_args="--align",
        )
        self.assertEqual(cmd[0], "fsl_mrs_preproc")
        self.assertIn("--data", cmd)
        self.assertIn("raw_svs.nii.gz", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("preproc_out", cmd)
        self.assertIn("--reference", cmd)
        self.assertIn("ref_svs.nii.gz", cmd)
        self.assertIn("--align", cmd)

    def test_segment_command_builder(self) -> None:
        """Test building fsl_mrs_segment command."""
        builder = MRSSegmentCommandBuilder()
        cmd = builder.build_segment_command(
            t1_path=Path("T1w.nii.gz"),
            output_dir=Path("seg_out"),
            data_path=Path("svs.nii.gz"),
        )
        self.assertEqual(cmd[0], "fsl_mrs_segment")
        self.assertIn("--t1", cmd)
        self.assertIn("T1w.nii.gz", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("seg_out", cmd)
        self.assertIn("svs.nii.gz", cmd)

    def test_fit_command_builder(self) -> None:
        """Test building fsl_mrs spectral fitting command."""
        builder = MRSFitCommandBuilder()
        cmd = builder.build_fit_command(
            data_path=Path("proc_svs.nii.gz"),
            output_dir=Path("fit_out"),
            basis_path=Path("basis_dir"),
            ref_path=Path("proc_ref.nii.gz"),
            tissue_frac_path=Path("frac.json"),
            algo=MRSFitAlgorithm.NEWTON,
            ppm_min=0.2,
            ppm_max=4.2,
            baseline_order=2,
            internal_ref="Cr",
            extra_args="--ignore Mac",
        )
        self.assertEqual(cmd[0], "fsl_mrs")
        self.assertIn("--data", cmd)
        self.assertIn("proc_svs.nii.gz", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("fit_out", cmd)
        self.assertIn("--basis", cmd)
        self.assertIn("basis_dir", cmd)
        self.assertIn("--h2o", cmd)
        self.assertIn("proc_ref.nii.gz", cmd)
        self.assertIn("--tissue_frac", cmd)
        self.assertIn("frac.json", cmd)
        self.assertIn("--algo", cmd)
        self.assertIn("Newton", cmd)
        self.assertIn("--ppm", cmd)
        self.assertIn("0.2", cmd)
        self.assertIn("4.2", cmd)
        self.assertIn("--baseline_order", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--internal_ref", cmd)
        self.assertIn("Cr", cmd)
        self.assertIn("--ignore", cmd)

    def test_preproc_command_builder_ignores_empty_or_dot_ref(self) -> None:
        """Test that empty string or dot ref_path does not add --reference flag."""
        builder = MRSPreprocCommandBuilder()
        cmd_dot = builder.build_preproc_command(
            data_path=Path("raw_svs.nii.gz"),
            output_dir=Path("preproc_out"),
            ref_path=Path("."),
        )
        self.assertNotIn("--reference", cmd_dot)

        cmd_empty = builder.build_preproc_command(
            data_path=Path("raw_svs.nii.gz"),
            output_dir=Path("preproc_out"),
            ref_path=Path(""),
        )
        self.assertNotIn("--reference", cmd_empty)

    def test_fit_command_builder_ignores_empty_or_dot_options(self) -> None:
        """Test that empty or dot optional paths do not add CLI flags."""
        builder = MRSFitCommandBuilder()
        cmd = builder.build_fit_command(
            data_path=Path("proc_svs.nii.gz"),
            output_dir=Path("fit_out"),
            basis_path=Path("."),
            ref_path=Path(""),
            tissue_frac_path=Path("."),
        )
        self.assertNotIn("--basis", cmd)
        self.assertNotIn("--h2o", cmd)
        self.assertNotIn("--tissue_frac", cmd)



class TestMRSManagersAndReport(BaseTest):
    """Test cases for tissue fractions, quantities, and report generation."""

    def test_ensure_tissue_fractions(self) -> None:
        """Test generating tissue_fractions.json."""
        with self.create_temp_dir() as temp_dir:
            json_path = Path(temp_dir) / "sub-01" / "tissue_fractions.json"
            manager = MRSTissueSegmentationManager()
            result = manager.ensure_tissue_fractions(json_path, "sub-01")
            self.assertTrue(result.exists())
            data = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(data.get("subject"), "sub-01")
            self.assertIn("GM", data.get("tissue_fractions", {}))

    def test_ensure_quantities_csv(self) -> None:
        """Test generating quantities.csv."""
        with self.create_temp_dir() as temp_dir:
            csv_path = Path(temp_dir) / "sub-01" / "quantities.csv"
            manager = MRSQuantitiesManager()
            result = manager.ensure_quantities_csv(csv_path, "sub-01")
            self.assertTrue(result.exists())
            content = result.read_text(encoding="utf-8")
            self.assertIn("metabolite,concentration_mM,CRLB_percent,subject", content)
            self.assertIn("tNAA", content)
            self.assertIn("tCr", content)

    def test_generate_report_html(self) -> None:
        """Test generating standalone HTML report."""
        with self.create_temp_dir() as temp_dir:
            html_path = Path(temp_dir) / "sub-01.html"
            csv_path = Path(temp_dir) / "quantities.csv"
            manager = MRSQuantitiesManager()
            manager.ensure_quantities_csv(csv_path, "sub-01")

            generator = MRSReportGenerator()
            result = generator.generate_report(
                output_html=html_path,
                subject="sub-01",
                data_path=Path("sub-01_svs.nii.gz"),
                quantities_path=csv_path,
            )
            self.assertTrue(result.exists())
            html_text = result.read_text(encoding="utf-8")
            self.assertIn("Mindquad MRS Quality Control Report", html_text)
            self.assertIn("sub-01", html_text)
            self.assertIn("tNAA", html_text)


class TestMRSRunner(BaseTest):
    """Test cases for MRSRunner execution orchestrator."""

    def setUp(self) -> None:
        """Initialize runner."""
        self.runner = MRSRunner()

    def test_prepare_environment(self) -> None:
        """Test environment preparation with resource limits."""
        with self.create_temp_dir() as temp_dir:
            tmp_path = Path(temp_dir) / ".tmp"
            env = self.runner.prepare_environment(tmp_path, threads=2)
            self.assertEqual(env.get("TMPDIR"), str(tmp_path))
            self.assertEqual(env.get("OMP_NUM_THREADS"), "2")
            self.assertEqual(env.get("OPENBLAS_NUM_THREADS"), "2")
            self.assertEqual(env.get("MKL_NUM_THREADS"), "2")

    def test_run_creates_expected_derivatives(self) -> None:
        """Test full execution creates marker, report, csv, and fractions."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            data_file = temp_path / "sub-01_svs.nii.gz"
            data_file.write_text("mock_data", encoding="utf-8")
            out_dir = temp_path / "derivatives" / "mrs" / "sub-01"
            marker_file = out_dir / ".mrs_complete"
            report_file = temp_path / "derivatives" / "mrs" / "sub-01.html"
            csv_file = out_dir / "quantities.csv"

            status = self.runner.run(
                data_path=data_file,
                output_dir=out_dir,
                subject="sub-01",
                threads=2,
                tmp_dir=temp_path / ".tmp",
                marker_path=marker_file,
                report_path=report_file,
                summary_csv=csv_file,
            )
            self.assertEqual(status, 0)
            self.assertTrue(marker_file.exists())
            self.assertTrue(report_file.exists())
            self.assertTrue(csv_file.exists())
            self.assertTrue((out_dir / "tissue_fractions.json").exists())


class TestMRSApp(BaseTest):
    """Test cases for MRSApp CLI wrapper."""

    def test_resolve_optional_path(self) -> None:
        """Test resolve_optional_path helper method."""
        app = MRSApp()
        self.assertIsNone(app.resolve_optional_path(None))
        self.assertIsNone(app.resolve_optional_path(""))
        self.assertIsNone(app.resolve_optional_path("."))
        self.assertIsNone(app.resolve_optional_path("   "))
        self.assertEqual(
            app.resolve_optional_path("/path/to/file"),
            Path("/path/to/file"),
        )

    def test_create_parser(self) -> None:
        """Test parser arguments configuration."""
        app = MRSApp()
        parser = app.create_parser()
        args = parser.parse_args([
            "--data", "svs.nii.gz",
            "--output-dir", "derivatives/mrs/sub-01",
            "--subject", "sub-01",
            "--threads", "2",
        ])
        self.assertEqual(str(args.data), "svs.nii.gz")
        self.assertEqual(str(args.output_dir), "derivatives/mrs/sub-01")
        self.assertEqual(args.subject, "sub-01")
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.fit_algo, "Newton")
        self.assertEqual(args.ppm_min, 0.2)
        self.assertEqual(args.ppm_max, 4.2)
        self.assertEqual(args.baseline_order, 2)
        self.assertEqual(args.internal_ref, "Cr")

    def test_run_cli(self) -> None:
        """Test running CLI application with valid arguments."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            data_file = temp_path / "sub-01_svs.nii.gz"
            data_file.write_text("svs", encoding="utf-8")
            out_dir = temp_path / "derivatives" / "mrs" / "sub-01"
            marker = out_dir / ".mrs_complete"

            app = MRSApp()
            exit_code = app.run([
                "--data", str(data_file),
                "--output-dir", str(out_dir),
                "--subject", "sub-01",
                "--reference", "",
                "--t1", "",
                "--basis", "",
                "--marker", str(marker),
                "--tmp-dir", str(temp_path / ".tmp"),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()

