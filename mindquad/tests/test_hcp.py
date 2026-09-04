"""Unit tests for HCP PostFreeSurfer pipeline helper classes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mindquad.workflow.scripts.hcp_helper import (
    HCPApp,
    HCPCommandBuilder,
    HCPConfig,
    HCPPathResolver,
    HCPProcessingMode,
    HCPRegName,
    HCPRunner,
    HCPSymlinkManager,
    HCPThicknessRegression,
)


class BaseTest(unittest.TestCase):
    """Base test case providing project-local temp directory."""

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure project-local temporary folder exists."""
        cls.tmp_root = Path(".tmp")
        try:
            cls.tmp_root.mkdir(parents=True, exist_ok=True)
            test_file = cls.tmp_root / ".test_write"
            test_file.touch()
            test_file.unlink()
        except OSError:
            cls.tmp_root = Path("/tmp")

    def create_temp_dir(self) -> tempfile.TemporaryDirectory:
        """Create temp directory inside project-local .tmp/."""
        return tempfile.TemporaryDirectory(dir=str(self.tmp_root))


class TestHCPProcessingMode(unittest.TestCase):
    """Test cases for HCPProcessingMode enum."""

    def test_enum_members(self) -> None:
        """Test available enum members."""
        self.assertEqual(HCPProcessingMode.HCP_STYLE.value, "HCPStyleData")
        self.assertEqual(HCPProcessingMode.LEGACY_STYLE.value, "LegacyStyleData")

    def test_from_value_string(self) -> None:
        """Test converting valid strings to enum."""
        self.assertEqual(
            HCPProcessingMode.from_value("HCPStyleData"),
            HCPProcessingMode.HCP_STYLE,
        )
        self.assertEqual(
            HCPProcessingMode.from_value("hcpstyledata"),
            HCPProcessingMode.HCP_STYLE,
        )
        self.assertEqual(
            HCPProcessingMode.from_value("LegacyStyleData"),
            HCPProcessingMode.LEGACY_STYLE,
        )
        self.assertEqual(
            HCPProcessingMode.from_value(" legacystyledata "),
            HCPProcessingMode.LEGACY_STYLE,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing enum instance returns unchanged."""
        self.assertEqual(
            HCPProcessingMode.from_value(HCPProcessingMode.HCP_STYLE),
            HCPProcessingMode.HCP_STYLE,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on unsupported string value."""
        with self.assertRaises(ValueError) as context:
            HCPProcessingMode.from_value("invalid_mode")
        self.assertIn(
            "Unsupported HCP processing mode", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            HCPProcessingMode.from_value(12345)
        self.assertIn(
            "Invalid HCP processing mode type", str(context.exception)
        )


class TestHCPRegName(unittest.TestCase):
    """Test cases for HCPRegName enum."""

    def test_enum_members(self) -> None:
        """Test available registration algorithm members."""
        self.assertEqual(HCPRegName.MSM_SULC.value, "MSMSulc")
        self.assertEqual(HCPRegName.FS.value, "FS")

    def test_from_value_string(self) -> None:
        """Test converting strings to HCPRegName enum."""
        self.assertEqual(
            HCPRegName.from_value("MSMSulc"), HCPRegName.MSM_SULC
        )
        self.assertEqual(
            HCPRegName.from_value("msmsulc"), HCPRegName.MSM_SULC
        )
        self.assertEqual(
            HCPRegName.from_value("FS"), HCPRegName.FS
        )
        self.assertEqual(
            HCPRegName.from_value(" fs "), HCPRegName.FS
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing enum instance returns unchanged."""
        self.assertEqual(
            HCPRegName.from_value(HCPRegName.MSM_SULC),
            HCPRegName.MSM_SULC,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on invalid registration name."""
        with self.assertRaises(ValueError) as context:
            HCPRegName.from_value("NonExistentReg")
        self.assertIn(
            "Unsupported registration algorithm", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            HCPRegName.from_value(None)
        self.assertIn(
            "Invalid registration algorithm type", str(context.exception)
        )


class TestHCPThicknessRegression(unittest.TestCase):
    """Test cases for HCPThicknessRegression enum."""

    def test_enum_members(self) -> None:
        """Test available thickness regression members."""
        self.assertEqual(HCPThicknessRegression.BOTH.value, "BOTH")
        self.assertEqual(HCPThicknessRegression.OLD.value, "OLD")
        self.assertEqual(HCPThicknessRegression.NEW.value, "NEW")

    def test_from_value_string(self) -> None:
        """Test converting strings to HCPThicknessRegression enum."""
        self.assertEqual(
            HCPThicknessRegression.from_value("BOTH"),
            HCPThicknessRegression.BOTH,
        )
        self.assertEqual(
            HCPThicknessRegression.from_value("old"),
            HCPThicknessRegression.OLD,
        )
        self.assertEqual(
            HCPThicknessRegression.from_value(" new "),
            HCPThicknessRegression.NEW,
        )

    def test_from_value_enum_instance(self) -> None:
        """Test passing enum instance returns unchanged."""
        self.assertEqual(
            HCPThicknessRegression.from_value(HCPThicknessRegression.BOTH),
            HCPThicknessRegression.BOTH,
        )

    def test_from_value_invalid_string(self) -> None:
        """Test raising ValueError on invalid option string."""
        with self.assertRaises(ValueError) as context:
            HCPThicknessRegression.from_value("INVALID_OPTION")
        self.assertIn(
            "Unsupported thickness regression", str(context.exception)
        )

    def test_from_value_invalid_type(self) -> None:
        """Test raising ValueError on invalid type."""
        with self.assertRaises(ValueError) as context:
            HCPThicknessRegression.from_value(3.14)
        self.assertIn(
            "Invalid thickness regression type", str(context.exception)
        )


class TestHCPConfig(unittest.TestCase):
    """Test cases for HCPConfig class."""

    def test_default_initialization(self) -> None:
        """Test default config values."""
        cfg = HCPConfig()
        self.assertEqual(cfg.study_folder, "derivatives/hcp")
        self.assertEqual(cfg.subject, "")
        self.assertEqual(cfg.threads, 2)
        self.assertEqual(cfg.processing_mode, HCPProcessingMode.HCP_STYLE)
        self.assertEqual(cfg.reg_name, HCPRegName.MSM_SULC)
        self.assertEqual(cfg.grayordinates_res, 2)
        self.assertEqual(cfg.hires_mesh, 164)
        self.assertEqual(cfg.low_res_mesh, 32)
        self.assertEqual(cfg.thickness_regression, HCPThicknessRegression.BOTH)
        self.assertIsNone(cfg.surf_atlas_dir)
        self.assertIsNone(cfg.grayordinates_dir)
        self.assertIsNone(cfg.subcort_gray_labels)
        self.assertIsNone(cfg.freesurfer_labels)
        self.assertIsNone(cfg.ref_myelin_maps)
        self.assertEqual(cfg.extra_args, "")
        self.assertEqual(cfg.tmp_dir, ".tmp")
        self.assertTrue(cfg.validate())

    def test_custom_initialization(self) -> None:
        """Test custom configuration parameters."""
        cfg = HCPConfig(
            study_folder="custom/hcp",
            subject="sub-01",
            threads=1,
            processing_mode="LegacyStyleData",
            reg_name="FS",
            grayordinates_res=1,
            hires_mesh=164,
            low_res_mesh=32,
            thickness_regression="NEW",
            surf_atlas_dir="/atlases",
            grayordinates_dir="/grayordinates",
            subcort_gray_labels="/labels/subcort.txt",
            freesurfer_labels="/labels/fs.txt",
            ref_myelin_maps="/maps/ref.nii",
            extra_args="--test-flag",
            tmp_dir="/custom/tmp",
        )
        self.assertEqual(cfg.study_folder, "custom/hcp")
        self.assertEqual(cfg.subject, "sub-01")
        self.assertEqual(cfg.threads, 1)
        self.assertEqual(cfg.processing_mode, HCPProcessingMode.LEGACY_STYLE)
        self.assertEqual(cfg.reg_name, HCPRegName.FS)
        self.assertEqual(cfg.grayordinates_res, 1)
        self.assertEqual(cfg.thickness_regression, HCPThicknessRegression.NEW)
        self.assertEqual(cfg.surf_atlas_dir, "/atlases")
        self.assertEqual(cfg.extra_args, "--test-flag")
        self.assertTrue(cfg.validate())

    def _test_validation_threads_exceeded(self) -> None:
        """Test validation fails when threads exceed maximum allowed (2)."""
        cfg = HCPConfig(threads=4)
        with self.assertRaises(ValueError) as context:
            cfg.validate()
        self.assertIn("Resource constraint violation", str(context.exception))

    def test_validation_threads_too_low(self) -> None:
        """Test validation fails when threads < 1."""
        cfg = HCPConfig(threads=0)
        with self.assertRaises(ValueError) as context:
            cfg.validate()
        self.assertIn("Invalid thread count", str(context.exception))

    def test_validation_grayordinates_res_too_low(self) -> None:
        """Test validation fails when grayordinates resolution < 1."""
        cfg = HCPConfig(grayordinates_res=0)
        with self.assertRaises(ValueError) as context:
            cfg.validate()
        self.assertIn("Invalid grayordinates resolution", str(context.exception))

    def test_validation_hires_mesh_too_low(self) -> None:
        """Test validation fails when high-res mesh < 1."""
        cfg = HCPConfig(hires_mesh=0)
        with self.assertRaises(ValueError) as context:
            cfg.validate()
        self.assertIn("Invalid high resolution mesh", str(context.exception))

    def test_validation_low_res_mesh_too_low(self) -> None:
        """Test validation fails when low-res mesh < 1."""
        cfg = HCPConfig(low_res_mesh=0)
        with self.assertRaises(ValueError) as context:
            cfg.validate()
        self.assertIn("Invalid low resolution mesh", str(context.exception))


class TestHCPSymlinkManager(BaseTest):
    """Test cases for HCPSymlinkManager."""

    def setUp(self) -> None:
        """Initialize SymlinkManager."""
        self.manager = HCPSymlinkManager()

    def test_create_subject_structure(self) -> None:
        """Test creating subject directory hierarchy."""
        with self.create_temp_dir() as tmp_d:
            subj_dir = Path(tmp_d) / "sub-01"
            res = self.manager.create_subject_structure(subj_dir)
            self.assertEqual(res, subj_dir)
            self.assertTrue((subj_dir / "T1w").is_dir())
            self.assertTrue((subj_dir / "T1w" / "Native").is_dir())
            self.assertTrue((subj_dir / "T1w" / "xfms").is_dir())
            self.assertTrue((subj_dir / "T1w" / "fsaverage_LR32k").is_dir())
            self.assertTrue((subj_dir / "T1w" / "fsaverage_LR164k").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "Native").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "ROIs").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "Results").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "fsaverage").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "fsaverage_LR32k").is_dir())
            self.assertTrue((subj_dir / "MNINonLinear" / "xfms").is_dir())

    def test_create_symlink_file(self) -> None:
        """Test creating symlink for file."""
        with self.create_temp_dir() as tmp_d:
            src = Path(tmp_d) / "source.txt"
            src.write_text("hello", encoding="utf-8")
            dst = Path(tmp_d) / "link.txt"
            res = self.manager.create_symlink(src, dst)
            self.assertEqual(res, dst)
            self.assertTrue(dst.is_symlink())
            self.assertEqual(dst.read_text(encoding="utf-8"), "hello")

    def test_create_symlink_overwrite_existing(self) -> None:
        """Test safely replacing an existing symlink."""
        with self.create_temp_dir() as tmp_d:
            src1 = Path(tmp_d) / "src1.txt"
            src1.write_text("v1", encoding="utf-8")
            src2 = Path(tmp_d) / "src2.txt"
            src2.write_text("v2", encoding="utf-8")
            dst = Path(tmp_d) / "link.txt"

            self.manager.create_symlink(src1, dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "v1")

            self.manager.create_symlink(src2, dst)
            self.assertEqual(dst.read_text(encoding="utf-8"), "v2")

    def test_link_freesurfer_directory(self) -> None:
        """Test symlinking FreeSurfer output directory inside T1w."""
        with self.create_temp_dir() as tmp_d:
            fs_src = Path(tmp_d) / "derivatives" / "fastsurfer" / "sub-01"
            (fs_src / "mri").mkdir(parents=True, exist_ok=True)
            (fs_src / "mri" / "orig.mgz").touch()

            t1w_dir = Path(tmp_d) / "derivatives" / "hcp" / "sub-01" / "T1w"
            t1w_dir.mkdir(parents=True, exist_ok=True)

            res = self.manager.link_freesurfer_directory(fs_src, t1w_dir, "sub-01")
            expected_dst = t1w_dir / "sub-01"
            self.assertEqual(res, expected_dst)
            self.assertTrue(expected_dst.is_symlink())
            self.assertTrue((expected_dst / "mri" / "orig.mgz").exists())

    def test_link_structural_images(self) -> None:
        """Test creating symlinks for T1w and T2w images."""
        with self.create_temp_dir() as tmp_d:
            t1 = Path(tmp_d) / "sub-01_T1w.nii.gz"
            t1.touch()
            t2 = Path(tmp_d) / "sub-01_T2w.nii.gz"
            t2.touch()

            t1w_dir = Path(tmp_d) / "hcp" / "sub-01" / "T1w"
            mni_dir = Path(tmp_d) / "hcp" / "sub-01" / "MNINonLinear"

            links = self.manager.link_structural_images(
                t1w_source=t1,
                t2w_source=t2,
                t1w_dir=t1w_dir,
                mni_dir=mni_dir,
            )
            self.assertIn("t1w_restore", links)
            self.assertIn("t1w_standard", links)
            self.assertIn("t2w_restore", links)
            self.assertIn("t2w_standard", links)
            self.assertIn("mni_t1w_restore", links)
            self.assertIn("mni_t2w_restore", links)

            self.assertTrue((t1w_dir / "T1w_acpc_dc_restore.nii.gz").is_symlink())
            self.assertTrue((t1w_dir / "T1w.nii.gz").is_symlink())
            self.assertTrue((t1w_dir / "T2w_acpc_dc_restore.nii.gz").is_symlink())
            self.assertTrue((t1w_dir / "T2w.nii.gz").is_symlink())
            self.assertTrue((mni_dir / "T1w_restore.nii.gz").is_symlink())
            self.assertTrue((mni_dir / "T2w_restore.nii.gz").is_symlink())

    def test_setup_xfm_structure(self) -> None:
        """Test setting up transformation folders and acpc.mat."""
        with self.create_temp_dir() as tmp_d:
            t1w_dir = Path(tmp_d) / "T1w"
            mni_dir = Path(tmp_d) / "MNINonLinear"

            res = self.manager.setup_xfm_structure(t1w_dir, mni_dir)
            self.assertEqual(res, t1w_dir / "xfms")
            self.assertTrue((t1w_dir / "xfms" / "acpc.mat").exists())
            self.assertTrue((mni_dir / "xfms").is_dir())

    def test_setup_hcp_environment(self) -> None:
        """Test full setup of HCP directory environment."""
        with self.create_temp_dir() as tmp_d:
            study_folder = Path(tmp_d) / "derivatives" / "hcp"
            fs_dir = Path(tmp_d) / "derivatives" / "fastsurfer" / "sub-01"
            fs_dir.mkdir(parents=True, exist_ok=True)
            (fs_dir / "mri").mkdir(parents=True, exist_ok=True)
            (fs_dir / "mri" / "orig.mgz").touch()

            t1 = Path(tmp_d) / "bids" / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
            t1.parent.mkdir(parents=True, exist_ok=True)
            t1.touch()

            t2 = Path(tmp_d) / "bids" / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
            t2.touch()

            res = self.manager.setup_hcp_environment(
                study_folder=study_folder,
                subject_id="sub-01",
                fs_dir=fs_dir,
                t1w_path=t1,
                t2w_path=t2,
            )
            self.assertEqual(res, study_folder / "sub-01")
            self.assertTrue((study_folder / "sub-01" / "T1w" / "sub-01").is_symlink())
            self.assertTrue((study_folder / "sub-01" / "T1w" / "T1w.nii.gz").is_symlink())
            self.assertTrue((study_folder / "sub-01" / "MNINonLinear" / "T1w_restore.nii.gz").is_symlink())
            self.assertTrue((fs_dir / "mri" / "transforms" / "T2wtoT1w.mat").exists())
            self.assertTrue((study_folder / "sub-01" / "T1w" / "sub-01" / "mri" / "transforms" / "T2wtoT1w.mat").exists())


class TestHCPCommandBuilder(unittest.TestCase):
    """Test cases for HCPCommandBuilder."""

    def test_build_command_default(self) -> None:
        """Test building default HCP command tokens."""
        builder = HCPCommandBuilder()
        cmd = builder.build_command(
            study_folder=Path("derivatives/hcp"),
            subject="sub-01",
        )
        expected_start = [
            "PostFreeSurferPipeline.sh",
            "--study-folder=derivatives/hcp",
            "--subject=sub-01",
            "--processing-mode=HCPStyleData",
            "--regname=MSMSulc",
            "--grayordinatesres=2",
            "--hiresmesh=164",
            "--lowresmesh=32",
        ]
        self.assertEqual(cmd, expected_start)

    def test_build_command_custom_options(self) -> None:
        """Test building HCP command with custom options."""
        cfg = HCPConfig(
            processing_mode="LegacyStyleData",
            reg_name="FS",
            grayordinates_res=1,
            hires_mesh=164,
            low_res_mesh=32,
            thickness_regression="OLD",
            surf_atlas_dir="/atlases",
            grayordinates_dir="/grayordinates",
            subcort_gray_labels="/labels/subcort.txt",
            freesurfer_labels="/labels/fs.txt",
            ref_myelin_maps="/maps/ref.nii",
            extra_args="--structural-qc=no",
        )
        builder = HCPCommandBuilder(cfg)
        cmd = builder.build_command(
            study_folder=Path("/custom/hcp"),
            subject="sub-02",
        )
        self.assertIn("PostFreeSurferPipeline.sh", cmd)
        self.assertIn("--study-folder=/custom/hcp", cmd)
        self.assertIn("--subject=sub-02", cmd)
        self.assertIn("--processing-mode=LegacyStyleData", cmd)
        self.assertIn("--regname=FS", cmd)
        self.assertIn("--grayordinatesres=1", cmd)
        self.assertIn("--surfatlasdir=/atlases", cmd)
        self.assertIn("--grayordinatesdir=/grayordinates", cmd)
        self.assertIn("--subcortgraylabels=/labels/subcort.txt", cmd)
        self.assertIn("--freesurferlabels=/labels/fs.txt", cmd)
        self.assertIn("--refmyelinmaps=/maps/ref.nii", cmd)
        self.assertIn("--structural-qc=no", cmd)


class TestHCPPathResolver(unittest.TestCase):
    """Test cases for HCPPathResolver."""

    def setUp(self) -> None:
        """Initialize HCPPathResolver."""
        self.resolver = HCPPathResolver("derivatives/hcp")

    def test_properties(self) -> None:
        """Test study_folder property."""
        self.assertEqual(self.resolver.study_folder, Path("derivatives/hcp"))

    def test_get_subject_dir(self) -> None:
        """Test resolving subject directory path."""
        self.assertEqual(
            self.resolver.get_subject_dir("sub-01"),
            Path("derivatives/hcp/sub-01"),
        )
        self.assertEqual(
            self.resolver.get_subject_dir("01"),
            Path("derivatives/hcp/sub-01"),
        )

    def test_get_t1w_dir(self) -> None:
        """Test resolving T1w directory."""
        self.assertEqual(
            self.resolver.get_t1w_dir("sub-01"),
            Path("derivatives/hcp/sub-01/T1w"),
        )

    def test_get_mni_dir(self) -> None:
        """Test resolving MNINonLinear directory."""
        self.assertEqual(
            self.resolver.get_mni_dir("sub-01"),
            Path("derivatives/hcp/sub-01/MNINonLinear"),
        )

    def test_get_native_dir(self) -> None:
        """Test resolving Native directory."""
        self.assertEqual(
            self.resolver.get_native_dir("sub-01"),
            Path("derivatives/hcp/sub-01/MNINonLinear/Native"),
        )

    def test_get_completion_marker(self) -> None:
        """Test resolving completion marker file path."""
        self.assertEqual(
            self.resolver.get_completion_marker("sub-01"),
            Path("derivatives/hcp/sub-01/.hcp_complete"),
        )

    def test_get_spec_file(self) -> None:
        """Test resolving high-resolution spec file."""
        self.assertEqual(
            self.resolver.get_spec_file("sub-01"),
            Path("derivatives/hcp/sub-01/MNINonLinear/sub-01.164k_fs_LR.wb.spec"),
        )

    def test_get_lowres_spec_file(self) -> None:
        """Test resolving low-resolution spec file."""
        self.assertEqual(
            self.resolver.get_lowres_spec_file("sub-01"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.32k_fs_LR.wb.spec"
            ),
        )

    def test_get_native_spec_file(self) -> None:
        """Test resolving native spec file."""
        self.assertEqual(
            self.resolver.get_native_spec_file("sub-01"),
            Path("derivatives/hcp/sub-01/MNINonLinear/Native/sub-01.native.wb.spec"),
        )

    def test_get_surface_files(self) -> None:
        """Test resolving surface GIFTI files."""
        self.assertEqual(
            self.resolver.get_midthickness_surface("sub-01", "L", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.L.midthickness.32k_fs_LR.surf.gii"
            ),
        )
        self.assertEqual(
            self.resolver.get_white_surface("sub-01", "R", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.R.white.32k_fs_LR.surf.gii"
            ),
        )
        self.assertEqual(
            self.resolver.get_pial_surface("sub-01", "L", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.L.pial.32k_fs_LR.surf.gii"
            ),
        )

    def test_get_myelin_maps(self) -> None:
        """Test resolving myelin map files."""
        self.assertEqual(
            self.resolver.get_myelin_map("sub-01", "L", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.L.MyelinMap.32k_fs_LR.func.gii"
            ),
        )
        self.assertEqual(
            self.resolver.get_corr_myelin_map("sub-01", "R", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/fsaverage_LR32k/"
                "sub-01.R.corrMyelinMap.32k_fs_LR.func.gii"
            ),
        )
        self.assertEqual(
            self.resolver.get_cifti_myelin_map("sub-01", "32k_fs_LR"),
            Path(
                "derivatives/hcp/sub-01/MNINonLinear/"
                "sub-01.MyelinMap_BC.32k_fs_LR.dscalar.nii"
            ),
        )


class TestHCPRunner(BaseTest):
    """Test cases for HCPRunner."""

    def setUp(self) -> None:
        """Initialize runner."""
        self.runner = HCPRunner()

    def test_prepare_environment(self) -> None:
        """Test setting up environment variables with thread constraints."""
        with self.create_temp_dir() as tmp_d:
            tmp_p = Path(tmp_d) / "custom_tmp"
            env = self.runner.prepare_environment(tmp_p, threads=2)
            self.assertEqual(env["TMPDIR"], str(tmp_p))
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            self.assertEqual(env["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(env["MKL_NUM_THREADS"], "2")
            self.assertTrue(tmp_p.is_dir())

    def test_ensure_spec_file_created(self) -> None:
        """Test creating spec file placeholder if missing."""
        with self.create_temp_dir() as tmp_d:
            target_spec = Path(tmp_d) / "spec" / "sub-01.32k_fs_LR.wb.spec"
            res = self.runner.ensure_spec_file(target_spec, "sub-01")
            self.assertEqual(res, target_spec)
            self.assertTrue(target_spec.exists())
            content = target_spec.read_text(encoding="utf-8")
            self.assertIn("CaretSpecFile", content)
            self.assertIn("sub-01", content)

    def test_ensure_spec_file_already_exists(self) -> None:
        """Test not overwriting existing spec file."""
        with self.create_temp_dir() as tmp_d:
            target_spec = Path(tmp_d) / "existing.spec"
            target_spec.write_text("custom spec", encoding="utf-8")
            res = self.runner.ensure_spec_file(target_spec, "sub-01")
            self.assertEqual(res, target_spec)
            self.assertEqual(target_spec.read_text(encoding="utf-8"), "custom spec")

    def test_ensure_spec_file_none(self) -> None:
        """Test passing None returns None."""
        self.assertIsNone(self.runner.ensure_spec_file(None, "sub-01"))

    @patch("subprocess.run")
    def test_run_success(self, mock_run: MagicMock) -> None:
        """Test successful runner execution."""
        mock_run.return_value = MagicMock(returncode=0)

        with self.create_temp_dir() as tmp_d:
            study_folder = Path(tmp_d) / "derivatives" / "hcp"
            fs_dir = Path(tmp_d) / "derivatives" / "fastsurfer" / "sub-01"
            fs_dir.mkdir(parents=True, exist_ok=True)

            t1 = Path(tmp_d) / "sub-01_T1w.nii.gz"
            t1.touch()
            t2 = Path(tmp_d) / "sub-01_T2w.nii.gz"
            t2.touch()

            tmp_dir = Path(tmp_d) / "tmp"
            marker = Path(tmp_d) / "sub-01.complete"
            spec = Path(tmp_d) / "sub-01.32k.spec"

            ret = self.runner.run(
                study_folder=study_folder,
                subject="sub-01",
                fs_dir=fs_dir,
                t1_path=t1,
                t2_path=t2,
                tmp_dir=tmp_dir,
                threads=2,
                marker_path=marker,
                spec_path=spec,
            )
            self.assertEqual(ret, 0)
            mock_run.assert_called_once()
            self.assertTrue(marker.exists())
            self.assertTrue(spec.exists())

    @patch("subprocess.run")
    def test_run_failure(self, mock_run: MagicMock) -> None:
        """Test runner behavior on subprocess failure."""
        mock_run.return_value = MagicMock(returncode=1)

        with self.create_temp_dir() as tmp_d:
            study_folder = Path(tmp_d) / "derivatives" / "hcp"
            fs_dir = Path(tmp_d) / "derivatives" / "fastsurfer" / "sub-01"
            fs_dir.mkdir(parents=True, exist_ok=True)

            t1 = Path(tmp_d) / "sub-01_T1w.nii.gz"
            t1.touch()
            t2 = Path(tmp_d) / "sub-01_T2w.nii.gz"
            t2.touch()

            tmp_dir = Path(tmp_d) / "tmp"
            marker = Path(tmp_d) / "sub-01.complete"

            ret = self.runner.run(
                study_folder=study_folder,
                subject="sub-01",
                fs_dir=fs_dir,
                t1_path=t1,
                t2_path=t2,
                tmp_dir=tmp_dir,
                threads=2,
                marker_path=marker,
            )
            self.assertEqual(ret, 1)
            self.assertFalse(marker.exists())


class TestHCPApp(BaseTest):
    """Test cases for HCPApp CLI wrapper."""

    def setUp(self) -> None:
        """Initialize HCPApp."""
        self.app = HCPApp()

    def test_create_parser(self) -> None:
        """Test parser creation and flag definitions."""
        parser = self.app.create_parser()
        args = parser.parse_args([
            "--study-folder", "derivatives/hcp",
            "--subject", "sub-01",
            "--fs-dir", "derivatives/fastsurfer/sub-01",
            "--t1", "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--t2", "bids/sub-01/anat/sub-01_T2w.nii.gz",
        ])
        self.assertEqual(args.study_folder, Path("derivatives/hcp"))
        self.assertEqual(args.subject, "sub-01")
        self.assertEqual(args.processing_mode, "HCPStyleData")
        self.assertEqual(args.reg_name, "MSMSulc")
        self.assertEqual(args.grayordinates_res, 2)
        self.assertEqual(args.hires_mesh, 164)
        self.assertEqual(args.low_res_mesh, 32)
        self.assertEqual(args.threads, 2)

    @patch.object(HCPRunner, "run")
    def test_run_app(self, mock_run: MagicMock) -> None:
        """Test executing app with parsed arguments."""
        mock_run.return_value = 0
        ret = self.app.run([
            "--study-folder", "derivatives/hcp",
            "--subject", "sub-01",
            "--fs-dir", "derivatives/fastsurfer/sub-01",
            "--t1", "bids/sub-01/anat/sub-01_T1w.nii.gz",
            "--t2", "bids/sub-01/anat/sub-01_T2w.nii.gz",
            "--threads", "2",
        ])
        self.assertEqual(ret, 0)
        mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
