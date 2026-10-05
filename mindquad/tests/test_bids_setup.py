"""Unit tests for BIDS initialization and organization scripts."""

import json
import tempfile
import unittest
from pathlib import Path

from mindquad.workflow.scripts.bids_init import BIDSInitializer, BIDSMetadata
from mindquad.workflow.scripts.bids_organizer import (
    BIDSFilenameBuilder,
    BIDSModality,
    BIDSOrganizer,
    DICOMSeriesClassifier,
    ScanClassification,
)
from mindquad.workflow.scripts.cohort import StudyCohort


class TestBIDSMetadata(unittest.TestCase):
    """Test cases for BIDSMetadata class."""

    def test_metadata_to_dict(self) -> None:
        """Test conversion of BIDSMetadata to dictionary."""
        metadata = BIDSMetadata(
            name="Test Dataset",
            bids_version="1.9.0",
            dataset_type="raw",
            license_str="MIT",
            authors=["Alice", "Bob"],
        )
        meta_dict = metadata.to_dict()
        self.assertEqual(meta_dict["Name"], "Test Dataset")
        self.assertEqual(meta_dict["BIDSVersion"], "1.9.0")
        self.assertEqual(meta_dict["DatasetType"], "raw")
        self.assertEqual(meta_dict["License"], "MIT")
        self.assertEqual(meta_dict["Authors"], ["Alice", "Bob"])


class BaseBIDSTest(unittest.TestCase):
    """Base test class ensuring temporary files use project-local .tmp/."""

    @classmethod
    def setUpClass(cls) -> None:
        """Ensure project-local .tmp directory exists."""
        cls.project_tmp_dir = Path(".tmp")
        cls.project_tmp_dir.mkdir(parents=True, exist_ok=True)

    def create_temp_dir(self) -> tempfile.TemporaryDirectory:
        """Create a temporary directory inside project-local .tmp/ or fallback."""
        try:
            return tempfile.TemporaryDirectory(dir=str(self.project_tmp_dir))
        except (OSError, PermissionError):
            return tempfile.TemporaryDirectory()


class TestBIDSInitializer(BaseBIDSTest):
    """Test cases for BIDSInitializer class."""

    def test_initialize_bids_root(self) -> None:
        """Test creating BIDS root structure and metadata files."""
        with self.create_temp_dir() as temp_dir:
            bids_root = Path(temp_dir) / "bids"
            initializer = BIDSInitializer()
            desc_path = initializer.initialize_bids_root(bids_root)

            self.assertTrue(desc_path.exists())
            self.assertTrue((bids_root / "README").exists())
            self.assertTrue((bids_root / ".bidsignore").exists())

            with open(desc_path, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
                self.assertIn("Name", data)
                self.assertIn("BIDSVersion", data)


class TestDICOMSeriesClassifier(BaseBIDSTest):
    """Test cases for DICOMSeriesClassifier class."""

    def setUp(self) -> None:
        """Set up test classifier."""
        self.classifier = DICOMSeriesClassifier()

    def test_classify_t1w(self) -> None:
        """Test classification of T1-weighted MPRAGE scan."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_mprage_01.json"
            json_file.write_text(
                json.dumps({"SeriesDescription": "t1_mprage_sag_p2_iso"}),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_mprage_01"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.ANAT)
            self.assertEqual(classification.suffix, "T1w")

    def test_classify_dwi(self) -> None:
        """Test classification of Diffusion Weighted scan."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_diff_02.json"
            json_file.write_text(
                json.dumps({"SeriesDescription": "ep2d_diff_dir-AP"}),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_diff_02"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.DWI)
            self.assertEqual(classification.suffix, "dwi")
            self.assertEqual(classification.entities.get("dir"), "AP")

    def test_classify_func_bold(self) -> None:
        """Test classification of functional BOLD scan."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_bold_03.json"
            json_file.write_text(
                json.dumps({"SeriesDescription": "ep2d_bold_task-rest"}),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_bold_03"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.FUNC)
            self.assertEqual(classification.suffix, "bold")
            self.assertEqual(classification.entities.get("task"), "rest")

    def test_classify_mrs_svs(self) -> None:
        """Test classification of MRS SVS sequence."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_svs_04.json"
            json_file.write_text(
                json.dumps({"SeriesDescription": "svs_press_w"}),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_svs_04"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.MRS)
            self.assertEqual(classification.suffix, "svs")

    def test_classify_fmap_bold_pepolar(self) -> None:
        """Test classification of fMRI spin-echo pepolar fieldmap."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_fmap_bold.json"
            json_file.write_text(
                json.dumps({
                    "SeriesDescription": "fMRI_SE_b0 pepolar=1",
                    "PhaseEncodingDirection": "j-",
                }),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_fmap_bold"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.FMAP)
            self.assertEqual(classification.suffix, "epi")
            self.assertEqual(classification.entities.get("acq"), "bold")
            self.assertEqual(classification.entities.get("dir"), "AP")

    def test_classify_fmap_dwi_pepolar(self) -> None:
        """Test classification of DTI pepolar fieldmap."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_fmap_dwi.json"
            json_file.write_text(
                json.dumps({
                    "SeriesDescription": "DTI_MB3_2x2x2mm_pepolar",
                    "PhaseEncodingDirection": "j",
                }),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_fmap_dwi"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.FMAP)
            self.assertEqual(classification.suffix, "epi")
            self.assertEqual(classification.entities.get("acq"), "dwi")
            self.assertEqual(classification.entities.get("dir"), "PA")

    def test_classify_task_name_from_am_task(self) -> None:
        """Test task name extraction from AM_task_FINAL description."""
        with self.create_temp_dir() as temp_dir:
            json_file = Path(temp_dir) / "test_task_am.json"
            json_file.write_text(
                json.dumps({"SeriesDescription": "AM_task_FINAL"}),
                encoding="utf-8",
            )
            classification = self.classifier.classify_series(
                json_file, "test_task_am"
            )
            self.assertIsNotNone(classification)
            self.assertEqual(classification.modality, BIDSModality.FUNC)
            self.assertEqual(classification.suffix, "bold")
            self.assertEqual(classification.entities.get("task"), "am")


class TestBIDSFilenameBuilder(unittest.TestCase):
    """Test cases for BIDSFilenameBuilder."""

    def test_build_bids_stem(self) -> None:
        """Test BIDS stem generation with strict canonical entity ordering."""
        builder = BIDSFilenameBuilder()
        classification = ScanClassification(
            modality=BIDSModality.FUNC,
            suffix="bold",
            entities={"dir": "AP", "run": "01", "task": "rest"},
        )
        stem = builder.build_bids_stem("19081001", classification)
        self.assertEqual(
            stem, "sub-19081001_task-rest_dir-AP_run-01_bold"
        )


class TestBIDSOrganizer(BaseBIDSTest):
    """Test cases for end-to-end BIDSOrganizer."""

    def test_organize_subject_single_runs(self) -> None:
        """Test organizing mock converted files into BIDS."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # Create mock T1w converted files
            t1_meta = {"SeriesDescription": "t1_mprage", "SeriesNumber": 1}
            (input_dir / "001_t1_mprage.json").write_text(
                json.dumps(t1_meta), encoding="utf-8"
            )
            (input_dir / "001_t1_mprage.nii.gz").write_text(
                "dummy nifti", encoding="utf-8"
            )

            # Create mock DWI converted files with bval/bvec
            dwi_meta = {"SeriesDescription": "dwi_dir-ap", "SeriesNumber": 2}
            (input_dir / "002_dwi_AP.json").write_text(
                json.dumps(dwi_meta), encoding="utf-8"
            )
            (input_dir / "002_dwi_AP.nii.gz").write_text(
                "dummy dwi", encoding="utf-8"
            )
            (input_dir / "002_dwi_AP.bval").write_text(
                "0 1000", encoding="utf-8"
            )
            (input_dir / "002_dwi_AP.bvec").write_text(
                "0 0\n0 1\n0 0", encoding="utf-8"
            )

            organizer = BIDSOrganizer()
            transferred = organizer.organize_subject(
                input_dir, bids_dir, "19081001"
            )

            self.assertEqual(transferred, 6)
            sub_anat = bids_dir / "sub-19081001" / "anat"
            sub_dwi = bids_dir / "sub-19081001" / "dwi"
            self.assertTrue(
                (sub_anat / "sub-19081001_T1w.nii.gz").exists()
            )
            self.assertTrue(
                (sub_anat / "sub-19081001_T1w.json").exists()
            )
            self.assertTrue(
                (sub_dwi / "sub-19081001_dir-AP_dwi.nii.gz").exists()
            )
            self.assertTrue(
                (sub_dwi / "sub-19081001_dir-AP_dwi.bval").exists()
            )
            self.assertTrue(
                (sub_dwi / "sub-19081001_dir-AP_dwi.bvec").exists()
            )

    def test_organize_subject_multiple_runs(self) -> None:
        """Test multiple runs of same sequence receive run-01, run-02."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # Create mock first BOLD run (SeriesNumber: 3)
            r1_meta = {
                "SeriesDescription": "ep2d_bold_task-rest",
                "SeriesNumber": 3,
            }
            (input_dir / "003_bold_rest.json").write_text(
                json.dumps(r1_meta), encoding="utf-8"
            )
            (input_dir / "003_bold_rest.nii.gz").write_text(
                "bold 1", encoding="utf-8"
            )

            # Create mock second BOLD run (SeriesNumber: 7)
            r2_meta = {
                "SeriesDescription": "ep2d_bold_task-rest",
                "SeriesNumber": 7,
            }
            (input_dir / "007_bold_rest.json").write_text(
                json.dumps(r2_meta), encoding="utf-8"
            )
            (input_dir / "007_bold_rest.nii.gz").write_text(
                "bold 2", encoding="utf-8"
            )

            organizer = BIDSOrganizer()
            transferred = organizer.organize_subject(
                input_dir, bids_dir, "19081001"
            )

            self.assertEqual(transferred, 4)
            func_dir = bids_dir / "sub-19081001" / "func"
            run1_nii = func_dir / "sub-19081001_task-rest_run-01_bold.nii.gz"
            run1_json = func_dir / "sub-19081001_task-rest_run-01_bold.json"
            run2_nii = func_dir / "sub-19081001_task-rest_run-02_bold.nii.gz"
            run2_json = func_dir / "sub-19081001_task-rest_run-02_bold.json"

            self.assertTrue(run1_nii.exists(), "Run 1 NIfTI must exist")
            self.assertTrue(run1_json.exists(), "Run 1 JSON must exist")
            self.assertTrue(run2_nii.exists(), "Run 2 NIfTI must exist")
            self.assertTrue(run2_json.exists(), "Run 2 JSON must exist")

    def test_organize_subject_with_fieldmaps_intended_for(self) -> None:
        """Test organizing subject with BOLD and DWI fieldmaps injects proper IntendedFor."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # 1. BOLD functional scan: AM_task_FINAL
            bold_meta = {
                "SeriesDescription": "AM_task_FINAL",
                "SeriesNumber": 6,
            }
            (input_dir / "006_AM_task.json").write_text(
                json.dumps(bold_meta), encoding="utf-8"
            )
            (input_dir / "006_AM_task.nii.gz").write_text(
                "mock bold", encoding="utf-8"
            )

            # 2. DWI scan: DTI_MB3_2x2x2mm
            dwi_meta = {
                "SeriesDescription": "DTI_MB3_2x2x2mm",
                "PhaseEncodingDirection": "j-",
                "SeriesNumber": 8,
            }
            (input_dir / "008_dti.json").write_text(
                json.dumps(dwi_meta), encoding="utf-8"
            )
            (input_dir / "008_dti.nii.gz").write_text(
                "mock dwi", encoding="utf-8"
            )

            # 3. fMRI spin-echo fieldmap: fMRI_SE_b0 pepolar=1
            fmap_bold_meta = {
                "SeriesDescription": "fMRI_SE_b0 pepolar=1",
                "PhaseEncodingDirection": "j-",
                "SeriesNumber": 7,
            }
            (input_dir / "007_fmap_bold.json").write_text(
                json.dumps(fmap_bold_meta), encoding="utf-8"
            )
            (input_dir / "007_fmap_bold.nii.gz").write_text(
                "mock fmap bold", encoding="utf-8"
            )

            # 4. DWI spin-echo fieldmap: DTI_MB3_2x2x2mm_pepolar
            fmap_dwi_meta = {
                "SeriesDescription": "DTI_MB3_2x2x2mm_pepolar",
                "PhaseEncodingDirection": "j",
                "SeriesNumber": 9,
            }
            (input_dir / "009_fmap_dwi.json").write_text(
                json.dumps(fmap_dwi_meta), encoding="utf-8"
            )
            (input_dir / "009_fmap_dwi.nii.gz").write_text(
                "mock fmap dwi", encoding="utf-8"
            )

            organizer = BIDSOrganizer()
            transferred = organizer.organize_subject(
                input_dir, bids_dir, "VNS001", session_label="01"
            )

            # Verify transferred count (4 pairs of .nii.gz and .json = 8 files)
            self.assertEqual(transferred, 8)

            ses_dir = bids_dir / "sub-VNS001" / "ses-01"
            fmap_dir = ses_dir / "fmap"

            # Check BOLD fieldmap exists with acq-bold
            bold_fmap_json = (
                fmap_dir / "sub-VNS001_ses-01_acq-bold_dir-AP_epi.json"
            )
            self.assertTrue(bold_fmap_json.exists())
            with open(bold_fmap_json, "r", encoding="utf-8") as fh:
                bold_fmap_data = json.load(fh)
            self.assertEqual(
                bold_fmap_data.get("IntendedFor"),
                "ses-01/func/sub-VNS001_ses-01_task-am_bold.nii.gz",
            )

            # Check DWI fieldmap exists with acq-dwi
            dwi_fmap_json = (
                fmap_dir / "sub-VNS001_ses-01_acq-dwi_dir-PA_epi.json"
            )
            self.assertTrue(dwi_fmap_json.exists())
            with open(dwi_fmap_json, "r", encoding="utf-8") as fh:
                dwi_fmap_data = json.load(fh)
            self.assertEqual(
                dwi_fmap_data.get("IntendedFor"),
                "ses-01/dwi/sub-VNS001_ses-01_dir-AP_dwi.nii.gz",
            )

    def test_organize_subject_single_session_fieldmap_intended_for(self) -> None:
        """Test organizing single-session dataset with DWI fieldmap (like pipeline_testing)."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # DWI scan: DTI_MB3_2x2x2mm
            dwi_meta = {
                "SeriesDescription": "DTI_MB3_2x2x2mm",
                "PhaseEncodingDirection": "j-",
                "SeriesNumber": 8,
            }
            (input_dir / "008_dti.json").write_text(
                json.dumps(dwi_meta), encoding="utf-8"
            )
            (input_dir / "008_dti.nii.gz").write_text(
                "mock dwi", encoding="utf-8"
            )

            # DWI Fieldmap: DTI_MB3_2x2x2mm_pepolar
            fmap_dwi_meta = {
                "SeriesDescription": "DTI_MB3_2x2x2mm_pepolar",
                "PhaseEncodingDirection": "j",
                "SeriesNumber": 7,
            }
            (input_dir / "007_fmap_dwi.json").write_text(
                json.dumps(fmap_dwi_meta), encoding="utf-8"
            )
            (input_dir / "007_fmap_dwi.nii.gz").write_text(
                "mock fmap dwi", encoding="utf-8"
            )

            organizer = BIDSOrganizer()
            transferred = organizer.organize_subject(
                input_dir, bids_dir, "19081001"
            )

            self.assertEqual(transferred, 4)

            sub_dir = bids_dir / "sub-19081001"
            fmap_json = (
                sub_dir / "fmap" / "sub-19081001_acq-dwi_dir-PA_epi.json"
            )
            self.assertTrue(fmap_json.exists())
            with open(fmap_json, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(
                data.get("IntendedFor"),
                "dwi/sub-19081001_dir-AP_dwi.nii.gz",
            )




class TestStudyCohort(BaseBIDSTest):
    """Test cases for StudyCohort BIDS detection and subject resolution."""

    def test_detect_bids_from_dataset_description(self) -> None:
        """Test BIDS detection when dataset_description.json is present."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "bids_data"
            raw_dir.mkdir()
            (raw_dir / "dataset_description.json").write_text("{}", encoding="utf-8")
            (raw_dir / "sub-01").mkdir()

            cohort = StudyCohort({"raw_data_dir": str(raw_dir)})
            self.assertTrue(cohort.is_bids)

    def test_detect_bids_from_modalities(self) -> None:
        """Test BIDS detection when modality subdirectories exist."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "bids_data"
            sub_dir = raw_dir / "sub-01"
            (sub_dir / "anat").mkdir(parents=True)

            cohort = StudyCohort({"raw_data_dir": str(raw_dir)})
            self.assertTrue(cohort.is_bids)

    def test_detect_bids_with_sessions(self) -> None:
        """Test BIDS detection with session subdirectories."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "bids_data"
            (raw_dir / "sub-01" / "ses-1" / "func").mkdir(parents=True)

            cohort = StudyCohort({"raw_data_dir": str(raw_dir)})
            self.assertTrue(cohort.is_bids)

    def test_detect_raw_dicom_not_bids(self) -> None:
        """Test non-BIDS raw DICOM folder returns is_bids=False."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "raw_dicoms"
            (raw_dir / "patient_A" / "series_1").mkdir(parents=True)

            cohort = StudyCohort({"raw_data_dir": str(raw_dir)})
            self.assertFalse(cohort.is_bids)

    def test_explicit_is_bids_config_override(self) -> None:
        """Test explicit is_bids configuration overrides auto-detection."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "raw_data"
            raw_dir.mkdir()

            cohort_true = StudyCohort({"raw_data_dir": str(raw_dir), "is_bids": True})
            self.assertTrue(cohort_true.is_bids)

            (raw_dir / "dataset_description.json").write_text("{}", encoding="utf-8")
            cohort_false = StudyCohort({"raw_data_dir": str(raw_dir), "is_bids": False})
            self.assertFalse(cohort_false.is_bids)

    def test_subjects_discovery_bids_filters_non_subjects(self) -> None:
        """Test subjects discovery in BIDS mode only includes sub-* directories."""
        with self.create_temp_dir() as temp_dir:
            raw_dir = Path(temp_dir) / "bids_dataset"
            (raw_dir / "sub-01" / "anat").mkdir(parents=True)
            (raw_dir / "sub-02" / "anat").mkdir(parents=True)
            (raw_dir / "derivatives").mkdir()
            (raw_dir / "code").mkdir()
            (raw_dir / "dataset_description.json").write_text("{}", encoding="utf-8")

            cohort = StudyCohort({"raw_data_dir": str(raw_dir)})
            self.assertEqual(cohort.subjects, ["sub-01", "sub-02"])
            self.assertEqual(cohort.bids_subjects, ["01", "02"])

    def test_get_bids_subject_label_strips_sub_prefix(self) -> None:
        """Test stripping sub- prefix from subject directory names."""
        cohort = StudyCohort({})
        self.assertEqual(cohort.get_bids_subject_label("sub-1"), "1")
        self.assertEqual(cohort.get_bids_subject_label("sub-01"), "01")
        self.assertEqual(cohort.get_bids_subject_label("1"), "1")
        self.assertEqual(cohort.get_bids_subject_label("sub-control_01"), "control01")

    def test_get_bids_subject_label_with_custom_mapping(self) -> None:
        """Test subject mapping respects configured dictionary and strips sub-."""
        cohort = StudyCohort({"subject_mapping": {"raw_sub": "sub-10"}})
        self.assertEqual(cohort.get_bids_subject_label("raw_sub"), "10")

    def test_get_bids_subject_label_with_none_mapping(self) -> None:
        """Test subject mapping safely handles None when left blank in YAML."""
        cohort = StudyCohort({"subject_mapping": None, "subjects": None})
        self.assertEqual(cohort.get_bids_subject_label("sub-01"), "01")
        self.assertEqual(cohort.get_bids_subject_label("VNS001_A"), "VNS001A")


if __name__ == "__main__":
    unittest.main()
