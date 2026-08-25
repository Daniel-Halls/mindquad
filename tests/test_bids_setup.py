"""Unit tests for BIDS initialization and organization scripts."""

import json
import tempfile
import unittest
from pathlib import Path

from workflow.scripts.bids_init import BIDSInitializer, BIDSMetadata
from workflow.scripts.bids_organizer import (
    BIDSFilenameBuilder,
    BIDSModality,
    BIDSOrganizer,
    DICOMSeriesClassifier,
    ScanClassification,
)


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
        """Create a temporary directory strictly inside project-local .tmp/."""
        return tempfile.TemporaryDirectory(dir=str(self.project_tmp_dir))


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
        # Task must precede dir, which must precede run
        self.assertEqual(
            stem, "sub-19081001_task-rest_dir-AP_run-01_bold"
        )


class TestBIDSOrganizer(BaseBIDSTest):
    """Test cases for end-to-end BIDSOrganizer."""

    def test_organize_subject_single_runs(self) -> None:
        """Test organizing mock converted files with single instances into BIDS."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # Create mock T1w converted files
            (input_dir / "001_t1_mprage.json").write_text(
                json.dumps({"SeriesDescription": "t1_mprage", "SeriesNumber": 1}),
                encoding="utf-8",
            )
            (input_dir / "001_t1_mprage.nii.gz").write_text("dummy nifti", encoding="utf-8")

            # Create mock DWI converted files with bval/bvec
            (input_dir / "002_dwi_AP.json").write_text(
                json.dumps({"SeriesDescription": "dwi_dir-ap", "SeriesNumber": 2}),
                encoding="utf-8",
            )
            (input_dir / "002_dwi_AP.nii.gz").write_text("dummy dwi", encoding="utf-8")
            (input_dir / "002_dwi_AP.bval").write_text("0 1000", encoding="utf-8")
            (input_dir / "002_dwi_AP.bvec").write_text("0 0\n0 1\n0 0", encoding="utf-8")

            organizer = BIDSOrganizer()
            transferred = organizer.organize_subject(
                input_dir, bids_dir, "19081001"
            )

            self.assertEqual(transferred, 6)
            self.assertTrue(
                (bids_dir / "sub-19081001" / "anat" / "sub-19081001_T1w.nii.gz").exists()
            )
            self.assertTrue(
                (bids_dir / "sub-19081001" / "anat" / "sub-19081001_T1w.json").exists()
            )
            self.assertTrue(
                (bids_dir / "sub-19081001" / "dwi" / "sub-19081001_dir-AP_dwi.nii.gz").exists()
            )
            self.assertTrue(
                (bids_dir / "sub-19081001" / "dwi" / "sub-19081001_dir-AP_dwi.bval").exists()
            )
            self.assertTrue(
                (bids_dir / "sub-19081001" / "dwi" / "sub-19081001_dir-AP_dwi.bvec").exists()
            )

    def test_organize_subject_multiple_runs(self) -> None:
        """Test that multiple runs of same sequence all receive run-01, run-02."""
        with self.create_temp_dir() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "dcm2niix_out"
            bids_dir = temp_path / "bids"
            input_dir.mkdir(parents=True)
            bids_dir.mkdir(parents=True)

            # Create mock first BOLD run (SeriesNumber: 3)
            (input_dir / "003_bold_rest.json").write_text(
                json.dumps({"SeriesDescription": "ep2d_bold_task-rest", "SeriesNumber": 3}),
                encoding="utf-8",
            )
            (input_dir / "003_bold_rest.nii.gz").write_text("bold 1", encoding="utf-8")

            # Create mock second BOLD run (SeriesNumber: 7)
            (input_dir / "007_bold_rest.json").write_text(
                json.dumps({"SeriesDescription": "ep2d_bold_task-rest", "SeriesNumber": 7}),
                encoding="utf-8",
            )
            (input_dir / "007_bold_rest.nii.gz").write_text("bold 2", encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
