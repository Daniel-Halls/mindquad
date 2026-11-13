"""Unit tests for FSL reorientation module (reorient_fsl)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np

from mindquad.workflow.scripts.reorient_fsl import (
    FSLReorientApp,
    FSLReorienter,
)


class TestFSLReorienter(unittest.TestCase):
    """Test suite for FSLReorienter core functionality."""

    def setUp(self) -> None:
        """Set up test fixtures for FSLReorienter tests.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self.reorienter = FSLReorienter()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up test fixtures.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self.temp_dir.cleanup()

    def test_compute_target_transform_lps_to_las(self) -> None:
        """Test compute_target_transform for standard QSIPrep LPS to LAS.

        Returns
        --------
            None: NoneType
                None return value.
        """
        affine = np.array([
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, -2.0, 0.0, 120.0],
            [0.0, 0.0, 2.0, -60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        shape = (10, 10, 10, 5)
        _, rot_mat = self.reorienter.compute_target_transform(affine, shape)

        expected_rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(np.abs(rot_mat), np.abs(expected_rot))
        self.assertAlmostEqual(rot_mat[1, 1], -1.0)

    def test_reorient_image(self) -> None:
        """Test reorient_image converts orientation axcodes to ('L', 'A', 'S').

        Returns
        --------
            None: NoneType
                None return value.
        """
        affine = np.array([
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, -2.0, 0.0, 120.0],
            [0.0, 0.0, 2.0, -60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        data = np.zeros((10, 10, 10, 3), dtype=np.float32)
        img = nib.Nifti1Image(data, affine)
        self.assertEqual(nib.aff2axcodes(img.affine), ("L", "P", "S"))

        reoriented, rot_mat = self.reorienter.reorient_image(img)
        self.assertEqual(nib.aff2axcodes(reoriented.affine), ("L", "A", "S"))
        self.assertEqual(rot_mat.shape, (3, 3))

    def test_load_and_validate_gradients(self) -> None:
        """Test loading bvals and bvecs in (3, N) format.

        Returns
        --------
            None: NoneType
                None return value.
        """
        bval_file = self.test_dir / "test.bval"
        bvec_file = self.test_dir / "test.bvec"
        np.savetxt(str(bval_file), np.array([0, 1000, 1000]), fmt="%g")
        np.savetxt(
            str(bvec_file),
            np.array([
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ]),
            fmt="%.4f",
        )

        bvals, bvecs = self.reorienter.load_and_validate_gradients(
            bval_file, bvec_file
        )
        self.assertEqual(bvals.shape, (3,))
        self.assertEqual(bvecs.shape, (3, 3))

    def test_load_and_validate_gradients_transposed(self) -> None:
        """Test loading (N, 3) bvecs transposes to (3, N).

        Returns
        --------
            None: NoneType
                None return value.
        """
        bval_file = self.test_dir / "test.bval"
        bvec_file = self.test_dir / "test.bvec"
        np.savetxt(str(bval_file), np.array([0, 1000, 1000, 1000]), fmt="%g")
        # Write (4, 3) matrix
        np.savetxt(
            str(bvec_file),
            np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
            fmt="%.4f",
        )

        bvals, bvecs = self.reorienter.load_and_validate_gradients(
            bval_file, bvec_file
        )
        self.assertEqual(bvals.shape, (4,))
        self.assertEqual(bvecs.shape, (3, 4))

    def test_load_and_validate_gradients_mismatch(self) -> None:
        """Test ValueError raised when count of bvals does not match bvecs.

        Returns
        --------
            None: NoneType
                None return value.
        """
        bval_file = self.test_dir / "test.bval"
        bvec_file = self.test_dir / "test.bvec"
        np.savetxt(str(bval_file), np.array([0, 1000]), fmt="%g")
        np.savetxt(str(bvec_file), np.zeros((3, 3)), fmt="%.4f")

        with self.assertRaises(ValueError):
            self.reorienter.load_and_validate_gradients(bval_file, bvec_file)

    def test_rotate_bvecs(self) -> None:
        """Test gradient direction rotation and unit normalization.

        Returns
        --------
            None: NoneType
                None return value.
        """
        bvecs = np.array([
            [0.0, 0.0, 2.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        rot_mat = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rotated = self.reorienter.rotate_bvecs(bvecs, rot_mat)

        self.assertAlmostEqual(rotated[0, 2], 1.0)
        self.assertAlmostEqual(rotated[1, 1], -1.0)
        self.assertAlmostEqual(rotated[0, 0], 0.0)

    def test_save_gradients(self) -> None:
        """Test saving gradients creates formatted text files.

        Returns
        --------
            None: NoneType
                None return value.
        """
        bvals = np.array([0, 1000])
        bvecs = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        out_bval = self.test_dir / "out.bval"
        out_bvec = self.test_dir / "out.bvec"

        self.reorienter.save_gradients(bvals, bvecs, out_bval, out_bvec)
        self.assertTrue(out_bval.is_file())
        self.assertTrue(out_bvec.is_file())

        loaded_bval = np.loadtxt(str(out_bval))
        loaded_bvec = np.loadtxt(str(out_bvec))
        np.testing.assert_allclose(loaded_bval, bvals)
        np.testing.assert_allclose(loaded_bvec, bvecs)

    def test_reorient_files(self) -> None:
        """Test reorient_files end-to-end on synthetic data.

        Returns
        --------
            None: NoneType
                None return value.
        """
        affine = np.array([
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, -2.0, 0.0, 120.0],
            [0.0, 0.0, 2.0, -60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        data = np.ones((8, 8, 8, 2), dtype=np.float32)
        mask_data = np.ones((8, 8, 8), dtype=np.uint8)

        dwi_path = self.test_dir / "sub-01_desc-preproc_dwi.nii.gz"
        mask_path = self.test_dir / "sub-01_desc-brain_mask.nii.gz"
        bval_path = self.test_dir / "sub-01_desc-preproc_dwi.bval"
        bvec_path = self.test_dir / "sub-01_desc-preproc_dwi.bvec"

        nib.save(nib.Nifti1Image(data, affine), str(dwi_path))
        nib.save(nib.Nifti1Image(mask_data, affine), str(mask_path))
        np.savetxt(str(bval_path), np.array([0, 1000]), fmt="%g")
        np.savetxt(
            str(bvec_path),
            np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            fmt="%.4f",
        )

        out_dwi = self.test_dir / "out" / "data.nii.gz"
        out_bval = self.test_dir / "out" / "bvals"
        out_bvec = self.test_dir / "out" / "bvecs"
        out_mask = self.test_dir / "out" / "nodif_brain_mask.nii.gz"

        res = self.reorienter.reorient_files(
            dwi_path=dwi_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            mask_path=mask_path,
            out_dwi=out_dwi,
            out_bval=out_bval,
            out_bvec=out_bvec,
            out_mask=out_mask,
        )

        self.assertEqual(res["dwi"], out_dwi)
        self.assertEqual(res["mask"], out_mask)
        self.assertTrue(out_dwi.is_file())
        self.assertTrue(out_bval.is_file())
        self.assertTrue(out_bvec.is_file())
        self.assertTrue(out_mask.is_file())

        # Original files must remain intact
        self.assertTrue(dwi_path.is_file())
        self.assertTrue(bval_path.is_file())
        self.assertTrue(bvec_path.is_file())
        self.assertTrue(mask_path.is_file())

        reoriented_dwi = nib.load(str(out_dwi))
        self.assertEqual(
            nib.aff2axcodes(reoriented_dwi.affine), ("L", "A", "S")
        )

    def test_reorient_subject(self) -> None:
        """Test reorient_subject discovers outputs and produces FSL data.

        Returns
        --------
            None: NoneType
                None return value.
        """
        subj_dir = self.test_dir / "sub-01" / "dwi"
        subj_dir.mkdir(parents=True, exist_ok=True)

        affine = np.array([
            [-2.0, 0.0, 0.0, 90.0],
            [0.0, -2.0, 0.0, 120.0],
            [0.0, 0.0, 2.0, -60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        dwi_path = subj_dir / "sub-01_space-T1w_desc-preproc_dwi.nii.gz"
        mask_path = subj_dir / "sub-01_space-T1w_desc-brain_mask.nii.gz"
        bval_path = subj_dir / "sub-01_space-T1w_desc-preproc_dwi.bval"
        bvec_path = subj_dir / "sub-01_space-T1w_desc-preproc_dwi.bvec"

        nib.save(
            nib.Nifti1Image(np.ones((6, 6, 6, 2)), affine), str(dwi_path)
        )
        nib.save(
            nib.Nifti1Image(np.ones((6, 6, 6)), affine), str(mask_path)
        )
        np.savetxt(str(bval_path), np.array([0, 1000]), fmt="%g")
        np.savetxt(
            str(bvec_path),
            np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            fmt="%.4f",
        )

        results = self.reorienter.reorient_subject(self.test_dir, "01")
        self.assertEqual(len(results), 1)

        # Check BIDS-convention files exist alongside original
        bids_fsl_dwi = (
            subj_dir / "sub-01_space-T1w_desc-fslstd_dwi.nii.gz"
        )
        self.assertTrue(bids_fsl_dwi.is_file())
        self.assertTrue(dwi_path.is_file())

        # Check standard FSL files exist in dwi/fsl/
        fsl_data = subj_dir / "fsl" / "data.nii.gz"
        self.assertTrue(fsl_data.is_file())

    def test_reorient_subject_multi_run(self) -> None:
        """Test reorient_subject handles multiple runs with distinct dirs.

        Returns
        --------
            None: NoneType
                None return value.
        """
        subj_dir = self.test_dir / "sub-02" / "dwi"
        subj_dir.mkdir(parents=True)

        for run_id in ["run-01", "run-02"]:
            prefix = f"sub-02_{run_id}_desc-preproc_dwi"
            dwi_path = subj_dir / f"{prefix}.nii.gz"
            bval_path = subj_dir / f"{prefix}.bval"
            bvec_path = subj_dir / f"{prefix}.bvec"

            affine = np.diag([-2.0, -2.0, 2.0, 1.0])
            data = np.zeros((4, 4, 4, 2), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, affine), str(dwi_path))
            np.savetxt(str(bval_path), np.array([0, 1000]), fmt="%g")
            np.savetxt(
                str(bvec_path),
                np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
                fmt="%.4f",
            )

        results = self.reorienter.reorient_subject(self.test_dir, "02")
        self.assertEqual(len(results), 2)
        fsl_dir_1 = results[0]["fsl_dir"]
        fsl_dir_2 = results[1]["fsl_dir"]
        self.assertNotEqual(fsl_dir_1, fsl_dir_2)
        self.assertTrue((fsl_dir_1 / "data.nii.gz").is_file())
        self.assertTrue((fsl_dir_2 / "data.nii.gz").is_file())


class TestFSLReorientApp(unittest.TestCase):
    """Test suite for FSLReorientApp CLI."""

    def setUp(self) -> None:
        """Set up CLI test fixtures.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self.app = FSLReorientApp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up CLI test fixtures.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self.temp_dir.cleanup()

    def test_run_missing_arguments(self) -> None:
        """Test run returns exit code 2 when required arguments are missing.

        Returns
        --------
            None: NoneType
                None return value.
        """
        ret = self.app.run([])
        self.assertEqual(ret, 2)

    @patch.object(FSLReorienter, "reorient_files")
    def test_run_file_mode(self, mock_reorient_files: MagicMock) -> None:
        """Test run executes file mode when dwi, bval, and bvec are provided.

        Parameters
        ------------
        mock_reorient_files: MagicMock
            Mocked reorient_files method.

        Returns
        --------
            None: NoneType
                None return value.
        """
        dwi = self.test_dir / "dwi.nii.gz"
        bval = self.test_dir / "dwi.bval"
        bvec = self.test_dir / "dwi.bvec"
        dwi.touch()
        bval.touch()
        bvec.touch()

        ret = self.app.run([
            "--dwi", str(dwi),
            "--bval", str(bval),
            "--bvec", str(bvec),
            "--outdir", str(self.test_dir / "fsl"),
        ])
        self.assertEqual(ret, 0)
        mock_reorient_files.assert_called_once()


if __name__ == "__main__":
    unittest.main()
