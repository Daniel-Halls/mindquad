"""Reorient QSIPrep DWI outputs and gradients to FSL convention."""

import argparse
import logging
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


class FSLReorienter:
    """Reorient QSIPrep outputs to FSL radiological convention (LAS+)."""

    def __init__(self) -> None:
        """Initialize FSLReorienter instance.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

    def compute_target_transform(
        self,
        affine: np.ndarray,
        shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute orientation transform and rotation matrix for LAS+.

        Parameters
        ------------
        affine: np.ndarray
            Image 4x4 affine matrix.
        shape: Tuple[int, ...]
            Image spatial dimension sizes.

        Returns
        --------
            Tuple[np.ndarray, np.ndarray]: tuple of arrays
                Tuple of orientation transform array and 3x3 rotation matrix.
        """
        orig_ornt = nib.io_orientation(affine)
        targ_ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))
        transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
        aff_new_to_old = nib.orientations.inv_ornt_aff(transform, shape)
        rot_old_to_new = np.linalg.inv(aff_new_to_old[:3, :3])
        return (transform, rot_old_to_new)

    def reorient_image(
        self,
        img: nib.Nifti1Image,
    ) -> Tuple[nib.Nifti1Image, np.ndarray]:
        """Reorient NIfTI image to FSL radiological orientation (LAS+).

        Parameters
        ------------
        img: nib.Nifti1Image
            Input NIfTI image object.

        Returns
        --------
            Tuple[nib.Nifti1Image, np.ndarray]: tuple object
                Tuple of reoriented NIfTI image and 3x3 voxel rotation matrix.
        """
        transform, rot_old_to_new = self.compute_target_transform(
            img.affine, img.shape
        )
        reoriented_img = img.as_reoriented(transform)
        return (reoriented_img, rot_old_to_new)

    def load_and_validate_gradients(
        self,
        bval_path: Path,
        bvec_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load bval and bvec files and validate dimension compatibility.

        Parameters
        ------------
        bval_path: Path
            Path to the input bval file.
        bvec_path: Path
            Path to the input bvec file.

        Returns
        --------
            Tuple[np.ndarray, np.ndarray]: tuple of arrays
                Tuple of flattened bval array and (3, N) bvec array.

        Raises
        -------
            ValueError: If shapes are invalid or counts do not match.
        """
        bvals = np.loadtxt(str(bval_path)).flatten()
        bvecs = np.loadtxt(str(bvec_path))

        if bvecs.ndim == 1:
            bvecs = bvecs[:, np.newaxis]

        if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
            bvecs = bvecs.T
        elif bvecs.shape[0] != 3:
            raise ValueError(
                f"Unexpected bvecs shape: {bvecs.shape}. "
                "Expected (3, N) or (N, 3)."
            )

        if bvecs.shape[1] != bvals.shape[0]:
            raise ValueError(
                f"Mismatch between number of bvals ({bvals.shape[0]}) "
                f"and bvecs ({bvecs.shape[1]})."
            )

        return (bvals, bvecs)

    def rotate_bvecs(
        self,
        bvecs: np.ndarray,
        rot_matrix: np.ndarray,
    ) -> np.ndarray:
        """Rotate bvecs using rotation matrix and normalize non-zero vectors.

        Parameters
        ------------
        bvecs: np.ndarray
            (3, N) diffusion gradient direction vectors.
        rot_matrix: np.ndarray
            3x3 rotation/permutation matrix.

        Returns
        --------
            np.ndarray: numpy ndarray
                Rotated and unit-normalized (3, N) bvec array.
        """
        rotated_bvecs = rot_matrix @ bvecs
        norms = np.linalg.norm(rotated_bvecs, axis=0)
        non_zero = norms > 1e-6
        rotated_bvecs[:, non_zero] /= norms[non_zero]
        return rotated_bvecs

    def save_gradients(
        self,
        bvals: np.ndarray,
        bvecs: np.ndarray,
        out_bval: Path,
        out_bvec: Path,
    ) -> Tuple[Path, Path]:
        """Save rotated bvals and bvecs in FSL format with trailing newline.

        Parameters
        ------------
        bvals: np.ndarray
            1D array of b-values.
        bvecs: np.ndarray
            (3, N) array of gradient directions.
        out_bval: Path
            Destination path for bval output file.
        out_bvec: Path
            Destination path for bvec output file.

        Returns
        --------
            Tuple[Path, Path]: tuple of paths
                Tuple of saved bval and bvec file paths.
        """
        out_bval.parent.mkdir(parents=True, exist_ok=True)
        out_bvec.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(out_bval),
            bvals[None, :],
            fmt="%g",
            delimiter=" ",
            newline="\n",
        )
        np.savetxt(
            str(out_bvec), bvecs, fmt="%.8f", delimiter=" ", newline="\n"
        )
        return (out_bval, out_bvec)

    def reorient_files(
        self,
        dwi_path: Path,
        bval_path: Path,
        bvec_path: Path,
        mask_path: Optional[Path],
        out_dwi: Path,
        out_bval: Path,
        out_bvec: Path,
        out_mask: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Reorient DWI volume, mask, and gradients to destination paths.

        Parameters
        ------------
        dwi_path: Path
            Path to input DWI NIfTI file.
        bval_path: Path
            Path to input bval file.
        bvec_path: Path
            Path to input bvec file.
        mask_path: Optional[Path]
            Optional path to input brain mask NIfTI file.
        out_dwi: Path
            Destination path for reoriented DWI file.
        out_bval: Path
            Destination path for reoriented bval file.
        out_bvec: Path
            Destination path for reoriented bvec file.
        out_mask: Optional[Path]
            Optional destination path for reoriented brain mask.
            Defaults to None.

        Returns
        --------
            Dict[str, Path]: dictionary object
                Dictionary mapping output keys to resolved output file paths.
        """
        self._logger.info("Loading DWI from %s", dwi_path)
        dwi_img = nib.load(str(dwi_path))
        dwi_las, rot_mat = self.reorient_image(dwi_img)

        out_dwi.parent.mkdir(parents=True, exist_ok=True)
        nib.save(dwi_las, str(out_dwi))
        self._logger.info("Saved reoriented DWI to %s", out_dwi)

        bvals, bvecs = self.load_and_validate_gradients(bval_path, bvec_path)
        rotated_bvecs = self.rotate_bvecs(bvecs, rot_mat)
        self.save_gradients(bvals, rotated_bvecs, out_bval, out_bvec)
        self._logger.info(
            "Saved reoriented bvals/bvecs to %s, %s", out_bval, out_bvec
        )

        result_dict: Dict[str, Path] = {
            "dwi": out_dwi,
            "bval": out_bval,
            "bvec": out_bvec,
        }

        if (
            mask_path is not None
            and mask_path.is_file()
            and out_mask is not None
        ):
            self._logger.info("Loading brain mask from %s", mask_path)
            mask_img = nib.load(str(mask_path))
            mask_las, _ = self.reorient_image(mask_img)
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            nib.save(mask_las, str(out_mask))
            self._logger.info("Saved reoriented brain mask to %s", out_mask)
            result_dict["mask"] = out_mask

        return result_dict

    def discover_subject_dwi_sets(
        self,
        subject_dir: Path,
    ) -> List[Dict[str, Path]]:
        """Discover preprocessed DWI file sets for a given subject directory.

        Parameters
        ------------
        subject_dir: Path
            Path to subject derivatives directory.

        Returns
        --------
            List[Dict[str, Path]]: list of dicts
                List of discovered DWI sets with dwi, bval, bvec, mask paths.
        """
        discovered_sets: List[Dict[str, Path]] = []
        dwi_dirs: List[Path] = []

        top_dwi = subject_dir / "dwi"
        if top_dwi.is_dir():
            dwi_dirs.append(top_dwi)

        for session_dir in sorted(subject_dir.glob("ses-*")):
            ses_dwi = session_dir / "dwi"
            if ses_dwi.is_dir():
                dwi_dirs.append(ses_dwi)

        for dwi_folder in dwi_dirs:
            for preproc_dwi in sorted(
                dwi_folder.glob("*desc-preproc_dwi.nii*")
            ):
                base_stem = (
                    preproc_dwi.name.replace(".nii.gz", "").replace(".nii", "")
                )
                base_prefix = base_stem.replace("_desc-preproc_dwi", "")

                bval_candidate = (
                    dwi_folder / f"{base_prefix}_desc-preproc_dwi.bval"
                )
                bvec_candidate = (
                    dwi_folder / f"{base_prefix}_desc-preproc_dwi.bvec"
                )

                if not bval_candidate.is_file():
                    bvals_found = sorted(
                        dwi_folder.glob(f"{base_prefix}*.bval")
                    )
                    bval_candidate = (
                        bvals_found[0] if bvals_found else bval_candidate
                    )

                if not bvec_candidate.is_file():
                    bvecs_found = sorted(
                        dwi_folder.glob(f"{base_prefix}*.bvec")
                    )
                    bvec_candidate = (
                        bvecs_found[0] if bvecs_found else bvec_candidate
                    )

                if not (bval_candidate.is_file() and bvec_candidate.is_file()):
                    continue

                mask_candidate: Optional[Path] = None
                masks = sorted(
                    dwi_folder.glob(f"{base_prefix}*desc-brain_mask.nii*")
                )
                if not masks:
                    masks = sorted(dwi_folder.glob("*desc-brain_mask.nii*"))
                if masks:
                    mask_candidate = masks[0]

                discovered_sets.append({
                    "dwi": preproc_dwi,
                    "bval": bval_candidate,
                    "bvec": bvec_candidate,
                    "mask": mask_candidate,
                    "dwi_dir": dwi_folder,
                    "base_prefix": Path(base_prefix),
                })

        return discovered_sets

    def reorient_subject(
        self,
        qsiprep_dir: Path,
        subject: str,
        out_dir: Optional[Path] = None,
    ) -> List[Dict[str, Path]]:
        """Reorient all discovered preprocessed DWI outputs for a subject.

        Parameters
        ------------
        qsiprep_dir: Path
            Path to root QSIPrep derivatives directory.
        subject: str
            Subject identifier string (with or without 'sub-' prefix).
        out_dir: Optional[Path]
            Optional target output directory. Defaults to None.

        Returns
        --------
            List[Dict[str, Path]]: list of dicts
                List of generated FSL-compatible file mappings.
        """
        clean_sub = subject.replace("sub-", "").strip()
        subj_dir = qsiprep_dir / f"sub-{clean_sub}"
        dwi_sets = self.discover_subject_dwi_sets(subj_dir)

        is_multi_set = len(dwi_sets) > 1
        results: List[Dict[str, Path]] = []
        for dwi_set in dwi_sets:
            dwi_folder = dwi_set["dwi_dir"]
            base_prefix_str = str(dwi_set["base_prefix"])

            # 1. Generate desc-fslstd files alongside original preproc files
            bids_dwi = dwi_folder / f"{base_prefix_str}_desc-fslstd_dwi.nii.gz"
            bids_bval = dwi_folder / f"{base_prefix_str}_desc-fslstd_dwi.bval"
            bids_bvec = dwi_folder / f"{base_prefix_str}_desc-fslstd_dwi.bvec"
            bids_mask: Optional[Path] = None
            if dwi_set.get("mask") is not None:
                bids_mask = (
                    dwi_folder
                    / f"{base_prefix_str}_desc-fslstd_brain_mask.nii.gz"
                )

            bids_results = self.reorient_files(
                dwi_path=dwi_set["dwi"],
                bval_path=dwi_set["bval"],
                bvec_path=dwi_set["bvec"],
                mask_path=dwi_set.get("mask"),
                out_dwi=bids_dwi,
                out_bval=bids_bval,
                out_bvec=bids_bvec,
                out_mask=bids_mask,
            )

            # 2. Populate standard FSL named files (data.nii.gz, bvals, bvecs)
            if out_dir is not None:
                fsl_target_dir = (
                    out_dir / base_prefix_str if is_multi_set else out_dir
                )
            else:
                fsl_target_dir = (
                    dwi_folder / "fsl" / base_prefix_str
                    if is_multi_set
                    else dwi_folder / "fsl"
                )
            fsl_target_dir.mkdir(parents=True, exist_ok=True)

            fsl_dwi = fsl_target_dir / "data.nii.gz"
            fsl_bval = fsl_target_dir / "bvals"
            fsl_bvec = fsl_target_dir / "bvecs"
            fsl_mask = fsl_target_dir / "nodif_brain_mask.nii.gz"

            # Create standard FSL files or symlinks from generated FSL images
            self._link_or_copy(bids_dwi, fsl_dwi)
            self._link_or_copy(bids_bval, fsl_bval)
            self._link_or_copy(bids_bvec, fsl_bvec)
            if bids_mask is not None and bids_mask.is_file():
                self._link_or_copy(bids_mask, fsl_mask)

            bids_results["fsl_dir"] = fsl_target_dir
            bids_results["fsl_dwi"] = fsl_dwi
            bids_results["fsl_bval"] = fsl_bval
            bids_results["fsl_bvec"] = fsl_bvec
            if bids_mask is not None:
                bids_results["fsl_mask"] = fsl_mask

            results.append(bids_results)

        if results:
            marker = subj_dir / ".reorient_fsl_complete"
            marker.write_text("FSL reorientation complete\n", encoding="utf-8")

        return results

    def _link_or_copy(self, source_path: Path, target_path: Path) -> None:
        """Create a symlink or copy file if symlinking is not possible.

        Parameters
        ------------
        source_path: Path
            Source file path.
        target_path: Path
            Destination file path.

        Returns
        --------
            None: NoneType
                None return value.
        """
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        try:
            target_path.symlink_to(source_path.resolve())
        except (OSError, PermissionError):
            import shutil
            shutil.copy2(str(source_path), str(target_path))


class FSLReorientApp:
    """CLI application for reorienting QSIPrep DWI to FSL convention."""

    def __init__(self) -> None:
        """Initialize FSLReorientApp instance.

        Returns
        --------
            None: NoneType
                None return value.
        """
        self._reorienter = FSLReorienter()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser for the reorient_fsl tool.

        Returns
        --------
            argparse.ArgumentParser: parser object
                Configured ArgumentParser instance.
        """
        parser = argparse.ArgumentParser(
            description=(
                "Reorient QSIPrep DWI outputs to FSL radiological "
                "standard (LAS+) and rotate bvals/bvecs for FSL tools "
                "like FSLeyes and dtifit."
            )
        )
        parser.add_argument(
            "-d", "--dwi",
            type=Path,
            default=None,
            help="Path to input QSIPrep preprocessed DWI (.nii.gz)",
        )
        parser.add_argument(
            "-b", "--bval",
            type=Path,
            default=None,
            help="Path to input .bval file",
        )
        parser.add_argument(
            "-r", "--bvec",
            type=Path,
            default=None,
            help="Path to input .bvec file",
        )
        parser.add_argument(
            "-m", "--mask",
            type=Path,
            default=None,
            help="Path to input brain mask (.nii.gz)",
        )
        parser.add_argument(
            "-o", "--outdir",
            type=Path,
            default=None,
            help="Output directory for FSL convention files",
        )
        parser.add_argument(
            "--qsiprep-dir",
            type=Path,
            default=None,
            help="Path to QSIPrep derivatives root directory",
        )
        parser.add_argument(
            "--subject",
            type=str,
            default=None,
            help="Subject ID to process (e.g. sub-01 or 01)",
        )
        parser.add_argument(
            "--run-dtifit",
            action="store_true",
            default=False,
            help="Run FSL dtifit on the reoriented files",
        )
        parser.add_argument(
            "--basename",
            type=str,
            default="dti",
            help="Output prefix for dtifit results (default: dti)",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Parse arguments and execute reorientation.

        Parameters
        ------------
        args: Optional[List[str]]
            Optional command line argument list. Defaults to None.

        Returns
        --------
            int: integer object
                Exit status code (0 for success, non-zero for failure).
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        # Mode 1: Subject-level processing from QSIPrep derivatives
        if parsed.qsiprep_dir is not None and parsed.subject is not None:
            results = self._reorienter.reorient_subject(
                qsiprep_dir=parsed.qsiprep_dir,
                subject=parsed.subject,
                out_dir=parsed.outdir,
            )
            if not results:
                logging.error(
                    "No preprocessed DWI sets found for subject %s in %s",
                    parsed.subject,
                    parsed.qsiprep_dir,
                )
                return 1

            if parsed.run_dtifit:
                for res in results:
                    fsl_d = res.get("fsl_dir")
                    if fsl_d:
                        self._run_dtifit(
                            fsl_dir=fsl_d,
                            basename=parsed.basename,
                        )
            return 0

        # Mode 2: Direct file-level processing
        if (
            parsed.dwi is not None
            and parsed.bval is not None
            and parsed.bvec is not None
        ):
            out_directory = (
                parsed.outdir
                if parsed.outdir is not None
                else parsed.dwi.parent / "fsl"
            )
            out_directory.mkdir(parents=True, exist_ok=True)

            out_dwi = out_directory / "data.nii.gz"
            out_bval = out_directory / "bvals"
            out_bvec = out_directory / "bvecs"
            out_mask = (
                (out_directory / "nodif_brain_mask.nii.gz")
                if parsed.mask is not None
                else None
            )

            self._reorienter.reorient_files(
                dwi_path=parsed.dwi,
                bval_path=parsed.bval,
                bvec_path=parsed.bvec,
                mask_path=parsed.mask,
                out_dwi=out_dwi,
                out_bval=out_bval,
                out_bvec=out_bvec,
                out_mask=out_mask,
            )

            if parsed.run_dtifit:
                self._run_dtifit(
                    fsl_dir=out_directory,
                    basename=parsed.basename,
                )
            return 0

        parser.print_help()
        logging.error(
            "Must provide either (--qsiprep-dir and --subject) "
            "or (--dwi, --bval, and --bvec)."
        )
        return 2

    def _run_dtifit(self, fsl_dir: Path, basename: str) -> None:
        """Execute FSL dtifit on prepared FSL directory files.

        Parameters
        ------------
        fsl_dir: Path
            Directory containing data.nii.gz, bvals, bvecs, and mask.
        basename: str
            Output file basename prefix.

        Returns
        --------
            None: NoneType
                None return value.
        """
        out_dwi = fsl_dir / "data.nii.gz"
        out_bval = fsl_dir / "bvals"
        out_bvec = fsl_dir / "bvecs"
        out_mask = fsl_dir / "nodif_brain_mask.nii.gz"
        out_prefix = fsl_dir / basename

        cmd = [
            "dtifit",
            f"--data={out_dwi}",
            f"--out={out_prefix}",
            f"--bvecs={out_bvec}",
            f"--bvals={out_bval}",
            "--save_tensor",
        ]
        if out_mask.is_file():
            cmd.append(f"--mask={out_mask}")

        logging.info("Running FSL dtifit: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    """CLI entry point function for reorient_fsl.

    Returns
    --------
        None: NoneType
            None return value.
    """
    app = FSLReorientApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
