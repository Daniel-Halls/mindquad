"""BIDS organizer script for neuroimaging data.

This module provides object-oriented classes and single-responsibility methods
to classify, rename, and structure converted NIfTI files into BIDS standard
directories.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BIDSModality(Enum):
    """Enumeration of BIDS modality folder names."""

    ANAT = "anat"
    FUNC = "func"
    DWI = "dwi"
    FMAP = "fmap"
    MRS = "mrs"


class ScanClassification:
    """Class representing classification results of an imaging series."""

    def __init__(
        self,
        modality: BIDSModality,
        suffix: str,
        entities: Dict[str, str],
    ) -> None:
        """Initialize ScanClassification instance.

        Args:
            modality: The target BIDS modality.
            suffix: The BIDS file suffix.
            entities: Key-value mapping of BIDS entities.
        """
        self._modality = modality
        self._suffix = suffix
        self._entities = entities

    @property
    def modality(self) -> BIDSModality:
        """Return the target BIDS modality."""
        return self._modality

    @property
    def suffix(self) -> str:
        """Return the BIDS suffix."""
        return self._suffix

    @property
    def entities(self) -> Dict[str, str]:
        """Return the BIDS entity dictionary."""
        return self._entities


class SeriesCandidate:
    """Class representing a candidate series file before BIDS transfer."""

    def __init__(
        self,
        json_path: Path,
        stem: str,
        classification: ScanClassification,
        series_number: int,
    ) -> None:
        """Initialize SeriesCandidate instance.

        Args:
            json_path: Path to the JSON metadata file.
            stem: Original file stem name.
            classification: ScanClassification object.
            series_number: DICOM series number for chronological ordering.
        """
        self._json_path = json_path
        self._stem = stem
        self._classification = classification
        self._series_number = series_number

    @property
    def json_path(self) -> Path:
        """Return the JSON file path."""
        return self._json_path

    @property
    def stem(self) -> str:
        """Return the original stem name."""
        return self._stem

    @property
    def classification(self) -> ScanClassification:
        """Return the scan classification."""
        return self._classification

    @property
    def series_number(self) -> int:
        """Return the series number."""
        return self._series_number


class DICOMSeriesClassifier:
    """Class to classify DICOM series based on JSON sidecars and file stems."""

    def __init__(self) -> None:
        """Initialize DICOMSeriesClassifier with default logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def read_json_metadata(self, json_path: Path) -> Dict[str, Any]:
        """Read and return metadata dictionary from a JSON file.

        Args:
            json_path: Path to the JSON sidecar.

        Returns:
            Dictionary containing JSON metadata.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as file_handle:
                metadata: Dict[str, Any] = json.load(file_handle)
                return metadata
        except (json.JSONDecodeError, OSError) as exc:
            self._logger.warning(
                "Could not parse JSON %s: %s", json_path, str(exc)
            )
            return {}

    def extract_series_number(self, json_path: Path, stem: str) -> int:
        """Extract series number for deterministic chronological ordering.

        Args:
            json_path: Path to JSON metadata file.
            stem: Filename stem.

        Returns:
            Extracted series number integer.
        """
        metadata = self.read_json_metadata(json_path)
        series_num = metadata.get("SeriesNumber")
        if series_num is not None:
            try:
                return int(series_num)
            except (ValueError, TypeError):
                pass
        match = re.search(r"(\d+)", stem)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return 0

    def _extract_series_description(
        self, metadata: Dict[str, Any], stem: str
    ) -> str:
        """Extract series description or fallback to filename stem.

        Args:
            metadata: JSON metadata dictionary.
            stem: Filename stem.

        Returns:
            Normalized series description string.
        """
        description = metadata.get("SeriesDescription", "")
        if not description:
            description = metadata.get("ProtocolName", "")
        if not description:
            description = stem
        return description.lower()

    def _detect_anat_suffix(self, description: str) -> Optional[str]:
        """Detect anatomical BIDS suffix from description.

        Args:
            description: Lowercase sequence description.

        Returns:
            BIDS suffix string or None.
        """
        if any(key in description for key in ["t1w", "mprage", "t1_"]):
            return "T1w"
        if any(key in description for key in ["t2w", "t2_spc", "t2_tse"]):
            return "T2w"
        if "flair" in description:
            return "FLAIR"
        if "pwi" in description or "asl" in description:
            return "asl"
        return None

    def _detect_func_suffix(self, description: str) -> Optional[str]:
        """Detect functional BIDS suffix from description.

        Args:
            description: Lowercase sequence description.

        Returns:
            BIDS suffix string or None.
        """
        if any(key in description for key in ["bold", "fmri", "ep2d_pace"]):
            return "bold"
        if "rest" in description and "dwi" not in description:
            return "bold"
        return None

    def _detect_dwi_suffix(self, description: str) -> Optional[str]:
        """Detect diffusion BIDS suffix from description.

        Args:
            description: Lowercase sequence description.

        Returns:
            BIDS suffix string or None.
        """
        if any(key in description for key in ["dwi", "diff", "dti", "ep2d_diff"]):
            return "dwi"
        return None

    def _detect_mrs_suffix(self, description: str) -> Optional[str]:
        """Detect MRS BIDS suffix from description.

        Args:
            description: Lowercase sequence description.

        Returns:
            BIDS suffix string or None.
        """
        if any(
            key in description
            for key in ["svs", "press", "steam", "slaser", "mrs", "mega_press"]
        ):
            return "svs"
        return None

    def _detect_fmap_suffix(self, description: str) -> Optional[str]:
        """Detect fieldmap BIDS suffix from description.

        Args:
            description: Lowercase sequence description.

        Returns:
            BIDS suffix string or None.
        """
        if "field_mapping" in description or "fmap" in description:
            if "phase" in description:
                return "phasediff"
            if "mag" in description:
                return "magnitude1"
            return "fieldmap"
        return None

    def _extract_entities(
        self, description: str, modality: BIDSModality
    ) -> Dict[str, str]:
        """Extract BIDS entities such as task or direction from description.

        Args:
            description: Lowercase sequence description.
            modality: Classified BIDS modality.

        Returns:
            Dictionary of extracted entities.
        """
        entities: Dict[str, str] = {}

        # Task entity for functional scans
        if modality == BIDSModality.FUNC:
            task_match = re.search(r"task-([a-zA-Z0-9]+)", description)
            if task_match:
                entities["task"] = task_match.group(1)
            elif "rest" in description:
                entities["task"] = "rest"
            else:
                entities["task"] = "task"

        # Direction entity
        if "dir-ap" in description or "_ap" in description:
            entities["dir"] = "AP"
        elif "dir-pa" in description or "_pa" in description:
            entities["dir"] = "PA"

        return entities

    def classify_series(
        self, json_path: Path, stem: str
    ) -> Optional[ScanClassification]:
        """Classify a series using its JSON sidecar and filename stem.

        Args:
            json_path: Path to the JSON metadata file.
            stem: Filename stem.

        Returns:
            ScanClassification object if successfully classified, else None.
        """
        metadata = self.read_json_metadata(json_path)
        description = self._extract_series_description(metadata, stem)

        # 1. Check Anatomical
        anat_suffix = self._detect_anat_suffix(description)
        if anat_suffix:
            entities = self._extract_entities(description, BIDSModality.ANAT)
            return ScanClassification(BIDSModality.ANAT, anat_suffix, entities)

        # 2. Check Diffusion
        dwi_suffix = self._detect_dwi_suffix(description)
        if dwi_suffix:
            entities = self._extract_entities(description, BIDSModality.DWI)
            return ScanClassification(BIDSModality.DWI, dwi_suffix, entities)

        # 3. Check Functional
        func_suffix = self._detect_func_suffix(description)
        if func_suffix:
            entities = self._extract_entities(description, BIDSModality.FUNC)
            return ScanClassification(BIDSModality.FUNC, func_suffix, entities)

        # 4. Check MRS
        mrs_suffix = self._detect_mrs_suffix(description)
        if mrs_suffix:
            entities = self._extract_entities(description, BIDSModality.MRS)
            return ScanClassification(BIDSModality.MRS, mrs_suffix, entities)

        # 5. Check Fieldmap
        fmap_suffix = self._detect_fmap_suffix(description)
        if fmap_suffix:
            entities = self._extract_entities(description, BIDSModality.FMAP)
            return ScanClassification(BIDSModality.FMAP, fmap_suffix, entities)

        self._logger.info("Unclassified sequence skipped: %s", stem)
        return None


class BIDSFilenameBuilder:
    """Class to construct standard BIDS file names with canonical entity ordering."""

    # Canonical BIDS entity ordering according to BIDS Specification v1.9.0
    BIDS_ORDERED_ENTITIES: Tuple[str, ...] = (
        "sub",
        "ses",
        "sample",
        "task",
        "acq",
        "ce",
        "trc",
        "rec",
        "dir",
        "run",
        "mod",
        "echo",
        "flip",
        "inv",
        "mt",
        "part",
        "proc",
        "hemi",
        "space",
        "split",
        "recording",
        "chunk",
        "desc",
    )

    def build_bids_stem(
        self, subject_label: str, classification: ScanClassification
    ) -> str:
        """Build standard BIDS filename stem with canonical entity ordering.

        Args:
            subject_label: BIDS subject identifier (e.g., 'sub-001').
            classification: ScanClassification containing modality and suffix.

        Returns:
            Complete BIDS stem string adhering to BIDS entity order.
        """
        clean_sub = (
            subject_label
            if subject_label.startswith("sub-")
            else f"sub-{subject_label}"
        )
        parts = [clean_sub]

        # Add entities strictly adhering to the canonical BIDS order
        present_entities = classification.entities
        for entity_key in self.BIDS_ORDERED_ENTITIES:
            if entity_key in present_entities:
                parts.append(f"{entity_key}-{present_entities[entity_key]}")

        # Add any unlisted custom entities in sorted order
        for key in sorted(present_entities.keys()):
            if key not in self.BIDS_ORDERED_ENTITIES:
                parts.append(f"{key}-{present_entities[key]}")

        # Add suffix
        parts.append(classification.suffix)
        return "_".join(parts)


class BIDSFileTransfer:
    """Class to copy or move converted files into BIDS structure."""

    def __init__(self) -> None:
        """Initialize BIDSFileTransfer with logger."""
        self._logger = logging.getLogger(self.__class__.__name__)

    def _copy_single_file(self, src: Path, dst: Path) -> Path:
        """Copy a single file safely from src to dst.

        Args:
            src: Source file path.
            dst: Destination file path.

        Returns:
            Destination path.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self._logger.debug("Copied %s -> %s", src.name, dst)
        return dst

    def transfer_associated_files(
        self,
        source_dir: Path,
        source_stem: str,
        target_dir: Path,
        target_stem: str,
    ) -> int:
        """Transfer all files associated with a given stem to target directory.

        Args:
            source_dir: Directory containing converted files.
            source_stem: Original filename stem.
            target_dir: Destination BIDS modality directory.
            target_stem: Target BIDS filename stem.

        Returns:
            Count of transferred files.
        """
        transferred_count = 0
        extensions = [
            ".nii.gz",
            ".nii",
            ".json",
            ".bval",
            ".bvec",
            ".tsv",
            ".mat",
        ]

        for ext in extensions:
            src_file = source_dir / f"{source_stem}{ext}"
            if src_file.exists():
                dst_file = target_dir / f"{target_stem}{ext}"
                self._copy_single_file(src_file, dst_file)
                transferred_count += 1

        return transferred_count


class BIDSOrganizer:
    """Class to organize converted NIfTI/JSON files for a subject into BIDS."""

    def __init__(self) -> None:
        """Initialize BIDSOrganizer with classifier and transfer components."""
        self._classifier = DICOMSeriesClassifier()
        self._filename_builder = BIDSFilenameBuilder()
        self._file_transfer = BIDSFileTransfer()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _find_json_sidecars(self, input_dir: Path) -> List[Path]:
        """Find and return all JSON sidecars in input directory.

        Args:
            input_dir: Path to directory containing converted files.

        Returns:
            Sorted list of JSON file paths.
        """
        return sorted(input_dir.glob("*.json"))

    def _collect_candidates(self, input_dir: Path) -> List[SeriesCandidate]:
        """Collect and classify all candidate series from JSON sidecars.

        Args:
            input_dir: Path to directory with converted files.

        Returns:
            List of classified SeriesCandidate objects.
        """
        json_sidecars = self._find_json_sidecars(input_dir)
        candidates: List[SeriesCandidate] = []

        for json_path in json_sidecars:
            stem = json_path.stem
            classification = self._classifier.classify_series(json_path, stem)
            if classification is not None:
                series_num = self._classifier.extract_series_number(
                    json_path, stem
                )
                candidates.append(
                    SeriesCandidate(
                        json_path, stem, classification, series_num
                    )
                )

        return candidates

    def _compute_group_key(
        self, classification: ScanClassification
    ) -> Tuple[str, str, Tuple[Tuple[str, str], ...]]:
        """Compute unique grouping key for a sequence ignoring run entity.

        Args:
            classification: ScanClassification instance.

        Returns:
            Tuple representing the grouping key.
        """
        non_run_entities = tuple(
            sorted(
                (k, v)
                for k, v in classification.entities.items()
                if k != "run"
            )
        )
        return (
            classification.modality.value,
            classification.suffix,
            non_run_entities,
        )

    def _group_candidates(
        self, candidates: List[SeriesCandidate]
    ) -> Dict[Tuple[str, str, Tuple[Tuple[str, str], ...]], List[SeriesCandidate]]:
        """Group candidates by modality, suffix, and non-run entities.

        Args:
            candidates: List of SeriesCandidate objects.

        Returns:
            Dictionary grouping candidates by unique key.
        """
        groups: Dict[
            Tuple[str, str, Tuple[Tuple[str, str], ...]], List[SeriesCandidate]
        ] = {}

        for candidate in candidates:
            key = self._compute_group_key(candidate.classification)
            if key not in groups:
                groups[key] = []
            groups[key].append(candidate)

        return groups

    def _assign_run_indices(
        self,
        groups: Dict[
            Tuple[str, str, Tuple[Tuple[str, str], ...]], List[SeriesCandidate]
        ],
    ) -> List[SeriesCandidate]:
        """Assign run indices to all sequences having multiple runs.

        Args:
            groups: Dictionary of candidate groups.

        Returns:
            Flat list of candidates with run indices properly assigned.
        """
        final_candidates: List[SeriesCandidate] = []

        for group_candidates in groups.values():
            # Sort chronologically by series number, then stem
            sorted_candidates = sorted(
                group_candidates,
                key=lambda c: (c.series_number, c.stem),
            )

            # If there are multiple runs, assign run index to ALL runs
            if len(sorted_candidates) > 1:
                for idx, cand in enumerate(sorted_candidates):
                    cand.classification.entities["run"] = f"{idx + 1:02d}"
                    final_candidates.append(cand)
            else:
                final_candidates.append(sorted_candidates[0])

        return final_candidates

    def organize_subject(
        self,
        input_dir: Path,
        bids_root: Path,
        subject_label: str,
    ) -> int:
        """Organize converted subject files into BIDS subject folder.

        Args:
            input_dir: Path to directory containing dcm2niix converted files.
            bids_root: Root directory of BIDS dataset.
            subject_label: Subject label (e.g. 'sub-001' or '001').

        Returns:
            Total count of organized files.
        """
        clean_sub = (
            subject_label
            if subject_label.startswith("sub-")
            else f"sub-{subject_label}"
        )
        subject_dir = bids_root / clean_sub

        candidates = self._collect_candidates(input_dir)
        if not candidates:
            self._logger.warning("No valid series candidates found in %s", input_dir)
            return 0

        groups = self._group_candidates(candidates)
        processed_candidates = self._assign_run_indices(groups)

        total_transferred = 0
        for candidate in processed_candidates:
            target_stem = self._filename_builder.build_bids_stem(
                clean_sub, candidate.classification
            )
            target_modality_dir = (
                subject_dir / candidate.classification.modality.value
            )

            transferred = self._file_transfer.transfer_associated_files(
                input_dir,
                candidate.stem,
                target_modality_dir,
                target_stem,
            )
            total_transferred += transferred

        self._logger.info(
            "Finished organizing subject %s: %d files organized.",
            clean_sub,
            total_transferred,
        )
        return total_transferred


class BIDSOrganizerApp:
    """CLI application runner for BIDS organization."""

    def __init__(self) -> None:
        """Initialize BIDSOrganizerApp."""
        self._organizer = BIDSOrganizer()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and return CLI argument parser.

        Returns:
            Configured ArgumentParser.
        """
        parser = argparse.ArgumentParser(
            description="Organize dcm2niix converted files into BIDS standard."
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            help="Directory with dcm2niix converted NIfTI/JSON files.",
        )
        parser.add_argument(
            "--bids-dir",
            type=Path,
            required=True,
            help="BIDS dataset root directory.",
        )
        parser.add_argument(
            "--subject",
            type=str,
            required=True,
            help="Subject label (e.g., 19081001 or sub-19081001).",
        )
        parser.add_argument(
            "--output-marker",
            type=Path,
            default=None,
            help="Optional marker file to touch upon completion.",
        )
        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Execute the CLI application.

        Args:
            args: Optional command line arguments.

        Returns:
            Exit code integer (0 for success, non-zero for error).
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if not parsed_args.input_dir.exists():
            logging.error("Input directory %s does not exist", parsed_args.input_dir)
            return 1

        parsed_args.bids_dir.mkdir(parents=True, exist_ok=True)

        organized_count = self._organizer.organize_subject(
            parsed_args.input_dir,
            parsed_args.bids_dir,
            parsed_args.subject,
        )

        if parsed_args.output_marker:
            parsed_args.output_marker.parent.mkdir(parents=True, exist_ok=True)
            parsed_args.output_marker.write_text(
                f"Organized {organized_count} files.\n",
                encoding="utf-8",
            )

        return 0


def main() -> None:
    """Main execution function for CLI script."""
    app = BIDSOrganizerApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
