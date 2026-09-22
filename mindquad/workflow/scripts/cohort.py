"""Study cohort and dataset format resolution module."""

from pathlib import Path
from typing import Any, Dict, List


class StudyCohort:
    """Cohort manager to discover subjects and BIDS status in study."""

    def __init__(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize StudyCohort with workflow configuration dictionary.

        Args:
            pipeline_config: Dictionary containing pipeline configuration.
        """
        self._config = pipeline_config or {}

    @property
    def raw_data_dir(self) -> Path:
        """Return the raw data directory path."""
        return Path(str(self._config.get("raw_data_dir", "")))

    @property
    def is_bids(self) -> bool:
        """Check whether raw_data_dir is already a BIDS formatted dataset."""
        for config_key in ("is_bids", "raw_is_bids", "bids_format"):
            config_val = self._config.get(config_key)
            if config_val is not None:
                return bool(config_val)

        if not self.raw_data_dir.exists() or not self.raw_data_dir.is_dir():
            return False

        # 1. Check for dataset_description.json in raw_data_dir
        if (self.raw_data_dir / "dataset_description.json").is_file():
            return True

        # 2. Check for sub-* directories containing BIDS modalities
        sub_dirs = [
            path_entry
            for path_entry in self.raw_data_dir.iterdir()
            if path_entry.is_dir() and path_entry.name.startswith("sub-")
        ]
        if sub_dirs:
            bids_modalities = {
                "anat",
                "func",
                "dwi",
                "fmap",
                "mrs",
                "perf",
                "meg",
                "eeg",
                "ieeg",
                "beh",
            }
            for subject_dir in sub_dirs:
                if any(
                    (subject_dir / modality_name).is_dir()
                    for modality_name in bids_modalities
                ):
                    return True
                # Also check ses-*/modality
                if any(
                    any(
                        (session_dir / modality_name).is_dir()
                        for modality_name in bids_modalities
                    )
                    for session_dir in subject_dir.iterdir()
                    if (
                        session_dir.is_dir()
                        and session_dir.name.startswith("ses-")
                    )
                ):
                    return True

        return False

    @property
    def subjects(self) -> List[str]:
        """Retrieve list of subject folder names from config or filesystem.

        Returns:
            List of raw subject directory identifiers.
        """
        configured_subjects = self._config.get("subjects")
        if configured_subjects:
            return [str(subject_item) for subject_item in configured_subjects]

        if self.raw_data_dir.exists():
            if self.is_bids:
                found_dirs = [
                    dir_entry.name
                    for dir_entry in self.raw_data_dir.iterdir()
                    if (
                        dir_entry.is_dir()
                        and dir_entry.name.startswith("sub-")
                    )
                ]
            else:
                found_dirs = [
                    dir_entry.name
                    for dir_entry in self.raw_data_dir.iterdir()
                    if (
                        dir_entry.is_dir()
                        and not dir_entry.name.startswith(".")
                    )
                ]
            if found_dirs:
                return sorted(found_dirs)
        raise FileNotFoundError(f"No subjects found in {self.raw_data_dir}")

    def get_bids_subject_label(self, raw_subject: str) -> str:
        """Map raw subject directory name to sanitized BIDS subject label.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Sanitized BIDS subject label (without 'sub-' prefix).
        """
        mapping: Dict[str, Any] = self._config.get("subject_mapping", {})
        mapped = mapping.get(raw_subject, raw_subject)
        mapped_str = str(mapped)
        if mapped_str.lower().startswith("sub-"):
            mapped_str = mapped_str[4:]
        elif mapped_str.lower().startswith("sub_"):
            mapped_str = mapped_str[4:]
        clean_label = mapped_str.replace("-", "").replace("_", "")
        return clean_label

    @property
    def bids_subjects(self) -> List[str]:
        """Return list of all BIDS subject labels (without sub- prefix).

        Returns:
            List of sanitized BIDS subject identifiers.
        """
        return [
            self.get_bids_subject_label(subject_name)
            for subject_name in self.subjects
        ]
