"""Study cohort and dataset format resolution module."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

    def _resolve_configured_subjects(
        self, configured_subjects: List[Any]
    ) -> List[str]:
        """Resolve configured subject entries to raw subject directory names.

        When the dataset is not yet in BIDS format and a subject_mapping is
        defined, entries in configured_subjects that specify BIDS subject IDs
        (e.g., 'VNS001') are expanded to all matching raw acquisition folder
        names (e.g., ['VNS001_A', 'VNS001_S']).

        Parameters
        ------------
        configured_subjects: List[Any]
            List of subject identifiers from configuration.

        Returns
        --------
            List[str]: list of strings
                Resolved list of raw subject directory identifiers.
        """
        raw_items = [str(subject_item) for subject_item in configured_subjects]
        mapping: Dict[str, Any] = self._config.get("subject_mapping") or {}
        if self.is_bids or not mapping:
            return raw_items

        resolved_subjects: List[str] = []
        for item_str in raw_items:
            if item_str in mapping:
                if item_str not in resolved_subjects:
                    resolved_subjects.append(item_str)
                continue

            clean_target = self._sanitize_bids_label(item_str)
            matched_keys = [
                str(map_key)
                for map_key in mapping
                if self.get_bids_subject_label(str(map_key)) == clean_target
            ]
            if matched_keys:
                for map_key in matched_keys:
                    if map_key not in resolved_subjects:
                        resolved_subjects.append(map_key)
            elif item_str not in resolved_subjects:
                resolved_subjects.append(item_str)

        return resolved_subjects

    @property
    def subjects(self) -> List[str]:
        """Retrieve list of subject folder names from config or filesystem.

        Returns
        --------
            List[str]: list of strings
                List of raw subject directory identifiers.
        """
        configured_subjects = self._config.get("subjects")
        if configured_subjects:
            return self._resolve_configured_subjects(list(configured_subjects))

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

    @staticmethod
    def _sanitize_bids_label(label: str) -> str:
        """Sanitize string to alphanumeric BIDS label without entity prefixes."""
        label_str = str(label).strip()
        for prefix in ("sub-", "sub_", "ses-", "ses_"):
            if label_str.lower().startswith(prefix):
                label_str = label_str[len(prefix):]
        return label_str.replace("-", "").replace("_", "")

    def get_subject_and_session(
        self, raw_subject: str
    ) -> Tuple[str, Optional[str]]:
        """Resolve BIDS subject and optional session for a raw subject folder.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Tuple of (sanitized_bids_subject, sanitized_bids_session_or_None).
        """
        mapping: Dict[str, Any] = self._config.get("subject_mapping") or {}
        mapped = mapping.get(raw_subject)

        subj_val = raw_subject
        ses_val = None

        if isinstance(mapped, dict):
            subj_val = mapped.get("subject", raw_subject)
            ses_val = mapped.get("session")
        elif isinstance(mapped, str):
            if "/" in mapped:
                parts = mapped.split("/", 1)
                subj_val = parts[0]
                ses_val = parts[1]
            else:
                subj_val = mapped
        elif mapped is not None:
            subj_val = str(mapped)

        clean_sub = self._sanitize_bids_label(str(subj_val))
        clean_ses = (
            self._sanitize_bids_label(str(ses_val))
            if ses_val is not None and str(ses_val).strip()
            else None
        )
        return clean_sub, clean_ses

    def get_bids_subject_label(self, raw_subject: str) -> str:
        """Map raw subject directory name to sanitized BIDS subject label.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Sanitized BIDS subject label (without 'sub-' prefix).
        """
        clean_sub, _ = self.get_subject_and_session(raw_subject)
        return clean_sub

    def get_bids_session_label(self, raw_subject: str) -> Optional[str]:
        """Map raw subject directory name to sanitized BIDS session label.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Sanitized BIDS session label (without 'ses-' prefix) or None.
        """
        _, clean_ses = self.get_subject_and_session(raw_subject)
        return clean_ses

    def get_raw_subjects_for_bids_subject(
        self, bids_subject: str
    ) -> List[str]:
        """Find all raw subject folders corresponding to a BIDS subject label.

        Args:
            bids_subject: Sanitized BIDS subject identifier.

        Returns:
            List of raw subject folder names in cohort subjects.
        """
        clean_target = self._sanitize_bids_label(bids_subject)
        return [
            raw_name
            for raw_name in self.subjects
            if self.get_bids_subject_label(raw_name) == clean_target
        ]

    @property
    def bids_subjects(self) -> List[str]:
        """Return unique list of all BIDS subject labels (without sub- prefix).

        Returns:
            List of unique sanitized BIDS subject identifiers in order of appearance.
        """
        seen = set()
        unique_bids_subjects: List[str] = []
        for raw_name in self.subjects:
            clean_sub = self.get_bids_subject_label(raw_name)
            if clean_sub not in seen:
                seen.add(clean_sub)
                unique_bids_subjects.append(clean_sub)
        return unique_bids_subjects
