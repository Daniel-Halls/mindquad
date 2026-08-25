"""Common helper functions and configuration parsing for Mindquad pipeline."""

from pathlib import Path
from typing import Any, Dict, List


class StudyCohort:
    """Cohort manager to discover and resolve subjects in study."""

    def __init__(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize StudyCohort with workflow configuration dictionary.

        Args:
            pipeline_config: Dictionary containing pipeline configuration.
        """
        self._config = pipeline_config

    @property
    def raw_data_dir(self) -> Path:
        """Return the raw data directory path."""
        default_raw = "/imgshare/tES-FUS/pilot/dif_pilot"
        return Path(str(self._config.get("raw_data_dir", default_raw)))

    @property
    def subjects(self) -> List[str]:
        """Retrieve list of subject folder names from config or filesystem.

        Returns:
            List of raw subject directory identifiers.
        """
        configured_subjects = self._config.get("subjects")
        if configured_subjects:
            return [str(s) for s in configured_subjects]

        if self.raw_data_dir.exists():
            found = [
                p.name
                for p in self.raw_data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
            if found:
                return sorted(found)
        return ["19081_001"]

    def get_bids_subject_label(self, raw_subject: str) -> str:
        """Map raw subject directory name to sanitized BIDS subject label.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Sanitized BIDS subject label (without 'sub-' prefix).
        """
        mapping: Dict[str, Any] = self._config.get("subject_mapping", {})
        mapped = mapping.get(raw_subject, raw_subject)
        clean_label = str(mapped).replace("-", "").replace("_", "")
        return clean_label

    @property
    def bids_subjects(self) -> List[str]:
        """Return list of all BIDS subject labels (without sub- prefix).

        Returns:
            List of sanitized BIDS subject identifiers.
        """
        return [self.get_bids_subject_label(s) for s in self.subjects]


def get_raw_data_dir() -> str:
    """Return the raw data directory from configuration."""
    cohort = StudyCohort(config)
    return str(cohort.raw_data_dir)


def get_bids_dir() -> str:
    """Return the BIDS dataset root directory from configuration."""
    return str(config.get("bids_dir", "bids"))


def get_derivatives_dir() -> str:
    """Return the root derivatives directory from configuration."""
    return str(config.get("derivatives_dir", "derivatives"))


def get_mriqc_dir() -> str:
    """Return the MRIQC output directory path."""
    return str(Path(get_derivatives_dir()) / "mriqc")


def get_work_dir() -> str:
    """Return the intermediate working directory from configuration."""
    return str(config.get("work_dir", "work"))


def get_tmp_dir() -> str:
    """Return the project-local temporary directory path."""
    return str(config.get("tmp_dir", ".tmp"))


def get_mriqc_modalities() -> str:
    """Return space-separated list of MRIQC modalities from config."""
    modalities = config.get("mriqc", {}).get(
        "modalities", ["T1w", "bold", "dwi"]
    )
    return " ".join(modalities)


def get_mriqc_extra_args() -> str:
    """Return extra CLI flags for MRIQC from config."""
    default_args = "--verbose-reports --no-sub"
    return str(config.get("mriqc", {}).get("args", default_args))


def get_subjects() -> List[str]:
    """Retrieve list of subject folder names from config or filesystem.

    Returns:
        List of subject directory identifiers.
    """
    cohort = StudyCohort(config)
    return cohort.subjects


def get_bids_subject_label(raw_subject: str) -> str:
    """Map raw subject folder name to sanitized BIDS subject label.

    Args:
        raw_subject: Raw subject directory name.

    Returns:
        Sanitized BIDS subject label (without 'sub-' prefix).
    """
    cohort = StudyCohort(config)
    return cohort.get_bids_subject_label(raw_subject)


def get_bids_subjects() -> List[str]:
    """Retrieve list of all sanitized BIDS subject labels.

    Returns:
        List of BIDS subject labels without 'sub-' prefix.
    """
    cohort = StudyCohort(config)
    return cohort.bids_subjects


def get_raw_subject_dir(wildcards: Any) -> str:
    """Resolve raw DICOM directory path for a subject wildcard.

    Args:
        wildcards: Snakemake rule wildcards containing 'subject'.

    Returns:
        Path to subject's raw DICOM directory.
    """
    cohort = StudyCohort(config)
    subject_wildcard = wildcards.subject
    mapping: Dict[str, Any] = config.get("subject_mapping", {})
    reverse_mapping = {
        cohort.get_bids_subject_label(k): k for k in mapping.keys()
    }
    raw_name = reverse_mapping.get(subject_wildcard, subject_wildcard)
    return str(cohort.raw_data_dir / raw_name)
