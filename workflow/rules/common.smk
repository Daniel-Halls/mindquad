"""Common helper functions and configuration parsing for Mindquad pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, List


def get_raw_data_dir() -> str:
    """Return the raw data directory from configuration."""
    return str(config.get("raw_data_dir", "/imgshare/tES-FUS/pilot/dif_pilot"))


def get_bids_dir() -> str:
    """Return the BIDS dataset root directory from configuration."""
    return str(config.get("bids_dir", "bids"))


def get_work_dir() -> str:
    """Return the intermediate working directory from configuration."""
    return str(config.get("work_dir", "work"))


def get_tmp_dir() -> str:
    """Return the project-local temporary directory path."""
    return str(config.get("tmp_dir", ".tmp"))


def get_subjects() -> List[str]:
    """Retrieve list of subject folder names from config or filesystem.

    Returns:
        List of subject directory identifiers.
    """
    configured_subjects = config.get("subjects")
    if configured_subjects:
        return [str(s) for s in configured_subjects]

    raw_path = Path(get_raw_data_dir())
    if raw_path.exists():
        found = [
            p.name
            for p in raw_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if found:
            return sorted(found)
    return ["19081_001"]


def get_bids_subject_label(raw_subject: str) -> str:
    """Map raw subject folder name to sanitized BIDS subject label.

    Args:
        raw_subject: Raw subject directory name.

    Returns:
        Sanitized BIDS subject label (without 'sub-' prefix).
    """
    mapping: Dict[str, Any] = config.get("subject_mapping", {})
    mapped = mapping.get(raw_subject, raw_subject)
    clean_label = str(mapped).replace("-", "").replace("_", "")
    return clean_label


def get_raw_subject_dir(wildcards: Any) -> str:
    """Resolve raw DICOM directory path for a subject wildcard.

    Args:
        wildcards: Snakemake rule wildcards containing 'subject'.

    Returns:
        Absolute or relative path to subject's raw DICOM directory.
    """
    raw_dir = get_raw_data_dir()
    subject_wildcard = wildcards.subject
    mapping: Dict[str, Any] = config.get("subject_mapping", {})
    reverse_mapping = {
        get_bids_subject_label(k): k for k in mapping.keys()
    }
    raw_name = reverse_mapping.get(subject_wildcard, subject_wildcard)
    return os.path.join(raw_dir, raw_name)


SUBJECTS = get_subjects()
BIDS_SUBJECTS = [get_bids_subject_label(s) for s in SUBJECTS]
