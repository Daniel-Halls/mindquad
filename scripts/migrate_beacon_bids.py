#!/usr/bin/env python3
"""Migration script to conform BEACONB_bids filenames to the BIDS specification.

Fixes non-standard filenames in BEACONB_bids:
1. In ses-02/anat/:
   sub-<id>_T1w.nii.gz -> sub-<id>_ses-02_T1w.nii.gz
2. In ses-02/func/:
   sub-<id>_task-rest_bold_e<N>.nii.gz -> sub-<id>_ses-02_task-rest_echo-<N>_bold.nii.gz
   sub-<id>_task-rest_bold_e<N>.json   -> sub-<id>_ses-02_task-rest_echo-<N>_bold.json
   sub-G2050_task-rest_bold.nii.gz    -> sub-G2050_ses-02_task-rest_bold.nii.gz
   sub-G2050_task-rest_bold.json      -> sub-G2050_ses-02_task-rest_bold.json
3. In func JSON sidecars:
   Ensures "TaskName" is present.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bids_migrator")


def get_bids_target(path: Path) -> Optional[Path]:
    """Calculate BIDS-compliant destination path for a given file."""
    name = path.name

    # 1. Anat: sub-<id>_T1w.nii.gz inside ses-<ses>/anat
    m_anat = re.match(r"^(sub-[^_]+)_T1w(\.nii(?:\.gz)?)$", name)
    if m_anat and "ses-" in str(path.parent):
        ses = path.parent.parent.name
        return path.with_name(f"{m_anat.group(1)}_{ses}_T1w{m_anat.group(2)}")

    # 2. Func multi-echo: sub-<id>_task-<task>_bold_e<echo>.(nii.gz|json)
    m_func_me = re.match(
        r"^(sub-[^_]+)_task-([^_]+)_bold_e(\d+)(\.nii(?:\.gz)?|\.json)$", name
    )
    if m_func_me:
        sub, task, echo, ext = m_func_me.groups()
        ses = path.parent.parent.name if "ses-" in str(path.parent) else ""
        ses_part = f"_{ses}" if ses.startswith("ses-") else ""
        return path.with_name(f"{sub}{ses_part}_task-{task}_echo-{echo}_bold{ext}")

    # 3. Func single-echo: sub-<id>_task-<task>_bold.(nii.gz|json) missing session
    m_func_se = re.match(
        r"^(sub-[^_]+)_task-([^_]+)_bold(\.nii(?:\.gz)?|\.json)$", name
    )
    if m_func_se:
        sub, task, ext = m_func_se.groups()
        ses = path.parent.parent.name if "ses-" in str(path.parent) else ""
        ses_part = f"_{ses}" if ses.startswith("ses-") else ""
        if ses_part:
            return path.with_name(f"{sub}{ses_part}_task-{task}_bold{ext}")

    return None


def migrate_dataset(bids_dir: Path, apply: bool = False) -> Tuple[int, int]:
    """Scan and migrate dataset to BIDS standard naming."""
    sub_dirs = sorted(
        [p for p in bids_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")]
    )
    logger.info("Discovered %d subject directories in %s", len(sub_dirs), bids_dir)

    renamed_count = 0
    json_updated_count = 0

    for sub_dir in sub_dirs:
        for file_path in sorted(sub_dir.rglob("*")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            target_path = get_bids_target(file_path)
            if target_path and target_path != file_path:
                renamed_count += 1
                if apply:
                    file_path.rename(target_path)
                    logger.debug("Renamed: %s -> %s", file_path.name, target_path.name)
                else:
                    logger.info("[DRY-RUN] Would rename: %s -> %s", file_path.name, target_path.name)

            effective_path = target_path if (apply and target_path) else file_path
            # Check JSON sidecars for TaskName
            if effective_path.name.endswith(".json") and "task-" in effective_path.name:
                try:
                    with open(effective_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if "TaskName" not in meta:
                        m_task = re.search(r"task-([a-zA-Z0-9]+)", effective_path.name)
                        if m_task:
                            task_name = m_task.group(1)
                            meta["TaskName"] = task_name
                            json_updated_count += 1
                            if apply:
                                with open(effective_path, "w", encoding="utf-8") as f:
                                    json.dump(meta, f, indent=4)
                                logger.debug("Added TaskName=%s to %s", task_name, effective_path.name)
                            else:
                                logger.info("[DRY-RUN] Would add TaskName=%s to %s", task_name, effective_path.name)
                except Exception as e:
                    logger.warning("Could not read/update JSON %s: %s", effective_path, e)

    return renamed_count, json_updated_count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Migrate BEACON dataset filenames to BIDS specification."
    )
    parser.add_argument(
        "--bids-dir",
        type=Path,
        default=Path("/imgshare/tES-FUS/BEACON/BEACONB_bids"),
        help="Path to BEACON BIDS root directory.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute renames and JSON updates. If omitted, runs in dry-run mode.",
    )
    args = parser.parse_args()

    if not args.bids_dir.exists():
        logger.error("BIDS directory not found: %s", args.bids_dir)
        sys.exit(1)

    mode_str = "APPLY" if args.apply else "DRY-RUN"
    logger.info("Starting migration in %s mode on %s", mode_str, args.bids_dir)
    renamed, updated_json = migrate_dataset(args.bids_dir, apply=args.apply)
    logger.info(
        "Finished migration. Total files to rename: %d, Total JSON sidecars to update: %d",
        renamed,
        updated_json,
    )
    if not args.apply:
        logger.info("Run with --apply to perform the changes.")


if __name__ == "__main__":
    main()
