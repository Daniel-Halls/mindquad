import sys
from pathlib import Path
"""Snakemake rules for BIDS dataset initialization and DICOM to BIDS conversion."""


if not (Path(get_bids_dir()) / "dataset_description.json").is_file():
    rule bids_init_dataset:
        """Initialize root BIDS directory and create dataset metadata files."""
        output:
            dataset_description=get_bids_dir() + "/dataset_description.json",
            readme=get_bids_dir() + "/README",
            bidsignore=get_bids_dir() + "/.bidsignore",
        params:
            python_bin=sys.executable,
            scripts_dir=get_scripts_dir(),
            bids_dir=get_bids_dir(),
            name=(config.get("dataset_description") or {}).get(
                "Name", "tES-FUS Multimodal Neuroimaging Study"
            ),
            bids_version=(config.get("dataset_description") or {}).get(
                "BIDSVersion", "1.9.0"
            ),
            license=(config.get("dataset_description") or {}).get("License", "CC0"),
        shell:
            """
            {params.python_bin} "{params.scripts_dir}"/bids_init.py \
                --bids-dir "{params.bids_dir}" \
                --name "{params.name}" \
                --bids-version "{params.bids_version}" \
                --license "{params.license}"
            """


if is_raw_data_bids():
    checkpoint organize_bids_subject:
        """Skip dcm2niix conversion and mark existing BIDS subject as organized."""
        input:
            subject_dir=get_raw_subject_dir,
        output:
            bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        shell:
            """
            mkdir -p "$(dirname "{output.bids_marker}")"
            touch "{output.bids_marker}"
            """
else:
    rule dcm2niix_convert_raw:
        """Convert raw subject DICOM directory to NIfTI using dcm2niix."""
        input:
            raw_dir=get_raw_dir_by_name,
        output:
            converted_marker=get_work_dir() + "/raw/{raw_subject}/dcm2niix/.converted",
        params:
            python_bin=sys.executable,
            env_cmd=get_tool_env_cmd("dcm2niix"),
            executable=get_tool_executable("dcm2niix", "dcm2niix"),
            out_dir=get_work_dir() + "/raw/{raw_subject}/dcm2niix",
            tmp_dir=get_tmp_dir(),
            args=(config.get("dcm2niix") or {}).get("args", "-z y -b y -ba y -f %p_%s"),
        threads: get_bids_threads()
        shell:
            """
            {params.env_cmd}
            mkdir -p "{params.tmp_dir}" "{params.out_dir}"
            rc=0
            TMPDIR="{params.tmp_dir}" {params.executable} {params.args} -o "{params.out_dir}" "{input.raw_dir}" || rc=$?
            if [ $rc -ne 0 ] && [ $rc -ne 8 ]; then
                exit $rc
            fi
            touch "{output.converted_marker}"
            """

    rule organize_bids_raw:
        """Organize converted NIfTI/JSON files for a raw directory into standard BIDS structure."""
        input:
            dataset_desc=get_bids_dir() + "/dataset_description.json",
            converted_marker=get_work_dir() + "/raw/{raw_subject}/dcm2niix/.converted",
        output:
            raw_marker=get_work_dir() + "/raw/{raw_subject}/.bids_organized",
        params:
            python_bin=sys.executable,
            scripts_dir=get_scripts_dir(),
            input_dir=get_work_dir() + "/raw/{raw_subject}/dcm2niix",
            bids_dir=get_bids_dir(),
            subject=lambda wildcards: get_bids_subject_label(wildcards.raw_subject),
            session_arg=lambda wildcards: (
                f"--session {get_raw_session_label(wildcards.raw_subject)}"
                if get_raw_session_label(wildcards.raw_subject)
                else ""
            ),
            min_bold_volumes=(config.get("bids") or {}).get("min_bold_volumes", 15),
        threads: get_bids_threads()
        shell:
            """
            {params.python_bin} "{params.scripts_dir}"/bids_organizer.py \
                --input-dir "{params.input_dir}" \
                --bids-dir "{params.bids_dir}" \
                --subject "{params.subject}" \
                {params.session_arg} \
                --min-bold-volumes {params.min_bold_volumes} \
                --output-marker "{output.raw_marker}"
            """

    checkpoint organize_bids_subject:
        """Mark BIDS subject as organized once all its raw session directories are processed."""
        input:
            dataset_desc=get_bids_dir() + "/dataset_description.json",
            raw_markers=get_organized_raw_markers_for_subject,
        output:
            bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        shell:
            """
            mkdir -p "$(dirname "{output.bids_marker}")"
            touch "{output.bids_marker}"
            """
