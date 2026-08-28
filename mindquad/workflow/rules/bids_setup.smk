"""Snakemake rules for BIDS dataset initialization and DICOM to BIDS conversion."""


rule bids_init_dataset:
    """Initialize root BIDS directory and create dataset metadata files."""
    output:
        dataset_description=get_bids_dir() + "/dataset_description.json",
        readme=get_bids_dir() + "/README",
        bidsignore=get_bids_dir() + "/.bidsignore",
    params:
        scripts_dir=get_scripts_dir(),
        bids_dir=get_bids_dir(),
        name=config.get("dataset_description", {}).get(
            "Name", "tES-FUS Multimodal Neuroimaging Study"
        ),
        bids_version=config.get("dataset_description", {}).get(
            "BIDSVersion", "1.9.0"
        ),
        license=config.get("dataset_description", {}).get("License", "CC0"),
    shell:
        """
        python "{params.scripts_dir}"/bids_init.py \
            --bids-dir "{params.bids_dir}" \
            --name "{params.name}" \
            --bids-version "{params.bids_version}" \
            --license "{params.license}"
        """


rule dcm2niix_convert_subject:
    """Convert raw subject DICOM directory to NIfTI using dcm2niix."""
    input:
        raw_dir=get_raw_subject_dir,
    output:
        converted_marker=get_work_dir() + "/sub-{subject}/dcm2niix/.converted",
    params:
        out_dir=get_work_dir() + "/sub-{subject}/dcm2niix",
        tmp_dir=get_tmp_dir(),
        args=config.get("dcm2niix", {}).get("args", "-z y -b y -ba y -f %p_%s"),
    threads: get_bids_threads()
    shell:
        """
        module load dcm2niix-img 2>/dev/null || module load dcm2niix 2>/dev/null || true
        mkdir -p "{params.tmp_dir}" "{params.out_dir}"
        TMPDIR="{params.tmp_dir}" dcm2niix {params.args} -o "{params.out_dir}" "{input.raw_dir}"
        touch "{output.converted_marker}"
        """


checkpoint organize_bids_subject:
    """Organize converted NIfTI/JSON files into standard BIDS structure."""
    input:
        dataset_desc=get_bids_dir() + "/dataset_description.json",
        converted_marker=get_work_dir() + "/sub-{subject}/dcm2niix/.converted",
    output:
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
    params:
        scripts_dir=get_scripts_dir(),
        input_dir=get_work_dir() + "/sub-{subject}/dcm2niix",
        bids_dir=get_bids_dir(),
        subject="{subject}",
    threads: get_bids_threads()
    shell:
        """
        python "{params.scripts_dir}"/bids_organizer.py \
            --input-dir "{params.input_dir}" \
            --bids-dir "{params.bids_dir}" \
            --subject "{params.subject}" \
            --output-marker "{output.bids_marker}"
        """
