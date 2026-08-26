"""Snakemake rules for MRIQC participant and group level quality control."""

from pathlib import Path


rule mriqc_participant:
    """Run MRIQC participant-level quality control on T1w, bold, and dwi modalities."""
    input:
        bids_dataset=get_bids_dir() + "/dataset_description.json",
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
    output:
        marker=get_mriqc_dir() + "/sub-{subject}/.mriqc_complete",
        report=get_mriqc_dir() + "/sub-{subject}.html",
    params:
        scripts_dir=get_scripts_dir(),
        bids_dir=get_bids_dir(),
        out_dir=get_mriqc_dir(),
        subject="{subject}",
        modalities=get_mriqc_modalities(),
        work_dir=lambda wildcards: str(
            Path(get_work_dir()) / "mriqc" / f"sub-{wildcards.subject}"
        ),
        tmp_dir=get_tmp_dir(),
        extra_args=get_mriqc_extra_args(),
    threads: 2
    shell:
        """
        python "{params.scripts_dir}"/mriqc_helper.py participant \
            --bids-dir "{params.bids_dir}" \
            --output-dir "{params.out_dir}" \
            --subject "{params.subject}" \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --threads {threads} \
            --modalities {params.modalities} \
            --marker "{output.marker}" \
            --report "{output.report}" \
            --extra-args "{params.extra_args}"
        """


rule mriqc_group:
    """Run MRIQC group-level summary reports across all processed subjects."""
    input:
        participant_markers=expand(
            get_mriqc_dir() + "/sub-{subject}/.mriqc_complete",
            subject=get_bids_subjects(),
        ),
    output:
        group_marker=get_mriqc_dir() + "/.mriqc_group_complete",
    params:
        scripts_dir=get_scripts_dir(),
        bids_dir=get_bids_dir(),
        out_dir=get_mriqc_dir(),
        modalities=get_mriqc_modalities(),
        work_dir=str(Path(get_work_dir()) / "mriqc" / "group"),
        tmp_dir=get_tmp_dir(),
        extra_args=get_mriqc_extra_args(),
    threads: 2
    shell:
        """
        python "{params.scripts_dir}"/mriqc_helper.py group \
            --bids-dir "{params.bids_dir}" \
            --output-dir "{params.out_dir}" \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --threads {threads} \
            --modalities {params.modalities} \
            --marker "{output.group_marker}" \
            --extra-args "{params.extra_args}"
        """
