"""Snakemake rules for MRIQC participant and group level quality control."""

from pathlib import Path


rule mriqc_participant:
    """Run MRIQC participant-level quality control on T1w, bold, and dwi modalities."""
    input:
        bids_dataset="bids/dataset_description.json",
        bids_marker="bids/sub-{subject}/.bids_organized",
    output:
        marker="derivatives/mriqc/sub-{subject}/.mriqc_complete",
        report="derivatives/mriqc/sub-{subject}.html",
    params:
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
        python workflow/scripts/mriqc_helper.py participant \
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
            "derivatives/mriqc/sub-{subject}/.mriqc_complete",
            subject=get_bids_subjects(),
        ),
    output:
        group_marker="derivatives/mriqc/.mriqc_group_complete",
    params:
        bids_dir=get_bids_dir(),
        out_dir=get_mriqc_dir(),
        modalities=get_mriqc_modalities(),
        work_dir=str(Path(get_work_dir()) / "mriqc" / "group"),
        tmp_dir=get_tmp_dir(),
        extra_args=get_mriqc_extra_args(),
    threads: 2
    shell:
        """
        python workflow/scripts/mriqc_helper.py group \
            --bids-dir "{params.bids_dir}" \
            --output-dir "{params.out_dir}" \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --threads {threads} \
            --modalities {params.modalities} \
            --marker "{output.group_marker}" \
            --extra-args "{params.extra_args}"
        """
