"""Snakemake rules for fMRIPrep functional and anatomical preprocessing."""

from pathlib import Path


rule fmriprep_participant:
    """Run fMRIPrep preprocessing integrating FastSurfer outputs."""
    input:
        bids_dataset="bids/dataset_description.json",
        bids_marker="bids/sub-{subject}/.bids_organized",
        fastsurfer_marker=(
            "derivatives/fastsurfer/sub-{subject}/.fastsurfer_complete"
        ),
    output:
        marker="derivatives/fmriprep/sub-{subject}/.fmriprep_complete",
        report="derivatives/fmriprep/sub-{subject}.html",
    params:
        bids_dir=get_bids_dir(),
        out_dir=get_fmriprep_dir(),
        subject="{subject}",
        fs_subjects_dir=get_fmriprep_fs_subjects_dir(),
        fs_license=get_fmriprep_fs_license(),
        output_spaces=get_fmriprep_output_spaces(),
        cifti_output=get_fmriprep_cifti_output(),
        mem_mb=get_fmriprep_mem_mb(),
        work_dir=lambda wildcards: str(
            Path(get_work_dir()) / "fmriprep" / f"sub-{wildcards.subject}"
        ),
        tmp_dir=get_tmp_dir(),
        extra_args=get_fmriprep_extra_args(),
    threads: 2
    shell:
        """
        module load fmriprep 2>/dev/null || true
        python workflow/scripts/fmriprep_helper.py \
            --bids-dir "{params.bids_dir}" \
            --output-dir "{params.out_dir}" \
            --subject "{params.subject}" \
            --fs-subjects-dir "{params.fs_subjects_dir}" \
            --fs-license "{params.fs_license}" \
            --output-spaces {params.output_spaces} \
            --cifti-output "{params.cifti_output}" \
            --mem-mb {params.mem_mb} \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --threads {threads} \
            --marker "{output.marker}" \
            --report "{output.report}" \
            --extra-args "{params.extra_args}"
        """
