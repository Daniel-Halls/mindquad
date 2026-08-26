"""Snakemake rules for QSIPrep diffusion MRI preprocessing."""

from pathlib import Path


rule qsiprep_participant:
    """Run QSIPrep diffusion MRI preprocessing integrating FastSurfer outputs."""
    input:
        bids_dataset=get_bids_dir() + "/dataset_description.json",
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        fastsurfer_marker=(
            get_fastsurfer_dir() + "/sub-{subject}/.fastsurfer_complete"
        ),
    output:
        marker=get_qsiprep_dir() + "/sub-{subject}/.qsiprep_complete",
        report=get_qsiprep_dir() + "/sub-{subject}.html",
    params:
        scripts_dir=get_scripts_dir(),
        bids_dir=get_bids_dir(),
        out_dir=get_qsiprep_dir(),
        subject="{subject}",
        fs_subjects_dir=get_qsiprep_fs_subjects_dir(),
        fs_license=get_qsiprep_fs_license(),
        output_resolution=get_qsiprep_output_resolution(),
        denoise_method=get_qsiprep_denoise_method(),
        unringing_method=get_qsiprep_unringing_method(),
        mem_mb=get_qsiprep_mem_mb(),
        work_dir=lambda wildcards: str(
            Path(get_work_dir()) / "qsiprep" / f"sub-{wildcards.subject}"
        ),
        tmp_dir=get_tmp_dir(),
        extra_args=get_qsiprep_extra_args(),
    threads: 2
    shell:
        """
        python "{params.scripts_dir}"/qsiprep_helper.py \
            --bids-dir "{params.bids_dir}" \
            --output-dir "{params.out_dir}" \
            --subject "{params.subject}" \
            --fs-subjects-dir "{params.fs_subjects_dir}" \
            --fs-license "{params.fs_license}" \
            --output-resolution {params.output_resolution} \
            --denoise-method "{params.denoise_method}" \
            --unringing-method "{params.unringing_method}" \
            --mem-mb {params.mem_mb} \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --threads {threads} \
            --marker "{output.marker}" \
            --report "{output.report}" \
            --extra-args "{params.extra_args}"
        """
