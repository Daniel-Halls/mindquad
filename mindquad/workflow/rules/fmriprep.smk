import sys
"""Snakemake rules for fMRIPrep functional and anatomical preprocessing."""

from pathlib import Path


rule fmriprep_participant:
    """Run fMRIPrep preprocessing integrating FastSurfer outputs."""
    input:
        bids_dataset=get_bids_dir() + "/dataset_description.json",
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        fastsurfer_marker=(
            get_fastsurfer_dir() + "/sub-{subject}/.fastsurfer_complete"
        ),
    output:
        marker=get_fmriprep_dir() + "/sub-{subject}/.fmriprep_complete",
        report=get_fmriprep_dir() + "/sub-{subject}.html",
    params:
        python_bin=sys.executable,
        env_cmd=get_tool_env_cmd("fmriprep"),
        executable=get_tool_executable("fmriprep", "fmriprep"),
        scripts_dir=get_scripts_dir(),
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
    threads: get_fmriprep_threads()
    shell:
        """
        {params.env_cmd}
        {params.python_bin} "{params.scripts_dir}"/fmriprep_helper.py \
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
            --extra-args "{params.extra_args}" \
            --fs-no-resume \
            --executable "{params.executable}"
        """
