import sys
"""Snakemake rules for T2w to T1w diffeomorphic coregistration."""

from pathlib import Path


rule coregister_t2_to_t1:
    """Run diffeomorphic SyN alignment of T2w to T1w structural scan."""
    input:
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        fastsurfer_marker=get_fastsurfer_dir() + "/sub-{subject}/.fastsurfer_complete",
    output:
        marker=(
            get_coregistration_dir() + "/sub-{subject}/.coregistration_complete"
        ),
        warped=(
            get_coregistration_dir() + "/sub-{subject}/"
            "anat/sub-{subject}_space-T1w_desc-coreg_T2w.nii.gz"
        ),
        report=get_coregistration_dir() + "/sub-{subject}.html",
    params:
        python_bin=sys.executable,
        scripts_dir=get_scripts_dir(),
        t1w=get_t1w_image,
        t2w=get_t2w_image,
        out_dir=lambda wildcards: str(
            Path(get_coregistration_dir()) / f"sub-{wildcards.subject}"
        ),
        subject="{subject}",
        tool=get_coregistration_tool(),
        metric=get_coregistration_metric(),
        transform_type=get_coregistration_transform_type(),
        step_length=get_coregistration_step_length(),
        tmp_dir=get_tmp_dir(),
        extra_args=get_coregistration_extra_args(),
    threads: get_coregistration_threads()
    shell:
        """
        {params.env_cmd}
        {params.python_bin} "{params.scripts_dir}"/coregistration_helper.py \
            --t1 "{params.t1w}" \
            --t2 "{params.t2w}" \
            --output-dir "{params.out_dir}" \
            --subject "{params.subject}" \
            --tool "{params.tool}" \
            --metric "{params.metric}" \
            --transform-type "{params.transform_type}" \
            --step-length "{params.step_length}" \
            --threads {threads} \
            --tmp-dir "{params.tmp_dir}" \
            --warped-output "{output.warped}" \
            --marker "{output.marker}" \
            --report "{output.report}" \
            --extra-args "{params.extra_args}"
        """
