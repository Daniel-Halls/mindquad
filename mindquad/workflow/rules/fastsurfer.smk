"""Snakemake rules for FastSurfer structural whole-brain segmentation and surface reconstruction."""

from pathlib import Path


rule fastsurfer_subject:
    """Run FastSurfer deep-learning whole-brain segmentation and surface reconstruction on T1w image."""
    input:
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        mriqc_group_marker=get_mriqc_dir() + "/.mriqc_group_complete",
    output:
        marker=get_fastsurfer_dir() + "/sub-{subject}/.fastsurfer_complete",
        seg=get_fastsurfer_dir() + "/sub-{subject}/mri/aparc.DKTatlas+aseg.deep.mgz",
    params:
        scripts_dir=get_scripts_dir(),
        t1w=get_t1w_image,
        sd=get_fastsurfer_dir(),
        sid="sub-{subject}",
        device=get_fastsurfer_device(),
        fs_license=get_fastsurfer_license(),
        extra_args=get_fastsurfer_extra_args(),
        tmp_dir=get_tmp_dir(),
    threads: 2
    shell:
        """
        module load extension/imaging fastsurfer-img 2>/dev/null || true
        python "{params.scripts_dir}"/fastsurfer_helper.py \
            --t1 "{params.t1w}" \
            --sd "{params.sd}" \
            --sid "{params.sid}" \
            --threads {threads} \
            --device "{params.device}" \
            --fs-license "{params.fs_license}" \
            --extra-args "{params.extra_args}" \
            --marker "{output.marker}" \
            --tmp-dir "{params.tmp_dir}"
        """
