"""Snakemake rules for FastSurfer structural whole-brain segmentation and surface reconstruction."""

from pathlib import Path


rule fastsurfer_subject:
    """Run FastSurfer deep-learning whole-brain segmentation and surface reconstruction on T1w image."""
    input:
        bids_marker="bids/sub-{subject}/.bids_organized",
        t1w=get_t1w_image,
    output:
        marker="derivatives/fastsurfer/sub-{subject}/.fastsurfer_complete",
        seg="derivatives/fastsurfer/sub-{subject}/mri/aparc.DKTatlas+aseg.deep.mgz",
    params:
        sd=get_fastsurfer_dir(),
        sid="sub-{subject}",
        device=get_fastsurfer_device(),
        fs_license=get_fastsurfer_license(),
        extra_args=get_fastsurfer_extra_args(),
        tmp_dir=get_tmp_dir(),
    threads: 2
    shell:
        """
        module load fastsurfer 2>/dev/null || true
        python workflow/scripts/fastsurfer_helper.py \
            --t1 "{input.t1w}" \
            --sd "{params.sd}" \
            --sid "{params.sid}" \
            --threads {threads} \
            --device "{params.device}" \
            --fs-license "{params.fs_license}" \
            --extra-args "{params.extra_args}" \
            --marker "{output.marker}" \
            --tmp-dir "{params.tmp_dir}"
        """
