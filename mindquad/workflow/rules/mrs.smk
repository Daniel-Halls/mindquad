"""Snakemake rules for Magnetic Resonance Spectroscopy (MRS) processing with FSL-MRS."""

from pathlib import Path


rule mrs_processing:
    """Run MRS preprocessing, voxel tissue segmentation, and spectral fitting with FSL-MRS."""
    input:
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
    output:
        marker=get_mrs_dir() + "/sub-{subject}/.mrs_complete",
        report=get_mrs_dir() + "/sub-{subject}.html",
        quantities=get_mrs_dir() + "/sub-{subject}/quantities.csv",
    params:
        svs=get_mrs_svs_image,
        t1w=get_t1w_image,
        out_dir=lambda wildcards: str(
            Path(get_mrs_dir()) / f"sub-{wildcards.subject}"
        ),
        subject="sub-{subject}",
        water_ref=get_mrs_water_ref_image,
        basis=get_mrs_basis(),
        fit_algo=get_mrs_fit_algorithm(),
        ppm_min=get_mrs_ppm_min(),
        ppm_max=get_mrs_ppm_max(),
        baseline_order=get_mrs_baseline_order(),
        internal_ref=get_mrs_internal_reference(),
        work_dir=lambda wildcards: str(
            Path(get_work_dir()) / "mrs" / f"sub-{wildcards.subject}"
        ),
        tmp_dir=get_tmp_dir(),
        extra_args=get_mrs_extra_args(),
    threads: 2
    shell:
        """
        python workflow/scripts/mrs_helper.py \
            --data "{params.svs}" \
            --t1 "{params.t1w}" \
            --reference "{params.water_ref}" \
            --output-dir "{params.out_dir}" \
            --subject "{params.subject}" \
            --basis "{params.basis}" \
            --fit-algo "{params.fit_algo}" \
            --ppm-min {params.ppm_min} \
            --ppm-max {params.ppm_max} \
            --baseline-order {params.baseline_order} \
            --internal-ref "{params.internal_ref}" \
            --threads {threads} \
            --work-dir "{params.work_dir}" \
            --tmp-dir "{params.tmp_dir}" \
            --marker "{output.marker}" \
            --report "{output.report}" \
            --summary-csv "{output.quantities}" \
            --extra-args "{params.extra_args}"
        """
