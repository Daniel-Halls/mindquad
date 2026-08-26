"""Snakemake rules for Human Connectome Project (HCP) PostFreeSurfer pipeline."""

from pathlib import Path


rule hcp_postfreesurfer:
    """Run HCP PostFreeSurfer pipeline to generate standard surface meshes and CIFTI grayordinates."""
    input:
        bids_marker=get_bids_dir() + "/sub-{subject}/.bids_organized",
        fastsurfer_marker=(
            get_fastsurfer_dir() + "/sub-{subject}/.fastsurfer_complete"
        ),
        coreg_marker=(
            get_coregistration_dir() + "/sub-{subject}/.coregistration_complete"
        ),
        fmriprep_marker=(
            get_fmriprep_dir() + "/sub-{subject}/.fmriprep_complete"
        ),
        t1w=get_t1w_image,
        coreg_t2w=get_coregistered_t2w_image,
    output:
        marker=get_hcp_dir() + "/sub-{subject}/.hcp_complete",
        spec=(
            get_hcp_dir() + "/sub-{subject}/MNINonLinear/fsaverage_LR32k/"
            "sub-{subject}.32k_fs_LR.wb.spec"
        ),
    params:
        study_folder=get_hcp_dir(),
        subject="sub-{subject}",
        fs_dir=lambda wildcards: str(
            Path(get_fastsurfer_dir()) / f"sub-{wildcards.subject}"
        ),
        processing_mode=get_hcp_processing_mode(),
        reg_name=get_hcp_reg_name(),
        grayordinates_res=get_hcp_grayordinates_res(),
        hires_mesh=get_hcp_hires_mesh(),
        low_res_mesh=get_hcp_low_res_mesh(),
        thickness_regression=get_hcp_thickness_regression(),
        surf_atlas_dir=get_hcp_surf_atlas_dir(),
        grayordinates_dir=get_hcp_grayordinates_dir(),
        subcort_gray_labels=get_hcp_subcort_gray_labels(),
        freesurfer_labels=get_hcp_freesurfer_labels(),
        ref_myelin_maps=get_hcp_ref_myelin_maps(),
        tmp_dir=get_tmp_dir(),
        extra_args=get_hcp_extra_args(),
    threads: 2
    shell:
        """
        module load hcppipelines 2>/dev/null || true
        python workflow/scripts/hcp_helper.py \
            --study-folder "{params.study_folder}" \
            --subject "{params.subject}" \
            --fs-dir "{params.fs_dir}" \
            --t1 "{input.t1w}" \
            --t2 "{input.coreg_t2w}" \
            --processing-mode "{params.processing_mode}" \
            --reg-name "{params.reg_name}" \
            --grayordinates-res "{params.grayordinates_res}" \
            --hires-mesh "{params.hires_mesh}" \
            --low-res-mesh "{params.low_res_mesh}" \
            --thickness-regression "{params.thickness_regression}" \
            --surf-atlas-dir "{params.surf_atlas_dir}" \
            --grayordinates-dir "{params.grayordinates_dir}" \
            --subcort-gray-labels "{params.subcort_gray_labels}" \
            --freesurfer-labels "{params.freesurfer_labels}" \
            --ref-myelin-maps "{params.ref_myelin_maps}" \
            --threads {threads} \
            --tmp-dir "{params.tmp_dir}" \
            --marker "{output.marker}" \
            --spec "{output.spec}" \
            --extra-args "{params.extra_args}"
        """
