"""Common helper functions and configuration parsing for Mindquad pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Tuple


class StudyCohort:
    """Cohort manager to discover and resolve subjects in study."""

    def __init__(self, pipeline_config: Dict[str, Any]) -> None:
        """Initialize StudyCohort with workflow configuration dictionary.

        Args:
            pipeline_config: Dictionary containing pipeline configuration.
        """
        self._config = pipeline_config

    @property
    def raw_data_dir(self) -> Path:
        """Return the raw data directory path."""
        return Path(str(self._config.get("raw_data_dir")))

    @property
    def subjects(self) -> List[str]:
        """Retrieve list of subject folder names from config or filesystem.

        Returns:
            List of raw subject directory identifiers.
        """
        configured_subjects = self._config.get("subjects")
        if configured_subjects:
            return [str(s) for s in configured_subjects]

        if self.raw_data_dir.exists():
            found = [
                p.name
                for p in self.raw_data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            ]
            if found:
                return sorted(found)
        raise FileNotFoundError(f"No subjects found in {self.raw_data_dir}")

    def get_bids_subject_label(self, raw_subject: str) -> str:
        """Map raw subject directory name to sanitized BIDS subject label.

        Args:
            raw_subject: Raw subject directory name.

        Returns:
            Sanitized BIDS subject label (without 'sub-' prefix).
        """
        mapping: Dict[str, Any] = self._config.get("subject_mapping", {})
        mapped = mapping.get(raw_subject, raw_subject)
        clean_label = str(mapped).replace("-", "").replace("_", "")
        return clean_label

    @property
    def bids_subjects(self) -> List[str]:
        """Return list of all BIDS subject labels (without sub- prefix).

        Returns:
            List of sanitized BIDS subject identifiers.
        """
        return [self.get_bids_subject_label(s) for s in self.subjects]


def get_raw_data_dir() -> str:
    """Return the raw data directory from configuration."""
    cohort = StudyCohort(config)
    return str(cohort.raw_data_dir)


def get_output_dir() -> str:
    """Return the global output directory prefix from configuration."""
    return str(Path(config.get("output_dir", ".")).resolve())


def get_scripts_dir() -> str:
    """Return the absolute path to the workflow scripts directory."""
    return str(Path(workflow.basedir) / "scripts")


def get_bids_dir() -> str:
    """Return the BIDS dataset root directory from configuration."""
    return str(Path(get_output_dir()) / config.get("bids_dir", "bids"))


def get_derivatives_dir() -> str:
    """Return the root derivatives directory from configuration."""
    return str(Path(get_output_dir()).resolve() / config.get("derivatives_dir", "derivatives"))


def get_mriqc_dir() -> str:
    """Return the MRIQC derivatives output directory path."""
    return str(Path(get_derivatives_dir()).resolve() / "mriqc")


def get_work_dir() -> str:
    """Return the intermediate working directory from configuration."""
    return str(Path(get_output_dir()) / config.get("work_dir", "work"))


def get_tmp_dir() -> str:
    """Return the project-local temporary directory path."""
    return str(Path(get_output_dir()) / config.get("tmp_dir", ".tmp"))


def get_mriqc_modalities() -> str:
    """Return space-separated list of MRIQC modalities from config."""
    modalities = config.get("mriqc", {}).get(
        "modalities", ["T1w", "bold"]
    )
    if isinstance(modalities, str):
        return modalities
    return " ".join(modalities)


def get_mriqc_threads() -> int:
    """Return configured MRIQC thread count."""
    return int(config.get("mriqc", {}).get("threads", 2))


def get_bids_threads() -> int:
    """Return configured BIDS setup thread count."""
    return int(config.get("bids", {}).get("threads", 2))


def get_mriqc_extra_args() -> str:
    """Return extra CLI flags for MRIQC from config."""
    default_args = "--verbose-reports --no-sub --notrack"
    return str(config.get("mriqc", {}).get("args", default_args))


def get_subjects() -> List[str]:
    """Retrieve list of subject folder names from config or filesystem.

    Returns:
        List of subject directory identifiers.
    """
    cohort = StudyCohort(config)
    return cohort.subjects


def get_bids_subject_label(raw_subject: str) -> str:
    """Map raw subject folder name to sanitized BIDS subject label.

    Args:
        raw_subject: Raw subject directory name.

    Returns:
        Sanitized BIDS subject label (without 'sub-' prefix).
    """
    cohort = StudyCohort(config)
    return cohort.get_bids_subject_label(raw_subject)


def get_bids_subjects() -> List[str]:
    """Retrieve list of all sanitized BIDS subject labels.

    Returns:
        List of BIDS subject labels without 'sub-' prefix.
    """
    cohort = StudyCohort(config)
    return cohort.bids_subjects


def get_raw_subject_dir(wildcards: Any) -> str:
    """Resolve raw DICOM directory path for a subject wildcard.

    Args:
        wildcards: Snakemake rule wildcards containing 'subject'.

    Returns:
        Path to subject's raw DICOM directory.
    """
    cohort = StudyCohort(config)
    subject_wildcard = wildcards.subject
    mapping: Dict[str, Any] = config.get("subject_mapping", {})
    reverse_mapping = {
        cohort.get_bids_subject_label(k): k for k in mapping.keys()
    }
    raw_name = reverse_mapping.get(subject_wildcard, subject_wildcard)
    return str(cohort.raw_data_dir / raw_name)


def get_fastsurfer_dir() -> str:
    """Return the FastSurfer derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "fastsurfer")


def get_fastsurfer_threads() -> int:
    """Return configured FastSurfer thread count capped at 2."""
    configured_threads = int(config.get("fastsurfer", {}).get("threads", 2))
    return configured_threads


def get_fastsurfer_device() -> str:
    """Return configured computing device for FastSurfer."""
    return str(config.get("fastsurfer", {}).get("device", "cpu"))


def get_fastsurfer_license() -> str:
    """Return configured FreeSurfer license file path if provided."""
    return str(config.get("fastsurfer", {}).get("fs_license", ""))


def get_fastsurfer_extra_args() -> str:
    """Return extra CLI flags for FastSurfer from configuration."""
    return str(config.get("fastsurfer", {}).get("args", ""))


def get_t1w_image(wildcards: Any) -> str:
    """Resolve T1w anatomical image path for a given subject wildcard.

    Args:
        wildcards: Snakemake wildcards containing 'subject'.

    Returns:
        Path to structural T1w NIfTI image.
    """
    subject_label = str(wildcards.subject).replace("sub-", "").strip()
    bids_root = Path(get_bids_dir())
    anat_dir = bids_root / f"sub-{subject_label}" / "anat"
    standard_t1 = anat_dir / f"sub-{subject_label}_T1w.nii.gz"

    if standard_t1.exists():
        return str(standard_t1)

    if anat_dir.exists():
        for pattern in ["*T1w*.nii.gz", "*T1w*.nii"]:
            matches = sorted(anat_dir.glob(pattern))
            if matches:
                return str(matches[0])

    return str(standard_t1)


def get_t2w_image(wildcards: Any) -> str:
    """Resolve T2w (or FLAIR) anatomical image path for a given subject wildcard.

    Args:
        wildcards: Snakemake wildcards containing 'subject'.

    Returns:
        Path to structural T2w or FLAIR NIfTI image.
    """
    subject_label = str(wildcards.subject).replace("sub-", "").strip()
    bids_root = Path(get_bids_dir())
    anat_dir = bids_root / f"sub-{subject_label}" / "anat"
    standard_t2 = anat_dir / f"sub-{subject_label}_T2w.nii.gz"
    standard_flair = anat_dir / f"sub-{subject_label}_FLAIR.nii.gz"

    if standard_t2.exists():
        return str(standard_t2)
    if standard_flair.exists():
        return str(standard_flair)

    if anat_dir.exists():
        for f in sorted(anat_dir.iterdir()):
            if "t2w" in f.name.lower() and f.name.endswith((".nii", ".nii.gz")):
                return str(f)
        for f in sorted(anat_dir.iterdir()):
            if "flair" in f.name.lower() and f.name.endswith((".nii", ".nii.gz")):
                return str(f)

    return str(standard_t2)


def get_fmriprep_dir() -> str:
    """Return the fMRIPrep derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "fmriprep")


def get_fmriprep_threads() -> int:
    """Return configured fMRIPrep thread count capped at 2."""
    configured_threads = int(config.get("fmriprep", {}).get("threads", 2))
    return configured_threads


def get_fmriprep_mem_mb() -> int:
    """Return configured fMRIPrep memory limit in MB."""
    return int(config.get("fmriprep", {}).get("mem_mb", 8000))


def get_fmriprep_output_spaces() -> str:
    """Return space-separated string of fMRIPrep output spaces from config."""
    spaces = config.get("fmriprep", {}).get(
        "output_spaces", ["MNI152NLin2009cAsym:res-2", "fsaverage5"]
    )
    if isinstance(spaces, list):
        return " ".join(spaces)
    return str(spaces)


def get_fmriprep_cifti_output() -> str:
    """Return configured CIFTI output resolution for fMRIPrep."""
    return str(config.get("fmriprep", {}).get("cifti_output", "91k"))


def get_fmriprep_fs_subjects_dir() -> str:
    """Return FreeSurfer/FastSurfer subjects directory for fMRIPrep."""
    fs_dir = config.get("fmriprep", {}).get("fs_subjects_dir", get_fastsurfer_dir())
    if not __import__("pathlib").Path(fs_dir).is_absolute():
        fs_dir = str(__import__("pathlib").Path(get_output_dir()) / fs_dir)
    return str(__import__("pathlib").Path(fs_dir).resolve())


def get_fmriprep_fs_license() -> str:
    """Return FreeSurfer license file path for fMRIPrep."""
    return str(
        config.get("fmriprep", {}).get("fs_license", get_fastsurfer_license())
    )


def get_fmriprep_extra_args() -> str:
    """Return extra CLI flags for fMRIPrep from configuration."""
    default_args = "--skip-bids-validation --notrack"
    return str(config.get("fmriprep", {}).get("args", default_args))


def get_qsiprep_dir() -> str:
    """Return the QSIPrep derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "qsiprep")


def get_qsiprep_threads() -> int:
    """Return configured QSIPrep thread count capped at 2."""
    configured_threads = int(config.get("qsiprep", {}).get("threads", 2))
    return configured_threads


def get_qsiprep_mem_mb() -> int:
    """Return configured QSIPrep memory limit in MB."""
    return int(config.get("qsiprep", {}).get("mem_mb", 8000))


def get_qsiprep_output_resolution() -> str:
    """Return configured QSIPrep output DWI resolution."""
    return str(config.get("qsiprep", {}).get("output_resolution", "1.5"))


def get_qsiprep_denoise_method() -> str:
    """Return configured QSIPrep DWI denoising method."""
    return str(config.get("qsiprep", {}).get("denoise_method", "dwidenoise"))


def get_qsiprep_unringing_method() -> str:
    """Return configured QSIPrep Gibbs unringing method."""
    return str(config.get("qsiprep", {}).get("unringing_method", "mrdegibbs"))


def get_qsiprep_separate_all_dwis() -> bool:
    """Return True if QSIPrep should process DWI runs separately."""
    return bool(config.get("qsiprep", {}).get("separate_all_dwis", False))


def get_qsiprep_fs_subjects_dir() -> str:
    """Return FreeSurfer/FastSurfer subjects directory for QSIPrep."""
    fs_dir = config.get("qsiprep", {}).get("fs_subjects_dir", get_fastsurfer_dir())
    if not __import__("pathlib").Path(fs_dir).is_absolute():
        fs_dir = str(__import__("pathlib").Path(get_output_dir()) / fs_dir)
    return str(__import__("pathlib").Path(fs_dir).resolve())


def get_qsiprep_fs_license() -> str:
    """Return FreeSurfer license file path for QSIPrep."""
    return str(
        config.get("qsiprep", {}).get("fs_license", get_fastsurfer_license())
    )


def get_qsiprep_eddy_config() -> str:
    """Return path to QSIPrep eddy parameters JSON file."""
    # First check if explicitly defined in config
    configured = config.get("qsiprep", {}).get("eddy_config")
    if configured:
        return str(configured)
    
    # Otherwise try to locate it in the workflow config_files directory
    default_path = Path(workflow.basedir) / "config_files" / "eddy_params.json"
    if default_path.exists():
        return str(default_path)
    return ""


def get_qsiprep_extra_args() -> str:
    """Return extra CLI flags for QSIPrep from configuration."""
    default_args = "--skip-bids-validation --notrack"
    return str(config.get("qsiprep", {}).get("args", default_args))

def get_coregistration_dir() -> str:
    """Return the Coregistration derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "coregistration")


def get_coregistration_threads() -> int:
    """Return configured Coregistration thread count capped at 2."""
    configured_threads = int(
        config.get("coregistration", {}).get("threads", 2)
    )
    return configured_threads


def get_coregistration_tool() -> str:
    """Return configured coregistration backend tool ('dipy' or 'ants')."""
    return str(config.get("coregistration", {}).get("tool", "dipy"))


def get_coregistration_metric() -> str:
    """Return configured similarity metric for diffeomorphic registration."""
    return str(config.get("coregistration", {}).get("metric", "CC"))


def get_coregistration_transform_type() -> str:
    """Return configured transformation model for coregistration."""
    return str(config.get("coregistration", {}).get("transform_type", "syn"))


def get_coregistration_step_length() -> float:
    """Return configured optimization step length for coregistration."""
    return float(config.get("coregistration", {}).get("step_length", 0.25))


def get_coregistration_extra_args() -> str:
    """Return extra CLI flags for coregistration from configuration."""
    return str(config.get("coregistration", {}).get("args", ""))


def get_coregistered_t2w_image(wildcards: Any) -> str:
    """Resolve coregistered T2w anatomical image path for a given subject wildcard.

    Args:
        wildcards: Snakemake wildcards containing 'subject'.

    Returns:
        Path to coregistered T2w NIfTI image or raw T2w image fallback.
    """
    subject_label = str(wildcards.subject).replace("sub-", "").strip()
    coreg_dir = Path(get_coregistration_dir()) / f"sub-{subject_label}" / "anat"
    warped_t2 = (
        coreg_dir / f"sub-{subject_label}_space-T1w_desc-coreg_T2w.nii.gz"
    )
    if warped_t2.exists():
        return str(warped_t2)
    return str(get_t2w_image(wildcards))


def get_hcp_dir() -> str:
    """Return the HCP derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "hcp")


def get_hcp_threads() -> int:
    """Return configured HCP processing thread count capped at 2."""
    configured_threads = int(config.get("hcp", {}).get("threads", 2))
    return configured_threads


def get_hcp_processing_mode() -> str:
    """Return configured HCP processing mode ('HCPStyleData' or 'LegacyStyleData')."""
    return str(config.get("hcp", {}).get("processing_mode", "HCPStyleData"))


def get_hcp_reg_name() -> str:
    """Return configured HCP surface registration name ('MSMSulc' or 'FS')."""
    return str(config.get("hcp", {}).get("reg_name", "MSMSulc"))


def get_hcp_grayordinates_res() -> int:
    """Return configured HCP grayordinates resolution in mm."""
    return int(config.get("hcp", {}).get("grayordinates_res", 2))


def get_hcp_hires_mesh() -> int:
    """Return configured high-resolution standard mesh vertex count in k."""
    return int(config.get("hcp", {}).get("hires_mesh", 164))


def get_hcp_low_res_mesh() -> int:
    """Return configured low-resolution standard mesh vertex count in k."""
    return int(config.get("hcp", {}).get("low_res_mesh", 32))


def get_hcp_thickness_regression() -> str:
    """Return configured HCP thickness regression method ('BOTH', 'OLD', 'NEW')."""
    return str(config.get("hcp", {}).get("thickness_regression", "BOTH"))


def get_hcp_surf_atlas_dir() -> str:
    """Return configured surface atlas templates directory path if set."""
    return str(config.get("hcp", {}).get("surf_atlas_dir", ""))


def get_hcp_grayordinates_dir() -> str:
    """Return configured grayordinates templates directory path if set."""
    return str(config.get("hcp", {}).get("grayordinates_dir", ""))


def get_hcp_subcort_gray_labels() -> str:
    """Return configured FreeSurfer subcortical gray label LUT path if set."""
    return str(config.get("hcp", {}).get("subcort_gray_labels", ""))


def get_hcp_freesurfer_labels() -> str:
    """Return configured FreeSurfer all labels LUT path if set."""
    return str(config.get("hcp", {}).get("freesurfer_labels", ""))


def get_hcp_ref_myelin_maps() -> str:
    """Return configured group myelin reference map path if set."""
    return str(config.get("hcp", {}).get("ref_myelin_maps", ""))


def get_hcp_extra_args() -> str:
    """Return extra CLI flags for HCP PostFreeSurfer from configuration."""
    return str(config.get("hcp", {}).get("args", ""))


def get_mrs_dir() -> str:
    """Return the MRS derivatives output directory path."""
    return str(Path(get_derivatives_dir()) / "mrs")


def get_mrs_threads() -> int:
    """Return configured MRS thread count capped at 2."""
    configured_threads = int(config.get("mrs", {}).get("threads", 2))
    return configured_threads


def get_mrs_basis() -> str:
    """Return configured basis set path for MRS fitting."""
    return str(config.get("mrs", {}).get("basis", ""))


def get_mrs_fit_algorithm() -> str:
    """Return configured MRS fitting algorithm ('Newton' or 'MH')."""
    return str(config.get("mrs", {}).get("fit_algorithm", "Newton"))


def get_mrs_ppm_range() -> str:
    """Return space-separated min and max ppm range for MRS fitting."""
    ppm = config.get("mrs", {}).get("ppm_range", [0.2, 4.2])
    if isinstance(ppm, (list, tuple)) and len(ppm) == 2:
        return f"{ppm[0]} {ppm[1]}"
    return str(ppm)


def get_mrs_ppm_min() -> float:
    """Return lower bound ppm value for MRS fitting."""
    ppm = config.get("mrs", {}).get("ppm_range", [0.2, 4.2])
    if isinstance(ppm, (list, tuple)) and len(ppm) >= 1:
        return float(ppm[0])
    return 0.2


def get_mrs_ppm_max() -> float:
    """Return upper bound ppm value for MRS fitting."""
    ppm = config.get("mrs", {}).get("ppm_range", [0.2, 4.2])
    if isinstance(ppm, (list, tuple)) and len(ppm) >= 2:
        return float(ppm[1])
    return 4.2


def get_mrs_baseline_order() -> int:
    """Return configured polynomial baseline order for MRS fitting."""
    return int(config.get("mrs", {}).get("baseline_order", 2))


def get_mrs_internal_reference() -> str:
    """Return internal reference metabolite name (e.g. 'Cr', 'tCr')."""
    return str(config.get("mrs", {}).get("internal_reference", "Cr"))


def get_mrs_extra_args() -> str:
    """Return extra CLI flags for MRS processing from configuration."""
    return str(config.get("mrs", {}).get("args", ""))


def get_mrs_svs_image(wildcards: Any) -> str:
    """Resolve MRS SVS NIfTI image path for a given subject wildcard.

    Args:
        wildcards: Snakemake wildcards containing 'subject'.

    Returns:
        Path to SVS NIfTI image file.
    """
    subject_label = str(wildcards.subject).replace("sub-", "").strip()
    bids_root = Path(get_bids_dir())
    mrs_dir = bids_root / f"sub-{subject_label}" / "mrs"
    standard_svs = mrs_dir / f"sub-{subject_label}_svs.nii.gz"

    if standard_svs.exists():
        return str(standard_svs)

    if mrs_dir.exists():
        for pattern in [
            "*svs*.nii.gz",
            "*svs*.nii",
            "*mrs*.nii.gz",
            "*mrs*.nii",
        ]:
            matches = sorted(mrs_dir.glob(pattern))
            if matches:
                return str(matches[0])

    return str(standard_svs)


def get_mrs_water_ref_image(wildcards: Any) -> str:
    """Resolve MRS water reference NIfTI image path for a given subject wildcard.

    Args:
        wildcards: Snakemake wildcards containing 'subject'.

    Returns:
        Path to water reference NIfTI image or empty string if not found.
    """
    subject_label = str(wildcards.subject).replace("sub-", "").strip()
    bids_root = Path(get_bids_dir())
    mrs_dir = bids_root / f"sub-{subject_label}" / "mrs"
    standard_ref = mrs_dir / f"sub-{subject_label}_ref.nii.gz"

    if standard_ref.exists():
        return str(standard_ref)

    if mrs_dir.exists():
        for pattern in [
            "*ref*.nii.gz",
            "*ref*.nii",
            "*water*.nii.gz",
            "*wref*.nii.gz",
            "*h2o*.nii.gz",
        ]:
            matches = sorted(mrs_dir.glob(pattern))
            if matches:
                return str(matches[0])

    return ""




def get_root_mounts(*paths: str) -> str:
    """Detect and return unique root directories from a list of paths for Singularity bindings."""
    roots = set()
    for p in paths:
        path = Path(p).resolve()
        if len(path.parts) > 1:
            roots.add(f"/{path.parts[1]}")
    if not roots:
        return ""
    return ",".join(f"{r}:{r}" for r in roots)

def _extract_load_components(load_val: Any) -> Tuple[str, str]:
    """Extract (module_string, file_path) from a load configuration value, supporting .sif and .sh."""
    if not load_val:
        return "", ""
        
    if isinstance(load_val, list):
        modules = []
        file_path = ""
        for item in load_val:
            s_item = str(item).strip()
            if s_item.endswith(".sif") or s_item.endswith(".sh"):
                file_path = s_item
            else:
                modules.append(s_item)
        return " ".join(modules), file_path
        
    load_str = str(load_val)
    if (load_str.endswith(".sif") or load_str.endswith(".sh")) and " " not in load_str.strip():
        return "", load_str.strip()
        
    import re
    match = re.search(r'(\S+\.(?:sif|sh))', load_str)
    if match:
        file_path = match.group(1)
        module_str = load_str.replace(file_path, "").strip()
        return module_str, file_path
        
    return load_str, ""

def get_tool_env_cmd(tool_name: str) -> str:
    """Return environment preparation command (e.g., module load) for a tool."""
    tool_cfg = config.get(tool_name, {})
    load_val = tool_cfg.get("load") or tool_cfg.get("module")
    
    module_str, sif_path = _extract_load_components(load_val)
    
    cmd_parts = []
    
    if sif_path and sif_path.endswith(".sif"):
        container_load = config.get("container", {}).get("load")
        if container_load:
            cmd_parts.append(f"module load {container_load} 2>/dev/null || true")
            
    if module_str:
        cleaned_str = module_str.strip()
        if cleaned_str.startswith("source "):
            cmd_parts.append(f"{cleaned_str} 2>/dev/null || true")
        else:
            if cleaned_str.startswith("module load "):
                cleaned_str = cleaned_str[len("module load "):].strip()
            for m in cleaned_str.split():
                cmd_parts.append(f"module load {m} 2>/dev/null || true")
        
    if not cmd_parts:
        return "true;"
        
    return "; ".join(cmd_parts) + ";"

def get_tool_executable(tool_name: str, default_bin: str) -> str:
    """Return the executable string, auto-wrapping in container engine if a .sif is provided,
    or returning custom script/executable if configured."""
    tool_cfg = config.get(tool_name, {})
    custom_exe = tool_cfg.get("executable") or tool_cfg.get("script") or tool_cfg.get("bin")
    if custom_exe:
        return str(custom_exe)

    load_val = tool_cfg.get("load") or tool_cfg.get("sif")
    _, target_path = _extract_load_components(load_val)
    
    if target_path:
        if target_path.endswith(".sif"):
            engine_cmd = config.get("container", {}).get("command", "singularity")
            out_dir = Path(get_output_dir()).resolve()
            bids_dir = Path(get_bids_dir()).resolve()
            
            mount_paths = [str(out_dir), str(bids_dir)]
            if tool_name == "qsiprep":
                mount_paths.append(get_qsiprep_eddy_config())
                
            binds = get_root_mounts(*mount_paths)
            bind_arg = f"-B {binds}" if binds else ""
            nv_arg = " --nv" if tool_name == "qsiprep" else ""
            return f"{engine_cmd} run --cleanenv{nv_arg} {bind_arg} {target_path}"
        elif target_path.endswith(".sh") or Path(target_path).is_file():
            return target_path
            
    return default_bin
