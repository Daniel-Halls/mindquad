#!/usr/bin/env python3
"""Module for computing combined metabolite SNR and appending it to FSL-MRS QC files."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_qc_data(fit_dir, qc_filename='qc.csv'):
    """Load the QC metrics table from an FSL-MRS fit directory.

    Args:
        fit_dir (str or Path): Path to the fit output directory.
        qc_filename (str, optional): Name of the QC file. Defaults to 'qc.csv'.

    Returns:
        pd.DataFrame: DataFrame containing QC metrics indexed by metabolite.

    Raises:
        FileNotFoundError: If the QC file does not exist in fit_dir.
    """
    qc_path = Path(fit_dir).expanduser().resolve() / qc_filename
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC file not found: {qc_path}")
    return pd.read_csv(qc_path, index_col=0)


def compute_combined_snr(qc_df, metabolites=('Glu', 'Gln')):
    """Compute combined root-sum-square SNR for a list of metabolites.

    Args:
        qc_df (pd.DataFrame): DataFrame containing QC metrics with 'SNR' column.
        metabolites (tuple or list of str, optional): Metabolite names to combine.
            Defaults to ('Glu', 'Gln').

    Returns:
        float: Combined SNR calculated using root-sum-square.

    Raises:
        KeyError: If 'SNR' column is not found in qc_df.
        ValueError: If any specified metabolite is not present in qc_df index.
    """
    if 'SNR' not in qc_df.columns:
        raise KeyError("Column 'SNR' was not found in QC DataFrame.")

    missing = [m for m in metabolites if m not in qc_df.index]
    if missing:
        raise ValueError(
            f"Metabolite(s) {missing} missing from QC index. "
            f"Available: {list(qc_df.index)}"
        )

    snr_values = qc_df.loc[list(metabolites), 'SNR'].astype(float).values
    return float(np.sqrt(np.sum(np.square(snr_values))))


def append_combined_snr(qc_df, combined_snr, combined_name='Glx'):
    """Append or update the combined SNR entry in the QC DataFrame.

    Args:
        qc_df (pd.DataFrame): DataFrame containing QC metrics.
        combined_snr (float): Calculated combined SNR to assign.
        combined_name (str, optional): Metabolite label for the combined entry.
            Defaults to 'Glx'.

    Returns:
        pd.DataFrame: Updated QC DataFrame with the combined entry.
    """
    qc_df.loc[combined_name, 'SNR'] = combined_snr
    return qc_df


def save_qc_data(qc_df, fit_dir, qc_filename='qc.csv'):
    """Save the updated QC DataFrame back to disk.

    Args:
        qc_df (pd.DataFrame): QC metrics DataFrame to save.
        fit_dir (str or Path): Path to the fit output directory.
        qc_filename (str, optional): Name of the QC file. Defaults to 'qc.csv'.

    Returns:
        Path: Path to the saved QC CSV file.
    """
    qc_path = Path(fit_dir).expanduser().resolve() / qc_filename
    qc_df.to_csv(qc_path)
    return qc_path


def calculate_posthoc_glx_snr(fit_dir, metabolites=('Glu', 'Gln'), combined_name='Glx', qc_filename='qc.csv'):
    """Calculate combined metabolite SNR and write the result into qc.csv.

    Orchestrates loading QC metrics, computing root-sum-square SNR for the
    requested metabolites, updating the table, and writing it back to disk.

    Args:
        fit_dir (str or Path): Path to the fit output directory containing qc.csv.
        metabolites (tuple or list of str, optional): Metabolites to combine.
            Defaults to ('Glu', 'Gln').
        combined_name (str, optional): Name of the combined metabolite entry.
            Defaults to 'Glx'.
        qc_filename (str, optional): Name of the QC file. Defaults to 'qc.csv'.

    Returns:
        float: Combined SNR calculated via root-sum-square.
    """
    qc_df = load_qc_data(fit_dir, qc_filename=qc_filename)
    combined_snr = compute_combined_snr(qc_df, metabolites=metabolites)
    qc_df = append_combined_snr(qc_df, combined_snr, combined_name=combined_name)
    save_qc_data(qc_df, fit_dir, qc_filename=qc_filename)
    return combined_snr


def parse_args():
    """Parse command-line arguments for calculating combined metabolite SNR.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate combined Glx (Glu + Gln) SNR and append to qc.csv."
    )
    parser.add_argument(
        '-f', '--fit-dir',
        dest='fit_dir',
        required=True,
        type=str,
        help="Path to FSL-MRS fit output directory containing qc.csv"
    )
    return parser.parse_args()


def main():
    """Execute CLI workflow for combined SNR calculation and QC file update.

    Returns:
        None
    """
    args = parse_args()
    calculate_posthoc_glx_snr(fit_dir=args.fit_dir)


if __name__ == '__main__':
    main()