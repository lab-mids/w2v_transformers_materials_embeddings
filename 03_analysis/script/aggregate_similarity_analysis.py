#!/usr/bin/env python
"""
Aggregate all per-material-system analysis CSVs into a single master file.

Usage:
    python aggregate_similarity_analysis.py \
        --analysis-dir analysis_results \
        --output analysis_results/master_analysis.csv
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def collect_analysis_files(analysis_dir: Path) -> List[Path]:
    """
    Collect all *_analysis.csv files under analysis_dir (recursively).
    """
    return sorted(analysis_dir.rglob("*_analysis.csv"))


def pretty_material_name(raw_key: str) -> str:
    """
    Turn a raw material system key like:
        "Ag_Au_Pd_Pt_Rh_material_system"
        "Cr_Mn_Fe_Co_Ni_O_700C_Per1_material_system"

    into:
        "Ag-Au-Pd-Pt-Rh"
        "Cr-Mn-Fe-Co-Ni-O-700C-Per1"

    Logic:
      1. Strip a trailing "_material_system" suffix if present.
      2. Replace underscores with hyphens.
    """
    core = raw_key
    if core.endswith("_material_system"):
        core = core[: -len("_material_system")]
    return core.replace("_", "-")


def load_and_annotate(path: Path, analysis_dir: Path) -> pd.DataFrame:
    """
    Load a single analysis CSV and add minimal material-system metadata.

    Added columns:
      - material_system: pretty name like "Ag-Pd" or "Cr-Mn-Fe-Co-Ni-O-700C-Per1"
    """
    df = pd.read_csv(path)

    # Ensure "method" is present as a column
    if "method" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "method"})

    # Extract material system key from filename
    # e.g. "Ag_Au_Pd_Pt_Rh_material_system_analysis.csv"
    #  -> raw_key = "Ag_Au_Pd_Pt_Rh_material_system"
    fname = path.name
    if not fname.endswith("_analysis.csv"):
        raise ValueError(f"Unexpected analysis filename pattern: {fname}")
    raw_key = fname[:-len("_analysis.csv")]

    df["material_system"] = pretty_material_name(raw_key)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-material-system analysis CSVs into one master file."
    )
    parser.add_argument(
        "--analysis-dir",
        "-d",
        type=str,
        default="analysis_results",
        help="Directory containing per-material-system *_analysis.csv files (default: analysis_results).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="analysis_results/master_analysis.csv",
        help="Path to write the aggregated CSV (default: analysis_results/master_analysis.csv).",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_analysis_files(analysis_dir)
    if not files:
        print(f"No *_analysis.csv files found under {analysis_dir}")
        return

    dfs = []
    for p in files:
        try:
            df = load_and_annotate(p, analysis_dir)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to error: {e}")

    if not dfs:
        print("No valid analysis files could be loaded.")
        return

    master = pd.concat(dfs, ignore_index=True)

    # Drop per-element distribution columns (keep only global metrics)
    suffixes_to_drop = (
        "_orig_min",
        "_orig_q1",
        "_orig_median",
        "_orig_q3",
        "_orig_max",
        "_pareto_min",
        "_pareto_q1",
        "_pareto_median",
        "_pareto_q3",
        "_pareto_max",
    )
    drop_cols = [
        c
        for c in master.columns
        if any(c.endswith(suf) for suf in suffixes_to_drop)
    ]
    if drop_cols:
        master = master.drop(columns=drop_cols)

    # Optional: sort for nicer viewing
    if "material_system" in master.columns and "method" in master.columns:
        master = master.sort_values(["material_system", "method"])

    master.to_csv(out_path, index=False)
    print(f"[INFO] Wrote master analysis to {out_path}")


if __name__ == "__main__":
    main()