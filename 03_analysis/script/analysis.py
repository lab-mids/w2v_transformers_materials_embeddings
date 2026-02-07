#!/usr/bin/env python
"""
Run multi-method similarity analysis from a config file.

Usage:
    python similarity_analysis.py --config config/similarity_analysis.yaml
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yaml

from pandas.api.types import is_numeric_dtype, is_integer_dtype


# ----------------------- Periodic table -----------------------

# All 118 element symbols (IUPAC)
ELEMENT_SYMBOLS: set[str] = {
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
}


# ----------------------- Method config -----------------------

@dataclass
class MethodConfig:
    """
    Configuration for one embedding / similarity method.

    Required:
      - name
      - root: directory with similarity files (non-Pareto)
      - pareto_root: directory with Pareto files

    Optional:
      - exts: set of file extensions to accept (".csv", ".pkl", etc.)
              If None, will be auto-detected from root/pareto_root.
    """
    name: str
    root: Path | str
    pareto_root: Path | str
    exts: Optional[set[str]] = None
    legend_group: Optional[str] = None

    root_path: Path = field(init=False)
    pareto_root_path: Path = field(init=False)

    def __post_init__(self):
        self.root_path = Path(self.root)
        self.pareto_root_path = Path(self.pareto_root)
        if self.legend_group is None:
            self.legend_group = self.name


# ----------------------- Analysis class -----------------------

class SimilarityAnalysisMultiMethod:
    """
    Analyze multiple methods (and their Pareto counterparts) per material system.

    For each 'key' (material system) we produce ONE analysis file in output_dir:
      - Each row = one method.
      - Only files whose filename CONTAINS "material_system" are used.
      - Only element columns (H, He, ..., Og) are summarized for box-plot stats.
    """

    def __init__(
        self,
        methods: List[MethodConfig],
        output_dir: str | Path,
    ):
        if not methods:
            raise ValueError("You must provide at least one MethodConfig in `methods`.")

        self.methods = self._prepare_methods(methods)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Fixed: group files by "...material_system" prefix
        self.prefix_re = re.compile(r"^(.*?material_system)")
        self.key_filter: Optional[re.Pattern] = None
        self.current_col: Optional[str] = None

        # Sanity check roots exist
        for m in self.methods:
            if not m.root_path.exists():
                raise FileNotFoundError(f"Root for {m.name} not found: {m.root_path}")
            if not m.pareto_root_path.exists():
                raise FileNotFoundError(f"Pareto root for {m.name} not found: {m.pareto_root_path}")

    # ----------------------- Methods prep & scanning -----------------------

    def _prepare_methods(self, methods: List[MethodConfig]) -> List[MethodConfig]:
        """
        Auto-detect extensions if not provided.
        """
        prepared: List[MethodConfig] = []
        for m in methods:
            if m.exts is None:
                exts = self._auto_detect_exts(m.root_path, m.pareto_root_path)
                if not exts:
                    exts = {".csv", ".pkl"}
                m.exts = exts
            prepared.append(m)
        return prepared

    def _auto_detect_exts(self, root: Path, pareto_root: Path) -> set[str]:
        exts: set[str] = set()
        for base in (root, pareto_root):
            if base.exists():
                for p in base.rglob("*"):
                    if p.is_file():
                        exts.add(p.suffix.lower())
        return exts

    def _scan_bucket(self, root: Path, exts: set[str]) -> Dict[str, List[Path]]:
        """
        Scan `root` for files with the given extensions, whose filenames
        contain "material_system".

        Key format:
          <subdir>/<basename_up_to_material_system>
        """
        out: Dict[str, List[Path]] = {}
        for p in root.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in exts
                and "material_system" in p.name
            ):
                try:
                    rel = p.relative_to(root)  # e.g. "MinDMaxC/Ag_...csv"
                except ValueError:
                    rel = Path(p.name)

                base_key = self._extract_key(p.stem)
                parent = rel.parent.as_posix() if isinstance(rel, Path) else "."

                if parent == ".":
                    key = base_key
                else:
                    key = f"{parent}/{base_key}"

                out.setdefault(key, []).append(p)

        return out

    def _extract_key(self, stem: str) -> str:
        """
        Extract a key from the stem using the material_system prefix.
        Fallback: drop the last underscore block.
        """
        m = self.prefix_re.search(stem)
        if m and m.group(1):
            return m.group(1)
        parts = stem.split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else stem

    # ----------------------- IO helpers -----------------------

    def _read_df(self, p: Path) -> Optional[pd.DataFrame]:
        try:
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p)
            obj = pd.read_pickle(p)
            if isinstance(obj, pd.DataFrame):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, pd.DataFrame):
                        return v
                return pd.DataFrame(obj)
            return pd.DataFrame(obj)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
            return None

    def _detect_current_col(self, df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty:
            return None
        if self.current_col and self.current_col in df.columns:
            return self.current_col
        for c in df.columns:
            if isinstance(c, str) and c.startswith("Current_at_"):
                return c
        return None

    # ----------------------- Metric helpers -----------------------

    def _pick_single_df(self, paths: List[Path]) -> Optional[pd.DataFrame]:
        if not paths:
            return None
        paths_sorted = sorted(paths)
        if len(paths_sorted) > 1:
            print(f"[WARN] Multiple files for same key/method; using first: {paths_sorted[0]}")
        return self._read_df(paths_sorted[0])

    def _metric_counts_and_best(
        self,
        df_sim: Optional[pd.DataFrame],
        df_par: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Compute:
          - original_count
          - pareto_count
          - current_col
          - best_current_original
          - best_current_pareto
          - best_current_rel_error_pct      (|best_par - best_orig| / |best_orig| * 100)
          - pareto_fraction_pct             (pareto_count / original_count * 100)

        All percentage values are rounded to 1 decimal place.
        """
        metrics: Dict[str, Any] = {}

        # basic counts
        metrics["original_count"] = int(df_sim.shape[0]) if df_sim is not None else 0
        metrics["pareto_count"] = int(df_par.shape[0]) if df_par is not None else 0

        # choose a current column: prefer similarity DF, fall back to Pareto DF
        cur_col = self._detect_current_col(df_sim) if df_sim is not None else None
        if cur_col is None and df_par is not None:
            cur_col = self._detect_current_col(df_par)
        metrics["current_col"] = cur_col

        def _best_val(df: Optional[pd.DataFrame]) -> float:
            if df is None or cur_col is None or cur_col not in df.columns:
                return np.nan
            s = pd.to_numeric(df[cur_col], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return np.nan
            idx = s.abs().idxmax()
            try:
                return float(s.loc[idx])
            except Exception:
                return np.nan

        best_orig = _best_val(df_sim)
        best_par = _best_val(df_par)

        metrics["best_current_original"] = best_orig
        metrics["best_current_pareto"] = best_par

        # helper to round percentages
        def _round_pct(x: float) -> float:
            if not isinstance(x, (int, float)) or np.isnan(x):
                return np.nan
            # 1 decimal place
            return float(np.round(x, 1))

        # relative percent error between Pareto best and original best
        if np.isnan(best_orig) or best_orig == 0:
            rel_err_pct = np.nan
        else:
            rel_err_pct = abs(best_par - best_orig) / abs(best_orig) * 100.0

        # percentage of candidates kept in Pareto vs original
        original_count = metrics["original_count"]
        pareto_count = metrics["pareto_count"]
        if original_count > 0:
            pareto_frac_pct = pareto_count / original_count * 100.0
        else:
            pareto_frac_pct = np.nan

        # store rounded percentages
        metrics["best_current_rel_error_pct"] = _round_pct(rel_err_pct)
        metrics["pareto_fraction_pct"] = _round_pct(pareto_frac_pct)

        return metrics

    def _metric_composition_summaries(
        self,
        df_sim: Optional[pd.DataFrame],
        df_par: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        For each element column (name exactly equals a valid element symbol),
        compute five-number summaries (min, Q1, median, Q3, max) for original
        and Pareto datasets.

        Additionally compute a global metric:
          - composition_iqr_narrowing_pct:
              average percentage narrowing of the interquartile ranges (Q3-Q1)
              across all elements with valid IQRs:
                  100 * mean(1 - IQR_pareto / IQR_orig)

              > 0  -> on average, element distributions are narrower in Pareto
              < 0  -> on average, element distributions became broader

        All percentage values are rounded to 1 decimal place.
        """
        metrics: Dict[str, Any] = {}

        def _element_numeric_cols(df: pd.DataFrame) -> List[str]:
            cols: List[str] = []
            for c in df.columns:
                if not isinstance(c, str):
                    continue
                if c not in ELEMENT_SYMBOLS:
                    continue
                try:
                    if pd.api.types.is_numeric_dtype(df[c]):
                        cols.append(c)
                except Exception:
                    continue
            return cols

        def _round_pct(x: float) -> float:
            if not isinstance(x, (int, float)) or np.isnan(x):
                return np.nan
            return float(np.round(x, 1))

        element_cols: set[str] = set()
        if df_sim is not None and not df_sim.empty:
            element_cols.update(_element_numeric_cols(df_sim))
        if df_par is not None and not df_par.empty:
            element_cols.update(_element_numeric_cols(df_par))

        if not element_cols:
            # no element composition info at all
            metrics["composition_iqr_narrowing_pct"] = np.nan
            return metrics

        def _five_number_stats(series: pd.Series) -> Dict[str, float]:
            s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return {
                    "min": np.nan,
                    "q1": np.nan,
                    "median": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                }
            q = s.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
            return {
                "min": float(q.loc[0.0]),
                "q1": float(q.loc[0.25]),
                "median": float(q.loc[0.5]),
                "q3": float(q.loc[0.75]),
                "max": float(q.loc[1.0]),
            }

        # collect per-element IQR shrink values
        shrink_values: List[float] = []

        for col in sorted(element_cols):
            # original dataset stats
            if df_sim is not None and col in df_sim.columns:
                stats_orig = _five_number_stats(df_sim[col])
            else:
                stats_orig = {
                    "min": np.nan,
                    "q1": np.nan,
                    "median": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                }

            metrics[f"{col}_orig_min"] = stats_orig["min"]
            metrics[f"{col}_orig_q1"] = stats_orig["q1"]
            metrics[f"{col}_orig_median"] = stats_orig["median"]
            metrics[f"{col}_orig_q3"] = stats_orig["q3"]
            metrics[f"{col}_orig_max"] = stats_orig["max"]

            # pareto dataset stats
            if df_par is not None and col in df_par.columns:
                stats_par = _five_number_stats(df_par[col])
            else:
                stats_par = {
                    "min": np.nan,
                    "q1": np.nan,
                    "median": np.nan,
                    "q3": np.nan,
                    "max": np.nan,
                }

            metrics[f"{col}_pareto_min"] = stats_par["min"]
            metrics[f"{col}_pareto_q1"] = stats_par["q1"]
            metrics[f"{col}_pareto_median"] = stats_par["median"]
            metrics[f"{col}_pareto_q3"] = stats_par["q3"]
            metrics[f"{col}_pareto_max"] = stats_par["max"]

            # compute IQR shrink for this element if possible
            q1_o, q3_o = stats_orig["q1"], stats_orig["q3"]
            q1_p, q3_p = stats_par["q1"], stats_par["q3"]

            if (
                not np.isnan(q1_o)
                and not np.isnan(q3_o)
                and not np.isnan(q1_p)
                and not np.isnan(q3_p)
            ):
                iqr_orig = q3_o - q1_o
                iqr_par = q3_p - q1_p
                if iqr_orig > 0:
                    shrink = 1.0 - (iqr_par / iqr_orig)
                    shrink_values.append(shrink)

        # aggregate over elements
        if shrink_values:
            avg_shrink_pct = float(np.mean(shrink_values) * 100.0)
            metrics["composition_iqr_narrowing_pct"] = _round_pct(avg_shrink_pct)
        else:
            metrics["composition_iqr_narrowing_pct"] = np.nan

        return metrics

    # ----------------------- Public API -----------------------

    def run(self) -> None:
        """
        Main entry: scans roots, computes metrics, writes per-key analysis CSV.
        """
        # method_maps[name] = {"config": MethodConfig, "base": {key: [Path]}, "pareto": {key: [Path]}}
        method_maps: Dict[str, Dict[str, Any]] = {}

        for m in self.methods:
            base_map = self._scan_bucket(m.root_path, m.exts)
            pareto_map = self._scan_bucket(m.pareto_root_path, m.exts)
            method_maps[m.name] = {
                "config": m,
                "base": base_map,
                "pareto": pareto_map,
            }

        # union of keys across all methods / base / pareto
        keys: List[str] = sorted({
            key
            for maps in method_maps.values()
            for kind in ("base", "pareto")
            for key in maps[kind].keys()
        })

        if self.key_filter:
            keys = [k for k in keys if self.key_filter.search(k)]
        if not keys:
            print("No matched keys found. Check roots / filename pattern.")
            return

        for key in keys:
            self._analyze_one_key(key, method_maps)

        print(f"Done. Analysis CSVs saved under: {self.output_dir}")

    # ----------------------- per-key analysis -----------------------

    def _analyze_one_key(self, key: str, method_maps: Dict[str, Dict[str, Any]]) -> None:
        # 1) Load per-method similarity & pareto DFs
        per_method_sim: Dict[str, Optional[pd.DataFrame]] = {}
        per_method_par: Dict[str, Optional[pd.DataFrame]] = {}

        for m in self.methods:
            maps = method_maps[m.name]
            sim_paths: List[Path] = maps["base"].get(key, [])
            par_paths: List[Path] = maps["pareto"].get(key, [])

            df_sim = self._pick_single_df(sim_paths)
            df_par = self._pick_single_df(par_paths)

            per_method_sim[m.name] = df_sim
            per_method_par[m.name] = df_par

        # 2) Per-method metrics
        per_method_metrics: Dict[str, Dict[str, Any]] = {}
        for m in self.methods:
            df_sim = per_method_sim[m.name]
            df_par = per_method_par[m.name]
            metrics = self._metric_counts_and_best(df_sim, df_par)
            comp_metrics = self._metric_composition_summaries(df_sim, df_par)
            metrics.update(comp_metrics)
            per_method_metrics[m.name] = metrics

        # 3) Build DataFrame (rows = methods)
        rows = []
        for m in self.methods:
            row = {"method": m.name}
            row.update(per_method_metrics[m.name])
            rows.append(row)
        df_out = pd.DataFrame(rows).set_index("method")

        # --- derive overpotential (mV) from current_col ---

        def _extract_overpotential(cur_name: Any) -> float:
            """
            Extract numeric overpotential from column name like 'Current_at_-300mV' -> -300.
            Returns NaN if pattern does not match.
            """
            if not isinstance(cur_name, str):
                return np.nan
            m = re.match(r"Current_at_(-?\d+\.?\d*)mV", cur_name)
            if not m:
                return np.nan
            try:
                return float(m.group(1))
            except ValueError:
                return np.nan

        if "current_col" in df_out.columns:
            df_out["overpotential (mV)"] = df_out["current_col"].apply(_extract_overpotential)
            # cast overpotential to integer mV (nullable Int64 so NaNs are allowed)
            df_out["overpotential (mV)"] = (
                pd.to_numeric(df_out["overpotential (mV)"], errors="coerce")
                  .round()
                  .astype("Int64")
            )
            # we no longer need the raw column name in the output
            df_out = df_out.drop(columns=["current_col"])

        # rename summary columns for nicer headers / units
        df_out = df_out.rename(
            columns={
                "original_count": "count/O",
                "pareto_count": "count/P",
                "best_current_original": "best/O (mA/cm^2)",
                "best_current_pareto": "best/P (mA/cm^2)",
                "best_current_rel_error_pct": "error (%)",
                "pareto_fraction_pct": "fraction (%)",
            }
        )

        # round all non-integer numeric columns to 1 decimal place
        for col in df_out.columns:
            s = df_out[col]
            if is_integer_dtype(s):
                # keep true integer columns (counts, overpotential) as they are
                continue
            if is_numeric_dtype(s):
                df_out[col] = s.round(1)

        # 4) Decide output path based on key structure
        key_path = Path(key)
        subdir = key_path.parent if key_path.parent.name != "" else Path(".")

        out_dir = self.output_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = key_path.name
        out_path = out_dir / f"{base_name}_analysis.csv"

        df_out.to_csv(out_path)
        print(f"[INFO] Wrote analysis for key '{key}' to {out_path}")


# ----------------------- CLI helpers -----------------------

def load_config(config_path: Path) -> dict:
    """
    Load YAML or JSON config file.
    """
    suffix = config_path.suffix.lower()
    with config_path.open("r") as f:
        if suffix in {".yml", ".yaml"}:
            return yaml.safe_load(f)
        elif suffix == ".json":
            return json.load(f)
        else:
            # default to YAML
            return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run similarity analysis for multiple methods.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML/JSON config file defining methods and output_dir.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    methods_cfg = cfg.get("methods", [])
    if not methods_cfg:
        raise ValueError("Config must contain a 'methods' list.")

    output_dir = cfg.get("output_dir", "analysis_results")

    methods: List[MethodConfig] = []
    for m in methods_cfg:
        methods.append(
            MethodConfig(
                name=m["name"],
                root=m["root"],
                pareto_root=m["pareto_root"],
                exts=set(m["exts"]) if m.get("exts") is not None else None,
            )
        )

    analyzer = SimilarityAnalysisMultiMethod(
        methods=methods,
        output_dir=output_dir,
    )
    analyzer.run()


if __name__ == "__main__":
    main()