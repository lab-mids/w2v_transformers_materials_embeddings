import re
from pathlib import Path
from typing import Sequence, Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class MaterialSystemPlotter:
    """
    Plot material systems for multiple methods.

    Directory structure (per method_dir):
        method_dir/
            material_systems_with_similarities/
                <material_system>_... .csv or .pkl
            material_systems_pareto_front/
                <material_system>_... .csv (Pareto subset)

    Produces one PDF per material system into output_dir.
    """

    def __init__(
        self,
        method_dirs: Sequence[str],
        output_dir: str,
        grid_shape=(2, 3),
        dpi: int = 300,
    ):
        self.method_dirs = [Path(d) for d in method_dirs]
        self.output_dir = Path(output_dir)
        self.grid_shape = grid_shape
        self.dpi = dpi

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.system_index: Dict[str, Dict[str, Any]] = {}

    def run(self):
        self._build_index()
        self._plot_all_systems()

    def _build_index(self):
        for method_dir in self.method_dirs:
            method_name = method_dir.name

            sim_dir = method_dir / "material_systems_with_similarities"
            pareto_dir = method_dir / "material_systems_pareto_front"

            if not sim_dir.is_dir():
                continue

            sim_files = list(sim_dir.glob("*.csv")) + list(sim_dir.glob("*.pkl"))
            for sim_path in sim_files:
                key = self._material_system_key(sim_path.stem)
                entry = self.system_index.setdefault(key, {"reference": None, "methods": {}})
                methods_entry = entry["methods"].setdefault(method_name, {})
                methods_entry["similarity"] = sim_path
                if entry["reference"] is None:
                    entry["reference"] = sim_path

            if not pareto_dir.is_dir():
                continue

            for pareto_path in pareto_dir.glob("*.csv"):
                key = self._material_system_key(pareto_path.stem)
                entry = self.system_index.setdefault(key, {"reference": None, "methods": {}})
                methods_entry = entry["methods"].setdefault(method_name, {})
                methods_entry["pareto"] = pareto_path

    @staticmethod
    def _material_system_key(stem: str) -> str:
        m = re.match(r"(.+material_system)", stem)
        return m.group(1) if m else stem

    def _plot_all_systems(self):
        for system_key, info in sorted(self.system_index.items()):
            ref_path = info.get("reference")
            if ref_path is None:
                continue

            try:
                df_ref = self._load_dataframe(ref_path)
            except Exception as e:
                print(f"[WARN] Failed to load reference for {system_key}: {e}")
                continue

            xcol, ycol = self._find_xy_columns(df_ref)
            if xcol is None or ycol is None:
                print(f"[INFO] Skipping {system_key} (no x/y columns)")
                continue

            current_col = self._find_current_column(df_ref)
            if current_col is None:
                print(f"[INFO] Skipping {system_key} (no Current_at_* column)")
                continue

            self._plot_single_system(
                system_key=system_key,
                df_ref=df_ref,
                xcol=xcol,
                ycol=ycol,
                current_col=current_col,
                methods_info=info["methods"],
            )

    def _plot_single_system(
        self,
        system_key: str,
        df_ref: pd.DataFrame,
        xcol: str,
        ycol: str,
        current_col: str,
        methods_info: Dict[str, Dict[str, Path]],
    ):
        rows, cols = self.grid_shape
        n_slots = rows * cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.ravel()

        vmin = df_ref[current_col].min()
        vmax = df_ref[current_col].max()

        x_min = df_ref[xcol].min()
        x_max = df_ref[xcol].max()
        y_min = df_ref[ycol].min()
        y_max = df_ref[ycol].max()

        loaded_pareto: Dict[str, pd.DataFrame] = {}
        for method_name, paths in methods_info.items():
            pareto_path = paths.get("pareto")
            if pareto_path is None:
                continue
            try:
                df_p = pd.read_csv(pareto_path)
            except Exception as e:
                print(f"[WARN] Failed to load pareto for {system_key} / {method_name}: {e}")
                continue

            if xcol not in df_p.columns or ycol not in df_p.columns:
                continue
            if current_col not in df_p.columns:
                continue

            loaded_pareto[method_name] = df_p
            x_min = min(x_min, df_p[xcol].min())
            x_max = max(x_max, df_p[xcol].max())
            y_min = min(y_min, df_p[ycol].min())
            y_max = max(y_max, df_p[ycol].max())

        x_span = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
        y_span = (y_max - y_min) if (y_max - y_min) != 0 else 1.0

        span = max(x_span, y_span)
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        x_min_plot = x_center - span / 2.0
        x_max_plot = x_center + span / 2.0
        y_min_plot = y_center - span / 2.0
        y_max_plot = y_center + span / 2.0

        pad = 0.02 * span
        x_min_plot -= pad
        x_max_plot += pad
        y_min_plot -= pad
        y_max_plot += pad

        used_axes = []

        ax0 = axes[0]
        used_axes.append(ax0)
        sc = ax0.scatter(
            df_ref[xcol],
            df_ref[ycol],
            c=df_ref[current_col],
            s=10,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )
        ax0.set_title("Original distribution", fontsize=10)
        ax0.set_xlabel(xcol)
        ax0.set_ylabel(ycol)
        ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, va="top", ha="left", fontweight="bold")

        subplot_index = 1
        label_char_code = ord("b")

        for method_name in sorted(methods_info.keys()):
            if method_name not in loaded_pareto:
                continue
            if subplot_index >= n_slots:
                print(f"[INFO] {system_key}: more methods than subplots; ignoring extras.")
                break

            df_pareto = loaded_pareto[method_name]

            ax = axes[subplot_index]
            used_axes.append(ax)
            subplot_index += 1

            ax.scatter(df_ref[xcol], df_ref[ycol], color="lightgrey", s=5, alpha=0.2)

            ax.scatter(
                df_pareto[xcol],
                df_pareto[ycol],
                c=df_pareto[current_col],
                s=15,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.2,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_title(f"{method_name}\nPareto subset", fontsize=10)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)

            label = f"({chr(label_char_code)})"
            label_char_code += 1
            ax.text(0.02, 0.98, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold")

        for i in range(subplot_index, n_slots):
            axes[i].axis("off")

        for ax in used_axes:
            ax.set_xlim(x_min_plot, x_max_plot)
            ax.set_ylim(y_min_plot, y_max_plot)
            ax.set_aspect("equal", adjustable="box")

        fig.subplots_adjust(right=0.9)
        fig.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(current_col)

        safe_key = re.sub(r"[^\w\-]+", "_", system_key)
        out_path = self.output_dir / f"{safe_key}_comparison.pdf"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Saved figure for {system_key} -> {out_path}")

    @staticmethod
    def _load_dataframe(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".pkl":
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported file extension: {path}")

    @staticmethod
    def _find_xy_columns(df: pd.DataFrame) -> (Optional[str], Optional[str]):
        candidates_x = ["x", "pos_x", "x_pos", "x_position"]
        candidates_y = ["y", "pos_y", "y_pos", "y_position"]

        cols_lower = {c.lower(): c for c in df.columns}

        xcol = next((cols_lower[c] for c in candidates_x if c in cols_lower), None)
        ycol = next((cols_lower[c] for c in candidates_y if c in cols_lower), None)
        return xcol, ycol

    @staticmethod
    def _find_current_column(df: pd.DataFrame) -> Optional[str]:
        for c in df.columns:
            cl = c.lower()
            if cl.startswith("current_at") or "current_at_" in cl:
                return c
        return None


def main(method_dirs, output_dir, grid_shape, dpi, done_flag):
    plotter = MaterialSystemPlotter(
        method_dirs=method_dirs,
        output_dir=output_dir,
        grid_shape=grid_shape,
        dpi=dpi,
    )
    plotter.run()

    # write a done flag so Snakemake has a single output to track
    done_path = Path(done_flag)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text("OK\n", encoding="utf-8")


if __name__ == "__main__":
    if "snakemake" not in globals():
        raise RuntimeError("This script is intended to be run via Snakemake.")

    grid_rows = int(snakemake.params["grid_rows"])
    grid_cols = int(snakemake.params["grid_cols"])

    main(
        method_dirs=snakemake.params["method_dirs"],
        output_dir=snakemake.params["output_dir"],
        grid_shape=(grid_rows, grid_cols),
        dpi=int(snakemake.params["dpi"]),
        done_flag=snakemake.output["done"],
    )